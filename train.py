import os
import io
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import DataLoader, IterableDataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import AutoTokenizer, Wav2Vec2Model, get_cosine_schedule_with_warmup

from model import SpeakerSeparator


LLM_NAME = os.environ.get("LLM_NAME", "Qwen/Qwen3-4B")
W2V_NAME = os.environ.get("W2V_NAME", "facebook/wav2vec2-base")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
SR = 16000
MAX_AUDIO_S = float(os.environ.get("MAX_AUDIO_S", "12"))
MAX_TEXT_TOKENS = int(os.environ.get("MAX_TEXT_TOKENS", "256"))
BSZ = int(os.environ.get("BSZ", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "8"))
LR = float(os.environ.get("LR", "1e-4"))
WARMUP = int(os.environ.get("WARMUP", "100"))
SEED = int(os.environ.get("SEED", "0"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "5000"))
OUT_DIR = os.environ.get("OUT_DIR", "./checkpoints")
LOG_EVERY = int(os.environ.get("LOG_EVERY", "10"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))


def shards():
    d = os.path.join(DATA_DIR, "data")
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.startswith("train-") and f.endswith(".parquet"))


def decode(row, tok):
    a = row["audio"]
    if isinstance(a, dict) and "bytes" in a and a["bytes"] is not None:
        wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
    else:
        wav = np.asarray(a["array"], dtype=np.float32)
        sr = a["sampling_rate"]
    if wav.ndim > 1:
        wav = wav.mean(-1)
    if sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR).astype(np.float32)
    n = int(MAX_AUDIO_S * SR)
    if wav.shape[0] > n:
        wav = wav[:n]
    ids = tok(row["transcript"], add_special_tokens=False, truncation=True, max_length=MAX_TEXT_TOKENS - 1).input_ids + [tok.eos_token_id]
    return wav, ids


class Stream(IterableDataset):
    def __init__(self, tok, rank, world):
        self.tok = tok
        self.rank = rank
        self.world = world

    def __iter__(self):
        ds = load_dataset("parquet", data_files=shards(), split="train", streaming=True)
        ds = split_dataset_by_node(ds, rank=self.rank, world_size=self.world)
        ds = ds.shuffle(seed=SEED + self.rank, buffer_size=512)
        for row in ds:
            if row.get("audio_duration") and row["audio_duration"] > MAX_AUDIO_S:
                continue
            try:
                wav, ids = decode(row, self.tok)
            except Exception:
                continue
            if wav.shape[0] < SR // 2 or len(ids) < 2:
                continue
            yield wav, ids


def collate(batch, pad_id):
    wavs, ids = zip(*batch)
    aT = max(w.shape[0] for w in wavs)
    audio = np.zeros((len(batch), aT), dtype=np.float32)
    amask = np.zeros((len(batch), aT), dtype=np.int64)
    for i, w in enumerate(wavs):
        audio[i, :w.shape[0]] = w
        amask[i, :w.shape[0]] = 1
    tT = max(len(x) for x in ids)
    input_ids = np.full((len(batch), tT), pad_id, dtype=np.int64)
    tmask = np.zeros((len(batch), tT), dtype=np.int64)
    labels = np.full((len(batch), tT), -100, dtype=np.int64)
    for i, x in enumerate(ids):
        input_ids[i, :len(x)] = x
        tmask[i, :len(x)] = 1
        labels[i, :len(x)] = x
    return {
        "audio": torch.from_numpy(audio),
        "audio_mask": torch.from_numpy(amask),
        "input_ids": torch.from_numpy(input_ids),
        "attention_mask": torch.from_numpy(tmask),
        "labels": torch.from_numpy(labels),
    }


def main():
    set_seed(SEED)
    acc = Accelerator(gradient_accumulation_steps=GRAD_ACCUM, mixed_precision="bf16")
    tok = AutoTokenizer.from_pretrained(LLM_NAME)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = SpeakerSeparator.from_pretrained(
        LLM_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        w2v_name=W2V_NAME,
    )
    w2v_pre = Wav2Vec2Model.from_pretrained(W2V_NAME, torch_dtype=torch.bfloat16)
    model.w2v.load_state_dict(w2v_pre.state_dict())
    del w2v_pre
    model.gradient_checkpointing_enable()

    ds = Stream(tok, acc.process_index, acc.num_processes)
    loader = DataLoader(ds, batch_size=BSZ, num_workers=NUM_WORKERS, collate_fn=lambda b: collate(b, tok.pad_token_id))

    nd = ("bias", "layer_norm.weight", "norm.weight", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in nd)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in nd)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95))
    sch = get_cosine_schedule_with_warmup(opt, WARMUP, MAX_STEPS)

    model, opt, loader, sch = acc.prepare(model, opt, loader, sch)
    model.train()

    step = 0
    micro = 0
    running = 0.0
    while step < MAX_STEPS:
        for batch in loader:
            with acc.accumulate(model):
                out = model(
                    audio=batch["audio"].to(torch.bfloat16),
                    audio_mask=batch["audio_mask"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                acc.backward(out.loss)
                if acc.sync_gradients:
                    acc.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad()
            running += out.loss.detach().float().item()
            micro += 1
            if acc.sync_gradients:
                step += 1
                if step % LOG_EVERY == 0 and acc.is_main_process:
                    print(f"step {step} loss {running/max(micro,1):.4f} lr {sch.get_last_lr()[0]:.2e}", flush=True)
                    running = 0.0
                    micro = 0
                if step >= MAX_STEPS:
                    break

    acc.wait_for_everyone()
    if acc.is_main_process:
        os.makedirs(OUT_DIR, exist_ok=True)
    unwrapped = acc.unwrap_model(model)
    state = acc.get_state_dict(model)
    if acc.is_main_process:
        unwrapped.save_pretrained(OUT_DIR, state_dict=state, safe_serialization=True)
        tok.save_pretrained(OUT_DIR)


if __name__ == "__main__":
    main()
