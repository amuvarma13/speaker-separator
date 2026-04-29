import os
import torch
import numpy as np
from datasets import load_dataset, Audio
from transformers import AutoTokenizer, Trainer, TrainingArguments, Wav2Vec2Model

from model import SpeakerSeparator


SR = 16000
LLM_NAME = os.environ.get("LLM_NAME", "InternalCan/4b-11nodes-c771-1fullepoch-0-a6c76f-2026-01-26-03-13-08")
TOK_NAME = os.environ.get("TOK_NAME", LLM_NAME)
W2V_NAME = os.environ.get("W2V_NAME", "facebook/wav2vec2-base")
FREEZE_W2V = int(os.environ.get("FREEZE_W2V", "1"))
DATA_DIR = os.environ.get("DATA_DIR", "./data_smoke")
DELAY = int(os.environ.get("DELAY", "0"))
BSZ = int(os.environ.get("BSZ", "1"))
GA = int(os.environ.get("GA", "8"))
LR = float(os.environ.get("LR", "5e-5"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "5000"))
WARMUP = int(os.environ.get("WARMUP", "100"))
SEED = int(os.environ.get("SEED", "0"))
MAX_AUDIO_S = float(os.environ.get("MAX_AUDIO_S", "12"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
LIMIT = int(os.environ.get("LIMIT", "0"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "5000"))
RUN_TAG = os.environ.get("RUN_TAG", "")

TOK_LEN = 151936
START_SPEECH = TOK_LEN + 1
END_SPEECH = TOK_LEN + 2
START_HUMAN = TOK_LEN + 3
END_HUMAN = TOK_LEN + 4
START_AI = TOK_LEN + 5
END_AI = TOK_LEN + 6
PAD = TOK_LEN + 7
START_TEXT = TOK_LEN + 10
END_TEXT = TOK_LEN + 11
AUDIO_START = TOK_LEN + 12

CB = ["semantic_codes", "cb_0", "cb_1", "cb_2", "cb_3", "cb_4", "cb_5", "cb_6"]
OFFS = [0, 16384, 16384 + 4096, 16384 + 2*4096, 16384 + 3*4096, 16384 + 4*4096, 16384 + 5*4096, 16384 + 6*4096]
NEW_VOCAB = AUDIO_START + 16384 + 7 * 4096


def shards():
    d = os.path.join(DATA_DIR, "data")
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".parquet"))


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds
        self.half = len(ds) // 2

    def __len__(self):
        return self.half

    def __getitem__(self, idx):
        return (self.ds[idx], self.ds[idx + self.half])


class Collator:
    def __init__(self, tok, delay):
        self.tok = tok
        self.delay = delay

    def __call__(self, batch):
        audios, idses, masks, labelses = [], [], [], []
        for r1, r2 in batch:
            wav1 = np.asarray(r1["audio"]["array"], dtype=np.float32)
            wav2 = np.asarray(r2["audio"]["array"], dtype=np.float32)
            n_max = int(MAX_AUDIO_S * SR)
            if wav1.shape[0] > n_max:
                wav1 = wav1[:n_max]
            if wav2.shape[0] > n_max:
                wav2 = wav2[:n_max]
            n = max(wav1.shape[0], wav2.shape[0])
            mix = np.zeros(n, dtype=np.float32)
            mix[:wav1.shape[0]] += wav1
            mix[:wav2.shape[0]] += wav2
            n_w2v = (mix.shape[0] // 320) // 4
            n_codes = min(len(r1["semantic_codes"]), len(r2["semantic_codes"]), n_w2v)
            prefix = [START_HUMAN, START_TEXT, END_TEXT, END_HUMAN, START_AI, START_SPEECH]
            seq = list(prefix)
            mask = [False] * len(prefix)
            for t in range(n_codes + self.delay):
                if 0 <= t < n_codes:
                    seq.append(PAD)
                    mask.append(True)
                ct = t - self.delay
                if 0 <= ct < n_codes:
                    for r in (r1, r2):
                        for i, name in enumerate(CB):
                            c = r[name][ct]
                            if c == -1:
                                continue
                            seq.append(c + OFFS[i] + AUDIO_START)
                            mask.append(False)
            seq.extend([END_SPEECH, END_AI])
            mask.extend([False, False])
            lbl = [-100 if m else s for s, m in zip(seq, mask)]
            audios.append(mix)
            idses.append(seq)
            masks.append(mask)
            labelses.append(lbl)
        aT = max(w.shape[0] for w in audios)
        audio = np.zeros((len(batch), aT), dtype=np.float32)
        for i, w in enumerate(audios):
            audio[i, :w.shape[0]] = w
        L = max(len(s) for s in idses)
        ids = np.full((len(batch), L), PAD, dtype=np.int64)
        am = np.zeros((len(batch), L), dtype=np.int64)
        m = np.zeros((len(batch), L), dtype=bool)
        labs = np.full((len(batch), L), -100, dtype=np.int64)
        for i in range(len(batch)):
            ids[i, :len(idses[i])] = idses[i]
            am[i, :len(idses[i])] = 1
            m[i, :len(masks[i])] = masks[i]
            labs[i, :len(labelses[i])] = labelses[i]
        return {
            "audio": torch.from_numpy(audio).to(torch.bfloat16),
            "input_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(am),
            "latent_mask": torch.from_numpy(m),
            "labels": torch.from_numpy(labs),
        }


def main():
    try:
        tok = AutoTokenizer.from_pretrained(TOK_NAME)
    except (OSError, EnvironmentError):
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    ds = load_dataset("parquet", data_files=shards(), split="train")
    if LIMIT > 0:
        ds = ds.select(range(min(LIMIT, len(ds))))
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))
    ds = PairDataset(ds)

    model = SpeakerSeparator.from_pretrained(
        LLM_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        w2v_name=W2V_NAME,
    )
    w2v_pre = Wav2Vec2Model.from_pretrained(W2V_NAME, torch_dtype=torch.bfloat16)
    model.w2v.load_state_dict(w2v_pre.state_dict())
    del w2v_pre
    if model.get_input_embeddings().weight.shape[0] < NEW_VOCAB:
        model.resize_token_embeddings(NEW_VOCAB)
    if FREEZE_W2V:
        for p in model.w2v.parameters():
            p.requires_grad = False

    tag = LLM_NAME.split("/")[-1][:24]
    run = f"sep_d{DELAY}_bs{BSZ}x{GA}_lr{LR:.0e}_{tag}" + (f"_{RUN_TAG}" if RUN_TAG else "")
    args = TrainingArguments(
        output_dir=f"./checkpoints/{run}",
        per_device_train_batch_size=BSZ,
        gradient_accumulation_steps=GA,
        learning_rate=LR,
        max_steps=MAX_STEPS,
        warmup_steps=WARMUP,
        logging_steps=1,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=True,
        dataloader_num_workers=NUM_WORKERS,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=run,
        seed=SEED,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=Collator(tok, DELAY),
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
