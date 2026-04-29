import os
import argparse
import torch
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer, Qwen3ForCausalLM
from model import SpeakerSeparator


SR = 16000
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
OFFS = [0, 16384, 16384 + 4096, 16384 + 2*4096, 16384 + 3*4096, 16384 + 4*4096, 16384 + 5*4096, 16384 + 6*4096]
CB_SIZES = [16384, 4096, 4096, 4096, 4096, 4096, 4096, 4096]


def load_audio(path, target_sr=SR):
    wav, file_sr = sf.read(path, always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if file_sr != target_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=target_sr)
    return wav


def sample_token(logits, temperature, top_k):
    if temperature == 0:
        return int(logits.argmax().item())
    if top_k > 0:
        k = min(top_k, logits.shape[-1])
        topv, _ = torch.topk(logits, k)
        thresh = topv[-1]
        logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
    if temperature != 1.0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate(model, tok, wav, text="", delay=2, max_audio_s=12.0, temperature=0.0, top_k=0, device="cuda"):
    text_ids = tok.encode(text, add_special_tokens=False) if text else []
    prefix = [START_HUMAN, START_TEXT, *text_ids, END_TEXT, END_HUMAN, START_AI, START_SPEECH]

    n_max = int(max_audio_s * SR)
    if wav.shape[0] > n_max:
        wav = wav[:n_max]
    audio = torch.from_numpy(np.asarray(wav, dtype=np.float32)).to(torch.bfloat16).unsqueeze(0).to(device)

    a = model.encode_audio(audio)
    n_frames = a.shape[1]

    embed = model.get_input_embeddings()
    embeds = embed(torch.tensor([prefix], device=device))

    def fwd():
        out = Qwen3ForCausalLM.forward(
            model,
            inputs_embeds=embeds,
            use_cache=False,
            return_dict=True,
        )
        return out.logits[0, -1].float()

    schedule = []
    for t in range(n_frames + delay):
        if 0 <= t < n_frames:
            schedule.append(("latent", t))
        ct = t - delay
        if 0 <= ct < n_frames:
            for i in range(len(CB_SIZES)):
                schedule.append(("code", (ct, i)))

    predicted = [[None] * len(CB_SIZES) for _ in range(n_frames)]

    for kind, val in schedule:
        if kind == "latent":
            embeds = torch.cat([embeds, a[:, val:val + 1]], dim=1)
        else:
            ct, i = val
            logits = fwd()
            lo = OFFS[i] + AUDIO_START
            hi = lo + CB_SIZES[i]
            code = sample_token(logits[lo:hi], temperature, top_k)
            predicted[ct][i] = code
            token_id = code + lo
            embeds = torch.cat([embeds, embed(torch.tensor([[token_id]], device=device))], dim=1)

    return np.asarray(predicted, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--text", default="")
    ap.add_argument("--w2v", default="facebook/wav2vec2-base")
    ap.add_argument("--tok", default=None)
    ap.add_argument("--max_audio_s", type=float, default=12.0)
    ap.add_argument("--delay", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok_name = args.tok or args.ckpt
    try:
        tok = AutoTokenizer.from_pretrained(tok_name)
    except (OSError, EnvironmentError):
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    model = SpeakerSeparator.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        w2v_name=args.w2v,
    ).to(args.device).eval()

    wav = load_audio(args.audio, target_sr=SR)
    print(f"audio: {wav.shape[0]} samples ({wav.shape[0]/SR:.2f}s @ {SR} Hz)")

    codes = generate(
        model, tok, wav,
        text=args.text,
        delay=args.delay,
        max_audio_s=args.max_audio_s,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )

    print(f"frames={codes.shape[0]} codebooks={codes.shape[1]}")
    print("first 5 frames:")
    print(codes[:5])
    print("last 5 frames:")
    print(codes[-5:])
    if args.out:
        np.save(args.out, codes)
        print(f"saved -> {args.out} shape={codes.shape} dtype={codes.dtype}")


if __name__ == "__main__":
    main()
