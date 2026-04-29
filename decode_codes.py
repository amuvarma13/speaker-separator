import os
import sys
import argparse
import numpy as np
import torch
import torchaudio
import safetensors.torch
import soundfile as sf

ORPHEUS_DIR = os.environ.get("ORPHEUS_DIR", "/workspace/orpheus-v1-inference/inference")
sys.path.insert(0, ORPHEUS_DIR)

import dualcodec
from utils import batch_decode
from decoder.decoder import Decoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--codec_weights", default="/workspace/speaker-separator/codec_model/model.safetensors")
    ap.add_argument("--codec_id", default="12hz_v1")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    arr = np.load(args.codes)
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError(f"expected (T, 8) int code matrix, got shape {arr.shape}")
    T = arr.shape[0]
    print(f"loaded codes: T={T} codebooks=8 dtype={arr.dtype}")

    semantic = torch.from_numpy(arr[:, 0:1].T.astype(np.int64)).unsqueeze(0).to(args.device)
    acoustic = torch.from_numpy(arr[:, 1:8].T.astype(np.int64)).unsqueeze(0).to(args.device)
    print(f"semantic: {semantic.shape}  acoustic: {acoustic.shape}")

    print(f"loading dualcodec model: {args.codec_id}")
    model = dualcodec.get_model(args.codec_id)
    model.dac.decoder = Decoder()
    print(f"loading codec weights: {args.codec_weights}")
    safetensors.torch.load_model(model, args.codec_weights, strict=False)
    model = model.to(args.device).eval()

    resampler = torchaudio.transforms.Resample(48000, 44100).to(args.device)

    print("decoding...")
    with torch.no_grad():
        audio = batch_decode(semantic, acoustic, model, resampler)
    wav = audio.squeeze().detach().cpu().float().numpy()
    print(f"wav shape: {wav.shape}  dur: {wav.shape[-1]/48000:.2f}s @ 48kHz")

    sf.write(args.out, wav, 48000)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
