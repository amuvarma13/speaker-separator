# speaker-separator

Minimal multimodal training scaffold for `Qwen/Qwen3-4B` that interleaves wav2vec2 latents (12.5 Hz) with audio codec token frames (8 codebooks) via an MLP projector into Qwen's `inputs_embeds`.

## Architecture

`SpeakerSeparator` subclasses `Qwen3ForCausalLM` directly (the LLM weights live on `self`, no submodule wrapping) and adds:

- `self.w2v` — `Wav2Vec2Model` (50 Hz hidden states)
- `self.proj` — 2-layer GELU MLP that stacks every 4 frames (50 Hz → 12.5 Hz) and projects to Qwen's `hidden_size`

The forward pass:

1. Encodes audio waveform → wav2vec → 12.5 Hz latents → MLP-project to Qwen hidden dim
2. Embeds the input token sequence (special tokens, text tokens, **PAD placeholders at latent positions**, audio codec tokens)
3. `masked_scatter`s the projected latents into the placeholder positions
4. Forwards through the Qwen backbone with `flash_attention_2`

All modules (Qwen, wav2vec, MLP) are trainable. No gradient checkpointing.

## Sequence layout

For every audio clip the collator builds:

```
[start_human, start_text, ...text_ids..., end_text, end_human, start_ai, start_speech,
   <interleaved: latent + 8 codes per frame, with optional delay>,
 end_speech, end_ai]
```

Each frame is 1 wav2vec latent followed by 8 codec tokens (`semantic_codes`, `cb_0..cb_6`). With `DELAY=D`, codes lag latents by `D` frames: the first `D` positions are latent-only, the last `D` positions are codes-only. Total length per audio = 1·N latents + 8·N codes = `9N` audio elements.

Audio token IDs are `code + offset_per_codebook + audio_tokens_start`, where offsets follow `[0, 16384, 16384+4096, ...]` (semantic codebook is 16384 entries, residual codebooks are 4096 each). `audio_tokens_start = 151948` (after the 11 special tokens).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
```

## Data

Datasets used (private, `InternalCan`):

- `InternalCan/10k_hrs_audio_with_tokens_small_example_ds` — small example (~5k rows) for development
- `InternalCan/10k_hrs_audio_with_tokens` — full dataset (use once uploaded)

Schema per row: `audio` (raw bytes), `text`, `semantic_codes`, `cb_0..cb_6` (12.5 Hz integer code lists).

Download via `huggingface_hub.snapshot_download` (multi-worker):

```bash
export HF_TOKEN=...
REPO=InternalCan/10k_hrs_audio_with_tokens_small_example_ds DATA_DIR=./data_internalcan WORKERS=16 \
  python download_data.py
```

A polling helper is provided for the bigger dataset:

```bash
./poll_big_dataset.sh &   # appends to poll_big.log every 10 min
```

## Train

```bash
./run.sh
```

Internally:

```bash
WANDB_API_KEY=... accelerate launch --num_processes 2 train.py
```

Hyperparameters are environment variables. Key ones:

| var          | default               | meaning                                              |
|--------------|-----------------------|------------------------------------------------------|
| `LLM_NAME`   | `Qwen/Qwen3-4B`       | base LLM                                             |
| `W2V_NAME`   | `facebook/wav2vec2-base` | audio encoder                                     |
| `DATA_DIR`   | `./data_internalcan`  | parquet shards directory                             |
| `DELAY`      | `0`                   | code-frame delay (in 12.5 Hz frames) vs wav2vec      |
| `BSZ`        | `1`                   | per-device batch size                                |
| `GA`         | `8`                   | gradient accumulation                                |
| `LR`         | `5e-5`                | peak learning rate                                   |
| `MAX_STEPS`  | `5000`                | total optimizer steps                                |
| `MAX_AUDIO_S`| `12`                  | clip audio to N seconds                              |
| `LIMIT`      | `0`                   | if >0, restrict dataset to first N rows (debug)      |

The W&B run name encodes `delay`, `bsz`, `grad_accum`, `lr`, and the LLM. Default project: `speaker-separator-pretrain`.

## Files

- `model.py` — `SpeakerSeparator(Qwen3ForCausalLM)`
- `train.py` — minimal HF Trainer + custom collator (interleave with delay)
- `download_data.py` — `snapshot_download` for any HF dataset
- `poll_big_dataset.sh` — polls the larger dataset until shards appear
- `run.sh` — `accelerate launch` wrapper
