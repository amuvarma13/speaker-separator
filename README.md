# speaker-separator

Minimal training scaffold for a Qwen3-4B model that consumes wav2vec2 latents projected through an MLP into Qwen's `inputs_embeds` at 12.5 Hz.

## Architecture

`SpeakerSeparator` subclasses `Qwen3ForCausalLM` (the LLM weights live on `self`) and adds:

- `self.w2v` — `Wav2Vec2Model` (50 Hz hidden states)
- `self.proj` — MLP that stacks every 4 frames (50 Hz → 12.5 Hz) and projects to `hidden_size`

Forward pass concatenates audio embeddings and text token embeddings along the time axis and feeds them to the Qwen backbone via `inputs_embeds`. Labels for audio positions are set to `-100`; CE loss is computed only over text tokens.

All modules (Qwen, wav2vec, MLP) are trainable.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
# flash-attn requires torch already installed; install last:
pip install flash-attn==2.6.3 --no-build-isolation
```

The LLM uses `attn_implementation="flash_attention_2"`.

## Data

Downloads ~50k rows (7 shards) of `parler-tts/mls_eng_10k`:

```bash
huggingface-cli login   # or: export HF_TOKEN=...
NUM_SHARDS=7 DATA_DIR=./data python download_data.py
```

## Train (multi-GPU FSDP)

```bash
HF_TOKEN=... ./run.sh
```

Equivalent direct invocation:

```bash
accelerate launch --config_file fsdp.yaml --num_processes 2 train.py
```

Hyperparameters are env vars (`BSZ`, `GRAD_ACCUM`, `LR`, `MAX_STEPS`, `MAX_AUDIO_S`, `MAX_TEXT_TOKENS`, `OUT_DIR`, ...).

## Files

- `model.py` — `SpeakerSeparator(Qwen3ForCausalLM)`
- `train.py` — accelerate + FSDP training loop
- `fsdp.yaml` — accelerate FSDP config
- `download_data.py` — pulls a few MLS-Eng-10k shards
- `run.sh` — convenience launcher
