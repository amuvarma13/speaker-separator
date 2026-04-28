# speaker-separator

Audio-conditioned LLM training: a `Qwen/Qwen3-4B`-derivative that consumes a speech waveform via `facebook/wav2vec2-base` latents and predicts an interleaved stream of audio-codec tokens (`semantic_codes` + `cb_0..cb_6`) at 12.5 Hz, with an optional `delay_frames` lag between the audio condition and the codes.

The repo is intentionally minimal — one ~50-line `model.py`, one ~150-line `train.py`, one `download_data.py`, a one-liner `run.sh`, and an `accelerate launch` invocation — patterned after [`canopy-labs-internal/Face-LLM-Trainer/Face/finetune.py`](https://github.com/canopy-labs-internal/Face-LLM-Trainer).

---

## Motivation

We want a single Qwen3-4B-shaped model that can:

1. **Listen** to raw 16 kHz speech (wav2vec2 base feature extractor → 50 Hz hidden states → stack-by-4 → 12.5 Hz → MLP → Qwen hidden dim).
2. **Emit / predict** the speaker's codec tokens — 8 codebook frames per 12.5 Hz step (1 semantic + 7 residuals from a 16384/4096 codec) — in the same Qwen vocabulary.
3. **Operate causally** on a single concatenated stream: text prompt, then per-frame `[wav2vec_latent, 8 codes]`, with the codes optionally delayed by `delay_frames` so the model can attend to a chunk of acoustic context before predicting the corresponding codes.

This is the scaffold for "speaker-separator"-style tasks (and more generally for any audio-conditioned LLM head over codec tokens). The acoustic encoder, the projector, and the LLM are all unified in one `Qwen3ForCausalLM` subclass; nothing is added as a wrapper module.

We deliberately **do not use gradient checkpointing**, **do not micromanage FSDP**, and **do not write a custom optimizer or scheduler** — HF Trainer defaults are excellent for this scale and the H200 GPUs have enough memory to run 4 B params + frozen wav2vec at reasonable batch sizes purely under DDP.

---

## Architecture (`model.py`)

```python
class SpeakerSeparator(Qwen3ForCausalLM):
    def __init__(self, config, w2v_name, down=4):
        super().__init__(config)
        self.w2v  = Wav2Vec2Model(Wav2Vec2Config.from_pretrained(w2v_name))
        self.proj = MLP(w2v_hidden * down → qwen_hidden → qwen_hidden)

    def forward(self, audio, input_ids, latent_mask, labels):
        a = self.encode_audio(audio)             # (B, F, qwen_hidden)
        e = self.get_input_embeddings()(input_ids)
        e = e.masked_scatter(latent_mask, a)     # latents into placeholder slots
        return super().forward(inputs_embeds=e, labels=labels)
```

- The LLM **is** the model (`SpeakerSeparator` *is* a `Qwen3ForCausalLM`); wav2vec and the MLP are submodules.
- Wav2vec output at 50 Hz is grouped into 4-frame stacks → 12.5 Hz, then projected to Qwen's hidden size.
- The collator emits the `input_ids` with a placeholder token at each "this position is a wav2vec latent" slot and a boolean `latent_mask`. The model substitutes the projected latents into those slots via `masked_scatter`.
- Labels at latent slots are `-100` (no LM loss); labels at code/text slots are the actual token IDs.
- `attn_implementation="flash_attention_2"` on the Qwen backbone.
- No gradient checkpointing anywhere.

## Sequence layout (`train.py` collator)

For each row we emit:

```
[start_human, start_text,  …text_ids…, end_text, end_human,
 start_ai, start_speech,
   < interleaved frames, with delay D, total length 9·N >
 end_speech, end_ai]
```

The interleaved block, with `N = min(len(codes), wav2vec_frames)`:

| step `t` ∈ | content |
|---|---|
| `[0, D)` | latent only |
| `[D, N)` | `latent`, `code_{t-D}_0 … code_{t-D}_7` |
| `[N, N+D)` | `code_{t-D}_0 … code_{t-D}_7` only |

Audio token IDs use per-codebook offsets:

```
audio_token = code + offset[i] + audio_tokens_start
audio_tokens_start = 151948
offset = [0, 16384, 16384+4096, 16384+2·4096, …, 16384+6·4096]
```

(Special control tokens — `start_text`, `end_speech`, …, the latent placeholder — sit between 151937 and 151947, which the InternalCan base model already provides in vocabulary.)

---

## Setup

```bash
git clone https://github.com/amuvarma13/speaker-separator.git
cd speaker-separator
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
```

Tested on Linux + 2× NVIDIA H200 (sm_90), CUDA 12.4, Python 3.11, torch 2.4.1+cu121, transformers 4.51.3, accelerate 1.0.1, flash-attn 2.6.3.

---

## Data

Two HF datasets (private, `InternalCan` org, gated by your `HF_TOKEN`):

| dataset | size | when to use |
|---|---|---|
| `InternalCan/10k_hrs_audio_with_tokens_small_example_ds` | ~5 k rows / 4.2 GB / 2 shards | development / smoke tests |
| `InternalCan/10k_hrs_audio_with_tokens` | ~7.9 M rows / **~1.5 TB** / 3170 shards | full pretraining |

The full training dataset is approximately **1.5 TB** on disk — comfortably within the few-TB ballpark and far below the ~369 TB free on `/workspace`.

Schema per row: `audio` (raw bytes), `text`, `semantic_codes`, `cb_0..cb_6` (12.5 Hz `int32` lists), `duration`, plus metadata.

Multi-worker download via `huggingface_hub.hf_hub_download` (one parquet shard per worker):

```bash
export HF_TOKEN=...
# small example (full)
REPO=InternalCan/10k_hrs_audio_with_tokens_small_example_ds \
DATA_DIR=./data_internalcan WORKERS=16 python download_data.py

# big dataset (a slice, e.g. first 10 shards)
REPO=InternalCan/10k_hrs_audio_with_tokens \
DATA_DIR=./data_internalcan_full NUM_SHARDS=10 WORKERS=32 \
python download_data.py
```

Everything writes to `/workspace/...` so the small local `/` (~1 TB) never fills up. The shared `/workspace` mount has hundreds of TB free.

---

## Base LLM

Default: **`InternalCan/4b-11nodes-c771-1fullepoch-0-a6c76f-2026-01-26-03-13-08`** — a Qwen3-4B continuation pre-trained with the audio-codebook embeddings already learnt. Vocabulary is already `197226`, so we **do not** call `resize_token_embeddings` (it's skipped automatically when the loaded model already has enough vocab).

By default we **freeze wav2vec2** (`FREEZE_W2V=1`); only the LLM and the MLP projector are trainable, since the LLM has strong codebook priors but no wav2vec alignment.

To start from raw `Qwen/Qwen3-4B` instead:

```bash
LLM_NAME=Qwen/Qwen3-4B FREEZE_W2V=0 ./run.sh
```

`resize_token_embeddings(197004)` will run automatically because the vanilla Qwen3 vocab is too small.

---

## Training

```bash
export HF_TOKEN=...
export WANDB_API_KEY=...
./run.sh
```

That expands to:

```bash
accelerate launch --num_processes 2 --main_process_port <random> train.py
```

The Trainer takes care of FP precision (`bf16=True`), DDP, data sampler, optimiser (`adamw_torch`), scheduler (linear with warmup), gradient clipping, checkpoint rotation (`save_total_limit=2`), and W&B logging.

### Recommended progression

1. **bs=1, 200 steps** — verifies the whole forward/backward path on the new architecture.
   ```bash
   BSZ=1 GA=1 MAX_STEPS=200 LIMIT=2000 DELAY=2 MAX_AUDIO_S=10 ./run.sh
   ```
2. **bs=2, 50 steps** — verifies multi-sample collation + DDP grad sync.
   ```bash
   BSZ=2 MAX_STEPS=50 ./run.sh
   ```
3. **bs=4, 30 steps** — current best on bare Qwen3-4B + everything trainable.
4. **bs=8 or higher** with the InternalCan base + `FREEZE_W2V=1` — the recommended throughput operating point once you trust the script.

### Hyperparameters (env vars)

| var | default | meaning |
|---|---|---|
| `LLM_NAME` | `InternalCan/4b-11nodes-…` | base LLM (Qwen3-4B-shaped) |
| `W2V_NAME` | `facebook/wav2vec2-base` | audio encoder |
| `FREEZE_W2V` | `1` | freeze wav2vec params |
| `DATA_DIR` | `./data_internalcan` | parquet shard root (`<dir>/data/*.parquet`) |
| `LIMIT` | `0` | if > 0, restrict to first N rows of the dataset |
| `DELAY` | `0` | code-frame lag in 12.5 Hz frames |
| `BSZ` | `1` | per-device batch size |
| `GA` | `8` | gradient accumulation |
| `LR` | `5e-5` | peak learning rate |
| `WARMUP` | `100` | warmup steps |
| `MAX_STEPS` | `5000` | total optimiser steps |
| `MAX_AUDIO_S` | `12` | clip audio to N seconds |
| `SAVE_STEPS` | `5000` | checkpoint cadence (~3 h at bs≈8) |
| `NUM_WORKERS` | `2` | DataLoader workers per process |
| `RUN_TAG` | `""` | extra suffix on the W&B run name |
| `WANDB_PROJECT` | `speaker-separator-pretrain` | W&B project |
| `WANDB_API_KEY` | *(required)* | W&B credentials |
| `HF_TOKEN` | *(required)* | HF credentials |

W&B run name format: `d{DELAY}_bs{BSZ}x{GA}_lr{LR}_{LLM_TAG}` (+ `_{RUN_TAG}` if set). Project: <https://wandb.ai/canopy-labs/speaker-separator-pretrain>.

---

## Verification (measured throughput)

`Qwen/Qwen3-4B`, all params trainable, bf16, flash-attn 2, `MAX_AUDIO_S=10`, `DELAY=2`, 2× H200:

| config | steady it/s | global samples/s | mem / GPU | shards / GPU-hour | audio-hours / GPU-hour |
|---|---|---|---|---|---|
| bs=1, ga=1 | ~4.0 | ~8 | 54 GB | ~6.0 | ~18.8 |
| bs=2, ga=1 | ~2.9 | ~12 | 58 GB | ~8.3 | ~26.0 |
| bs=4, ga=1 | ~1.85 | ~15 | 92 GB | ~10.6 | ~33.5 |

Loss came down from 19 → 10 in 200 bs=1 steps with everything trainable, confirming the audio-token embeddings + projector were learning. With the InternalCan base + frozen wav2vec the start loss is much lower because codebook embeddings are already pretrained.

---

## Disk space and checkpointing

| mount | size | free | role |
|---|---|---|---|
| `/workspace` (network) | 755 TB | ~369 TB | datasets, checkpoints, HF cache |
| `/` (local overlay) | ~1 TB | ~972 GB | venv only |

`run.sh` exports `HF_HOME=/workspace/.hf_cache`, `HF_DATASETS_CACHE=/workspace/.hf_cache/datasets`, `TRANSFORMERS_CACHE=/workspace/.hf_cache/transformers` so nothing dataset- or weight-shaped lands on `/`.

Per checkpoint we write the full bf16 model (~8 GB) plus optimizer state plus tokenizer. With `save_total_limit=2` the on-disk ceiling is ~60 GB per run. The full ~1.5 TB training dataset fits on `/workspace` hundreds of times over.

`SAVE_STEPS=5000` is calibrated to roughly **one checkpoint every ~3 hours** at the recommended operating point (bs≈8, frozen w2v).

---

## Files

- `model.py` — `SpeakerSeparator(Qwen3ForCausalLM)` (no comments, no docstrings, no main).
- `train.py` — minimal HF `Trainer` + a custom collator that builds the interleaved sequence with `delay_frames`.
- `download_data.py` — multi-thread `hf_hub_download` for any HF dataset; supports `REPO`, `NUM_SHARDS`, `WORKERS`.
- `run.sh` — `accelerate launch` wrapper that exports the env vars and a random `--main_process_port`.
- `requirements.txt` — pinned versions of torch / transformers / accelerate / flash-attn / wandb / hf_xet / etc.

---

## Original instructions, verbatim

Every requirement that shaped this repository, in chronological order. Each `>` block is a verbatim user prompt; my notes follow.

### Round 1 — initial scaffold

> create a new repo push to akv13 github call it speaker separator - its essentially going to be a training script. HEre is what you need to do:
>
> 1. Set up venv && requirements ensure you use flash attention
> 2. Set up a minimal multigpu training script for qwen3 4b using accelerate launch
> 3. Set up a minimal training script to project wav2vec latents through an mlp to 12.5Hz latents in the input_embed of qwen
> 4. Download a few shard - 50k rows of text to speech from mls-eng-10k from huggingface do not download the whole dataset just a few shards and train using fsdp (you should be subclassing pretrainedmodel i.e. qwen 4b not adding it as a module - train all modules and use fsdp and flash_attn on the llm part
> 5. remove all comments main function doc strings make the model.py minimal make the training script minimal
>
> gh token - `<provided>`
> hf_key - `<provided>`
>
> do not ask for help or questions - i am leaving to run and you must accomplish all these tasks without asking for clarification completely independently

(GitHub PAT belongs to `amuvarma13`, so the repo lives at `amuvarma13/speaker-separator` rather than `akv13/speaker-separator`.)

> for the fsdp we should try to just use autowrap not a specific config

> we shouldnt need to explicitly call accelerateor

> get rid of fsdp.yaml the only fsdp setting in training args should be auto_wrap

> dont use an iterable dataset use load_dataset explicitly remember its only 50k rows

> remove all use of accelerator in the training script and use default optimiser scheduler and hf trainer to simplify things

(That sequence ended with HF `Trainer` + DDP defaults, no `fsdp.yaml`. FSDP was later dropped entirely once the new model + frozen wav2vec made it unnecessary on H200s.)

### Round 2 — switch dataset and architecture

> okay you will build a new training script:
>
> 1. `InternalCan/10k_hrs_audio_with_tokens_small_example_ds` download this dataset and investigate its structure
> 2. You will notice there are audio and codes semantic cb0 cb1 etc up. to 6
> 3. We will essentiall build a training script which interleaves wav2vec latents and the frames of codes
> 4. The codes are also at 12.5 Hz and correspond to the whole audio like wav2vec after we stack and mlp them
> 5. We will be adding offsets to each of the codes in the following way
>     ```python
>     offsets = [0, 16384, 16384+4096, 16384+2*4096, 16384+3*4096, 16384+4*4096, 16384+5*4096, 16384+6*4096]
>     audio_tokens_start = 151948   # = tokeniser_length(151936) + 12
>     audio_token = code + offset[cb_i] + audio_tokens_start
>     ```
>    (with `start_of_*` / `end_of_*` / `pad` / `start_of_text` / `end_of_text` at `tokeniser_length + 1..11`)
> 6. We also want a parameter called delay frames - where we pass have a delay between wav2vec frames and code frames - where the code frames are delayed by "delay frames"
> 7. Do not use gradient checkpointing ever - write a script where you get this working on a batch size of 1 and then once that can train for 200 steps get it working on a batch size of more than 1
> 8. Do not ask me any questions and carry out this task completely independently
> 9. The dataset I shared earlier is a test to help you get the script working - a larger dataset is currently uploading - `InternalCan/10k_hrs_audio_with_tokens` poll it every 5-10 minutes to see if its uploaded and ultimately once and only once your small dataset script is working download this dataset and use it.
> 7. (sic) You must always use accelerate launch and load_dataset or snapshot download on multiple workers to download the dataset even if it takes a bit of time
>
> Follow all these steps independently without looking to me for guidance

> Unlike this script we are doing latent frame of 8 codes latent frame of 8 codes etc
> Check that we are with +- the matching number of code frames and stacked and mlped wav2vec latents.

### Round 3 — style, observability, README

> 1. for code style minimalism and amount of comments/non standard config - we want to minimise this `https://github.com/canopy-labs-internal/Face-LLM-Trainer.git` this repo has `https://github.com/canopy-labs-internal/Face-LLM-Trainer/blob/main/Face/finetune.py` this script which is similarish to what we want but look at the minimalism and taking advantage of as much of the hf stuff on default setting
> 2. also link up training to wandb `wandb login <key>` and make sure delay and hyper params are in the run name pick a sensible project name
> 3. You can create separate tasks to download the entire set of data for either dataset but get the script working with even smaller subsets of data
> 4. Push to github with sensible commit messages after important points including with readmes and include a comprehensive summary of the exact instructions you were given in constructing this repo
>
> once again do not ask me questions and conduct this task autonomously!

### Round 4 — base-model swap

> `InternalCan/4b-11nodes-c771-1fullepoch-0-a6c76f-2026-01-26-03-13-08` - you can use this instead of base qwen - it has a very strong prior for codebook emebddings but not wav2vec latents - instead unfreeze the llm and the projector consider increasing the batch size to see if it speeds up training also with this model you dont need to resize token embeddings probably.

### Round 5 — checkpointing cadence and disk

> addditionally choose number of steps to save so that you save checkpoits every 3 hours roughly (close enough is fine) and check that we have enough disk space

> how much disk space do we have left will we fit theentire new dataset/.

### Round 6 — final docs request

> Can you write a detailed readme with instructions with how to get training started our motivations what we are looking to accomplish and also can you include a section with all my prompts and instructions and push to github
