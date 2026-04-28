import os
from huggingface_hub import snapshot_download

REPO = os.environ.get("REPO", "InternalCan/10k_hrs_audio_with_tokens_small_example_ds")
OUT = os.environ.get("DATA_DIR", "./data_internalcan")
WORKERS = int(os.environ.get("WORKERS", "16"))


def main():
    os.makedirs(OUT, exist_ok=True)
    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=OUT,
        max_workers=WORKERS,
        allow_patterns=["data/*.parquet", "README.md"],
    )
    print("done", OUT)


if __name__ == "__main__":
    main()
