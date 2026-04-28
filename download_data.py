import os
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ThreadPoolExecutor

REPO = os.environ.get("REPO", "InternalCan/10k_hrs_audio_with_tokens_small_example_ds")
OUT = os.environ.get("DATA_DIR", "./data_internalcan")
WORKERS = int(os.environ.get("WORKERS", "16"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "0"))


def main():
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files = sorted(
        f for f in api.list_repo_files(REPO, repo_type="dataset")
        if f.startswith("data/") and f.endswith(".parquet")
    )
    if NUM_SHARDS > 0:
        files = files[:NUM_SHARDS]
    print(f"downloading {len(files)} shards from {REPO} to {OUT}")

    def _get(path):
        return hf_hub_download(
            repo_id=REPO,
            repo_type="dataset",
            filename=path,
            local_dir=OUT,
            token=os.environ.get("HF_TOKEN"),
        )

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(_get, files))
    print("done")


if __name__ == "__main__":
    main()
