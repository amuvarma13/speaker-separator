import os
from huggingface_hub import hf_hub_download

REPO = "parler-tts/mls_eng_10k"
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "7"))
OUT = os.environ.get("DATA_DIR", "./data")


def main():
    os.makedirs(OUT, exist_ok=True)
    for i in range(NUM_SHARDS):
        name = f"data/train-{i:05d}-of-00317.parquet"
        p = hf_hub_download(repo_id=REPO, filename=name, repo_type="dataset", local_dir=OUT)
        print(p)


if __name__ == "__main__":
    main()
