#!/usr/bin/env bash
TOKEN="${HF_TOKEN:?set HF_TOKEN}"
REPO="${REPO:-InternalCan/10k_hrs_audio_with_tokens}"
LOG=/workspace/speaker-separator/poll_big.log
while true; do
  N=$(curl -s -H "Authorization: Bearer $TOKEN" "https://huggingface.co/api/datasets/$REPO/tree/main/data" 2>/dev/null | python3 -c "import sys,json
try:
  d=json.load(sys.stdin)
  print(len([x for x in d if x.get('path','').endswith('.parquet')]))
except Exception:
  print(-1)" 2>/dev/null)
  echo "[$(date '+%F %T')] big dataset has $N parquet files in data/" >> $LOG
  if [ "$N" -ge "5" ] 2>/dev/null; then
    echo "[$(date '+%F %T')] big dataset has >=5 shards, exiting poll" >> $LOG
    break
  fi
  sleep 600
done
