#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${DATA_ROOT:-/home/xxj/Data}"
gpu_id="${GPU_ID:-0}"
num_workers="${NUM_WORKERS:-8}"

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s {levircd|levircdplus|cdd|whucd|sysucd}\n' "$0" >&2
  exit 2
fi

case "$1" in
  levircd) dataset_root="$data_root/LEVIR_CD"; data_name="LEVIR+"; run_lr="5e-4"; run_epochs=200; image_a="A"; image_b="B"; label_dir="label" ;;
  levircdplus) dataset_root="$data_root/LEVIR_CD_PLUS/cut_data"; data_name="LEVIR+"; run_lr="5e-4"; run_epochs=200; image_a="A"; image_b="B"; label_dir="label" ;;
  cdd) dataset_root="$data_root/CDD"; data_name="CDD"; run_lr="5e-4"; run_epochs=250; image_a="A"; image_b="B"; label_dir="OUT" ;;
  whucd) dataset_root="$data_root/WHUCD"; data_name="WHU"; run_lr="2e-4"; run_epochs=100; image_a="A"; image_b="B"; label_dir="label" ;;
  sysucd) dataset_root="$data_root/SYSU_CD"; data_name="SYSU"; run_lr="5e-4"; run_epochs=100; image_a="time1"; image_b="time2"; label_dir="label" ;;
  *) printf 'Unknown dataset key: %s\n' "$1" >&2; exit 2 ;;
esac

for split in train val test; do
  for folder in "$image_a" "$image_b" "$label_dir"; do
    if [[ ! -d "$dataset_root/$split/$folder" ]]; then
      printf 'Dataset directory missing: %s\nRun prepare_server_data.sh first or set DATA_ROOT.\n' "$dataset_root/$split/$folder" >&2
      exit 1
    fi
  done
done

if [[ ! -f "$repo_root/model/repvit_m0_6.pth" ]]; then
  printf 'Bundled RepViT pretrained weights are missing: %s\n' "$repo_root/model/repvit_m0_6.pth" >&2
  exit 1
fi

work_dir="$repo_root/FlickCD_work_dirs/$1"
mkdir -p "$work_dir"
cd "$repo_root"
export CUDA_VISIBLE_DEVICES="$gpu_id"
python train.py \
  --title FlickCD \
  --data_name "$data_name" \
  --mode train \
  --train_dataset_path "$dataset_root/train" \
  --val_dataset_path "$dataset_root/val" \
  --test_dataset_path "$dataset_root/test" \
  --image_dir_a "$image_a" \
  --image_dir_b "$image_b" \
  --label_dir "$label_dir" \
  --savedir "$work_dir" \
  --gpu_id "$gpu_id" \
  --num_workers "$num_workers" \
  --learning_rate "$run_lr" \
  --epochs "$run_epochs"
