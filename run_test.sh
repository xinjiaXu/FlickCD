#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${DATA_ROOT:-/home/xxj/Data}"
gpu_id="${GPU_ID:-0}"
num_workers="${NUM_WORKERS:-8}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'Usage: %s {levircd|levircdplus|cdd|whucd|sysucd} [CHECKPOINT]\n' "$0" >&2
  exit 2
fi

case "$1" in
  levircd) dataset_root="$data_root/LEVIR_CD"; data_name="LEVIR+"; image_a="A"; image_b="B"; label_dir="label"; official_tag="" ;;
  levircdplus) dataset_root="$data_root/LEVIR_CD_PLUS/cut_data"; data_name="LEVIR+"; image_a="A"; image_b="B"; label_dir="label"; official_tag="LEVIR-CD+" ;;
  cdd) dataset_root="$data_root/CDD"; data_name="CDD"; image_a="A"; image_b="B"; label_dir="OUT"; official_tag="CDD" ;;
  whucd) dataset_root="$data_root/WHUCD"; data_name="WHU"; image_a="A"; image_b="B"; label_dir="label"; official_tag="WHU" ;;
  sysucd) dataset_root="$data_root/SYSU_CD"; data_name="SYSU"; image_a="time1"; image_b="time2"; label_dir="label"; official_tag="SYSU" ;;
  *) printf 'Unknown dataset key: %s\n' "$1" >&2; exit 2 ;;
esac

for split in train val test; do
  for folder in "$image_a" "$image_b" "$label_dir"; do
    if [[ ! -d "$dataset_root/$split/$folder" ]]; then
      printf 'Dataset directory missing: %s\n' "$dataset_root/$split/$folder" >&2
      exit 1
    fi
  done
done

work_dir="$repo_root/FlickCD_work_dirs/$1"
if [[ $# -eq 2 ]]; then
  checkpoint="$(realpath "$2")"
elif [[ -f "$work_dir/FlickCD/best_model.pth" ]]; then
  checkpoint="$work_dir/FlickCD/best_model.pth"
elif [[ -n "$official_tag" && -f "$repo_root/checkpoint/FlickCD_$official_tag/best_model.pth" ]]; then
  checkpoint="$repo_root/checkpoint/FlickCD_$official_tag/best_model.pth"
else
  printf 'No test checkpoint found. Pass CHECKPOINT explicitly.\n' >&2
  exit 1
fi

cd "$repo_root"
export CUDA_VISIBLE_DEVICES="$gpu_id"
python train.py \
  --title FlickCD \
  --data_name "$data_name" \
  --mode test \
  --train_dataset_path "$dataset_root/train" \
  --val_dataset_path "$dataset_root/val" \
  --test_dataset_path "$dataset_root/test" \
  --image_dir_a "$image_a" \
  --image_dir_b "$image_b" \
  --label_dir "$label_dir" \
  --savedir "$work_dir" \
  --gpu_id "$gpu_id" \
  --num_workers "$num_workers" \
  --test_checkpoint "$checkpoint"
