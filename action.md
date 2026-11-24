# FlickCD Server Actions

Server checkout: `/home/xxj/code/DINOCLIP_comparative_test/FlickCD`  
Dataset root: `/home/xxj/Data`

## Pull reproduce branch

First time:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/FlickCD
git fetch origin
git switch --track -c reproduce origin/reproduce
```

Later updates:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/FlickCD
git switch reproduce
git pull --ff-only
```

## Environment and optional pretrained weights

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/FlickCD
bash setup_server_env.sh
conda activate flickcd-rs
bash prepare_server_data.sh /home/xxj/Data
```

Training uses `model/repvit_m0_6.pth` when present. To train without downloading it, set `PRETRAINED=none`; the model then initializes the RepViT backbone randomly. This is a scratch-trained configuration, not the ImageNet-pretrained setup. To use the pretrained setup, download the official RepViT-M0.6 checkpoint from the [THU-MIG RepViT repository](https://github.com/THU-MIG/RepViT). Evaluation checkpoints remain excluded from Git; train a dataset first, then test with that run's output.

```bash
mkdir -p model
wget -O model/repvit_m0_6.pth https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m0_6_distill_300e.pth
```

## Train

```bash
bash run_train.sh levircd
bash run_train.sh levircdplus
bash run_train.sh cdd
bash run_train.sh whucd
bash run_train.sh sysucd
```

The wrappers use dataset settings: SYSU 100 epochs/lr `5e-4`, WHU 100/lr `2e-4`, LEVIR-CD+ 200/lr `5e-4`, and CDD 250/lr `5e-4`; LEVIR-CD uses the LEVIR+ window setup. Set `EPOCHS`, `BATCH_SIZE`, `NUM_WORKERS`, `GPU_ID`, and `PATCH_SIZE` to override runtime values.

LEVIR-CD and LEVIR-CD-PLUS are tiled into aligned 256x256 patches while reading for training, validation, and testing. Test predictions are stitched into each original image canvas before pixel metrics are computed. No crop files are created and source data is unchanged. Large inputs must be divisible by `PATCH_SIZE`.

Training validates once per epoch and retains only the top three validation-F1 models: `best_model.pth`, `best_model_2.pth`, and `best_model_3.pth`. Ranking persists on resume. One `checkpoint.pth.tar` is overwritten for resuming. At completion FlickCD tests `best_model.pth`. `GPU_ID` chooses a physical GPU; the process uses logical `cuda:0` after restricting visibility.

For standalone testing, run:

```bash
bash run_test.sh levircd
bash run_test.sh levircdplus
bash run_test.sh cdd
bash run_test.sh whucd
bash run_test.sh sysucd
```

For LEVIR-CD, train first; the repository has no bundled LEVIR-CD checkpoint. Pass a checkpoint explicitly when needed, for example `bash run_test.sh levircd FlickCD_work_dirs/levircd/FlickCD/best_model.pth`.

Test batches default to 4; adjust with `TEST_BATCH_SIZE` if GPU memory is limited.
