# FlickCD server setup

FlickCD reads the existing dataset directories directly under `/home/xxj/Data`. It does not copy images, labels, or list files. Outputs are stored under `FlickCD_work_dirs/<dataset>/`.

## Environment

From `/home/xxj/code/DINOCLIP_comparative_test/FlickCD`:

```bash
bash setup_server_env.sh
conda activate flickcd-rs
```

The README asks for Python 3.8 and `timm==0.5.4`. The server setup uses Python 3.10 and PyTorch 2.7.1/CUDA 12.8 for the RTX 5090; `requirements_server.txt` keeps the required timm version and pins NumPy/OpenCV to compatible wheels. Before training, download the official RepViT-M0.6 ImageNet checkpoint to the ignored path expected by FlickCD:

```bash
wget -O model/repvit_m0_6.pth https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m0_6_distill_300e.pth
```

The `.pth` file is excluded from Git and must be fetched separately after cloning.

## Data lists and folder layout

The code will use `train.txt`, `val.txt`, and `test.txt` when they are beside their split folders. For LEVIR-CD+ it also checks one directory above `cut_data`. If a list is absent, the code falls back to a sorted list of filenames in that split's label folder. It never creates or modifies the dataset lists.

| Key | Dataset root | Image A | Image B | Label | Model window name |
| --- | --- | --- | --- | --- | --- |
| `levircd` | `/home/xxj/Data/LEVIR_CD` | `A` | `B` | `label` | `LEVIR+` |
| `levircdplus` | `/home/xxj/Data/LEVIR_CD_PLUS/cut_data` | `A` | `B` | `label` | `LEVIR+` |
| `cdd` | `/home/xxj/Data/CDD` | `A` | `B` | `OUT` | `CDD` |
| `whucd` | `/home/xxj/Data/WHUCD` | `A` | `B` | `label` | `WHU` |
| `sysucd` | `/home/xxj/Data/SYSU_CD` | `time1` | `time2` | `label` | `SYSU` |

The original code has window settings for SYSU, WHU, LEVIR+ and CDD, but no separate LEVIR-CD setting. Since both LEVIR datasets use 256x256 patches, `levircd` uses the existing LEVIR+ window setting without changing the model implementation.

Check that the five server roots are present:

```bash
bash prepare_server_data.sh /home/xxj/Data
```

## Train

```bash
bash run_train.sh levircd
bash run_train.sh levircdplus
bash run_train.sh cdd
bash run_train.sh whucd
bash run_train.sh sysucd
```

The wrappers use dataset settings: SYSU 100 epochs/lr `5e-4`, WHU 100/lr `2e-4`, LEVIR+ 200/lr `5e-4`, and CDD 250/lr `5e-4`. LEVIR-CD uses LEVIR+ window settings. Other defaults are batch size 32, input size 256, weight decay `1e-2`, and seed 2333. Override with `EPOCHS`, `BATCH_SIZE`, `NUM_WORKERS`, `GPU_ID`, or `PATCH_SIZE`.

LEVIR-CD and LEVIR-CD-PLUS are tiled into aligned 256x256 patches on read. Test predictions are stitched into each original image canvas before metrics are calculated. No crop files are generated and source files remain unchanged; dimensions larger than `PATCH_SIZE` must be divisible by it.

Training validates once per epoch and retains the top three models by validation F1 (`best_model.pth`, `best_model_2.pth`, and `best_model_3.pth`). The ranking is included in the resume checkpoint. One `checkpoint.pth.tar` is overwritten each epoch. Training tests `best_model.pth` at completion. `GPU_ID` selects a physical GPU and is mapped to logical `cuda:0`.

## Test

```bash
bash run_test.sh whucd
```

For SYSU, WHU, LEVIR-CD+, and CDD, the wrapper uses the repository's bundled `checkpoint/FlickCD_<dataset>/best_model.pth` if present. For a trained checkpoint or another file, pass its path:

```bash
bash run_test.sh levircd FlickCD_work_dirs/levircd/FlickCD/best_model.pth
```

The test command reports the model's precision, recall, OA, F1, IoU, and kappa metrics.

Test batch defaults to 4. Set `TEST_BATCH_SIZE` to lower it if memory is constrained.
