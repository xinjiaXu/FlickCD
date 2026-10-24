# FlickCD server setup

FlickCD reads the existing dataset directories directly under `/home/xxj/Data`. It does not copy images, labels, or list files. Outputs are stored under `FlickCD_work_dirs/<dataset>/`.

## Environment

From `/home/xxj/code/DINOCLIP_comparative_test/FlickCD`:

```bash
bash setup_server_env.sh
conda activate flickcd-rs
```

The README asks for Python 3.8 and `timm==0.5.4`. The server setup uses Python 3.10 and PyTorch 2.7.1/CUDA 12.8 for the RTX 5090; `requirements_server.txt` keeps the required timm version and pins NumPy/OpenCV to compatible wheels. Training uses the repository-bundled `model/repvit_m0_6.pth` encoder weights.

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

The wrappers use the per-dataset values shown in the upstream `trainAndtest.sh` examples: SYSU 100 epochs/lr `5e-4`, WHU 100/lr `2e-4`, LEVIR+ 200/lr `5e-4`, and CDD 250/lr `5e-4`. LEVIR-CD uses the LEVIR+ values as a 256x256 dataset not listed in the upstream examples. Other defaults stay as in `train.py`: batch size 32, input size 256, weight decay `1e-2`, and seed 2333. Workers default to 8 on the server; choose a different GPU with `GPU_ID=1`.

Training evaluates validation data during the run and tests the best-F1 checkpoint at the end. It saves `best_model.pth` and resumable `checkpoint.pth.tar` in `FlickCD_work_dirs/<key>/FlickCD/`. A later training run resumes that checkpoint automatically when present.

## Test

```bash
bash run_test.sh whucd
```

For SYSU, WHU, LEVIR-CD+, and CDD, the wrapper uses the repository's bundled `checkpoint/FlickCD_<dataset>/best_model.pth` if present. For a trained checkpoint or another file, pass its path:

```bash
bash run_test.sh levircd FlickCD_work_dirs/levircd/FlickCD/best_model.pth
```

The test command reports the model's precision, recall, OA, F1, IoU, and kappa metrics.
