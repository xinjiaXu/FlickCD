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

## Environment and pretrained weights

```bash
bash setup_server_env.sh
conda activate flickcd-rs
bash prepare_server_data.sh /home/xxj/Data
```

The RepViT encoder checkpoint `model/repvit_m0_6.pth` is included in the repository. No separate full-model weight download is required for training. Four released evaluation checkpoints are included under `checkpoint/` for SYSU, WHU, LEVIR-CD+, and CDD.

## Train

```bash
bash run_train.sh levircd
bash run_train.sh levircdplus
bash run_train.sh cdd
bash run_train.sh whucd
bash run_train.sh sysucd
```

The wrappers use the values in the repository's `trainAndtest.sh` examples: SYSU 100 epochs/lr `5e-4`, WHU 100/lr `2e-4`, LEVIR-CD+ 200/lr `5e-4`, and CDD 250/lr `5e-4`. LEVIR-CD uses the LEVIR+ model-window setup because there is no separate LEVIR-CD entry in the source. Other defaults are batch 32, input 256, weight decay `1e-2`, seed 2333; workers default to 8. `GPU_ID=1` selects GPU 1.

The train command evaluates test after training. For standalone testing, run:

```bash
bash run_test.sh levircd
bash run_test.sh levircdplus
bash run_test.sh cdd
bash run_test.sh whucd
bash run_test.sh sysucd
```

For LEVIR-CD, train first; the repository has no bundled LEVIR-CD checkpoint. Pass a checkpoint explicitly when needed, for example `bash run_test.sh levircd FlickCD_work_dirs/levircd/FlickCD/best_model.pth`.
