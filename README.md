# DEHRL

## Requirements

In order to install requirements, follow:

```bash

# create env
conda create -n ehrl

# source in env
source ~/.bashrc
source activate ehrl

conda install pytorch torchvision -c soumith
pip install opencv-contrib-python
conda install scikit-image
pip install --upgrade imutils
```

Run commands here:

To run shared policy without reward_bounty:
```bash
source ~/.bashrc
source activate ehrl
```

## OverCooked

### Level 1

Goal-type: any (works, done)
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp delta_model --obs-type 'image' --env-name "OverCooked" --reward-level 1 --setup-goal any --new-overcooked --num-hierarchy 2 --num-subpolicy 5 --hierarchy-interval 4 --num-steps 128 128 --reward-bounty 0.1875 --distance mass_center --transition-model-mini-batch-size 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux delta_model_r_0
```

Goal-type: fix (works, done)
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp delta_model --obs-type 'image' --env-name "OverCooked" --reward-level 1 --setup-goal fix --new-overcooked --num-hierarchy 2 --num-subpolicy 5 --hierarchy-interval 4 --num-steps 128 128 --reward-bounty 0.1875 --distance mass_center --transition-model-mini-batch-size 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux delta_model_r_0
```

Goal-type: random (works, done)
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp delta_model --obs-type 'image' --env-name "OverCooked" --reward-level 1 --setup-goal random --new-overcooked --num-hierarchy 2 --num-subpolicy 5 --hierarchy-interval 4 --num-steps 128 128 --reward-bounty 0.1875 --distance mass_center --transition-model-mini-batch-size 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux delta_model_r_0
```

### Level 2

Goal-type: any (logic verified, running)
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp delta_model --obs-type 'image' --env-name "OverCooked" --reward-level 2 --setup-goal any --new-overcooked --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 4 12 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux delta_model_r_0
```

Goal-type: fix (logic verified, running)
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp delta_model --obs-type 'image' --env-name "OverCooked" --reward-level 2 --setup-goal fix --new-overcooked --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 4 12 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux delta_model_r_0
```

Goal-type: random (works, running)
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp delta_model --obs-type 'image' --env-name "OverCooked" --reward-level 2 --setup-goal random --new-overcooked --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 4 12 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux delta_model_r_0
```

## MineCraft

```bash
CUDA_VISIBLE_DEVICES=1 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 1 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp add_minecraft_new --obs-type 'image' --env-name "MineCraft" --num-hierarchy 3 --num-subpolicy 8 8 --hierarchy-interval 4 4 --num-steps 128 128 128 --reward-bounty 1 --distance l1 --transition-model-mini-batch-size 64 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 10 --aux delta_model_r_0
```
