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

Goal-type: any (works, done)
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp delta_model --obs-type 'image' --env-name "OverCooked" --reward-level 2 --setup-goal any --new-overcooked --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 4 12 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux delta_model_r_0
```

Goal-type: fix (works, done)
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp delta_model --obs-type 'image' --env-name "OverCooked" --reward-level 2 --setup-goal fix --new-overcooked --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 4 12 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux delta_model_r_0
```

Goal-type: random (works, done)
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp delta_model --obs-type 'image' --env-name "OverCooked" --reward-level 2 --setup-goal random --new-overcooked --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 4 12 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux delta_model_r_0
```

## MineCraft

Number of level: 2
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 1 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp minecraft_build_more_kinds_of_blocks --obs-type 'image' --env-name "MineCraft" --num-hierarchy 2 --num-subpolicy 8 --hierarchy-interval 4 --num-steps 128 128 --reward-bounty 1 --distance l1 --transition-model-mini-batch-size 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 10 --aux r_2
```

Number of level: 4
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 1 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp minecraft_build_more_kinds_of_blocks --obs-type 'image' --env-name "MineCraft" --num-hierarchy 4 --num-subpolicy 8 8 8 --hierarchy-interval 4 4 4 --num-steps 128 128 128 128 --reward-bounty 1 --distance l1 --transition-model-mini-batch-size 64 64 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 10 --aux r_2
```

Number of level: 5
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 1 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp minecraft_build_more_kinds_of_blocks --obs-type 'image' --env-name "MineCraft" --num-hierarchy 5 --num-subpolicy 8 8 8 8 --hierarchy-interval 4 4 4 4 --num-steps 128 128 128 128 128 --reward-bounty 1 --distance l1 --transition-model-mini-batch-size 64 64 64 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 10 --aux r_2
```

## Atari

### Game: MontezumaRevengeNoFrameskip-v4

2 levels (best, also last run, no enemy, get down and close the the key)
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp delta_model --obs-type 'image' --env-name "MontezumaRevengeNoFrameskip-v4" --num-hierarchy 2 --num-subpolicy 5 --hierarchy-interval 4 --num-steps 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux delta_model_r_0
```

3 levels (best run: have enemy, try to get down for one time)
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp delta_model --obs-type 'image' --env-name "MontezumaRevengeNoFrameskip-v4" --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 4 4 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux delta_model_r_0
```

4 levels (a few very lone episode where agent can get to the bottom and move around, but why there is not a enemy down at the bottom)
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp delta_model --obs-type 'image' --env-name "MontezumaRevengeNoFrameskip-v4" --num-hierarchy 4 --num-subpolicy 5 5 5 --hierarchy-interval 4 4 4 --num-steps 128 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux delta_model_r_0
```

#### Extrinsic reward is added to intrinsic reward at every level, called combine reward in aux info.

2 levels (running w5-0)
First see the agent jump over the monster: [link](https://www.dropbox.com/s/smhjguxx0reeocl/level-2%20H-0_F-21738496_observation.avi?dl=0)
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp try_atar_2 --obs-type 'image' --env-name "MontezumaRevengeNoFrameskip-v4" --num-hierarchy 2 --num-subpolicy 5 --hierarchy-interval 4 --num-steps 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux delta_model_combine_reward_r_0
```

2 levels with longer interval and more num-subpolicy (running w5-1, seems to be better then above run from the episode length)
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp try_atar_2 --obs-type 'image' --env-name "MontezumaRevengeNoFrameskip-v4" --num-hierarchy 2 --num-subpolicy 16 --hierarchy-interval 16 --num-steps 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux delta_model_combine_reward_r_0
```

#### I need to figure out if the mass center is doing well and find a suitable hierarchy interval.

I think a reasonable choice is (running on W4-2).
Some positive results are shown: [link](https://www.dropbox.com/s/85iffkcfmicglkd/H-0_F-8511488_observation.avi?dl=0)
The key is reachable when the skull is not presented, video is [here](https://www.dropbox.com/s/abcjmzxu9eubcbd/H-0_F-67778560_observation.avi?dl=0).
Thus, the problem is on the skull.
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp find_params --obs-type 'image' --env-name "MontezumaRevengeNoFrameskip-v4" --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 8 4 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux delta_model_combine_reward_r_0
```

#### I am adding inverse mask model to solve the problem of uncontrollable parts in state

Running on W4-2.
Nothing meaningful shows up.
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp inverse_mask_model --obs-type 'image' --env-name "MontezumaRevengeNoFrameskip-v4" --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 8 4 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --inverse-mask --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux delta_model_combine_reward_r_0
```

I will need to study on the OverCooked before switching on this game.
Goal-type: any.
Running on W4-2.
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp inverse_mask_model --obs-type 'image' --env-name "OverCooked" --reward-level 1 --setup-goal any --new-overcooked --num-hierarchy 2 --num-subpolicy 5 --hierarchy-interval 4 --num-steps 128 128 --reward-bounty 0.1875 --distance mass_center --transition-model-mini-batch-size 64 --inverse-mask --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_0
```

Try to fix the bug on sleeping log.
Running on W4-2.
Log curve works properly now, [here](https://www.dropbox.com/s/8zvv39yzffyksi1/Screen%20Recording%202018-10-09%20at%2015.56.01.mov?dl=0).
Video log works OK, [here](https://www.dropbox.com/s/u02ktr6y7f28gzn/H-1_F-655360_state_prediction.avi?dl=0).
```bash
CUDA_VISIBLE_DEVICES=3 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp bug_on_sleeping_log --obs-type 'image' --env-name "OverCooked" --reward-level 1 --setup-goal any --new-overcooked --num-hierarchy 2 --num-subpolicy 5 --hierarchy-interval 4 --num-steps 128 128 --reward-bounty 0.1875 --distance mass_center --transition-model-mini-batch-size 64 --inverse-mask --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_0
```

Grad on one scalar.
Running on W4-2.
The mask is not satisfying, [here](https://www.dropbox.com/s/o3fjjyqbqqk8jk7/H-1_F-1053696_state_prediction.avi?dl=0).
The loss_inverse_mask_model drops as the score want up [here](https://www.dropbox.com/s/29644li2wphlc26/Screen%20Recording%202018-10-09%20at%2018.50.10.mov?dl=0), thus I think the training of inverse_mask_model is fine. But we properly want a more strict investigation.
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp grad_on_scalar --obs-type 'image' --env-name "OverCooked" --reward-level 1 --setup-goal any --new-overcooked --num-hierarchy 2 --num-subpolicy 5 --hierarchy-interval 4 --num-steps 128 128 --reward-bounty 0.1875 --distance mass_center --transition-model-mini-batch-size 64 --inverse-mask --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_0
```

I think I need to remove the BatchNorm and then take a look.
Running on W4-2.
The mask makes some senses, but not very ideal, see [here](https://www.dropbox.com/s/pkd6a0b94f23gj8/H-1_F-1292288_state_prediction.avi?dl=0).
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp grad_on_scalar --obs-type 'image' --env-name "OverCooked" --reward-level 1 --setup-goal any --new-overcooked --num-hierarchy 2 --num-subpolicy 5 --hierarchy-interval 4 --num-steps 128 128 --reward-bounty 0.1875 --distance mass_center --transition-model-mini-batch-size 64 --inverse-mask --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux mask_model_no_bn_r_0
```

Add a log to see if inverse model is predicting correctly.
Running on W4-2.
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp grad_on_scalar --obs-type 'image' --env-name "OverCooked" --reward-level 1 --setup-goal any --new-overcooked --num-hierarchy 2 --num-subpolicy 5 --hierarchy-interval 4 --num-steps 128 128 --reward-bounty 0.1875 --distance mass_center --transition-model-mini-batch-size 64 --inverse-mask --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_1
```

Add normalize term to inverse_mask_model.
Running on W4-5.
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp grad_on_scalar --obs-type 'image' --env-name "OverCooked" --reward-level 1 --setup-goal any --new-overcooked --num-hierarchy 2 --num-subpolicy 5 --hierarchy-interval 4 --num-steps 128 128 --reward-bounty 0.1875 --distance mass_center --transition-model-mini-batch-size 64 --inverse-mask --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_2
```

The inverse model memorize the actions. We need to go random.
Running on W4-2.
I think it seems to be ok, maybe. [here](https://www.dropbox.com/s/ynvql04wrci0zfc/H-1_F-2779136_state_prediction.avi?dl=0)
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp grad_on_scalar --obs-type 'image' --env-name "OverCooked" --reward-level 1 --setup-goal random --new-overcooked --num-hierarchy 2 --num-subpolicy 5 --hierarchy-interval 4 --num-steps 128 128 --reward-bounty 0.1875 --distance mass_center --transition-model-mini-batch-size 64 --inverse-mask --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_2
```

Try on MontezumaRevengeNoFrameskip.
Running on W4-0
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp grad_on_scalar --obs-type 'image' --env-name "MontezumaRevengeNoFrameskip-v4" --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 8 4 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --inverse-mask --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_0
```

I will add [adm model](https://www.dropbox.com/s/mftt1ll0q39dalz/File%202018-10-7%2C%209%2036%2022%20AM.pdf?dl=0).
Running on W4-0
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp adm_model --obs-type 'image' --env-name "MontezumaRevengeNoFrameskip-v4" --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 8 4 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --inverse-mask --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_0
```

I thing the inverse model is only needed for bottom layer and it should be shared by all layers.
And the mask should be generated by the current observation.
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp adm_model --obs-type 'image' --env-name "MontezumaRevengeNoFrameskip-v4" --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 8 4 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --inverse-mask --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_1
```
Probaly ok? [here](https://www.dropbox.com/s/fe3108b5ycxiztk/H-1_F-3033088_state_prediction.avi?dl=0)

I think I can add the mask and see how it goes.
W4-3.
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp adm_model --obs-type 'image' --env-name "MontezumaRevengeNoFrameskip-v4" --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 8 4 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --inverse-mask --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_2
```

Test on more Atari games.
Pong.
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp adm_model --obs-type 'image' --env-name "PongNoFrameskip-v4" --num-hierarchy 2 --num-subpolicy 3 --hierarchy-interval 4 --num-steps 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 --inverse-mask --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_0
```

I am trying to use the no_conv_adm structure.
Weird, learns to focus on the skull? [here](https://www.dropbox.com/s/r4pwnizv240ywnc/H-1_F-458752_state_prediction.avi?dl=0)
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp adm_model --obs-type 'image' --env-name "MontezumaRevengeNoFrameskip-v4" --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 8 4 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --inverse-mask --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_3
```

Try simple adm.
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp adm_model --obs-type 'image' --env-name "MontezumaRevengeNoFrameskip-v4" --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 8 4 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --inverse-mask --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_5
```
* Seems to be nice! [here](https://www.dropbox.com/s/8gjmn9rn0yhkc2n/H-1_F-449536_state_prediction.avi?dl=0).
* See if binarize_mask_torch is better. I think it is reasonable, note that the action is not predictable when the agent is in the sky (here)[https://www.dropbox.com/s/ghd4wess14u0igz/H-1_F-305152_state_prediction.avi?dl=0].

Try other size of mask = 4.
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp adm_model --obs-type 'image' --env-name "MontezumaRevengeNoFrameskip-v4" --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 8 4 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --inverse-mask --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_6
```

Try other size of mask = 3.
```bash
CUDA_VISIBLE_DEVICES=2 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp adm_model --obs-type 'image' --env-name "MontezumaRevengeNoFrameskip-v4" --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 8 4 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --inverse-mask --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_7
```

Full ADM Model

--num-grid 7
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp full_adm_model --obs-type 'image' --env-name "PongNoFrameskip-v4" --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 8 4 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --inverse-mask --num-grid 7 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_0
```

--num-grid 3
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp full_adm_model --obs-type 'image' --env-name "PongNoFrameskip-v4" --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 8 4 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --inverse-mask --num-grid 3 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_0
```

Try MuJoCo. (IceClear, try run this.)
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 1 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp adm_model --obs-type 'image' --env-name "Reacher-v2" --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 8 4 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_0
```
