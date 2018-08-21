# pytorch-a2c-ppo-acktr

## Update 10/06/2017: added enjoy.py and a link to pretrained models!
## Update 09/27/2017: now supports both Atari and MuJoCo/Roboschool!

This is a PyTorch implementation of
* Advantage Actor Critic (A2C), a synchronous deterministic version of [A3C](https://arxiv.org/pdf/1602.01783v1.pdf)
* Proximal Policy Optimization [PPO](https://arxiv.org/pdf/1707.06347.pdf)
* Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation [ACKTR](https://arxiv.org/abs/1708.05144)

Also see the OpenAI posts: [A2C/ACKTR](https://blog.openai.com/baselines-acktr-a2c/) and [PPO](https://blog.openai.com/openai-baselines-ppo/) for more information.

This implementation is inspired by the OpenAI baselines for [A2C](https://github.com/openai/baselines/tree/master/baselines/a2c), [ACKTR](https://github.com/openai/baselines/tree/master/baselines/acktr) and [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo1). It uses the same hyper parameters and the model since they were well tuned for Atari games.

Please use this bibtex if you want to cite this repository in your publications:

    @misc{pytorchrl,
      author = {Kostrikov, Ilya},
      title = {PyTorch Implementations of Reinforcement Learning Algorithms},
      year = {2018},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/ikostrikov/pytorch-a2c-ppo-acktr}},
    }

## Supported (and tested) environments (via [OpenAI Gym](https://gym.openai.com))
* [Atari Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
* [MuJoCo](http://mujoco.org)
* [PyBullet](http://pybullet.org) (including Racecar, Minitaur and Kuka)
* [DeepMind Control Suite](https://github.com/deepmind/dm_control) (via [dm_control2gym](https://github.com/martinseilair/dm_control2gym))

I highly recommend PyBullet as a free open source alternative to MuJoCo for continuous control tasks.

All environments are operated using exactly the same Gym interface. See their documentations for a comprehensive list.

To use the DeepMind Control Suite environments, set the flag `--env-name dm.<domain_name>.<task_name>`, where `domain_name` and `task_name` are the name of a domain (e.g. `hopper`) and a task within that domain (e.g. `stand`) from the DeepMind Control Suite. Refer to their repo and their [tech report](https://arxiv.org/abs/1801.00690) for a full list of available domains and tasks. Other than setting the task, the API for interacting with the environment is exactly the same as for all the Gym environments thanks to [dm_control2gym](https://github.com/martinseilair/dm_control2gym).

## Requirements

* Python 3 (it might work with Python 2, but I didn't test it)
* [PyTorch](http://pytorch.org/)
* [Visdom](https://github.com/facebookresearch/visdom)
* [OpenAI baselines](https://github.com/openai/baselines)

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
pip install -e .
```

Run commands here:

To run shared policy without reward_bounty:
```bash
source ~/.bashrc
source activate ehrl
```

## OverCooked

### Level 1

```bash
CUDA_VISIBLE_DEVICES=1 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp mass_center --obs-type 'image' --env-name "OverCooked" --reward-level 1 --num-hierarchy 2 --num-subpolicy 5 --hierarchy-interval 4 --num-steps 128 128 --reward-bounty 10 --distance mass_center --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux r_0
```

### Level 2

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 8 --actor-critic-mini-batch-size 256 --actor-critic-epoch 4 --exp two_sequence --obs-type 'image' --env-name "OverCooked" --reward-level 2 --num-hierarchy 3 --num-subpolicy 5 5 --hierarchy-interval 4 12 --num-steps 128 128 128 --reward-bounty 1 --distance mass_center --transition-model-mini-batch-size 64 64 --train-mode together --encourage-ac-connection none --clip-reward-bounty --clip-reward-bounty-active-function linear --log-behavior-interval 5 --aux shuffle_show_goal_r_0
```

## Option-Critic
This repo include The Option-Critic Architecture in [DeepRL](https://github.com/ShangtongZhang/DeepRL) as comparison. To run Option-Critic Architecture on OverCooked game:

### Level 0

```bash
CUDA_VISIBLE_DEVICES=1 python option_critic.py --exp two_sequence --obs-type 'image' --env-name "OverCooked" --reward-level 0 --log-behavior-interval 5 --aux r_0
```

```base
tensorboard --logdir=../results/t-MovementBandits-v0/
```
