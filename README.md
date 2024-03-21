# Physics Simulators

Reinforcement Learning tutorial for physics-based environments:
- MuJoCo gym
- MuJoCo envpool
- MuJoCo XLA (brax)
- Isaac Sim

Train policies for MuJoCo gym, MuJoCo envpool, and brax using forked version of [rl_games](https://github.com/yurangja99/rl_games.git). 

Train policies for Isaac Sim using [OmniIsaacGymEnvs](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs.git)

## Start using Docker
How to run this repository is described in [`DOCKER.md`](./DOCKER.md). 

## Trained checkpoints
Simple MLP policies are trained by PPO algorithm for `Humanoid` environment provided by (1) MuJoco, (2) MuJoCo envpool, (3) brax, and (4) Isaac Sim. 

Each agent is trained 2000 iters for 1 hours (brax) to 33 hours (mujoco gym). 

![](./results/Humanoid/learning_curve.png)

You can run trained checkpoints for each simulator, but first you have to set environment following the [instruction](#start-using-docker). 

After setting environment, you can play the policies. If you want rendering of the rollouts, set `params.config.player.render` in the config file. 

```bash
# play trained policy in mujoco gym
cd /workspace/physics-simulators/rl_games
poetry run python runner.py --play --file rl_games/configs/mujoco/humanoid.yaml --checkpoint ../results/Humanoid/mujoco/nn/Humanoid-v4_ray.pth

# play trained policy in mujoco envpool
cd /workspace/physics-simulators/rl_games
poetry run python runner.py --play --file rl_games/configs/mujoco/humanoid_envpool.yaml --checkpoint ../results/Humanoid/envpool/nn/Humanoid-v4_envpool.pth

# play trained policy in brax
cd /workspace/physics-simulators/rl_games
poetry run python runner.py --play --file rl_games/configs/brax/ppo_humanoid.yaml --checkpoint ../results/Humanoid/brax/nn/Humanoid_brax.pth

# play trained policy in isaac sim
cd /workspace/physics-simulators/OmniIsaacGymEnvs/omniisaacgymenvs
sudo /isaac-sim/python.sh scripts/rlgames_train.py task=Humanoid headless=True test=True num_envs=64 checkpoint=../../results/Humanoid/isaac-sim/nn/Humanoid.pth
```

Average reward of trained policies are:

||MuJoCo Gym|MuJoCo Envpool|Brax|Isaac Sim|
|-|-|-|-|-|
|Training Time (hours)|33|8|1|7.75|
|Test Episodes|2000|2000|2076|1944|
|Average Test Reward|8230.41|11791.60|8999.30|8668.70|
|Average Test Steps|935.26|958.99|965.31|976.45|
