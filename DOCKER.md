# Physics-Simulators setting with Docker

First, you have to install [Isaac Sim](https://developer.nvidia.com/isaac-sim) and [MuJoCo](https://mujoco.org/). 

Install MuJoCo 2.2.0 at `~/.mujoco` directory from [download homepage](https://www.roboti.us/download.html). 

Install Isaac Sim docker image following the [instruction](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html). (You need Nvidia account!)

Then, you have `nvcr.io/nvidia/isaac-sim:2023.1.1` image. 

Run the Isaac Sim docker with x11 settings. 

```bash
xhost +
docker run -dit --gpus '"device=0"' --name phys-sims --network=host --ipc=host \
-e "ACCEPT_EULA=Y" \
-e "PRIVACY_CONSENT=Y" \
-e DISPLAY=$DISPLAY \
-e USER=$USER \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $HOME/.Xauthority:/root/.Xauthority:rw \
-v ~/.mujoco:/root/.mujoco \
-v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
-v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
-v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
-v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
-v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
-v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
-v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
-v ~/docker/isaac-sim/documents:/root/Documents:rw \
-w /workspace \
nvcr.io/nvidia/isaac-sim:2023.1.1
```

Initialize the container with the following scripts:

```bash
################################################
# update apt
apt update -y 
apt install -y sudo 
# install basic packages and python3
sudo apt install -y curl wget nano git gcc x11-apps 
# clone physics-simulators
cd /workspace
git clone --recurse-submodules https://github.com/yurangja99/physics-simulators.git
################################################
# install OmniIsaacGymEnvs
cd /workspace/physics-simulators/OmniIsaacGymEnvs
sudo /isaac-sim/python.sh -m pip install -e . # ignore version errors.
################################################
# install cuda
cd /workspace
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
echo "export PATH=/usr/local/cuda-11.8/bin:${PATH}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}" >> ~/.bashrc
source ~/.bashrc
nvcc -V # to check cuda
# install cudnn (nvidia login needed!)
wget https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/11.x/cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz/
sudo tar -xvf cudnn-linux-x86_64-8.9.7.29_cuda11-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn # to check cudnn
# set mujoco env_var
echo "export LD_LIBRARY_PATH=/usr/lib/nvidia:/root/.mujoco/mujoco-2.2.0/bin:${LD_LIBRARY_PATH}" >> ~/.bashrc 
echo "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so" >> ~/.bashrc 
source ~/.bashrc 
# install python for mujoco
sudo apt install -y python3 python3-pip # 3.7 <= python < 3.11
echo "alias python=python3" >> ~/.bashrc
source ~/.bashrc
# install packages
sudo apt install -y libgl1-mesa-glx libxrandr2 libxinerama1 libosmesa6-dev libglfw3 patchelf libglew-dev libglib2.0-0 
# install poetry
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="/root/.local/bin:${PATH}"' >> ~/.bashrc
source ~/.bashrc
cd /workspace/physics-simulators/rl_games
poetry lock
poetry install
# install torch and jaxlib
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --default-timeout=1000
poetry run pip install -U "jax[cuda]==0.4.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Then, environment setting is done!

You can try some demos: 

```bash
# training isaac sim 
cd /workspace/physics-simulators/OmniIsaacGymEnvs/omniisaacgymenvs
sudo /isaac-sim/python.sh scripts/rlgames_train.py task=Humanoid
sudo /isaac-sim/python.sh scripts/rlgames_train.py task=ShadowHand headless=True

# training mujoco
cd /workspace/physics-simulators/rl_games
poetry run python runner.py --train --file rl_games/configs/mujoco/humanoid.yaml

# training mujoco envpool
cd /workspace/physics-simulators/rl_games
poetry run python runner.py --train --file rl_games/configs/mujoco/humanoid_envpool.yaml

# training brax
cd /workspace/physics-simulators/rl_games
poetry run python runner.py --train --file rl_games/configs/brax/ppo_humanoid.yaml
```
