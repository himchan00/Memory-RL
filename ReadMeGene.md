# Memory-RL MuJoCo Environment Setup (Full Guide)

This file explains all steps needed to create the environment for Memory-RL with MuJoCo.

## Part 1 — Create Conda Environment

We use Python 3.8.18 for compatibility.

```bash
conda create -y -n hist python=3.8.18
conda activate hist
```

## Part 2 — Install Python Requirements

Install all packages from the pinned requirements.txt (see Part 8 for contents).

```bash
pip install -r requirements.txt
```

## Part 3 — Install PyTorch with CUDA 11.3

torch==1.11.0+cu113 supports sm_86 (RTX 3090).

```bash
pip uninstall -y torch triton torchtriton 2>/dev/null || true
pip install --no-cache-dir torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

Check installation:

```bash
python - <<'PY'
import torch
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

## Part 4 — Install MuJoCo 2.1.0 (Custom Path, No sudo)

Select a folder to install MuJoCo:

```bash
export MUJOCO_ROOT=/your/storage/path/mujoco
mkdir -p "$MUJOCO_ROOT"
cd "$MUJOCO_ROOT"

wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
```

## Part 5 — Install Runtime Libraries

```bash
conda install -y -c conda-forge   glew glfw libglvnd mesalib   xorg-libx11 xorg-libxau xorg-libxdmcp xorg-libxext xorg-libxrender   xorg-libxrandr xorg-libxinerama xorg-libxcursor xorg-libxi
```

For software rendering:

```bash
conda install -y -c conda-forge osmesa
```

## Part 6 — Environment Variables for MuJoCo

Create `env.mujoco.sh`:

```bash
cat > env.mujoco.sh <<'EOF'
export MUJOCO_ROOT=${MUJOCO_ROOT:-/your/storage/path/mujoco}
export MUJOCO_PY_MUJOCO_PATH="$MUJOCO_ROOT/mujoco210"
export NVIDIA_LIB_DIR=${NVIDIA_LIB_DIR:-/lib/x86_64-linux-gnu}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MUJOCO_PY_MUJOCO_PATH/bin:$NVIDIA_LIB_DIR"
EOF
```

Activate it:

```bash
source env.mujoco.sh
```

If EGL fails, try:

```bash
export MUJOCO_GL=osmesa
source env.mujoco.sh
```

## Part 7 — Test the Setup

```bash
source env.mujoco.sh
python - <<'PY'
import gym, mujoco_py
env = gym.make("HalfCheetah-v2")
obs = env.reset()
for _ in range(5):
    obs, r, done, info = env.step(env.action_space.sample())
    if done: env.reset()
print("MuJoCo and Gym are working")
PY
```

## Run Training

After setup:

```bash
source env.mujoco.sh
python main.py --config_env configs/envs/half_cheetah_vel.py --config_rl configs/rl/sac_default.py --train_episodes 15000 --config_seq configs/seq_models/hist_default.py --config_seq.sampled_seq_len -1 --seed 0 --device 0 --run_name test
```

## Notes

- MUJOCO_ROOT and NVIDIA_LIB_DIR are variables. Adjust for your system.
- packaging==20.9 is needed to avoid Transformers 4.5.1 errors.
- If you change Python or CUDA version, you may need a different torch build.
