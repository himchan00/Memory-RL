#!/usr/bin/env bash
# Install SYMBOLIC dm_alchemy (arXiv:2102.02926) into a conda env. Symbolic-only:
# no Docker/Unity/GL. dm_alchemy is archived and its setup.py needs pkg_resources
# (dropped by setuptools>=82), so we skip it: pip-install deps as wheels, compile
# the protobufs in place, and expose the package via a .pth — leaving the env's
# numpy/scipy/torch untouched.
#
# Usage: bash scripts/install_dm_alchemy.sh [conda_env_name=mate] [clone_dir=$HOME/dm_alchemy]
#
# Troubleshooting:
#   * protobuf "Descriptors cannot be created directly":
#       export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
#   * numpy-2 alias error (np.bool/np.int) from a dep: patch to the modern alias.
#   * `import dm_alchemy` uses the `docker` pkg but needs no Docker daemon.
set -euo pipefail

ENV="${1:-mate}"
CLONE_DIR="${2:-$HOME/dm_alchemy}"
run() { conda run --no-capture-output -n "$ENV" "$@"; }

echo "[1/5] Pinning setuptools<82 (torch requires it; do NOT upgrade it)..."
run python -m pip install "setuptools<82"

echo "[2/5] Installing DeepMind + build deps as wheels (numpy/scipy/torch untouched)..."
run python -m pip install --only-binary=:all: \
  dm-env dm-env-rpc dm-tree absl-py frozendict portpicker \
  grpcio grpcio-tools docker

echo "[3/5] Fetching archived dm_alchemy source into '$CLONE_DIR'..."
if [ ! -d "$CLONE_DIR/.git" ]; then
  git clone https://github.com/google-deepmind/dm_alchemy.git "$CLONE_DIR"
else
  echo "      (already present, skipping clone)"
fi
# setup.py normally generates _version.py; create a stub if absent.
[ -f "$CLONE_DIR/dm_alchemy/_version.py" ] || \
  echo "__version__ = '1.0.0'" > "$CLONE_DIR/dm_alchemy/_version.py"

echo "[4/5] Compiling protobufs in place + exposing dm_alchemy via a .pth..."
INCLUDE=$(run python -c "import grpc_tools,os;print(os.path.join(os.path.dirname(grpc_tools.__file__),'_proto'))")
( cd "$CLONE_DIR" && run python -m grpc_tools.protoc -I. -I"$INCLUDE" \
    --python_out=. --grpc_python_out=. \
    dm_alchemy/protos/*.proto dm_alchemy/encode/*.proto )
SP=$(run python -c "import site;print(site.getsitepackages()[0])")
echo "$CLONE_DIR" > "$SP/dm_alchemy.pth"
echo "      -> $SP/dm_alchemy.pth"

echo "[5/5] Smoke test: import symbolic env, reset, step 200x, check shapes..."
run python - <<'PY'
import numpy as np
from dm_alchemy import symbolic_alchemy
from dm_alchemy.types import utils
LEVEL = "perceptual_mapping_randomized_with_rotation_and_random_bottleneck"
env = symbolic_alchemy.get_symbolic_alchemy_level(
    LEVEL, seed=0, num_trials=10, max_steps_per_trial=20, observe_used=True,
    end_trial_action=False,
    see_chemistries={"chem_gt": utils.ChemistrySeen(content=utils.ElementContent.GROUND_TRUTH)})
ts = env.reset()
obs = ts.observation
print("obs keys:", list(obs.keys()))
print("symbolic_obs:", np.asarray(obs["symbolic_obs"]).shape)
print("chem_gt:", np.asarray(obs["chem_gt"]).shape)
aspec = env.action_spec(); print("action_spec:", aspec.minimum, "..", aspec.maximum)
n_act = int(aspec.maximum) + 1
r, steps = 0.0, 0
rng = np.random.RandomState(0)
while True:
    ts = env.step(int(rng.randint(n_act)))
    r += float(ts.reward or 0.0); steps += 1
    if ts.last(): break
print(f"episode: steps={steps} reward={r:.3f} last={ts.last()}")
assert np.asarray(obs["symbolic_obs"]).shape == (39,), "symbolic_obs must be (39,)"
assert np.asarray(obs["chem_gt"]).shape == (28,), "chem_gt must be (28,)"
assert n_act == 40, f"expected 40 actions, got {n_act}"
assert steps == 200, f"expected 200 steps, got {steps}"
print("OK: symbolic dm_alchemy is working.")
PY

echo "Done. Symbolic Alchemy is installed in conda env '$ENV'."
