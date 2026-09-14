#!/usr/bin/env bash
# Reproduce the reference Alchemy settings. One command per saved baseline.
#
#   ./scripts/repro_alchemy.sh <target> [extra flags...]
#
# Targets (see docs/alchemy_trials.md G):
#   rl_sanity        memoryless on all_fixed        302.1  (wiring control)
#   rl_oracle_aux    oracle + aux_canon             297.8  +0.877
#   rl_mate_cpc      MATE + aux_cpc                 183.0  +0.067  <- best MATE
#   rl_mate          MATE, no aux                   176.6  +0.022  <- baseline
#
# The world-model targets (wm_*) live on the world-model branch together with
# scripts/train_world_model.py; they are not reachable from here.
#
# Everything after <target> is appended to the command, so a new technique is
# added without editing this file:
#   ./scripts/repro_alchemy.sh rl_mate --config_seq.seq_model.use_gate=True \
#       --run_name mate_gate
#
# NOTE ON DISK. RL runs write to a LOCAL disk by default. /NFS is a shared
# mount and a run rewrites its whole replay buffer at every checkpoint; 31
# finished runs once wrote 326 GB there and slowed the mount for everyone.
# Override with RL_SAVE_DIR=... if /HDD1 is not present on this box.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/NFS/workspaces/g.chung/miniconda3/envs/mate/bin/python}"
RL_SAVE_DIR="${RL_SAVE_DIR:-/HDD1/g.chung/Memory-RL/logs/repro}"
DEVICE="${DEVICE:-0}"
SEED="${SEED:-42}"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
cd "$ROOT"

target="${1:?usage: repro_alchemy.sh <target> [extra flags...]}"; shift || true

# ---- RL ------------------------------------------------------------------
# Common to every RL baseline. structured_potions is the one non-privileged
# re-encoding that paid (+32.6); PopArt and the invalid-action mask are on in
# every run these numbers come from.
rl() {  # rl <run_name> <seq_config> [flags...]
  local name=$1 seq=$2; shift 2
  mkdir -p "$RL_SAVE_DIR"
  set -x
  $PY main.py \
    --config_env=configs/envs/alchemy.py \
    --config_rl=configs/rl/dqn_default.py \
    --config_seq="$seq" \
    --config_env.structured_potions=True \
    --config_rl.critic_lr=3e-5 \
    --config_rl.use_popart=True \
    --config_rl.mask_alchemy_invalid_actions=True \
    --config_seq.normalize_inputs=True \
    --config_seq.use_pe=True \
    --config_seq.conditioning_hidden_dim=128 \
    --train_episodes=24000 --k=1 --seed="$SEED" \
    --device="$DEVICE" --run_name="$name" --save_dir="$RL_SAVE_DIR" "$@"
}

MATE=configs/seq_models/mate_default.py
MARKOV=configs/seq_models/markov_default.py


case "$target" in
  # Wiring control: no memory, chemistry fixed across the episode, so the task
  # needs no meta-learning at all. Reaches ~90% of the ceiling. If a new
  # technique looks broken, check that this still passes before blaming it.
  rl_sanity)
    rl repro_fx_markov "$MARKOV" --config_env.env_name=all_fixed "$@" ;;

  # The project's strongest result: a 33-dim supervision label (latent stone
  # coords + potion types + bottleneck graph), excised from the observation
  # before the network sees it. Privileged -- an upper bound, not a method.
  # is_oracle appends the true chemistry to the observation. Without that
  # flag this is a plain memoryless agent, not the oracle.
  rl_oracle_aux)
    rl repro_nbaux_oracle "$MARKOV" \
      --config_env.env_name=no_bottleneck \
      --config_env.aux_canon_target=True \
      --config_seq.seq_model.is_oracle=True \
      --config_rl.aux_canon_weight=1.0 \
      --config_rl.aux_canon_parts=both \
      --config_rl.aux_canon_site=joint "$@" ;;

  # Best memory-model result. Same 33-dim label as above, InfoNCE instead of
  # regression. site=joint, matching the run these numbers come from.
  rl_mate_cpc)
    rl repro_nbcpc_mate "$MATE" \
      --config_env.env_name=no_bottleneck \
      --config_env.aux_canon_target=True \
      --config_rl.aux_canon_weight=0.0 \
      --config_rl.aux_cpc_weight=1.0 \
      --config_rl.aux_canon_parts=both \
      --config_rl.aux_canon_site=joint "$@" ;;

  # The plain MATE baseline every technique should be compared against.
  rl_mate)
    rl repro_nb_mate "$MATE" --config_env.env_name=no_bottleneck "$@" ;;


  *) echo "unknown target: $target" >&2
     sed -n '4,20p' "${BASH_SOURCE[0]}" >&2
     exit 2 ;;
esac
