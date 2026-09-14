"""The SAC aux path must agree with the DQN one, index for index.

`policies/models/aux_canon.py` re-implements what `policy_rnn_dqn` grew inline.
Two copies of a LEAK GUARD is how one of them silently rots, so this asserts
they compute the same thing on the same config: the same stripped width, the
same label slice, and byte-identical stripped observations.

It also checks the guard actually holds on the SAC side -- that nothing the
network sees contains the label block.

    python scripts/verify_aux_canon_parity.py
"""
import os, sys, torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torchkit.pytorch_utils as ptu
from configs.envs import alchemy as env_cfg
from configs.rl import dqn_default, sac_default
from configs.seq_models import mate_default, markov_default
from envs.alchemy import AUX_CANON_DIM, get_symbolic_alchemy_layout
from policies.models.policy_rnn_dqn import ModelFreeOffPolicy_DQN_RNN
from policies.models.policy_rnn_sac import ModelFreeOffPolicy_SAC_RNN

A, T, B = 40, 8, 3


def env_config(oracle):
    ce = env_cfg.get_config(); del ce.create_fn
    ce.structured_potions = True
    ce.aux_canon_target = True
    ce.num_trials, ce.max_steps_per_trial = 2, 4
    return ce


def seq_config(oracle, context_dim):
    cs = (markov_default if oracle else mate_default).get_config()
    del cs.update_fn
    cs.compile = False
    cs.seq_model.max_seq_length = T + 2
    cs.seq_model.context_dim = context_dim
    cs.conditioning_hidden_dim = 128
    if oracle:
        cs.seq_model.is_oracle = True
    return cs


def build(oracle):
    ce = env_config(oracle)
    context_dim = 28 if oracle else 0
    layout = get_symbolic_alchemy_layout(True, True)
    obs_dim = layout.symbolic_obs_dim + 1 + AUX_CANON_DIM + context_dim

    cr_d = dqn_default.get_config(); del cr_d.update_fn
    cr_d.mask_alchemy_invalid_actions = True
    cr_d.aux_canon_weight = 1.0
    cr_d.aux_canon_site = "joint" if oracle else "memory_obs"
    cr_d.aux_canon_parts = "potion"
    cr_d.init_eps, cr_d.end_eps, cr_d.schedule_steps = 1.0, 0.01, 100

    cr_s = sac_default.get_config(); del cr_s.update_fn
    cr_s.mask_alchemy_invalid_actions = True
    cr_s.aux_canon_weight = 1.0
    cr_s.aux_canon_site = cr_d.aux_canon_site
    cr_s.aux_canon_parts = "potion"
    cr_s.critic_lr = cr_s.actor_lr = 3e-5

    dqn = ModelFreeOffPolicy_DQN_RNN(
        obs_dim=obs_dim, action_dim=A, config_seq=seq_config(oracle, context_dim),
        config_rl=cr_d, config_env=ce, freeze_critic=False,
    ).to(ptu.device)
    sac = ModelFreeOffPolicy_SAC_RNN(
        obs_dim=obs_dim, action_dim=A, config_seq=seq_config(oracle, context_dim),
        config_rl=cr_s, config_env=ce, freeze_critic=False,
        continuous_action=False,
    ).to(ptu.device)
    return dqn, sac, obs_dim


def check(oracle):
    tag = "ORACLE (markov)" if oracle else "MATE"
    print(f"\n=== {tag} ===")
    dqn, sac, obs_dim = build(oracle)
    print(f"  obs_dim {obs_dim}   net_obs_dim  dqn {dqn.net_obs_dim} / sac {sac.net_obs_dim}")
    print(f"  label slice          dqn [{dqn._aux_start}:{dqn._aux_end}] "
          f"/ sac [{sac._aux_start}:{sac._aux_end}]")
    assert dqn.net_obs_dim == sac.net_obs_dim, "stripped width differs"
    assert (dqn._aux_start, dqn._aux_end) == (sac._aux_start, sac._aux_end), \
        "label slice differs"
    assert dqn._alchemy_split_kwargs == sac._alchemy_split_kwargs, \
        f"split kwargs differ:\n  dqn {dqn._alchemy_split_kwargs}\n  sac {sac._alchemy_split_kwargs}"

    obs = torch.randn((T + 2, B, obs_dim), device=ptu.device)
    a = dqn._strip_aux_target(obs)
    b = sac.strip_aux_target(obs)
    assert a.shape == b.shape and torch.equal(a, b), "stripped observations differ"
    print(f"  stripped obs identical: True   width {a.shape[-1]}")

    # the label block must not survive anywhere in what the network sees
    label = obs[..., dqn._aux_start:dqn._aux_end]
    print(f"  label block width    : {label.shape[-1]} (= {AUX_CANON_DIM})")
    kept = torch.cat((obs[..., :dqn._aux_start], obs[..., dqn._aux_end:]), -1)
    assert torch.equal(b, kept), "SAC strip does not match the definition"
    print("  leak guard holds     : True")

    tgt_d = dqn._aux_target_slice(obs)
    tgt_s = sac.aux_target_slice(obs)
    assert torch.equal(tgt_d, tgt_s), "label slices differ"
    print("  label slice identical: True")


if __name__ == "__main__":
    ptu.set_gpu_mode(torch.cuda.is_available(), int(os.environ.get("MATE_DEVICE", "3")))
    check(oracle=False)
    check(oracle=True)
    print("\nOK")
