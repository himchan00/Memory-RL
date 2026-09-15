"""Smoke the aux_cpc path end to end after the constructor/label-encoding fix.

The port called AuxCanonCPC(embed_dim=..., use_stone=..., ...) against a
signature of (embed_size, label_size, proj_dim, tau), and then handed forward()
the RAW 33-dim label block where g() expects the ENCODED label (potion slots
one-hot with an ABSENT column). Both are fixed; this checks the widths line up
and one real training step produces finite CPC metrics.
"""
import os, sys, numpy as np, torch
sys.path.insert(0, "/NFS/workspaces/g.chung/Memory-RL")
import torchkit.pytorch_utils as ptu
ptu.set_gpu_mode(torch.cuda.is_available(), int(os.environ.get("MATE_DEVICE", "0")))
from configs.envs import alchemy as env_cfg
from configs.rl import dqn_default
from configs.seq_models import mate_default
from envs.alchemy import SymbolicAlchemyEnv, get_symbolic_alchemy_layout
from policies.models.policy_rnn_dqn import ModelFreeOffPolicy_DQN_RNN
from policies.models.aux_cpc import encode_label

T, B, A = 12, 4, 40
ce = env_cfg.get_config(); del ce.create_fn
ce.env_name = "no_bottleneck"; ce.structured_potions = True; ce.aux_canon_target = True
cr = dqn_default.get_config(); del cr.update_fn
cr.init_eps, cr.end_eps, cr.schedule_steps = 1.0, 0.01, 1000
cr.aux_cpc_weight = 1.0; cr.aux_canon_weight = 0.0
cr.mask_alchemy_invalid_actions = True; cr.use_popart = True
cs = mate_default.get_config(); del cs.update_fn
cs.seq_model.max_seq_length = T + 2
cs.seq_model.context_dim = 0
cs.seq_model.is_oracle = False
cs.use_pe = True; cs.max_norm = 0.2
cs.seq_model.hidden_size = 256; cs.seq_model.n_layer = 1
cs.conditioning_hidden_dim = 128; cs.conditioning_n_layer = 1; cs.compile = False

layout = get_symbolic_alchemy_layout(True, True)
obs_dim = layout.symbolic_obs_dim + 1 + 33
agent = ModelFreeOffPolicy_DQN_RNN(obs_dim=obs_dim, action_dim=A,
        config_seq=cs, config_rl=cr, config_env=ce).to(ptu.device)
agent.train()
al = agent.alchemy
print(f"  cpc widths       embed_in={al.cpc.f[0].in_features} "
      f"label_in={al.cpc.g[0].in_features}")

envs = [SymbolicAlchemyEnv(level_name="perceptual_mapping_randomized", num_trials=2,
        max_steps_per_trial=8, observe_used=True, add_trial_flag=True,
        structured_potions=True, aux_canon_target=True) for _ in range(B)]
rng = np.random.default_rng(0)
obs = [e.reset(seed=700 + i)[0] for i, e in enumerate(envs)]
rows = []
for _ in range(T + 1):
    rows.append(np.stack(obs)); nxt = []
    for e, o in zip(envs, obs):
        s, _, te, tr, _ = e.step(int(rng.integers(A)))
        nxt.append(e.reset(seed=int(rng.integers(1 << 30)))[0] if (te or tr) else s)
    obs = nxt
raw = torch.as_tensor(np.stack(rows).astype(np.float32), device=ptu.device)
observs, next_observs = raw[:-1], raw[1:]
enc = encode_label(al.target_slice(observs), al.use_stone, al.use_potion, al.use_graph)
print(f"  encoded label    {enc.shape[-1]} dims (must equal label_in)")
assert enc.shape[-1] == al.cpc.g[0].in_features

actions = torch.nn.functional.one_hot(torch.randint(A, (T, B), device=ptu.device), A).float()
rewards = torch.randn((T, B, 1), device=ptu.device)
terms = torch.zeros((T, B, 1), device=ptu.device)
masks = torch.ones((T, B, 1), device=ptu.device)
tt = torch.arange(T, device=ptu.device).view(-1, 1).expand(T, B)
out = agent._compute_loss(actions, rewards, observs, next_observs, terms, masks, tt)
d = out[1] if isinstance(out, tuple) else out
aux = {k: round(float(d[k]), 4) for k in d if "cpc" in k.lower() or "aux" in k.lower()}
print("  aux metrics     ", aux)
assert aux and all(np.isfinite(v) for v in aux.values())
print("SMOKE OK")
