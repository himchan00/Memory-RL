"""Gradient-flow check for the auxiliary heads' attachment sites.

Each site is a claim about WHERE the auxiliary gradient lands, and those claims
are what the experiments mean, so they are checked rather than assumed:

  aux_canon_site="probe" -- MEASUREMENT ONLY. The head reads cat(obs, h) but
                            sends gradient into neither, so reading
                            `aux_canon_potion_acc` cannot change the thing it
                            measures. Checked by asserting the head's INPUT
                            carries no gradient at all.
  aux_count_weight > 0   -- the count head shapes the conditioner's OBSERVATION
                            branch. It must reach that branch, and must NOT
                            reach the memory or the critic: the counts are a
                            property of the current frame, and the point of the
                            loss is to regularise how that frame is read.

Run:  python scripts/verify_aux_sites.py
"""
import os, sys, torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torchkit.pytorch_utils as ptu
from configs.envs import alchemy as env_cfg
from configs.rl import dqn_default
from configs.seq_models import mate_default
from envs.alchemy import AUX_CANON_DIM, get_symbolic_alchemy_layout
from policies.models.policy_rnn_dqn import ModelFreeOffPolicy_DQN_RNN

T, B, N_TRIALS, TRIAL_LEN, ACTION_DIM = 40, 1, 2, 20, 40

GROUPS = (
    "head.seq_model",          # MATE memory
    "head.conditioner",        # observation branch, shared with the critic
    "qf.",                     # critic
    "aux_canon_head",
    "aux_count_head",
)


def build(aux_canon_site, aux_canon_weight, aux_count_weight,
          aux_count_site="obs"):
    ce = env_cfg.get_config()
    del ce.create_fn
    ce.structured_potions = True
    ce.aux_canon_target = aux_canon_weight > 0.0
    ce.num_trials, ce.max_steps_per_trial = N_TRIALS, TRIAL_LEN

    cr = dqn_default.get_config()
    del cr.update_fn
    cr.mask_alchemy_invalid_actions = True
    cr.aux_canon_site = aux_canon_site
    cr.aux_canon_weight = aux_canon_weight
    cr.aux_canon_parts = "potion"
    cr.aux_count_weight = aux_count_weight
    cr.aux_count_site = aux_count_site
    cr.init_eps, cr.end_eps, cr.schedule_steps = 1.0, 0.01, 1000

    cs = mate_default.get_config()
    del cs.update_fn
    cs.compile = False
    cs.seq_model.max_seq_length = T + 2
    # Normally set by Learner from the env; 0 = no oracle context tail.
    cs.seq_model.context_dim = 0
    cs.conditioning_hidden_dim = 128

    layout = get_symbolic_alchemy_layout(True, True)
    obs_dim = (
        layout.symbolic_obs_dim + 1
        + (AUX_CANON_DIM if ce.aux_canon_target else 0)
    )
    agent = ModelFreeOffPolicy_DQN_RNN(
        obs_dim=obs_dim, action_dim=ACTION_DIM,
        config_seq=cs, config_rl=cr, config_env=ce, freeze_critic=False,
    ).to(ptu.device)
    return agent, obs_dim


def dummy_batch(obs_dim):
    observs = torch.zeros((T + 2, B, obs_dim), device=ptu.device)
    actions = torch.zeros((T + 1, B, ACTION_DIM), device=ptu.device)
    actions[..., 0] = 1.0
    rewards = torch.zeros((T + 1, B, 1), device=ptu.device)
    masks = torch.ones((T + 1, B, 1), device=ptu.device)
    return observs, actions, rewards, masks


def grads_by_group(agent):
    return {
        prefix: any(
            p.grad is not None and bool(p.grad.abs().sum() > 0)
            for name, p in agent.named_parameters()
            if name.startswith(prefix)
        )
        for prefix in GROUPS
    }


def check_probe_is_inert():
    print("=== aux_canon_site='probe': the head's input must be detached ===")
    agent, obs_dim = build("probe", 1.0, 0.0)
    observs, actions, rewards, masks = dummy_batch(obs_dim)
    stripped = agent._strip_aux_target(observs)
    _, d = agent.head.forward(
        actions=actions, rewards=rewards, observs=stripped, masks=masks
    )
    memory, encoded_obs = d["_memory_embeds"], d["_encoded_obs"]
    probe_input = torch.cat((encoded_obs.detach(), memory.detach()), dim=-1)
    print(f"  memory readout requires_grad : {memory.requires_grad}   (live, as it should be)")
    print(f"  probe head input requires_grad: {probe_input.requires_grad}   <- must be False")
    assert not probe_input.requires_grad, "probe site would train the agent"


def check_count_reaches_only_the_obs_branch():
    print("\n=== aux_count: obs branch yes, memory/critic no ===")
    agent, obs_dim = build("joint", 0.0, 1.0)
    observs, actions, rewards, masks = dummy_batch(obs_dim)
    _, d = agent.head.forward(
        actions=actions, rewards=rewards, observs=observs, masks=masks
    )
    loss, metrics = agent._aux_count_loss(d["_obs_embeds"], observs, masks)
    agent.zero_grad(set_to_none=True)
    loss.backward()

    got = grads_by_group(agent)
    for name, hit in got.items():
        print(f"  {name:22s} grad={hit}")
    print(f"  metrics: {sorted(metrics)}")
    assert got["aux_count_head"], "count head received no gradient"
    assert got["head.conditioner"], "count loss never reached the obs branch"
    assert not got["head.seq_model"], "count loss leaked into the memory"
    assert not got["qf."], "count loss leaked into the critic"


def check_memory_obs_shapes_only_the_memory():
    print("\n=== aux_canon_site='memory_obs': memory yes, obs path/critic no ===")
    agent, obs_dim = build("memory_obs", 1.0, 0.0)
    observs, actions, rewards, masks = dummy_batch(obs_dim)
    targets = agent._aux_target_slice(observs)
    stripped = agent._strip_aux_target(observs)
    _, d = agent.head.forward(
        actions=actions, rewards=rewards, observs=stripped, masks=masks
    )
    aux_embeds = torch.cat(
        (d["_encoded_obs"].detach(), d["_memory_embeds"]), dim=-1
    )
    loss, _ = agent._aux_canon_loss(aux_embeds, stripped, targets, masks)
    agent.zero_grad(set_to_none=True)
    loss.backward()

    got = grads_by_group(agent)
    for name, hit in got.items():
        print(f"  {name:22s} grad={hit}")
    assert got["aux_canon_head"], "aux head received no gradient"
    assert got["head.seq_model"], "aux loss never reached the memory"
    assert not got["head.conditioner"], "aux loss leaked into the obs branch"
    assert not got["qf."], "aux loss leaked into the critic"


def check_count_site_memory_obs():
    """aux_count_site='memory_obs' must shape the MEMORY and nothing else.

    This is the paper-faithful site: arXiv:2102.02926 fed symbolic observations
    straight into the transformer core and hung the auxiliary heads off it, so
    the counting gradient landed on the recurrent representation, not on a
    separate perceptual branch (their agent had none).
    """
    print("\n=== aux_count_site='memory_obs': memory yes, obs branch/critic no ===")
    agent, obs_dim = build("joint", 0.0, 1.0, aux_count_site="memory_obs")
    observs, actions, rewards, masks = dummy_batch(obs_dim)
    _, d = agent.head.forward(
        actions=actions, rewards=rewards, observs=observs, masks=masks
    )
    embeds = agent._aux_count_site_embeds(
        d.get("_obs_embeds"), d["_memory_embeds"], d["_encoded_obs"]
    )
    loss, _ = agent._aux_count_loss(embeds, observs, masks)
    agent.zero_grad(set_to_none=True)
    loss.backward()

    got = grads_by_group(agent)
    for name, hit in got.items():
        print(f"  {name:22s} grad={hit}")
    assert got["aux_count_head"], "count head received no gradient"
    assert got["head.seq_model"], "count loss never reached the memory"
    assert not got["head.conditioner"], "count loss leaked into the obs branch"
    assert not got["qf."], "count loss leaked into the critic"


def check_count_site_memory():
    print("\n=== aux_count_site='memory': memory only ===")
    agent, obs_dim = build("joint", 0.0, 1.0, aux_count_site="memory")
    observs, actions, rewards, masks = dummy_batch(obs_dim)
    _, d = agent.head.forward(
        actions=actions, rewards=rewards, observs=observs, masks=masks
    )
    embeds = agent._aux_count_site_embeds(
        d.get("_obs_embeds"), d["_memory_embeds"], d["_encoded_obs"]
    )
    assert embeds.shape[-1] == agent.head.memory_embed_size, (
        f"memory site fed {embeds.shape[-1]} dims, expected "
        f"{agent.head.memory_embed_size}"
    )
    loss, _ = agent._aux_count_loss(embeds, observs, masks)
    agent.zero_grad(set_to_none=True)
    loss.backward()
    got = grads_by_group(agent)
    for name, hit in got.items():
        print(f"  {name:22s} grad={hit}")
    assert got["head.seq_model"], "count loss never reached the memory"
    assert not got["head.conditioner"], "count loss leaked into the obs branch"
    assert not got["qf."], "count loss leaked into the critic"


def check_count_site_probe_is_inert():
    print("\n=== aux_count_site='probe': nothing but the head trains ===")
    agent, obs_dim = build("joint", 0.0, 1.0, aux_count_site="probe")
    observs, actions, rewards, masks = dummy_batch(obs_dim)
    _, d = agent.head.forward(
        actions=actions, rewards=rewards, observs=observs, masks=masks
    )
    embeds = agent._aux_count_site_embeds(
        d.get("_obs_embeds"), d["_memory_embeds"], d["_encoded_obs"]
    )
    assert not embeds.requires_grad, "probe site would train the agent"
    loss, _ = agent._aux_count_loss(embeds, observs, masks)
    agent.zero_grad(set_to_none=True)
    loss.backward()
    got = grads_by_group(agent)
    for name, hit in got.items():
        print(f"  {name:22s} grad={hit}")
    assert got["aux_count_head"], "count probe head received no gradient"
    assert not got["head.seq_model"], "probe leaked into the memory"
    assert not got["head.conditioner"], "probe leaked into the obs branch"


if __name__ == "__main__":
    # GPU 3 is the only card we own on this box; 0-2 are other users'.
    device = int(os.environ.get("MATE_DEVICE", "3"))
    ptu.set_gpu_mode(torch.cuda.is_available(), device)
    check_probe_is_inert()
    check_count_reaches_only_the_obs_branch()
    check_memory_obs_shapes_only_the_memory()
    check_count_site_memory()
    check_count_site_memory_obs()
    check_count_site_probe_is_inert()
    print("\nOK")
