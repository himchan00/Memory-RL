"""Where does an MSC loss actually reach, and does any of it touch the policy?

`detach_z` decides whether the contrastive loss can change `z`, the
per-transition embedding that IS MATE's memory: m_t = (init + Σ z_i)/(w + t).
Detached, the loss keeps training SOMETHING -- but whether that something is
read by the policy differs between the two MSC variants, and getting it wrong
costs a whole 24k-episode arm:

  v2 detached      trains msc.weight and msc.log_kappa. Both appear only inside
                   the loss, so the auxiliary is INERT: the agent is vanilla
                   MATE with a probe trained beside it. Not an experiment.
  legacy detached  trains log_gains and head. `log_gains` is multiplied into
                   the memory the policy reads (mate_vanilla.py), so the arm is
                   real -- it asks whether re-weighting the memory's axes is
                   enough without changing how transitions are encoded.

    python scripts/verify_msc_reach.py
"""
import contextlib
import io
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torchkit.pytorch_utils as ptu
from configs.seq_models import mate_msc_default, mate_msc_v2_default
from policies.seq_models import SEQ_MODELS

PASS, FAIL = "  PASS", "  FAIL"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{PASS if ok else FAIL}  {name}" + (f"   {detail}" if detail else ""))


T, B, H = 24, 8, 64


def build(mod, **over):
    cfg = mod.get_config()
    del cfg.update_fn
    cfg.seq_model.hidden_size = H
    cfg.seq_model.max_seq_length = T + 4
    cfg.seq_model.n_layer = 1
    for k, v in over.items():
        cfg.seq_model[k] = v
    d = cfg.seq_model.to_dict()
    d.pop("context_dim", None)
    with contextlib.redirect_stdout(io.StringIO()):
        m = SEQ_MODELS["mate"](input_size=H, dropout_emb=0.0, dropout_ff=0.0, **d)
    return m.to(ptu.device).train()


def reach(m):
    """(params the aux loss moves, params the policy's memory depends on)."""
    x = ptu.randn((T, B, H))
    h0 = m.get_zero_internal_state(batch_size=B)
    out, _, info = m(x, h0, mask=torch.ones((T, B, 1), device=ptu.device))
    m.zero_grad()
    info["_aux_loss"].backward(retain_graph=True)
    aux = {n for n, p in m.named_parameters()
           if p.grad is not None and p.grad.abs().sum() > 0}
    m.zero_grad()
    out.pow(2).mean().backward()
    pol = {n for n, p in m.named_parameters()
           if p.grad is not None and p.grad.abs().sum() > 0}
    return aux, pol


def main():
    for mod, tag, over in ((mate_msc_v2_default, "v2", {}),
                           (mate_msc_default, "legacy-temporal",
                            {"msc_view": "temporal"})):
        aux, pol = reach(build(mod, msc_detach_z=False, **over))
        emb = {n for n in aux if n.startswith("embedder")}
        check(f"[{tag}] detach_z=False reaches the embedder", bool(emb),
              f"{len(emb)} embedder tensors")
        check(f"[{tag}] detach_z=False reaches the policy", bool(aux & pol),
              f"{len(aux & pol)} shared")

        aux, pol = reach(build(mod, msc_detach_z=True, **over))
        emb = {n for n in aux if n.startswith("embedder")}
        check(f"[{tag}] detach_z=True spares the embedder", not emb)
        check(f"[{tag}] detach_z=True still trains something", bool(aux),
              ", ".join(sorted(aux)))
        shared = aux & pol
        if tag == "v2":
            check("[v2] detach_z=True is INERT for the policy", not shared,
                  "nothing it trains is read by the policy")
        else:
            check("[legacy] detach_z=True still moves the policy via gains",
                  "msc.log_gains" in shared, ", ".join(sorted(shared)))
            aux2, pol2 = reach(build(mod, msc_detach_z=True,
                                     msc_learn_gains=False, **over))
            check("[legacy] detach_z=True + learn_gains=False is INERT too",
                  not (aux2 & pol2),
                  "the control that isolates the gains")

    print()
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED")
    else:
        print(f"{results.count(False)} of {len(results)} CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    ptu.set_gpu_mode(torch.cuda.is_available(), int(os.environ.get("MATE_DEVICE", "2")))
    main()
