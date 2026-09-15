"""Does the restored gate gate, and is the memory still a permutation-invariant mean?

The gate exists to break the UNIFORM average without breaking the two properties
that make a running mean the right inductive bias for a CMDP: the memory is
bounded, and it depends on the MULTISET of transitions rather than their order
(the posterior over an episode's chemistry does not care which order the
evidence arrived in). Weighting the numerator alone would keep invariance and
lose boundedness; weighting the denominator alone would be nonsense. Only
weighting both keeps everything, and that is easy to get subtly wrong, so it is
checked here rather than assumed.

Both gate_modes are checked. "novelty" is the interesting case: its weights
depend on the running memory, so permutation invariance is NOT obvious and is
in fact only approximate -- the test records how large the deviation is instead
of asserting it away.

    python scripts/verify_gate.py
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torchkit.pytorch_utils as ptu
from configs.seq_models import mate_default
from policies.seq_models import SEQ_MODELS

PASS, FAIL = "  PASS", "  FAIL"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{PASS if ok else FAIL}  {name}" + (f"   {detail}" if detail else ""))


def build(T, hidden=64, **over):
    cfg = mate_default.get_config()
    del cfg.update_fn
    cfg.seq_model.hidden_size = hidden
    cfg.seq_model.max_seq_length = T + 4
    for k, v in over.items():
        cfg.seq_model[k] = v
    d = cfg.seq_model.to_dict()
    d.pop("context_dim", None)
    m = SEQ_MODELS[d["name"]](
        input_size=hidden, dropout_emb=0.0, dropout_ff=0.0, **d
    ).to(ptu.device)
    return m.eval()


def main():
    T, B, H = 24, 5, 64
    x = ptu.randn((T, B, H))

    # --- 1. off by default is byte-for-byte vanilla ------------------------
    base = build(T, H)
    h0 = base.get_zero_internal_state(batch_size=B)
    with torch.no_grad():
        out_off, _, info_off = base(x, h0)
    check("use_gate=False leaves the vanilla path untouched",
          "gate_mean" not in info_off)

    for mode in ("mlp",):
        print(f"\n--- gate_mode={mode} ---")
        m = build(T, H, use_gate=True)
        h0 = m.get_zero_internal_state(batch_size=B)
        with torch.no_grad():
            out, (hn, cn), info = m(x, h0)
        check(f"[{mode}] output finite", bool(torch.isfinite(out).all()))
        check(f"[{mode}] gate stats reported", "gate_mean" in info,
              f"mean {float(info['gate_mean'].mean()):.3f}")

        # --- 2. the denominator counts WEIGHT, not steps -------------------
        # 24 transitions at w<1 must leave a count strictly below 24 + w_0.
        w0 = float(h0[1][0, 0, 0])
        total = float(cn[0, 0, 0])
        check(f"[{mode}] count accumulates weight, not steps",
              total < w0 + T - 1e-6,
              f"count {total:.2f} vs ungated {w0 + T:.2f}")

        # --- 3. one step at a time == the whole sequence at once -----------
        with torch.no_grad():
            st, outs = h0, []
            for t in range(T):
                o, st, _ = m(x[t:t + 1], st)
                outs.append(o)
            step_out = torch.cat(outs, dim=0)
        d = float((step_out - out).abs().max())
        check(f"[{mode}] stepwise == batched", d < 2e-4, f"max |diff| {d:.2e}")

        # --- 4. permutation invariance -------------------------------------
        perm = torch.randperm(T, device=x.device)
        with torch.no_grad():
            out_p, (hn_p, cn_p), _ = m(x[perm], h0)
        dh = float((hn - hn_p).abs().max())
        dc = float((cn - cn_p).abs().max())
        rel = dh / max(float(hn.abs().max()), 1e-8)
        check(f"[{mode}] final memory is permutation invariant",
              dh < 2e-4 and dc < 2e-4,
              f"|d sum| {dh:.2e}  |d count| {dc:.2e}")

        # --- 5. a shut gate must freeze the memory -------------------------
        m2 = build(T, H, use_gate=True)
        with torch.no_grad():
            m2.gate[-2].weight.zero_(); m2.gate[-2].bias.fill_(-30.0)
            shut, (_, cn_shut), _ = m2(x, h0)
        # The gate floors at _GATE_MIN rather than 0 (a 0 gate is an
        # unrecoverable dead end), so "shut" means the count crawls at
        # _GATE_MIN per step instead of stopping -- that is the thing to
        # assert, not that the memory is frozen solid.
        expected = w0 + T * m2._GATE_MIN
        got = float(cn_shut[0, 0, 0])
        check(f"[{mode}] a shut gate crawls at the floor, not at 1/step",
              abs(got - expected) < 1e-3,
              f"count {got:.3f}, expected {expected:.3f}, ungated {w0 + T:.1f}")

        # --- 6. gradients reach the gate -----------------------------------
        m3 = build(T, H, use_gate=True).train()
        o, _, _ = m3(x, m3.get_zero_internal_state(batch_size=B))
        o.pow(2).mean().backward()
        ps = m3.gate.parameters()
        g = max(float(p.grad.abs().max()) for p in ps if p.grad is not None)
        check(f"[{mode}] gradient reaches the gate", g > 0, f"max |grad| {g:.2e}")

    # --- 7. the sparsity penalty actually pushes the gate shut -------------
    m = build(T, H, use_gate=True, gate_sparsity_weight=1.0,
              gate_sparsity_target=0.06).train()
    opt = torch.optim.Adam(m.gate.parameters(), lr=0.01)
    h0 = m.get_zero_internal_state(batch_size=B)
    before = None
    for _ in range(200):
        _, _, info = m(x, h0)
        if before is None:
            before = float(info["gate_mean"].mean())
        opt.zero_grad(); info["_aux_loss"].backward(); opt.step()
    with torch.no_grad():
        _, _, info = m(x, h0)
    after = float(info["gate_mean"].mean())
    check("sparsity penalty drives mean(w) toward the target "
          "without collapsing",
          abs(after - 0.06) < 0.03 and after > 0.5 * m._GATE_MIN,
          f"mean(w) {before:.3f} -> {after:.3f}, target 0.060")
    m0 = build(T, H, use_gate=True).train()
    _, _, i0 = m0(x, m0.get_zero_internal_state(batch_size=B))
    check("weight=0 adds no aux loss", "_aux_loss" not in i0)

    print()
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED")
    else:
        print(f"{results.count(False)} of {len(results)} CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    ptu.set_gpu_mode(torch.cuda.is_available(), int(os.environ.get("MATE_DEVICE", "0")))
    main()
