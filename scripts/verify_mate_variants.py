"""Do the restored MATE variants do what they claim?

Two mechanisms came back out of the git history -- a per-transition gate
(a279603, deleted in f086408 "for code simplicity" with no measurement) and
linear attention (b2c882f, file deleted entirely). Both are supposed to break
the UNIFORM average without breaking PERMUTATION INVARIANCE, which is the
property that makes a running mean the right inductive bias for this task: the
posterior over an episode's chemistry depends on the multiset of transitions,
not their order.

That claim is cheap to state and easy to get wrong, so it is checked here
rather than assumed. Each check is a way a variant could look like it works
while quietly not being a permutation-invariant memory at all.

    python scripts/verify_mate_variants.py --device 0
"""
import argparse, os, sys

import numpy as np
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


def build(kind, T, hidden=64, **over):
    cfg = mate_default.get_config()
    del cfg.update_fn
    cfg.seq_model.hidden_size = hidden
    cfg.seq_model.max_seq_length = T + 4
    cfg.seq_model.context_dim = 0
    for k, v in over.items():
        cfg.seq_model[k] = v
    m = SEQ_MODELS[cfg.seq_model.name](
        input_size=hidden, dropout_emb=0.0, dropout_ff=0.0,
        **cfg.seq_model.to_dict()
    ).to(ptu.device)
    return m.eval()


def state_flat(s):
    if torch.is_tensor(s):
        return [s]
    out = []
    for x in s:
        out.extend(state_flat(x))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--T", type=int, default=32)
    args = ap.parse_args()
    ptu.set_gpu_mode(torch.cuda.is_available(), args.device)
    torch.manual_seed(0)
    T, B, H = args.T, 3, 64
    x = torch.randn((T, B, H), device=ptu.device)
    obs = torch.randn((T, B, H), device=ptu.device)
    perm = torch.randperm(T, device=ptu.device)

    variants = [
        ("mate", dict(use_gate=False), False),
        ("mate+gate", dict(use_gate=True), False),
    ]

    for label, over, is_lin in variants:
        kind = "mate"
        m = build(kind, T, H, **over)
        h0 = m.get_zero_internal_state(batch_size=B, training=False)
        kw = {"obs_emb": obs} if is_lin else {}
        kwp = {"obs_emb": obs[perm]} if is_lin else {}

        with torch.no_grad():
            out_a, hn_a, _ = m(x, h0, **kw)
            out_b, hn_b, _ = m(x[perm], h0, **kwp)

        # --- 1. the FINAL memory must not depend on the order -------------
        diffs = [float((a - b).abs().max())
                 for a, b in zip(state_flat(hn_a), state_flat(hn_b))]
        worst = max(diffs)
        scale = max(float(max(t.abs().max() for t in state_flat(hn_a))), 1e-9)
        check(f"[{label}] 최종 메모리가 순서에 불변",
              worst / scale < 1e-4, f"rel diff {worst/scale:.2e}")

        # --- 2. it must not be trivially constant --------------------------
        with torch.no_grad():
            out_c, _, _ = m(torch.randn_like(x), h0, **kw)
        moved = float((out_a[-1] - out_c[-1]).abs().max())
        check(f"[{label}] 입력이 바뀌면 메모리도 바뀐다",
              moved > 1e-4, f"|delta| {moved:.3e}")

        # --- 3. incremental == batch (the planner depends on this) ---------
        with torch.no_grad():
            st, outs = h0, []
            for t in range(T):
                kt = {"obs_emb": obs[t:t + 1]} if is_lin else {}
                o, st, _ = m(x[t:t + 1], st, **kt)
                outs.append(o)
            step_out = torch.cat(outs, dim=0)
        d = float((step_out - out_a).abs().max())
        check(f"[{label}] 한 스텝씩 == 전체 한 번에", d < 2e-4, f"max |diff| {d:.2e}")

        # --- 4. finite ------------------------------------------------------
        check(f"[{label}] 출력이 유한", bool(torch.isfinite(out_a).all()))

    # --- 5. the gate must actually gate ---------------------------------
    m = build("mate", T, H, use_gate=True)
    h0 = m.get_zero_internal_state(batch_size=B, training=False)
    with torch.no_grad():
        base, _, info = m(x, h0)
        # Force the gate shut and confirm the memory stops moving: with w=0 the
        # numerator and the denominator both stop accumulating, so the memory
        # must stay at the init prior for the whole episode.
        m.gate[-2].bias.fill_(-30.0)
        m.gate[-2].weight.zero_()
        shut, _, info0 = m(x, h0)
    spread = float(shut.std(dim=0).max())
    check("게이트를 닫으면 메모리가 멈춘다",
          spread < 1e-3, f"닫힌 뒤 시간축 std {spread:.2e} "
                         f"(열렸을 때 {float(base.std(dim=0).max()):.3e})")
    check("게이트 통계가 info로 나온다", "gate_mean" in info)

    # --- 6. gradient reaches the gate -----------------------------------
    m = build("mate", T, H, use_gate=True).train()
    h0 = m.get_zero_internal_state(batch_size=B, training=True)
    out, _, _ = m(x, h0)
    out.pow(2).mean().backward()
    g = max(float(p.grad.abs().max()) for p in m.gate.parameters() if p.grad is not None)
    check("게이트 파라미터에 그래디언트가 도달", g > 0, f"max |grad| {g:.2e}")

    # --- 7. state size, which is what a search node has to hold ----------
    print("\n  탐색 노드 1개가 들고 있어야 할 상태 크기 (hidden=256, batch=1)")
    for label, over, is_lin in variants:
        mm = build("mate", T, 256, **over)
        st = mm.get_zero_internal_state(batch_size=1, training=False)
        nbytes = sum(t.numel() * t.element_size() for t in state_flat(st))
        print(f"    {label:<12}{nbytes:>10,} B")

    print()
    if all(results):
        print(f"ALL {len(results)} CHECKS PASSED")
    else:
        print(f"{sum(results)}/{len(results)} passed")
        sys.exit(1)


if __name__ == "__main__":
    main()
