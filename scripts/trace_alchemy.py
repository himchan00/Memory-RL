"""Step-by-step trace of one Symbolic Alchemy episode: hidden truth vs. agent view.

Pedagogical tool, not part of training. Replays one episode with a random
policy (what an early-training agent experiences) and prints, side by side,
the perceptual frame the agent observes and the latent frame it cannot see --
then summarizes what is available as CONTEXT: the observed transition deltas
(what MATE/LSTM/GPT accumulate) versus the 28-dim ``chem_gt`` answer key
(oracle only, written in the wrong coordinate frame).

Usage::

    python scripts/trace_alchemy.py --seed 6 --show_steps 14
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.alchemy import SymbolicAlchemyEnv, decode_action
from envs.alchemy_baselines import RandomStonePotionPolicy
from dm_alchemy.types import stones_and_potions as sp, graphs

LEVEL = "perceptual_mapping_randomized_with_rotation_and_random_bottleneck"
AXIS = "xyz"
W = 80


def rule(char="=", text=None):
    if text is None:
        print(char * W)
    else:
        print("\n" + char * W + f"\n{text}\n" + char * W)


def latent_str(c):
    return "(" + ",".join(f"{int(v):+d}" for v in c) + ")"


def seen_str(v):
    return "[" + " ".join(f"{x:+.0f}" for x in v) + "]"


def potion_name(dim, direction):
    return f"{AXIS[dim]}{'+' if direction > 0 else '-'}"


def blocked_edges(dm):
    adj = np.asarray(graphs.convert_graph_to_adj_mat(dm._chemistry.graph))
    out = []
    for i in range(8):
        ci = np.asarray(sp.index_to_coords(i))
        for j in range(i + 1, 8):
            cj = np.asarray(sp.index_to_coords(j))
            if int(np.abs(ci - cj).sum()) == 2 and adj[i, j] == graphs.NO_EDGE:
                out.append(f"{latent_str(ci)} <-> {latent_str(cj)}")
    return out


def state(dm):
    chem = dm._chemistry
    stones, potions = [], []
    for s in dm.game_state.existing_stones():
        perceived = sp.unalign(chem.stone_map.apply_inverse(s.latent_stone()),
                               chem.rotation)
        stones.append({
            "slot": dm.game_state.get_stone_ind(stone_inst=s.idx),
            "seen": np.asarray(perceived.perceived_coords, float),
            "reward": dm._reward_weights(s.latent),
            "latent": np.asarray(s.latent, float),
        })
    for p in dm.game_state.existing_potions():
        perceived = chem.potion_map.apply_inverse(p.latent_potion())
        potions.append({
            "slot": dm.game_state.get_potion_ind(potion_inst=p.idx),
            "type": perceived.index(),
            "latent": potion_name(p.dimension, p.direction),
        })
    return stones, potions


def print_state(dm):
    stones, potions = state(dm)
    print("      " + "-" * 68)
    print("      슬롯 |  에이전트가 보는 좌표  보상  |  (숨은 잠재좌표)")
    for s in stones:
        print(f"        {s['slot']}  |     {seen_str(s['seen'])}       {s['reward']:+3.0f}"
              f"  |     {latent_str(s['latent'])}")
    counts = {}
    for p in potions:
        counts[(p["type"], p["latent"])] = counts.get((p["type"], p["latent"]), 0) + 1
    desc = "   ".join(f"타입{t}x{n}" for (t, _), n in sorted(counts.items()))
    truth = "   ".join(f"타입{t}={lat}" for (t, lat), _ in sorted(counts.items()))
    print(f"      물약 {len(potions)}개: {desc}")
    print(f"                  (숨은 진실: {truth})")
    print("      " + "-" * 68)


def main(seed=6, show_steps=14, policy_seed=3):
    env = SymbolicAlchemyEnv(LEVEL, num_trials=10, max_steps_per_trial=20)
    env.reset(seed=seed)
    dm = env._env  # the raw dm_alchemy env holds the hidden chemistry

    # ---------------------------------------------------------------- truth
    rule("=", "PART 1.  이 EPISODE의 숨은 정답 — 10 trial 내내 고정, 에이전트는 못 봄")

    print("\n[1] 잠재 큐브: 8개 꼭짓점, 보상은 좌표의 합")
    for st in sp.possible_latent_stones():
        c = st.latent_coords
        r = dm._reward_weights(c)
        note = "   <<< 목표: 합 +3 에 보너스 +12" if r > 3 else ""
        print(f"      {latent_str(c)}  ->  {r:+3.0f}{note}")

    blocked = blocked_edges(dm)
    print(f"\n[2] 막힌 모서리 (bottleneck): 12개 중 {len(blocked)}개가 막힘")
    for e in blocked:
        print(f"      X  {e}   <- 이 두 꼭짓점 사이는 물약으로 못 건너감")

    print("\n[3] 지각 변환: 잠재좌표를 가려버리는 부분")
    chem = dm._chemistry
    print(f"      회전행렬  = {chem.rotation.tolist()}")
    print(f"      부호뒤집기 = {chem.stone_map.latent_pos_dir}")
    print("\n      물약 라벨 -> 실제 효과 (에이전트는 이 표를 절대 못 봄):")
    for d in range(3):
        for s in (-1, 1):
            pp = chem.potion_map.apply_inverse(sp.LatentPotion(d, s))
            print(f"        타입{pp.index()} (obs 값 {pp.index()/3.0-1.0:+.3f})"
                  f"  ->  실제로는 잠재축 {potion_name(d, s)} 로 뒤집음")

    # ------------------------------------------------------------ agent view
    rule("=", "PART 2.  에이전트가 받는 관측 39-dim — 위 정답은 한 글자도 없음")
    obs = np.asarray(dm.observation()["symbolic_obs"], float)
    print("\n  돌 3슬롯 x 5 + 물약 12슬롯 x 2 = 39")
    for i in range(3):
        print(f"    돌 슬롯{i}  {np.round(obs[5*i:5*i+5], 3)}"
              + ("   <- (지각좌표 x3, 보상/15, 사용됨)" if i == 0 else ""))
    print(f"    물약 0~5  {np.round(obs[15:27], 3)}   <- (타입값, 사용됨) x 6")
    print("\n  주의: 4번째 숫자(보상)는 잠재 보상이 그대로 새어나온 값입니다.")
    print("       chemistry를 몰라도 145점이 나오는 이유입니다.")

    # ---------------------------------------------------------------- replay
    rule("=", f"PART 3.  TRIAL 0 을 한 스텝씩 — 무작위 정책 (= 학습 초기 에이전트)")
    policy = RandomStonePotionPolicy(seed=policy_seed)
    policy.reset()
    print("\n  시작 상태:")
    print_state(dm)

    learned = {}   # 에이전트가 관측만으로 알아낼 수 있는 것
    blocked_seen = []
    total = 0.0
    for step in range(env.max_episode_steps):
        before, potions_before = state(dm)
        action = policy.act(dm)
        d = decode_action(action)
        _, reward, _, _, _ = env.step(action)
        total += reward
        after, _ = state(dm)
        if step >= show_steps:
            continue

        if d.kind == "no_op":
            print(f"    step {step:2d}  아무것도 안 함")
            continue
        if d.kind == "cash":
            s0 = next((s for s in before if s["slot"] == d.stone_index), None)
            print(f"    step {step:2d}  슬롯{d.stone_index} 돌 -> 가마솥.  "
                  f"보상 {reward:+.0f}   (잠재좌표 {latent_str(s0['latent'])})")
            continue

        b = next((s for s in before if s["slot"] == d.stone_index), None)
        a = next((s for s in after if s["slot"] == d.stone_index), None)
        p = next((q for q in potions_before if q["slot"] == d.potion_index), None)
        moved = not np.allclose(a["latent"], b["latent"])
        print(f"\n    step {step:2d}  슬롯{d.stone_index} 돌을 [타입{p['type']}] 물약에 담금")
        print(f"            에이전트 시점:  {seen_str(b['seen'])} 보상{b['reward']:+3.0f}"
              f"   ->   {seen_str(a['seen'])} 보상{a['reward']:+3.0f}"
              f"    {'<< 바뀜' if moved else '<< 아무 변화 없음'}")
        print(f"            숨은 진실   :  {latent_str(b['latent'])}"
              f"   ->   {latent_str(a['latent'])}   (타입{p['type']} = 잠재축 {p['latent']})")
        if moved:
            delta = a["seen"] - b["seen"]
            axes = np.nonzero(np.abs(delta) > 1e-6)[0]
            note = (f"타입{p['type']} 은 지각축 {[int(a) for a in axes]} 를 "
                    f"{'+' if delta[axes[0]] > 0 else '-'} 방향으로 움직인다")
            print(f"            >> 에이전트가 배우는 것: {note}")
            learned.setdefault(p["type"], note)
        else:
            fact = f"{latent_str(b['latent'])} 에서 타입{p['type']} 은 막혀 있다"
            print(f"            >> 에이전트가 배우는 것: {fact}")
            blocked_seen.append(fact)

    print(f"\n  ... (trial 0 나머지 스텝 생략)   episode 총 return {total:.1f}")

    # --------------------------------------------------------------- context
    rule("=", "PART 4.  그래서 무엇이 CONTEXT 가 될 수 있는가")

    print("\n[A] 관측된 transition — MATE / LSTM / GPT 가 실제로 쓰는 것")
    print("    한 스텝은 (o_t, a_t, r_t, o_t+1 - o_t) 로 인코딩됩니다.")
    print("    chemistry 증거는 전부 마지막 항 (delta) 안에 들어 있습니다.")
    print(f"\n    위 {show_steps} 스텝만으로 에이전트가 알아낸 것:")
    for t, note in sorted(learned.items()):
        print(f"      * {note}")
    for f in blocked_seen[:4]:
        print(f"      * {f}")
    print(f"\n    아직 모르는 물약 타입: {sorted(set(range(6)) - set(learned))}")
    print("    -> trial 0~2 에서 이걸 모으고, trial 7~9 에서 써먹습니다.")
    print("       그 차이가 adaptation 지표입니다.")

    print("\n[B] chem_gt 28-dim — oracle 에게만 주는 정답지")
    g = np.asarray(dm.observation()["chem_gt"], float)
    print(f"    0-11  graph      {np.round(g[0:12],1)}")
    print(f"    12-17 dim_map    {np.round(g[12:18],1)}   <- 6가지 순열 one-hot")
    print(f"    18-20 dir_map    {np.round(g[18:21],1)}")
    print(f"    21-23 stone_map  {np.round(g[21:24],1)}")
    print(f"    24-27 rotation   {np.round(g[24:28],1)}   <- 4가지 회전 one-hot")
    print("\n    문제: 이건 '잠재 프레임' 언어이고 관측은 '지각 프레임' 언어입니다.")
    print("    둘을 잇는 데 회전행렬 곱 + 순열 적용 + 계산된 인덱스 gather 가 필요합니다.")
    print("    그래서 정답지를 통째로 받고도 oracle 이 161 점에 머무릅니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=6,
                        help="episode seed; 6 has 5 blocked edges (a hard, illustrative board)")
    parser.add_argument("--show_steps", type=int, default=14,
                        help="how many steps of trial 0 to print")
    parser.add_argument("--policy_seed", type=int, default=3)
    args = parser.parse_args()
    main(seed=args.seed, show_steps=args.show_steps, policy_seed=args.policy_seed)
