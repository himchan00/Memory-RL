# Symbolic Alchemy — 현황과 실행 계획

> 2026-08-31 · branch `alchemy_test` · 대시보드 HTML의 텍스트판
> (터미널/에디터/GitHub 어디서나 열립니다)

**한 줄 결론.** oracle은 정답지를 받고도 planner보다 **142점** 뒤집니다.
이건 upper bound가 망가져 있다는 뜻입니다. 천장이 161인 상태에서 MATE 152를
측정하면 "memory가 나쁘다"가 아니라 **"decoder가 161에서 막혀 있다"** 를 측정한
것입니다. oracle을 먼저 고치지 않으면 MATE 비교는 의미가 없습니다.

---

## 01. Symbolic Alchemy는 어떤 환경인가

DeepMind의 meta-RL 벤치마크 (arXiv:2102.02926). 돌(stone)을 물약(potion)에
담가 값을 올리고, 가마솥에 넣어 점수로 바꿉니다.

| 항목 | 값 |
|---|---|
| 잠재 공간 | 3D 큐브의 **8개 꼭짓점** (±1, ±1, ±1) |
| 보상 | 좌표 합. `(+1,+1,+1)`만 보너스 +12 → **+15**. 나머지 −3 / −1 / +1 / +3 |
| 물약 | 한 축의 부호를 뒤집음. 단, 그 모서리가 **막혀 있을 때(bottleneck)** 는 무효 |
| 1 trial | 돌 3개 + 물약 12개, 20 스텝 |
| 1 episode | **10 trial × 20 step = 200 step** |
| 행동 | 40개 = no-op 1 + (돌 3 × [가마솥 1 + 물약 12]) |

### 메타-RL 구조 — 여기가 핵심

**chemistry(회전·부호뒤집기·막힌 모서리)는 1 episode 내내 고정**이고,
돌/물약만 trial마다 새로 뽑힙니다.

```
trial:  0   1   2   3   4   5   6   7   8   9
        └── 잠재 chemistry 알아내기 ──┘   └ 알아낸 걸로 최적 플레이 ┘
```

그래서 **adaptation = mean(trial 7–9) − mean(trial 0–2)** 가 "episode 안에서
학습이 일어났는가"를 재는 지표입니다. return이 145~292로 큰 이유는 **10 trial
합산**이기 때문이지, trial당 평균이 아닙니다.

---

## 02. 에이전트가 실제로 보는 것

관측은 **39차원 벡터**입니다 (+ trial flag 1 = 40).

- 돌 슬롯 3개 × 5 feature = 15
- 물약 슬롯 12개 × 2 feature = 24

### 함정 3가지

1. **잠재 좌표는 절대 안 보입니다.** 매 episode 랜덤 회전(4가지) + 축별 부호
   뒤집기가 걸린 **perceptual frame**만 보입니다. 같은 돌이 seed마다 전혀 다른
   숫자로 나타납니다.
2. **슬롯은 교환 가능합니다.** 돌/물약이 어느 슬롯에 들어가는지는 iid 추첨이라
   정보가 0인데, MLP는 슬롯 순서를 의미 있는 축으로 봅니다.
3. **보상은 이미 새고 있습니다.** `perceived_stone.reward`가
   `LatentStone → AlignedStone → PerceivedStone`을 그대로 통과합니다
   (dm_alchemy 소스에서 확인). 즉 관측에 각 돌의 **진짜 잠재 보상**이 들어 있습니다.
   chemistry 없이도 145점이 나오는 이유이고, chemistry의 한계 가치는 오직
   **"어떤 물약을 써야 값이 오르는가"** 를 계획하는 데에만 있다는 뜻입니다.

### oracle이 받는 `chem_gt` 28차원 분해

| 차원 | 내용 | 형태 |
|---|---|---|
| 0–11 | graph — 큐브 12개 모서리의 개폐 | binary 12 |
| 12–17 | potion `dim_map` | **6가지 순열에 대한 one-hot** |
| 18–20 | potion `dir_map` | ±1 × 3 |
| 21–23 | stone map 축별 부호 | ±1 × 3 |
| 24–27 | rotation | **4가지 회전에 대한 one-hot** (회전 행렬은 미포함) |

정보는 완전합니다. 문제는 **쓸 수 있느냐**입니다.

---

## 03. 성능 — 스크립트 베이스라인 (1024 episode, seed 100–115)

`python scripts/eval_alchemy.py` 로 재생성합니다 (16 프로세스, 약 1분).

| 정책 | return | adaptation | normalized |
|---|---|---|---|
| `uniform_random` | 17.05 ± 0.55 | +0.22 | −0.903 |
| `random_stone_potion` (휴리스틱) | **145.20 ± 1.52** | +0.42 | **0.000** |
| `chemistry_oracle` (planner) | **287.08 ± 1.55** | +0.05 | **1.000** |
| — 학습 결과 — | | | |
| oracle_markov (정답지 받음) | 161.2 | | 0.113 |
| MATE | 152.4 | | 0.051 |
| canonicalized oracle (과거, 재현 불가) | 237.0 | | 0.647 |

`normalized = (return − 145.2) / (287.1 − 145.2)`.
논문 값(145.7 / 288.5)과 일치하므로 구현은 검증되었습니다.

스크립트 정책 셋은 adaptation ≈ 0 — episode 안에서 학습을 안 하니 당연하고,
이게 지표가 제대로 동작한다는 증거입니다.

### planner를 새로 만든 이유

dm_alchemy의 `IdealObserverBot`은 우리 설정에서 **사용 불가**입니다.

| 설정 | 소요 |
|---|---|
| 물약 6 × 돌 3 | 2.2 초 |
| 물약 8 × 돌 3 | 136 초 |
| **물약 12 × 돌 3 (우리 설정)** | **900초 내 미완료** |

dm_alchemy가 대신 배포한 녹화 trace는 **비회전 레벨 전용**입니다(1000개 평가
chemistry 전부 identity rotation). 논문의 `search_oracle`은 공개되지
않았습니다. 그래서 잠재 공간 memoized DFS로 직접 구현했고, 200 episode에 대해
행동 단위로 `search_oracle`과 완전히 일치함을 확인했습니다 (127–187 ms/episode).

---

## 04. 진단 — 142점 격차는 어디서 오는가

```
휴리스틱   148.6
   │  +12.5  (9%)   ← 정답지를 줘서 얻은 것
raw oracle 161.2
   │  +75.8  (54%)  ← 프레임 정렬만 해줬을 때 회복되는 몫
canonical  237.0
   │  +52.2  (37%)  ← 남은 계획 능력
planner    289.2
```

**격차의 절반 이상은 planning이 아니라 frame misalignment입니다.**

### 왜 못 쓰는가 — 요구되는 연산 사슬

`chem_gt`는 **잠재 프레임**, `symbolic_obs`는 **perceptual 프레임**입니다.
둘을 잇는 데 필요한 연산:

1. 4-way rotation one-hot → 회전 행렬 lookup → **3×3 matvec**
2. 축별 부호 뒤집기 → 잠재 노드
3. 6-way permutation one-hot → 순열 적용
4. `latent_dir = perceived_dir * dir_map[latent_dim]` — **계산된 인덱스로 gather**
   (MLP에게 가장 어려운 연산)
5. 8노드 그래프 탐색 + 물약 12개를 돌 3개에 20스텝 내 배분

이걸 전부 **width-128 · 2-layer conditioner MLP + width-128 · 2-layer critic**
안에서 해내야 합니다. 안 되는 게 정상입니다.

### 원인 3가지

| # | 원인 | 영향 |
|---|---|---|
| 1 | **frame misalignment** — 정답지가 다른 좌표계로 쓰여 있음 | 54% |
| 2 | **slot permutation** — 정보 0인 슬롯 순서를 축으로 취급 | MATE에 특히 치명적 |
| 3 | **decoder capacity** — 128×2로는 위 연산 사슬이 안 들어감 | 나머지 |

**2번이 MATE에 더 아픈 이유:** `full_transition=True`라 transition이
`(o_t, a_t, r_t, o_{t+1} − o_t)`이고, 그 **delta가 곧 chemistry 증거**입니다.
MATE의 running mean은 이 증거를 **12가지 슬롯 의존 기저로 쓰인 임베딩끼리**
평균내고 있습니다.

---

## 05. 저장소 상태

| 항목 | 상태 |
|---|---|
| `envs/alchemy_baselines.py` | ✅ 신규 — planner + 휴리스틱 + uniform |
| `scripts/eval_alchemy.py` | ✅ 신규 — 1024ep / seed 100–115 / normalized score |
| `eval/adaptation` 로깅 | ✅ 신규 — `learner.py` |
| `amlt/alchemy.yaml` | ✅ 갱신 — 아래 표대로 |
| 237.02를 냈던 permutation 코드 | ❌ 커밋 `7b76c44 "remove permutation training"`에서 제거됨. **재현 불가** |

### `amlt/alchemy.yaml` 갱신 내역

| 플래그 | 이전 | 현재 |
|---|---|---|
| `train_episodes` | 200000 | 24000 |
| `normalize_inputs` | false | true |
| `use_popart` | 없음 | true |
| `mask_alchemy_invalid_actions` | 없음 | true |
| `critic_lr` | 없음 (1e-4) | 3e-5 |
| `use_pe` | 없음 | true |
| `conditioning_hidden_dim` | 없음 (256) | 128 |
| `max_norm` | 1.0 오버라이드 | 기본값 0.2 |
| jobs | mate/gpt/lstm/oracle | oracle/markov/mate/mate_msc_ema_v2/gpt/lstm |

oracle·mate 두 설정 모두 `torch.compile` 기본값으로 end-to-end 스모크 실행을
마쳤습니다.

---

## 06. 실행 계획

### ✅ P2 — 인프라 (완료)

측정 기준을 먼저 세웁니다. 이게 없으면 어떤 실험 결과도 해석할 수 없습니다.

- planner + 휴리스틱을 저장소 API로 편입
- 평가 프로토콜 스크립트 + normalized score
- adaptation 지표 로깅
- amlt 레시피 갱신

### ▶ P0 — oracle 천장 올리기 (다음)

`--config_env.canonicalize_oracle=True` 를 **환경 쪽에서** 재구현합니다.
프레임 정렬을 신경망에 시키지 말고 환경이 미리 해서 주는 것입니다.

- 돌 슬롯 → 잠재 좌표(3) + 보상(1) + 존재(1)
- 물약 슬롯 → 잠재 축 one-hot(3) + 방향(1) + 존재(1)
- context는 graph 12차원만 남김

**목표: 237 재현.** oracle 전용 진단 기능이며, 이게 되면 격차의 54%가 프레임
문제였다는 진단이 확정됩니다.

### ⏸ P1 — slot-equivariant encoder (사용자 결정 대기)

슬롯 순서를 무시하는 인코더로 바꿉니다.

| 후보 | 내용 | 비용 |
|---|---|---|
| **DeepSets** (권장) | 슬롯별 공유 MLP → 합/평균 pooling. 순서 무관이 구조적으로 보장 | 낮음 |
| cross-attention | 학습된 query가 슬롯을 attend | 높음 |

함께 필요한 것: 물약 타입 one-hot(6) + absent flag, conditioner/critic 용량
재탐색, MATE용 slot-equivariant transition embedder.

---

## 부록 — 재현 명령

```bash
# 베이스라인 표 재생성 (약 1분)
python scripts/eval_alchemy.py --out logs/alchemy_baselines.json

# 학습 결과를 표에 얹어서 위치 확인
python scripts/eval_alchemy.py --compare oracle_markov=161.2 --compare mate=152.4
```

dm_alchemy는 archived 특수 설치입니다: `bash scripts/install_dm_alchemy.sh mate`
