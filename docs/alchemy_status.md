# Symbolic Alchemy — 현황과 실행 계획

> 2026-08-31 · branch `alchemy_test` · 대시보드 HTML의 텍스트판
> (터미널/에디터/GitHub 어디서나 열립니다)

**한 줄 결론.** oracle은 정답지를 받고도 planner보다 **129점** 뒤졌습니다
(실측 158.2 / 141.9 폭 기준 **9.2%** 지점). 천장이 158인 상태에서 MATE 152를
재면 "memory가 나쁘다"가 아니라 **"decoder가 158에서 막혀 있다"** 를 잰
것이므로, MATE 비교 이전에 oracle부터 고쳐야 했습니다.

**그 1단계를 끝냈습니다.** 관측을 잠재 프레임으로 되돌려 주자
(`canonicalize_oracle`) oracle이 **158.2 → 192.8** (normalized 0.09 → 0.34,
seed 2개) 로 올랐습니다. MATE(152)와의 여유가 6점(노이즈)에서 **41점**
(측정 가능)으로 벌어졌으므로 이제 자가 비교가 의미를 갖습니다.

**무엇이 병목인지 세 실험으로 좁혔습니다 (05절).** γ 0.99 → 0.999 는 0.2점
차이 → **RL 지평은 아닙니다.** RL을 들어내고 planner를 지도학습으로 흉내내면,
지각 프레임에서는 118.5(floor 아래)이지만 잠재 프레임으로 되돌려 주면
164.9(floor 위) → **frame은 실재하는 병목**입니다. 그리고 그 프레임 정렬을
RL oracle에 직접 넣자(`canonicalize_oracle=True`) **158.2 → 192.8**,
normalized 0.09 → **0.34 (3.6배)** 로 올랐습니다 (seed 2개, 폭 1.5점 vs 효과
34.6점).

즉 129점 격차의 분해는 **frame 27% : search 73%** 입니다. 이 문서가 이전에
적었던 "53%는 frame"(출처 불명 237.0 기반)은 폐기했고, 그 뒤 잠시 적었던
"86%는 search"는 *지도학습* 기준이라 RL 기준으로는 과대평가였습니다.

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

## 02. 실전 플레이 — 실제 episode 한 판을 한 스텝씩

`python scripts/trace_alchemy.py --seed 6` 의 출력입니다. 학습 초기 에이전트를
흉내내기 위해 **무작위 정책**으로 플레이했습니다.

### ① 이 episode의 숨은 규칙 (10 trial 내내 고정, 에이전트는 못 봄)

**잠재 큐브** — 돌의 진짜 정체는 8개 꼭짓점 중 하나. 보상 = 좌표 합.

```
(-1,-1,-1) -3      (+1,+1,-1) +1
(+1,-1,-1) -1      (+1,-1,+1) +1
(-1,+1,-1) -1      (-1,+1,+1) +1
(-1,-1,+1) -1      (+1,+1,+1) +15   <- 보너스 +12. 목표는 이 한 점.
```

**Bottleneck** — 12개 모서리 중 5개가 막힘.

```
X (-1,-1,-1) <-> (+1,-1,-1)      X (-1,+1,-1) <-> (+1,+1,-1)
X (-1,-1,-1) <-> (-1,+1,-1)      X (+1,-1,+1) <-> (+1,+1,+1)   <- 목표 바로 옆
X (+1,-1,-1) <-> (+1,+1,-1)
```

막힌 모서리에 물약을 쓰면 아무 일도 안 일어나고 물약만 소모됩니다.
이 "no-op 관측"이 bottleneck을 배우는 **유일한 경로**입니다.

**지각 변환** — 이 환경의 전부.

```
회전행렬  = [[1, 0, -1], [0, 1, 0], [1, 0, 1]]   <- x·z 를 섞음
부호뒤집기 = [+1, -1, -1]
```

| 관측 라벨 | obs 값 | 실제 효과 |
|---|---|---|
| 타입0 | −1.000 | 잠재축 `x+` |
| 타입1 | −0.667 | 잠재축 `x−` |
| 타입2 | −0.333 | 잠재축 `z+` |
| 타입3 | +0.000 | 잠재축 `z−` |
| 타입4 | +0.333 | 잠재축 `y−` |
| 타입5 | +0.667 | 잠재축 `y+` |

다음 episode에서는 이 표가 완전히 달라집니다. **"타입0"이라는 라벨 자체에는
아무 의미가 없고**, 이번 판에서 관측한 효과로만 의미가 생깁니다.

### ② Trial 0 시작 상태 — 같은 장면을 두 프레임으로

```
슬롯 | 에이전트가 보는 것        | (숨은 진실)
  0  | [+0 +1 -1]  보상  +1     | (+1,-1,+1)
  1  | [+0 -1 -1]  보상 +15     | (+1,+1,+1)   <- 이미 정답. 그냥 넣으면 됨
  2  | [+0 -1 +1]  보상  -1     | (-1,+1,-1)   <- 올려야 함

물약 12개: 타입0x5  타입1x2  타입2x1  타입3x2  타입4x1  타입5x1
```

보상 숫자는 지각 변환을 거치지 않고 잠재 보상이 그대로 나오므로, 슬롯1이
15점짜리라는 걸 chemistry 없이도 이미 압니다 (아래 "함정 3").

### ③ Trial 0 을 한 스텝씩

**step 0 — 첫 정보 획득.** 잠재적으로는 z축 *하나만* 뒤집혔는데, 회전 때문에
에이전트 눈에는 두 축이 동시에 움직입니다.

```
step 0  슬롯0 돌 + [타입3] 물약
  에이전트: [+0 +1 -1] 보상 +1  ->  [+1 +1 +0] 보상 -1   << 바뀜
  숨은진실: (+1,-1,+1)         ->  (+1,-1,-1)          (타입3 = z-)
  >> 배우는 것: "타입3은 지각축 0과 2를 + 방향으로 민다"
     보상이 +1 -> -1 로 내려갔습니다. 잘못 골랐습니다.
```

**step 1 — 아무 일도 안 일어남. 이것도 정보입니다.**

```
step 1  슬롯2 돌 + [타입5] 물약
  에이전트: [+0 -1 +1] 보상 -1  ->  [+0 -1 +1] 보상 -1   << 변화 없음
  숨은진실: (-1,+1,-1)         ->  (-1,+1,-1)          (타입5 = y+)
  >> 배우는 것: "(이 위치)에서 타입5는 막혔다"
```

**step 2–10 — 무작위 정책의 낭비. memory의 가치가 드러나는 지점입니다.**

```
step  2  슬롯0 + 타입0  ->  변화 없음  (막힘)
step  3  슬롯0 + 타입0  ->  변화 없음  (똑같은 걸 또 시도)
step  4  슬롯2 + 타입4  ->  변화 없음  (막힘)
step  5  슬롯0 + 타입0  ->  변화 없음  (세 번째)
step  7  슬롯0 + 타입1  ->  변화 없음
step  8  슬롯0 + 타입0  ->  변화 없음  (네 번째)
step  9  슬롯2 + 타입1  ->  변화 없음
step 10  슬롯0 + 타입3  ->  변화 없음
```

슬롯0의 돌 `(+1,-1,-1)`은 사실상 갇혔습니다. 나가는 3개 모서리 중 2개가
막혔는데, 무작위 정책은 그걸 모르니 같은 실패를 네 번 반복합니다.
**메모리가 있는 에이전트라면 step 3부터는 이 시도를 하지 않습니다.**

**step 6, 11 — 성공 경로. 2스텝이면 충분했습니다.**

```
step  6  슬롯2 + [타입2]:  [+0 -1 +1](-1) -> [-1 -1 +0](+1)
                           (-1,+1,-1)     -> (-1,+1,+1)     z+ 적용
step 11  슬롯2 + [타입0]:  [-1 -1 +0](+1) -> [+0 -1 -1](+15)  << 정답 도달!
                           (-1,+1,+1)     -> (+1,+1,+1)     x+ 적용
step 12  슬롯1 돌 -> 가마솥  +15
step 13  슬롯2 돌 -> 가마솥  +15
         슬롯0의 갇힌 돌은 -1 이라 버립니다.
```

−1짜리 돌을 단 2스텝만에 15점으로 만들었습니다. chemistry를 알았다면 이
2스텝을 **맨 처음 2스텝에** 했을 것입니다. 그 차이가 곧 `adaptation` 입니다.
(이 판은 bottleneck 5개짜리 어려운 판이라 episode 총 return 111.0)

### ④ 그래서 무엇이 CONTEXT 가 될 수 있는가 — 두 종류

**[A] 관측된 transition — MATE / LSTM / GPT 가 실제로 쓰는 것**

```
( o_t , a_t , r_t , o_{t+1} - o_t )
                    └──────┬─────┘
              chemistry 증거는 전부 여기
```

delta가 곧 실험 결과입니다. step 0이면 `a_t` = "슬롯0 + 타입3",
`delta` = `[+1, 0, +1]` → "타입3은 지각축 0,2를 + 로 민다".
delta가 **0벡터**면 → "이 상태에서 이 물약은 막혔다".

위 14스텝만으로 축적된 것:

```
✓ 타입0 -> 지각축 (0,2) 를 + 로       ✓ (-1,+1,-1) 에서 타입5 막힘
✓ 타입2 -> 지각축 (0,2) 를 - 로       ✓ (+1,-1,-1) 에서 타입0 막힘
✓ 타입3 -> 지각축 (0,2) 를 + 로       ✓ (-1,+1,-1) 에서 타입4 막힘
? 아직 모르는 물약 타입: 1, 4, 5
```

MATE는 이 transition 임베딩들의 running mean을 memory로 씁니다. "어떤 실험을
했고 뭐가 나왔나"의 **순서 없는 집합**이라는 점은 이 문제에 잘 맞습니다.
문제는 슬롯 순서가 정보 0인데 MLP 임베더가 이를 의미 있는 축으로 취급한다는
것입니다 → 아래 "원인 2" / P1 (DeepSets)의 근거.

**[B] `chem_gt` 28차원 — oracle 에게만 주는 정답지**

정보는 완전하지만 **잠재 프레임** 언어이고 관측은 **지각 프레임** 언어입니다.
분해와 필요한 연산 사슬은 아래 03 / 05 절에서 이어집니다.
결과: 정답지를 통째로 받은 oracle이 **158.2점** (천장의 9.2%, 24k episode 실측).
관측까지 잠재 프레임으로 되돌려 주면 **192.8점** (33.6%) — 05절 실험 C.

---

## 03. 에이전트가 실제로 보는 것

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

## 04. 성능 — 스크립트 베이스라인 (1024 episode, seed 100–115)

`python scripts/eval_alchemy.py` 로 재생성합니다 (16 프로세스, 약 1분).

| 정책 | return | adaptation | normalized |
|---|---|---|---|
| `uniform_random` | 17.05 ± 0.55 | +0.22 | −0.903 |
| `random_stone_potion` (휴리스틱) | **145.20 ± 1.52** | +0.42 | **0.000** |
| `chemistry_oracle` (planner) | **287.08 ± 1.55** | +0.05 | **1.000** |
| — 학습 결과 — | | | |
| oracle_markov, γ=0.99 (본 저장소 실측, 24k ep) | 158.36 ± 1.45 | | 0.093 |
| oracle_markov, γ=0.999 (본 저장소 실측, 24k ep) | 158.13 ± 1.37 | | 0.091 |
| **oracle_markov + canonicalize, seed 42** (실측, 24k ep) | **193.49 ± 1.86** | | **0.340** |
| **oracle_markov + canonicalize, seed 43** (실측, 24k ep) | **192.04 ± 1.57** | | **0.330** |
| oracle_markov (인수인계 값, 출처 미확인) | 161.2 | | 0.113 |
| MATE (인수인계 값, 출처 미확인) | 152.4 | | 0.051 |
| — 비-RL 진단 (05절) — | | | |
| BC, perceived frame | 118.5 | | −0.188 |
| BC, latent frame | 164.9 | | 0.138 |

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

## 05. 진단 — 129점 격차는 어디서 오는가

이 절은 이전에 출처 불명의 237.0에 기대어 "격차의 53%는 frame"이라고
적었습니다. 그 수치는 근거가 없었으므로(아래 06절 주석), 두 개의 실험으로
직접 측정해 대체했습니다.

**실험 A — RL이 부족한 것인가?** γ만 0.99 → 0.999 로 바꿔(유효 지평 100 →
1000 스텝) 같은 seed·같은 스택으로 24000 episode 를 돌렸습니다.

| discount | return (마지막 20 eval) | normalized |
|---|---|---|
| γ = 0.99 (control) | 158.36 ± 1.45 | 0.093 |
| γ = 0.999 (treatment) | 158.13 ± 1.37 | 0.091 |

차이 **0.2점** — 표준오차의 1/7. 신용 할당 지평은 병목이 아닙니다.
덤으로 이 158.2 는 이 저장소에서 직접 잰 최초의 oracle 수치이고, 인수인계
값 161.2 를 대략 확인해 줍니다.

**실험 B — frame인가 search인가?** RL을 통째로 들어내고, planner의 행동을
지도학습으로 흉내내게 했습니다 (`scripts/bc_diagnostic.py`). 두 조건은
**입력 프레임 하나만** 다릅니다: 68차원, 인코딩, 용량, episode 수 모두 동일하고
`latent` 조건은 돌 좌표 9개 + 물약 타입 12개, 총 21/68 차원만 잠재 프레임으로
되돌려 놓은 것입니다 (보상·used·trial flag·`chem_gt` 는 바이트 단위로 동일).

| frame / model | params | test acc | non-no-op acc | return | normalized |
|---|---|---|---|---|---|
| perceived / small | 118k | 0.733 | 0.398 | 118.5 | −0.188 |
| perceived / large | 3260k | 0.723 | 0.382 | 114.1 | −0.219 |
| **latent** / small | 118k | 0.795 | **0.540** | **164.9** | **0.138** |
| **latent** / large | 3260k | 0.803 | 0.561 | 163.9 | 0.132 |
| planner (상한) | | 1.000 | 1.000 | 288.1 | 1.007 |

no-op이 전체 스텝의 56%라 raw accuracy는 부풀려집니다. 판별력이 있는 건
**non-no-op accuracy** 열입니다.

**실험 C — 그 프레임 정렬을 RL oracle에 직접 넣으면?** (= P0) 실험 B의
`latent` 변환을 그대로 env에 옮겨(`--config_env.canonicalize_oracle=True`)
oracle을 RL로 학습시켰습니다. 옮긴 변환이 BC 기준과 **비트 단위로 동일**한지
`scripts/verify_canonicalize.py` 로 먼저 확인했습니다 (1200 스텝, `max |env −
bc_reference| = 0.000e+00`, 액션 마스크 불일치 0, 100% 스텝에서 perceived와
다름 = no-op 아님).

| 조건 | seed | return (마지막 20 eval) | normalized |
|---|---|---|---|
| control (perceived, γ=0.99) | 42 | 158.36 ± 1.45 | 0.093 |
| control (perceived, γ=0.999) | 42 | 158.13 ± 1.37 | 0.091 |
| **canonicalized** | 42 | **193.49 ± 1.86** | **0.340** |
| **canonicalized** | 43 | **192.04 ± 1.57** | **0.330** |

**+34.6점, normalized 3.6배.** seed 간 폭(1.5점)의 23배라 노이즈가 아닙니다.
10-eval 블록 평균도 단조에 가깝게 올라간 뒤 평평해집니다
(canon s42 `183.0 → 186.9 → 194.3 → 192.7`, control `155.2 → 154.0 → 159.2 →
157.5`) — 아직 학습 중이라 높게 나온 값이 아닙니다.

### 측정된 사다리

```
휴리스틱 floor              145.2   (0.000)  ← chemistry 안 씀
RL oracle (정답지 O)         158.2   (0.092)  ← 실측, γ와 무관
latent BC (프레임까지 O)      164.9   (0.138)  ← 실측, 지도학습 + 정답 라벨
RL oracle + canonicalize    192.8   (0.336)  ← 실측, seed 2개
planner                    287.1   (1.000)  ← ceiling
```

**129점 격차의 분해 (RL 기준):** frame 34.6점 = **27%**, 남은 94.3점 =
**73%가 search**. (지도학습 기준으로는 5:95 였는데, BC가 RL 성능을
과소평가하기 때문입니다 — 아래 주의 참조.)

**결론 1 — frame은 실재하는 병목입니다.** 프레임만 되돌려도 non-no-op
accuracy 가 0.398 → 0.540 (상대 +36%), return 이 118.5 → 164.9 (+46.4)로
뜁니다. perceived BC는 휴리스틱 floor **아래**인데 latent BC는 floor 위이자
RL oracle 위입니다. 프레임 정렬은 값싸고 확실한 이득입니다.

**결론 2 — 그러나 frame이 격차의 절반은 아닙니다.** 프레임을 완전히 정렬해
준 RL oracle도 normalized 0.336 에서 멈춥니다. 남은 **73%** 는 search 입니다 —
8노드 그래프 탐색과 20스텝 배분은 프레임을 정리해 준다고 사라지지 않습니다.
문서가 적었던 "53%는 frame" 은 **과대평가**였습니다.

**결론 3 — 용량은 답이 아닙니다.** 28배 키우면 perceived에서는 오히려
나빠지고 (non-no-op −0.016, return −4.4; train loss 1.17 → 0.37 인데 test acc는
12 epoch에서 정점 = 전형적 과적합), latent에서도 accuracy만 +0.021 오르고
return은 오르지 않습니다.

**주의 — 164.9는 RL의 상한이 아니었습니다 (확인됨).** RL oracle(158.2)이
perceived BC(118.5)를 이미 이겼으므로 BC는 달성 가능한 RL 성능을
**과소평가**한다고 적어 두었는데, 실험 C가 이를 확인했습니다: canonicalized
RL oracle은 **192.8** 로 latent BC(164.9)를 28점 넘어섭니다. BC 기준의
frame:search = 5:95 대신 **RL 기준 27:73** 을 쓰십시오.

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
| 1 | **frame misalignment** — 정답지가 다른 좌표계로 쓰여 있음 | ✅ **해결됨(oracle 한정).** RL oracle 158.2 → **192.8** (+34.6, 격차의 27%). BC로는 118.5 → 164.9 |
| 2 | **slot permutation** — 정보 0인 슬롯 순서를 축으로 취급 | MATE에 특히 치명적 (미측정) |
| 3 | **search** — 8노드 그래프 탐색 + 20스텝 배분 | 남은 **73%**. 프레임을 다 정리해 준 뒤에도 남는 부분 |
| 4 | **decoder capacity** — 128×2로는 위 연산 사슬이 안 들어감 | ❌ **반증됨.** 28배 키워도 return이 오르지 않음 (05절) |

**2번이 MATE에 더 아픈 이유:** `full_transition=True`라 transition이
`(o_t, a_t, r_t, o_{t+1} − o_t)`이고, 그 **delta가 곧 chemistry 증거**입니다.
MATE의 running mean은 이 증거를 **12가지 슬롯 의존 기저로 쓰인 임베딩끼리**
평균내고 있습니다.

---

## 06. 저장소 상태

| 항목 | 상태 |
|---|---|
| `envs/alchemy_baselines.py` | ✅ 신규 — planner + 휴리스틱 + uniform |
| `scripts/eval_alchemy.py` | ✅ 신규 — 1024ep / seed 100–115 / normalized score |
| `scripts/trace_alchemy.py` | ✅ 신규 — 02절의 스텝별 트레이스 생성기 (교육용) |
| `scripts/bc_diagnostic.py` | ✅ 신규 — RL 없이 frame vs search를 가르는 05절 실험 |
| `eval/adaptation` 로깅 | ✅ 신규 — `learner.py` |
| `amlt/alchemy.yaml` | ✅ 갱신 — 아래 표대로 |
| 237.02를 냈던 코드 | ❓ **저장소에 존재한 적 없음.** 아래 주석 참조 |

> **237.02에 대한 정정.** 이 문서는 이전에 "237을 냈던 permutation 코드가 커밋
> `7b76c44`에서 제거되어 재현 불가"라고 적었으나, 확인 결과 **사실이 아닙니다.**
> `7b76c44`가 제거한 `permutation_training`은 transition 을 **시간축으로 섞는**
> MATE 전용 augmentation(running-sum 의 순서 불변성을 이용)이고, frame
> canonicalization 과 무관합니다. `git log --all -S "canonicalize"` 결과
> `canonicalize_oracle` 은 이 저장소 히스토리에 **한 번도 존재하지 않았습니다**
> (제 문서 커밋에만 등장). 따라서 237.02 는 출처를 확인할 수 없는 인수인계
> 수치이며, 이 수치에 기대어 세운 "격차의 53%는 프레임 문제" 라는 분해도
> 근거가 없습니다. 그 분해는 05절의 BC 실험(직접 측정)으로 대체했고, 측정
> 결과 frame의 몫은 53%보다 **작습니다**.

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

## 07. 실행 계획

### ✅ P2 — 인프라 (완료)

측정 기준을 먼저 세웁니다. 이게 없으면 어떤 실험 결과도 해석할 수 없습니다.

- planner + 휴리스틱을 저장소 API로 편입
- 평가 프로토콜 스크립트 + normalized score
- adaptation 지표 로깅
- amlt 레시피 갱신

### ✅ P0 — oracle 천장 올리기 (1단계 완료)

프레임 정렬을 신경망에 시키지 말고 환경이 미리 해서 주는 것입니다. 05절 BC
실험의 `latent_obs` 를 그대로 환경으로 옮겼고 (`envs/alchemy.py:_canonicalize`),
`scripts/verify_canonicalize.py` 로 BC 기준과 비트 단위 동일함을 확인한 뒤
학습했습니다.

세 개의 레버가 있고, 첫 번째는 측정이 끝났습니다:

| 레버 | 플래그 | 특권? | 결과 |
|---|---|---|---|
| 돌·물약을 잠재 프레임으로 | `canonicalize_oracle` | 🔒 oracle 전용 | ✅ 158.2 → **192.8** (+34.6) |
| 물약 ordinal 스칼라 → 축 one-hot(3)+방향(1) | `structured_potions` | 🆓 **누구나 사용 가능** | 측정 중 |
| context를 graph 12차원만 남김 | `context_graph_only` | 🔒 oracle 전용 | 측정 중 |

`structured_potions` 는 순수 재인코딩이라 **MATE에도 쓸 수 있습니다** —
셋 중 유일하게 최종 결과에 그대로 반영 가능한 레버입니다.

**남은 기대치.** 이 세 레버를 다 써도 천장이 planner 근처까지 가지는
**않습니다.** 프레임을 완전히 정리해 준 뒤에도 격차의 **73%가 search** 로
남아 있고 (05절), 그건 표현 문제가 아니라 계획 문제입니다.

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
python scripts/eval_alchemy.py --compare oracle_markov=158.2 \
  --compare oracle_canonicalized=192.8 --compare bc_latent=164.9

# 05절의 frame-vs-search 진단 (수집 + 4개 조건 학습 + 롤아웃, GPU 1장 ~1시간)
python scripts/bc_diagnostic.py

# oracle RL 실측 재현 (24k episode). γ는 결과에 영향이 없었습니다.
python main.py --config_env=configs/envs/alchemy.py \
  --config_env.env_name=rotation_random_bottleneck \
  --config_rl=configs/rl/dqn_default.py \
  --config_seq=configs/seq_models/markov_default.py \
  --config_seq.seq_model.is_oracle=True \
  --train_episodes=24000 --k=1 --seed=42 \
  --config_rl.critic_lr=3e-5 --config_rl.use_popart=True \
  --config_rl.mask_alchemy_invalid_actions=True \
  --config_seq.normalize_inputs=True --config_seq.use_pe=True \
  --config_seq.conditioning_hidden_dim=128 --device=0 --run_name=oracle

# P0 실험 C 재현 (192.8). 위 명령에 플래그 하나만 추가하면 됩니다.
#   --config_env.canonicalize_oracle=True
# 학습 전에 변환이 BC 기준과 동일한지 먼저 확인하십시오 (약 1분, CPU)
python scripts/verify_canonicalize.py

# 02절의 스텝별 트레이스 (다른 판을 보려면 --seed 를 바꾸세요)
python scripts/trace_alchemy.py --seed 6 --show_steps 14
```

dm_alchemy는 archived 특수 설치입니다: `bash scripts/install_dm_alchemy.sh mate`
