# Alchemy 시도 전체 목록

작성 2026-09-14. 인계용 — **어떤 레버를 이미 당겼고, 무엇이 남아 있는가.**
개별 실험의 논증 과정은 `docs/alchemy_experiment_log.md`, 스케일 분석은
`docs/mate_memory_scale.md`, 환경 설명은 `docs/alchemy_status.md`에 있다.

## 0. 읽는 법

절대 return은 의미가 없다. `symbolic_obs`가 각 돌의 진짜 보상값을 이미
흘리므로, 화학을 전혀 쓰지 않는 정책도 높은 점수를 받는다. 레벨별 바닥/천장으로
정규화한다 (`scripts/eval_alchemy.py`, 1024 에피소드).

| 레벨 | 바닥 (화학 미사용) | 천장 (잠재공간 플래너) |
|---|---:|---:|
| `rotation_random_bottleneck` (논문 레벨) | 145.2 | 287.1 |
| `random_bottleneck` | 146.95 | 287.20 |
| `no_bottleneck` (가장 쉬움) | 173.48 | 315.25 |

`normalized = (return − 바닥) / (천장 − 바닥)`.

숫자 관례: `no_bottleneck` 표는 wandb 마지막 5회 평가 평균,
`rotation_random_bottleneck` 표는 ledger 기록값(집계 창이 달라 1~2점 차이가 난다).
세계모델 표는 해당 스텝의 검증셋 값이다.

**한 줄 요약.** 병목은 기억 용량이 아니라 **지각 프레임 → 잠재 프레임 맵**이다.
그 맵은 에피소드마다 새로 뽑히고, 해당 약을 실제로 써 본 **단 하나의 전이**에서만
알 수 있다. 평균 집계는 그 전이를 `1/(w+200)`로 희석한다.

---

## A. 환경·관측 쪽 레버

관측을 바꾼다. 오라클(진짜 화학을 입력으로 받는 에이전트)에 고정해 측정했다.
레벨 `rotation_random_bottleneck`, 기준 158.4.

| 레버 | 특권? | return | 판정 |
|---|---|---:|---|
| `canonicalize_oracle` (관측을 잠재 프레임으로 재작성) | 특권 | 193.5 | **+35** |
| `structured_potions` (약 인코딩 구조화) | 비특권 | 225.3 | **+32** (누적) |
| `context_graph_only` | 특권 | 232.3 | +7 (누적) |
| `structured_stones` | 비특권 | 156.4 | 0 |
| 약 인수분해 (factored potions) | 비특권 | 154.7 | 0 |
| `add_trial_phase` | 비특권 | 152.8 | − |
| 슬롯 재인코딩 | 비특권 | 149.8 | − |
| 위 비특권 레버 전부 | 비특권 | 150.4 | − |
| `canon_potion_acc` (프레임 맵을 알려진 정확도로 열화) | 특권 | 167.7 @0.43 | 측정 도구 |

**읽는 법.** 특권 레버(잠재 프레임을 쥐여주는 것)만 크게 듣는다. 비특권 재인코딩
9가지는 전부 0 또는 음수 — 단 하나 `structured_potions`만 예외이고, 이건 이후
모든 런의 기본값이 되었다.

## B. 학습 신호 쪽 — 보조 손실

관측은 그대로 두고 손실만 바꾼다. **프로젝트 최대의 양수 결과가 여기 있다.**

### B.1 `aux_canon` — 잠재 프레임 회귀 (논문의 "Predict: Chemistry" 계열, 특권)

21~33차원 레이블(잠재 돌 좌표 9 + 잠재 약 타입 12 + 병목 그래프 12)을 관측에
붙였다가 네트워크가 보기 전에 잘라낸다. 입력은 늘지 않고 학습 신호만 바뀐다.

| 런 | 메모리 | weight | site | return |
|---|---|---:|---|---:|
| `orc_sp_base` | 오라클 | off | — | 156.5 |
| `orc_aux_w0p1` | 오라클 | 0.1 | joint | 229.8 |
| **`orc_aux_w1`** | 오라클 | 1.0 | joint | **233.1** |
| `orc_aux_w10` | 오라클 | 10 | joint | 233.9 |
| `dqn_orc_aux_potion` | 오라클 | 1.0 | joint, `parts=potion` | **97.8** |
| `mem_mate_sp_s42` | MATE | off | — | 150.4 |
| `matep_aux_w0p1` | MATE | 0.1 | joint | 149.2 |
| `matep_aux_w1` | MATE | 1.0 | joint | 122.6 |
| `matep_aux_w10` | MATE | 10 | joint | 67.0 |
| `mem_site_w1` | MATE | 1.0 | memory | 140.9 |
| `mo_site_w1` | MATE | 1.0 | memory_obs | 146.3 |
| `mem_w1_both` | MATE | 1.0 | memory_obs, both | 151.3 |
| `mem_w1_potion` | MATE | 1.0 | memory_obs, potion | 146.4 |
| `mem_w10_potion` | MATE | 10 | memory_obs, potion | 136.7 |

세 가지가 읽힌다.

1. **오라클에서 +77.** 특권적 관측 재작성 두 개가 사준 것(+76)을 학습 레이블
   하나로 재현한다. 정보는 언제나 거기 있었고 꺼낼 유인만 없었다.
2. **레이블의 어느 절반인지가 전부다.** `parts=potion`으로 돌 좌표 9개를 빼면
   233.1 → **97.8**, 바닥보다 47점 아래다. §4의 +77은 약 지도 감독이 아니라
   **"지각→잠재 프레임 맵을 계산하라"는 아홉 개의 돌 좌표 회귀**였다.
3. **MATE에는 전부 음수.** 같은 레이블이 오라클에는 "이미 가진 입력으로 맵을
   계산하라", 메모리 모델에는 "그 맵을 발명하라"가 된다. 같은 바이트, 반대 효과.

`aux_canon_site`는 이 세션 이전에 추가된 것으로, 보조 그래디언트가 어디에
떨어지는지를 고른다 (`scripts/verify_aux_sites.py`가 검증).

| site | 헤드가 읽는 것 | 그래디언트가 닿는 곳 |
|---|---|---|
| `joint` | `conditioner(obs, h)` | 인코더 + conditioner + 메모리 |
| `memory` | `h_t`만 | 메모리만 (현재 프레임을 못 봐서 맵을 못 배움) |
| `memory_obs` | `cat(obs.detach(), h_t)` | **메모리만**, 프레임은 읽을 수 있음 |
| `probe` | 위와 같으나 메모리도 detach | 헤드만 (순수 측정) |

`memory_obs`가 공유 트렁크 간섭을 고쳤다 — 손실분 27.8점 중 23.7점 회복,
가중치 10에서의 붕괴는 완전 소멸. 그런데 **회복해도 return은 오르지 않는다.**

### B.2 `aux_cpc` — 같은 레이블, 회귀 대신 대조 (이 세션 이전)

메인 레포의 contrastive learning을 이식한 InfoNCE 버전. 같은 site/parts를 공유한다.

| 런 | 메모리 | return | 정규화 |
|---|---|---:|---:|
| `nb_mate` (대조군) | MATE | 176.6 | +0.022 |
| **`nbcpc_mate`** | MATE | **183.0** | **+0.067** |
| `nbaux_mate` (회귀) | MATE | 165.6 | −0.056 |
| `nbcpc_oracle` | 오라클 | 271.5 | +0.692 |
| `nbaux_oracle` (회귀) | 오라클 | 297.8 | +0.877 |

**메모리 모델에서 양수가 나온 유일한 보조 손실이다** (+0.045). 오라클에서는
회귀가 더 낫다.

### B.3 `aux_count` — "Predict: Features" (논문 §4.3 과제 1+2, 비특권)

지각 카테고리별 돌 개수(27) + 색깔별 약 개수(6). 에이전트 자신의 관측의 결정적
함수라 환경에서 아무것도 받지 않고 누출도 불가능하다. 논문이 **특권 정보 없이
ideal observer에 근접한 유일한 조건**으로 보고한 과제다.

| 런 | weight | site | return | 대조군 대비 |
|---|---:|---|---:|---:|
| `mate_probe_ctrl` | off | — | 153.0 | — |
| `mate_count_w0p1` | 0.1 | obs | 153.0 | **±0** |
| `mate_count_w1` | 1.0 | obs | 107.8 | **−45** |
| `oracle_count_w1` | 1.0 | obs | 135.7 | **−21** |

**구현 실패가 아니다.** 세는 과제 자체는 거의 완벽히 풀린다 — 약 개수 오차
1.37 → 0.006, 돌 1.72 → 0.059. 세는 법은 배우고 그 대가로 아무것도 얻지 못한다.

가중치 1이 해로운 이유: 학습 초기에 보조항이 critic 항의 두 배(0.267 vs 0.133)라,
가치 함수가 발언권을 갖기 전에 공유 관측 분기가 세기 위해 재편된다.

**단, 붙이는 위치가 논문과 다르다.** 논문은 symbolic 관측을 transformer core에
직접 넣었으므로(§4.1) 보조 헤드가 **메모리**에 붙는다. 위 표는 전부 conditioner의
**관측 분기**다. 그 차이를 메우려고 이 세션에 `config_rl.aux_count_site`를
추가했다 (`obs` | `memory` | `memory_obs` | `probe`, 그래디언트 흐름 검증 완료).
`nbc_*` 런이 그 측정이다.

---

## C. 알고리즘·최적화·행동공간 쪽

| # | 개입 | 건드린 것 | 결과 |
|---|---|---|---|
| C.1 | 할인율 0.99 → 0.999 | RL 지평 100배 | −0.3. **기각** |
| C.2 | 네트워크 폭 4배 (256) | 용량 | +3.4. **기각** |
| C.3 | critic lr 3e-5 → 1e-4 | 최적화 | −21. **기각** |
| C.4 | 그래디언트 클립 0.2 → 12.0 (60배) | 최적화 | +0.4. **기각** |
| C.5 | NO_OP 행동 마스킹 | 행동공간 | 3개 런 모두 악화, 대조군 −24.5. **기각** |
| C.6 | dueling + factored Q 헤드 | critic 구조 | 147.1 (대조 146.4). **기각** |
| C.7 | 디코딩된 약 사후확률을 critic에 주입 | 정보 전달 | 147.3. **기각** |
| C.8 | PopArt 가치 정규화 | 보상 스케일 | 기본값으로 채택 |
| C.9 | SAC-discrete 부활 | 알고리즘 축 | 아래 |

**C.9 알고리즘 축은 보조 손실 아래에서만 존재한다.**

| | aux 없음 | aux `parts=both` |
|---|---:|---:|
| DQN 오라클 | 156.5 | **233.1** |
| SAC-discrete 오라클 | 159.9 | 160.4 |

보조 손실 없이는 두 학습 규칙이 3.4점 차이다. 있으면 **73점** 차이다. 조밀한
레이블에서 프레임 맵을 배우는 것은 DDQN이 하고 이 SAC-discrete가 못 하는 일이다.
메모리 축에서는 알고리즘이 무관하다 (MATE, aux 없음: DQN 153.0 / SAC-d 153.1).

**C.5 부기.** NO_OP 마스킹의 현금화율이 세 런 모두 정확히 0.150 = 3돌/20스텝에
고정되었다 — 마스크가 현금화를 강제했을 뿐 *어느* 돌을 현금화할지는 개선하지 못했다.

**예산 산술.** MATE는 약 슬롯 120개 중 ~103개(86%)를 쓰지만 돌 30개 중
~16개(53%)만 현금화한다. 오라클은 ~22개(73%). **실험할 기회가 부족한 게 아니라
돌을 +15로 바꾸지 못한다.**

---

## D. 세계모델 + 트리서치

근거: arXiv:2208.11535 (Pinon+). Transformer 세계모델 + MuZero식 트리서치,
정책망·가치망 없음. **1e6 trajectory / 750k step**이라 우리 예산 안에 들어오는
유일한 성공 세팅이다 (V-MPO 쪽은 symbolic 기준 **1e9 에피소드** = 우리 처리량으로
런 하나에 1.6~11년).

데이터: 무작위 정책 100만 에피소드 (학습 95만 / 검증 5만), `rotation_random_bottleneck`.
손실 = `obs_CE + λ·chem_CE + reward_CE`.

프로브 두 개(메모리만 detach해서 읽음):
- `perm` — 약 슬롯 → 잠재축 6지선다 (우연 16.7%) = **프레임 맵**
- `sign` — 6개 부호 비트 (우연 50%) = **부호 구조**

### D.1 λ=10, 10만 스텝 — 결정적 대비

| 런 | 메모리 | CHEM_CE ↓ | perm 종료 | sign 상승 |
|---|---|---:|---:|---:|
| **`L_gpt_w10`** | GPT-2 3층 | **0.0571** | **0.999** | +0.254 |
| `S_iw1` | MATE | 0.2109 | 0.166 | +0.000 |
| `S_iw10` | MATE `init_weight=10` | 0.2152 @40k | 0.167 | +0.002 |
| `S_gate` | MATE + 전이별 게이트 | 0.2110 | 0.165 | +0.000 |
| `S_norm` | MATE + 메모리 LayerNorm | 0.2072 | 0.164 | −0.001 |
| `S_linattn` | 선형 어텐션 | 0.2079 | 0.165 | −0.001 |
| `ctl_lstm_w10` | LSTM (5만 데이터) | 0.2060 | 0.168 | +0.131 |
| `ctl_markov_w10` | 무기억 (5만 데이터) | 0.2333 | — | +0.000 |

GPT는 4만 스텝에 perm 96.6%, 6만에 99.8%, 80만 스텝 최종 CHEM_CE **0.0486**.
**고정 크기 순환 요약은 종류를 막론하고 프레임 맵에서 실패한다.** LSTM은 부호는
얻지만(+0.131) 프레임 맵은 못 얻는다.

### D.2 λ=0, 40만 스텝 — MATE 처방 비교

| 런 | 처방 | obs_CE ↓ | CHEM_CE ↓ | perm 상승 | sign 상승 |
|---|---|---:|---:|---:|---:|
| `Z_markov` | 무기억 대조군 | 0.0600 | 0.2469 | +0.000 | +0.000 |
| `Z_mate` | 기준 MATE | 0.0569 | 0.2465 | −0.001 | +0.252 |
| `H_base` | + init 초기화 수정 | 0.0622 | 0.2434 | −0.003 | +0.251 |
| **`H_gate`** | + 전이별 게이트 | 0.0666 | **0.2361** | −0.002 | +0.244 |
| `Z_mate_gate` | + 전이별 게이트 (init 수정 전) | 0.0556 | **0.2227** | −0.003 | +0.249 |
| `H_proj` | + hyperspherical norm. | 0.0680 | 0.2553 | −0.004 | +0.253 |
| `H_proj_gate` | + 둘 다 | 0.0578 | 0.2515 | −0.003 | +0.253 |
| `Z_mate_norm` | + 메모리 LayerNorm | 0.0679 | 0.2505 | +0.000 | +0.210 |
| `Z_mate_both` | 게이트 + LayerNorm | 0.0627 | 0.2544 | −0.004 | +0.251 |
| **`Z_gpt`** | GPT-2 3층 | **0.0410** | **0.1688** | **+0.475** | **+0.341** |

**MATE의 모든 변종이 무기억 대조군(0.2469)에서 ±0.02 안에 갇혀 있다.**

### D.3 트리서치 — 아직 판정 불가

24 에피소드, `rotation_random_bottleneck`.

| 동역학 모델 | 예산 100 | 예산 500 | 예산 효과 |
|---|---:|---:|---:|
| 완벽한 모델 (진짜 환경) | 86.0 | 131.0 | **+45.0** |
| `gpt` (논문 구성) | 97.8 | 97.0 | −0.8 |
| `lstm` | 105.5 | 103.6 | −1.8 |
| `mate` | 93.4 | 91.7 | −1.7 |
| **논문 (학습된 Transformer)** | **79.3** | **161.8** | — |

완벽한 모델이 예산에 반응하므로 탐색의 *방향*은 건강하다. 그러나 절대 수준이
논문의 *학습된* 모델보다 낮으므로 **"모델이 나쁘다"와 "탐색이 약하다"를 구분할 수
없다.** 위 학습 모델은 전부 5만 에피소드로 학습한 수렴 전 모델이며, 100만 에피소드로
수렴시킨 `Z_gpt` / `L_gpt_w10`으로는 아직 재평가하지 않았다.

검증: `scripts/verify_planner.py` 14/14, `scripts/verify_mate_variants.py` 15/15.

---

## E. MATE 자체의 테크닉 — 시도 여부

**이것이 이 문서의 핵심이다.** 레포에 존재하는 MATE 관련 노브 전부와, 그것이
Alchemy에서 시도되었는지.

| 테크닉 | 플래그 | RL에서? | 세계모델에서? | 결과 |
|---|---|---|---|---|
| 임베더 깊이 | `seq_model.n_layer` | ❌ 1 고정 | ❌ 1 고정 | 미측정 |
| 은닉 폭 | `seq_model.hidden_size` | ❌ 256 고정 | ❌ 256 고정 | 미측정 |
| **RFF 커널 평균** | `use_rff`, `kernel`, `learn_kernel` | ❌ | ❌ | **전혀 안 해봄** |
| 학습되는 초기 사전분포 | `learn_init_emb` | ✅ 항상 True | ✅ 항상 True | 기본값 |
| 초기 `w` | `init_weight` | ⚠️ 배선만 | ✅ 10, 40 | λ=10에서 무효 |
| **전이별 게이트** | `use_gate` | ⚠️ 배선만 | ✅ | λ=0 −0.007, λ=10 0.000 |
| **hyperspherical norm.** | `project_output` | ❌ **미포팅** | ✅ | 드리프트 5.05→1.00배, CHEM_CE +0.012 악화 |
| 메모리 LayerNorm | `norm_memory` | ❌ 세계모델 전용 | ✅ | sign +0.252 → +0.210 (악화) |
| **MSC (대조 정규화)** | `msc_enable`, `msc_objective`, … | ❌ | ❌ | **전혀 안 해봄** |
| MSC + EMA 인코더 | `mate_msc_ema_v2` | ❌ | ❌ | **전혀 안 해봄** |
| 선형 어텐션 변종 | `mate_linattn` | ❌ | ✅ | λ=10 무효. 노드 상태 263KB로 탐색 불가 |
| 롤아웃 z 캐시 | `use_rollout_z_cache` | ❌ | ❌ | main 브랜치에만 있음, 미이식 |
| 입력 노이즈 | `noise_ratio` | ❌ 0.0 고정 | ❌ | 미측정 |
| 입력 정규화 | `normalize_inputs` | ✅ 항상 True | ✅ | 기본값 |
| 관측 지름길 | `obs_shortcut` | ✅ 항상 True | ✅ 어블레이션 | **없으면 MATE만 크게 손해** |
| 전체 전이 입력 | `full_transition` | ✅ 항상 True | ✅ | 같음 |
| positional encoding | `use_pe` | ✅ 항상 True | — | 어블레이션 없음 |
| conditioning 종류 | `conditioning` (concat/film/hypernet) | ⚠️ 오라클 컨텍스트에만 | ❌ | film 150.2 / hypernet 130.5 (둘 다 −) |
| conditioning 깊이 | `conditioning_n_layer` | ❌ 1 고정 | — | 미측정 |
| **슬롯 등변 인코더** | `alchemy_slot_encoder` (DeepSets) | ❌ **구현만, 미실행** | ❌ | **전혀 안 해봄** |
| 리셋 전이 처리 | `skip_reset_transition` 등 | ✅ 기본값 | — | 미측정 |

범례: ✅ 시도함 / ⚠️ 코드는 연결됐으나 Alchemy 런 없음 / ❌ 안 해봄

**주의 두 가지.**

1. **RL 경로와 세계모델 경로가 다르다.** `project_output`과 `norm_memory`는
   `policies/models/world_model.py`에만 있고 `policies/models/recurrent_head.py`에는
   없다. RL에서 hyperspherical normalization을 쓰려면 먼저 이식해야 한다.
   `use_gate`와 `init_weight`는 `mate_vanilla.py`에 있으므로 RL에서 바로 쓸 수 있다
   (`--config_seq.seq_model.use_gate=True`).
2. **전 Alchemy RL 런에서 MATE는 `use_rff=False`, `msc_enable` 없음,
   `alchemy_slot_encoder=False`, `conditioning=concat`으로 돌았다.** 즉 MATE의
   고유 테크닉 중 Alchemy RL에서 검증된 것은 사실상 하나도 없다.

---

## F. 기각된 가설 (측정으로)

| 가설 | 어떻게 기각했나 |
|---|---|
| **그래디언트 희석** | `∂mₜ/∂zᵢ = 1/(w+t)`가 균등하니 전이별 그래디언트도 균등할 것이라 봤으나, `zᵢ`는 이후 모든 메모리에 **누적**된다. 실측 최대/평균 MATE 4.73 vs GPT 3.12 — MATE가 오히려 더 집중적 |
| **스케일 표류** | hyperspherical normalization이 5.05배 → 1.00배로 완전 제거. CHEM_CE는 오히려 0.012 악화 |
| **탐색 붕괴** | `diagnose_exploration.py`: 반복률 21.8% vs 균등 18.3%, 쓸모없는 약 비율 49.2% vs 균등 49.8%. 메모리를 0.43까지 가르쳐도 행동은 0.7%p 변함 |
| **critic이 메모리를 못 읽음** | `mate_feed` — 디코딩을 이미 끝낸 사후확률을 critic에 직접 먹여도 +0.9 |
| **데이터 품질** | `probe_memory_accumulation.py` — 메모리 고정, 데이터만 교체(학습 정책 vs 균등). 두 곡선 다 안 오름 |
| **약 지도가 병목** | `parts=potion`이 233.1 → 97.8. 값은 프레임 쪽에 있고 약 지도는 거의 무가치 |

---

## G. 안정 세팅 — 저장된 기준선

`scripts/repro_alchemy.sh`에 그대로 실행 가능한 명령이 있다.

### G.1 RL

| 목적 | 런 | 명령 | return / 정규화 |
|---|---|---|---|
| **MATE 최고** | `nbcpc_mate` | `repro_alchemy.sh rl_mate_cpc` | 183.0 / **+0.067** |
| ↑ 주의: `aux_cpc_weight=1.0`, `aux_canon_weight=0.0`, `site=joint` | | | |
| MATE 기준 | `nb_mate` | `repro_alchemy.sh rl_mate` | 176.6 / +0.022 |
| **오라클 천장** | `nbaux_oracle` | `repro_alchemy.sh rl_oracle_aux` | 297.8 / **+0.877** |
| ↑ 주의: `--config_seq.seq_model.is_oracle=True` 없으면 그냥 무기억 에이전트다 | | | |
| 배선 검증 | `fx_markov` (`all_fixed`) | `repro_alchemy.sh rl_sanity` | 302.1 / ≈+0.90 |

`rl_sanity`가 배선 대조군이다 — 무기억 에이전트가 화학 고정 레벨에서 천장의
90%에 도달하므로 환경·버퍼·DQN 경로 자체는 건강하다. **새 테크닉이 안 될 때
여기부터 확인할 것.**

### G.2 세계모델

| 목적 | 런 | 명령 | 지표 |
|---|---|---|---|
| **GPT 성공 레퍼런스** | `L_gpt_w10` | `repro_alchemy.sh wm_gpt_w10` | CHEM_CE 0.0486, perm **1.000** |
| MATE 최고 (λ=0) | `Z_mate_gate` | `repro_alchemy.sh wm_mate_gate` | CHEM_CE 0.2227 |
| MATE 기준 (λ=0) | `Z_mate` | `repro_alchemy.sh wm_mate` | CHEM_CE 0.2465 |
| MATE 기준 (λ=10) | `S_iw1` | `repro_alchemy.sh wm_mate_w10` | CHEM_CE 0.2109, perm 0.166 |
| 무기억 대조군 | `Z_markov` | `repro_alchemy.sh wm_markov` | CHEM_CE 0.2469 |

체크포인트는 `logs/wm_ckpt/`, 결과 JSON은 `logs/wm_runs/`에 있다.

### G.3 실행 시 주의

- **`--save_dir`를 로컬 디스크로 지정할 것** (`/HDD1/g.chung/Memory-RL/logs`).
  `/NFS`는 공유 마운트이고, 체크포인트마다 리플레이 버퍼를 통째로 다시 쓰기 때문에
  31개 런이 326GB를 써서 마운트를 느리게 만든 적이 있다. `CLAUDE.md`의
  "Where runs may write" 참고.
- 세계모델 데이터 `logs/wm_data/wm_rrb_1m_mm`는 8GB uint8 memmap이라 프로세스 간
  페이지 캐시를 공유한다. `.npz`(32GB float32) 쪽은 4개를 동시에 띄우면 OOM 난다.

---

## H. 아직 안 해본 것

우선순위 순.

1. **수렴한 `L_gpt_w10`으로 트리서치 재평가.** D.3의 모든 행이 수렴 전 모델
   기준이다. 프레임 맵을 100% 복원하는 모델로 다시 재면 가장 크게 달라질 지점.
2. **완벽한 모델이 예산 500에서 131.0인 이유.** 논문의 학습 모델(161.8)보다도
   낮으므로 탐색 쪽 결함이 남아 있다. 롤아웃 깊이 / 평가 정책 / 에피소드 수(24,
   SEM≈8)를 좁혀야 한다.
3. **MATE 고유 테크닉 (E절의 ❌ 행).** RFF 커널 평균, MSC, 슬롯 등변 인코더가
   전부 미측정이다. 그 중 **슬롯 등변 인코더**가 이 과제와 가장 잘 맞는다 —
   돌 3슬롯과 약 12슬롯은 순서에 의미가 없는 집합인데 MLP는 고정 위치로 읽는다.
   `structured_potions`가 재인코딩만으로 +32를 벌어준 것이 같은 계열의 증거다.
4. **`project_output`을 RL 경로에 이식.** 현재 세계모델 전용이다.
5. **`no_bottleneck`에서 GPT/LSTM에 보조 손실.** 메모리 모델 중 MATE에만 걸어봤다.
6. **시드 반복.** 거의 모든 결론이 단일 시드(42)다.
