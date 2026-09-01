# Symbolic Alchemy — 다음 실험 방향

작성 2026-09-01 · branch `alchemy_test` · 이전 기록은 `docs/alchemy_status.md`,
시각화는 `~/alchemy_dashboard/index.html` §10–§11

---

## 0. 한 줄 요약

oracle 천장 작업(158.4 → 232.3)은 끝났다. 그 과정에서 **비교 축을 잘못
세우고 있었다는 것**이 드러났으므로, 이 문서는 축을 다시 세우고 그 위에
남은 실험을 배치한다.

---

## 1. 축 재정리 — 이것이 이 문서의 존재 이유

지금까지 "MATE vs V-MPO", "우리 MATE 152.4 ≈ 논문 V-MPO 155.4 → 거의 재현"
같은 서술을 했다. **성립하지 않는 비교다.** 두 개의 직교하는 축이 있다.

| 축 | 뜻 | 후보 |
|---|---|---|
| **A. 학습 알고리즘** | 가치/정책을 어떻게 갱신하는가 | DDQN(우리), V-MPO, IMPALA, PPO |
| **B. 메모리 아키텍처** | 에피소드 맥락을 무엇으로 표현하는가 | MATE, Transformer-XL, LSTM, **GT chemistry**, Markov(없음) |

**GT chemistry는 알고리즘이 아니라 메모리 축의 한 항목이다** — 공짜로
주어지는 완벽한 메모리, 즉 **메모리 축의 천장**. MATE가 하려는 일이 정확히
그것을 근사하는 것이다.

### 격자

| 알고리즘 ＼ 메모리 | MATE | Transformer-XL | LSTM | GT (완벽) |
|---|---|---|---|---|
| **DDQN** (우리) | 측정 중 | 측정 중 | 측정 중 | **158.4** raw / **232.3** 레버 |
| **V-MPO** (논문) | — | 155.4 | — | **≈284** (Fig 4) |
| **IMPALA** (논문) | — | — | 140.2 | — |
| 스크립트 봇 | — | — | — | 288.5 (우리 287.1) |

바닥(무작위 휴리스틱): 논문 145.7 / 우리 145.2.
정규화: `(return − 145.2) / (287.1 − 145.2)`.

### 이 정리가 고쳐주는 것

**158.4 vs ≈284는 메모리를 GT(완벽)로 고정한 채 알고리즘만 다른 비교다.**
양쪽 입력이 동일하다 — 가공 안 한 symbolic 관측 + 가공 안 한 `chem_gt`.
논문 Fig 4의 GT 막대에는 우리가 넣은 레버가 하나도 없다.

따라서 이전에 "알고리즘 + 스케일 + 미발견 인코딩"이라고 뭉뚱그린 것은
정정되어야 한다. 메모리 축이 양쪽 다 천장에 못 박혀 있으므로 이것은
**알고리즘 축의 순수 측정치**다.

> **레버 3종이 벌어준 +74점은 정보를 더 준 게 아니라, 논문이 애초에
> 필요로 하지 않았던 학습 난이도를 대신 깎아준 것이다.**

그리고 여기서 따라 나오는 가장 중요한 결론:

> **메모리 축의 상한 = 그 알고리즘의 oracle 점수.**
> DDQN + raw 인코딩에서 그 상한은 **158**이다. MATE가 아무리 좋아져도
> 넘을 수 없다. **알고리즘 축을 고치면 모든 메모리 아키텍처의 천장이
> 한꺼번에 올라간다.**

### 이 정리가 드러낸 구멍

1. **MATE를 이 환경에서 측정한 적이 없다.** `logs/alchemy/` 아래 11개 런이
   전부 `markov + is_oracle=True`다. 152.4는 `docs/alchemy_status.md:272`에
   스스로 "인수인계 값, 출처 미확인"이라 표시해둔 숫자다.
2. **Transformer-XL / LSTM 칸이 비어 있다.** 그런데 `gpt_default.py`,
   `lstm_default.py`는 이미 있다. DDQN 고정 하에 셋을 나란히 돌리면
   **논문에 없는 통제된 MATE vs Transformer 비교**가 처음으로 성립한다
   (논문의 Transformer는 V-MPO 아래에 있어 축이 섞여 있다).

---

## 2. 지금 돌고 있는 것 — 메모리 축, DDQN 고정

`bash /tmp/launch_mem_axis.sh` · 24,000 ep · seed 42 · 예상 70분

비특권 레버 `structured_potions` **하나만** 켰다. `canonicalize_oracle`과
`context_graph_only`는 껐다 — 둘 다 특권이고 `is_oracle` 없이는 의미가 없다.
나머지 안정화 스택은 oracle 런과 동일하게 맞춰 **232.3 천장과 직접 비교**되게 했다.

| 런 | 메모리 | GPU | 로그 |
|---|---|---|---|
| `mem_mate_sp_s42` | MATE | 0 | `/tmp/mem_mate.log` |
| `mem_gpt_sp_s42` | GPT (Transformer) | 1 | `/tmp/mem_gpt.log` |
| `mem_lstm_sp_s42` | LSTM | 0 | `/tmp/mem_lstm.log` |

**특권 없음 확인.** 기동 로그가 셋 다 `obs_dim 76 act_dim 40`.
76 = 40 + 36 (물약 블록 12×2 → 12×5)이고 28차원 `chem_gt`가 더해져 있지 않다.
`main.py:84`의 `is_oracle = config_seq.seq_model.get("is_oracle", False)`가
mate/gpt/lstm 설정에서 `False`로 떨어지기 때문이다.

**함정 (실제로 밟았다).** ml_collections는 `--config_seq=<파일>`이 어떤
`--config_seq.<서브플래그>`보다 **먼저** 와야 한다. 순서가 틀리면
`Found --config_seq.X in argv before a value for --config_seq was specified`로
즉사한다. 1차 기동이 이걸로 세 런 모두 죽었다.

**읽는 법:** 천장 232.3 / 바닥 145.2.
- 셋 다 ~150 → 메모리 축 전체가 이 예산에서 작동하지 않는다는 뜻.
  알고리즘/스케일 문제이지 MATE 고유 문제가 아니다.
- MATE만 낮다 → 처음으로 MATE 고유의 결함 증거. slot permutation(§4-P1)이 1순위 용의자.
- 셋 다 200+ → 인수인계 값 152.4가 틀렸던 것. 로드맵 전체 재검토.

---

## 3. 레포에 실제로 있는 알고리즘 (조사 완료)

| 레지스트리 항목 | 학습 규칙 | Alchemy에서 |
|---|---|---|
| `dqn` | off-policy · **1-step TD** · Double DQN · PopArt · 행동 마스킹 | 지금 쓰는 것 |
| `sac` | off-policy · 1-step · twin-Q · 엔트로피 자동 α | **불가** — `continuous_action = True`가 `policy_rnn_sac.py:43`에 하드코딩 → 이산 40액션에서 shape 에러 |

레지스트리에 이 둘뿐이다. GAE · advantage · importance ratio · 엔트로피 계수 ·
n-step — **트리 전체에 하나도 없다.**

### 삭제된 구현 (복원 가능, 직접 확인함)

```
107f469  refactor: remove PPO implementation and associated rollout buffer logic
  configs/rl/ppo_default.py         |  28 ----   GAE lam=0.95, eps_clip=0.2, ppo_epochs=10
  policies/models/policy_rnn_ppo.py | 189 ----
  policies/rl/ppo.py                |  34 ----
  buffers/rollout_buffer.py         |  49 ----   values/logprobs/advantages/returns

git show 107f469^:policies/models/policy_rnn_ppo.py   # 189줄
git show d8356fe^:policies/rl/sacd.py                 # SAC-discrete 87줄
```

**SAC 파일 안은 이미 90% 이산용이다.** `if not self.continuous_action:` 분기들
(`:308`, `:329`, `:390`)이 이미 올바른 SAC-discrete 수식이고,
`CategoricalPolicy`도 `actor.py:158`에 완성돼 있는데 **아무 데서도 import를
안 한다.** 도달 불가능한 상태로 완성돼 있다.

### 문서 정정 필요

`CLAUDE.md`가 가리키는 `policies/rl/dqn.py`, `policies/rl/sac.py`,
`RL_ALGORITHMS` 레지스트리는 **존재하지 않는다** — `c8e3629`에서 디렉터리째
삭제됐다. 실제 경로는 `policies/models/policy_rnn_dqn.py` /
`policy_rnn_sac.py`, 레지스트리는 `AGENT_CLASSES`.

---

## 4. 로드맵

### P0 — 메모리 축 3런 (진행 중, §2)

### P1 — n-step / λ-return  ★ 다음 순서 추천

| | |
|---|---|
| 비용 | **2~3시간** |
| 테스트하는 것 | 1-step 부트스트랩이 병목인가 |
| 왜 1순위 | off-policy를 유지한 채 신용 할당만 바꾼다. 샘플 효율을 희생하지 않는다 |

버퍼(`buffers/rollout_buffer.py`)가 이미 에피소드를 시간축으로 연속 저장한다 —
`rewards`, `terminals`, `masks`가 모두 `(max_episode_len+1, num_episodes, 1)`.
n-step 타깃은 `_compute_loss` 안에서 이미 반환된 텐서에 대한 shift + discount일
뿐이고 **저장소 변경이 0**이다. DQN·SAC 양쪽에 동시 적용된다.

주의할 점 하나: `_window` (`:143-167`)가 BPTT 윈도를 자르므로, 윈도 꼬리
근처의 n-step 부트스트랩은 윈도 경계로 clamp하거나 1-step으로 폴백해야 한다.

**검증 방법:** `oracle_full` 설정(232.3)에 n-step만 얹는다. 메모리가 GT로
고정돼 있으므로 움직임이 있다면 그건 순수하게 알고리즘 축의 이득이다.
- 움직인다 → 알고리즘 격차가 실재. PPO 구축이 정당화된다.
- 안 움직인다 → 남은 52점은 스케일. 여기서 알고리즘 작업은 헛수고이고
  MSC 쪽(P3)으로 간다.

### P2 — 갈림길 (P1 결과에 따라)

**P2a. SAC-discrete 부활** — 반나절. 두 번째 off-policy 데이터점.
`git show d8356fe^:policies/rl/sacd.py`가 줄 단위 참조가 된다. 필요한 변경:
action space에서 `continuous_action`을 읽기, `build_actor` → `CategoricalPolicy`,
`build_critic` → `output_size=action_dim` (`input_size`에 `+action_dim` 제거),
`forward_actor`가 `(probs, log_probs)` 반환, `use_target_actor=False`,
`target_entropy`를 `log(A)`로 스케일, `prepare_recurrent_batch` 두 곳에
`discrete_action_dim` 추가. 추가로 행동 마스킹(~2시간)은 DQN과 달리
**softmax 이전에 logit을 마스킹**해야 한다.

**P2b. PPO 복원** — 1~2일. 논문 계열(on-policy AC)의 직접 프로브.
189줄이 구 `RL_ALGORITHMS` 레지스트리와 `pos_offset`/`memory_mask`/MSC
이전 `RNN_head` API 기준이라 배선 재작업이 필요하고, 버퍼의
`values`/`logprobs`/`advantages`/`returns` 배열과 on-policy(최근 에피소드만)
샘플러를 되살려야 한다.

> **경고.** PPO가 우리 예산에서 DDQN을 이길 거라 기대하면 안 된다.
> on-policy는 샘플을 한 번 쓰고 버리는데 우리 예산은 4.8e6 스텝이다.
> 논문이 2e10을 쓴 건 우연이 아니고 on-policy AC가 빛나는 지점이 거기다.
> PPO가 158보다 낮게 나와도 그건 "on-policy가 틀렸다"가 아니라 "예산이
> 부족하다"의 증거일 뿐이라, **해석이 안 되는 실험이 될 위험**이 있다.
> P1 없이 P2b로 직행하지 말 것.

**P2c. V-MPO / IMPALA** — 각 3~5일+. 밑바탕 전무(V-trace 없음, actor-learner
분리 없음, on-policy 경로 없음). IMPALA는 단일 프로세스 `AsyncVectorEnv` +
리플레이 설계와 정면으로 충돌한다. **현 시점 권장하지 않음.**

### P3 — slot-equivariant 인코더 (DeepSets)

비특권. 돌 3슬롯과 물약 12슬롯은 **순서에 의미가 없는 집합**인데 MLP는 고정된
위치로 읽는다. "슬롯 3의 물약"과 "슬롯 7의 같은 물약"을 별개로 배우고 있다는 뜻.
MATE에 특히 아픈데, `full_transition=True`라 transition의 델타가 곧 법칙의
증거이고 MATE는 그 증거들을 **12가지 슬롯 의존 기저 위에서 평균**내고 있기
때문이다. 43%로 추정한 "탐색" 몫을 좁혀줄 후보이기도 하다.

### P4 — MSC (CPC 보조 손실)

논문이 **특권 정보 없이** 제대로 된 메타러닝을 얻어낸 유일한 경우가
symbolic + predict-features 보조 과제(≈265)였다. 이 레포의 `mate_msc_v2` /
`mate_msc_ema_v2`가 같은 계열이다. **방향이 논문 결과와 일치한다.**
P1이 "스케일 문제"로 결론나면 여기가 남은 유일한 길이다.

---

## 5. 스케일 격차 — 정직하게

로그 타임스탬프 실측: 24,000 에피소드(4.8e6 스텝)에 2런 동시 실행으로 40~45분.
박스 전체 처리량 약 **4,000 steps/s** (Alchemy는 CPU 바운드, `n_env=64`).

논문의 2e10 스텝 → **박스를 통째로 써서 한 런에 약 2개월.** 논문은 조건·시드를
여럿 돌렸으니 실제 격차는 더 크다. **그대로 재현은 불가능하다.**

다만 열려 있는 질문: **GT 조건이 정말 2e10을 필요로 했는가?** GT를 주면 문제가
훨씬 쉬워지니 훨씬 일찍 수렴했을 수도 있는데, 논문에 그 조건의 학습 곡선이
없어서 모른다. 이건 우리가 on-policy AC를 만들어야만 알 수 있다.

---

## 6. 미결 (사용자 결정 대기)

- `requirements.txt`에 `torch==2.11.0+cu126` 고정, 또는 `main.py`에
  `torch.cuda.is_available()` assert 추가 (3회 제기, 1회 기각)
- `CLAUDE.md` 정정: (a) `policies/rl/*.py` / `RL_ALGORITHMS` 경로가 존재하지
  않음, (b) `chemistry_oracle`가 `IdealObserverBot`을 "대체"한다는 서술 —
  논문 기준 Oracle(288.5)과 Ideal Observer(284.4)는 별개 레퍼런스이고
  우리 287.1은 Oracle 쪽에 대응
- `docs/alchemy_status.md`에 4건의 사실 정정 반영 (프레임 체인 / "100%" 주장 /
  reward 스케일 `/3` / 물약은 "뒤집기"가 아니라 "만들기")
- 대시보드 §04 레버 2 카드를 관측값 덤프 대신 평이한 설명으로 교체
- `alchemy_test` 미푸시 커밋 8개
