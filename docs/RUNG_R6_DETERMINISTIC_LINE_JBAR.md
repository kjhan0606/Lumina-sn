# 단 R6 — 결정론 팔의 정본 line-J̄ (사전등록, 2026-08-08)

L1-3(결정론 부트스트랩 + 두 팔 합류)의 **막는 조각**.
R7 판정이 이 자리를 정확히 지목했다:

```
[R7][PHASE] lane=DET iter=0 phase=view  r=1  line_status=-1 line_r=0
[R7][PHASE] lane=DET iter=0 phase=a208  r=1  o=1
[A2-09][BLOCKED] R7_PUBLICATION_BLOCKED rc=3 blocked_stale_line=1
```

---

## 계약 (하나)

> **결정론 팔은 연속체 장과 **같은 원자적 commit** 안에서, MC 팔과 **같은 q-set 정체성**
> (q_set_hash · profile_id · profile_hash)을 갖는 line-J̄ 블록을 발행한다.**

---

## 1. 실측 — 무엇이 있고 무엇이 없나

| | 상태 |
|---|---|
| 결정론 line-J̄ **생산자** | **있다.** `cmfgen_lineres_jbar`(`lumina_cmfgen.c:4043-4490`)가 미세격자 `fs.J` 를 선 프로파일로 적분해 `opac->jbar_line_det[l*NS+s]` 를 채운다 |
| 그 생산자의 **적용범위** | ★**UV 펌프 창에 한정**. 창 밖 선은 센티널 `-1.0`(`:4100`, `:4472`) |
| 결정론 **commit** | `cmfgen_commit_jnu`(`:3399`)가 **연속체만** 올린다 ⟹ `line_n=0` ⟹ `radiation_field.c:655` 에서 line generation 이 0 으로 남는다 |
| MC commit 의 line 필드 | `line_n·line_id·line_q_set_hash·line_profile_id·line_profile_hash·line_sum·line_sumsq·line_count·line_n_packets·line_error_latch` (`lumina_main.c:532`) |
| 게이트 | `LUMINA_CMF_LINERES_JBAR` — 즉 현재는 **노브 뒤에 있다** |

## 2. ★설계 질문 (Codex 소관 — 운전석이 답을 정하지 않는다)

1. **MC 추정자 모양의 계약에 무잡음 양을 어떻게 넣는가.**
   `line_sum/sumsq/count/n_packets` 는 표본 통계다. 결정론 값은 표본이 아니다.
   `statistic_kind` 에 결정론 종류가 이미 있는가?  없다면 무엇이 정직한가 —
   `count=1, sumsq` 를 0 으로 두는 것은 **분산 0 을 주장**하는 것이므로 의미론을 명시해야 한다.
2. **부분 적용범위를 어떻게 발행하는가.**
   생산자가 UV 창만 채운다.  창 밖 선을 (a) 창을 넓혀 전부 채우거나 (b) **선별 validity**
   (VALID/UNSAMPLED)로 정직하게 올리거나 — **어느 쪽도 조용한 0 이나 센티널 누출은 안 된다.**
   ⚠`-1.0` 센티널이 소비자에 그대로 새면 음의 J̄ 다.
3. **q-set 정체성을 두 팔이 어떻게 공유하는가.**
   MC 는 `line_qset` 을 만들어 해시를 낸다.  결정론 팔이 **같은 해시**를 내려면 같은 선 집합·
   같은 프로파일 정의를 써야 한다.  누가 q-set 의 소유자인가?
4. **노브 뒤에서 꺼낸다.**  `LUMINA_CMF_LINERES_JBAR` 가 꺼져 있으면 line-J̄ 가 없다 ⟹
   a209 가 막힌다.  그러면 그것은 노브가 아니라 **계약의 일부**다(사건 측도 단과 같은 계급).

## 3. 기대 변경집합 (사전등록)

1. `cmfgen_commit_jnu` 가 연속체와 line 블록을 **하나의 원자적 commit** 으로 올린다.
2. 결정론 provenance 가 명시된다(`CMFGEN_REPLAY` 계열과 구분되는 이름).
3. 적용범위가 **선별 validity** 로 표현된다 — 센티널이 소비자에 새지 않는다.
4. q-set 정체성 공유 — 두 팔의 `q_set_hash`·`profile_id`·`profile_hash` 가 **일치**한다.
5. `LUMINA_CMF_LINERES_JBAR` 노브 제거 또는 계약화.

**변경하지 않는 것**: `cmfgen_lineres_jbar` 의 적분 물리, MC 팔의 line 누적.

## 4. 음성 대조

| # | 주입 | 기대 |
|---|---|---|
| **N6-1** | line 블록을 뺀 commit(현행) | `A2-09 blocked_stale_line=1` — **오늘의 상태가 그대로 대조다** |
| **N6-2** | q-set 해시를 한 글자 바꾼다 | `LINE_JBAR_VIEW` 가 `QHASH_MISMATCH` 로 거부 |
| **N6-3** | 창 밖 선을 **VALID 로 위장**해 올린다(센티널 -1 그대로) | 소비자가 음의 J̄ 를 거부해야 한다.  통과하면 FAIL |
| **N6-4** | ★창 밖 선을 정직하게 `UNSAMPLED` 로 올린다 | a209 는 **통과**하고, 그 선을 쓰는 SE 만 이름 있는 사유로 막혀야 한다. "부분 적용범위 = 전체 차단" 이면 게이트 과잉 |

★N6-4 가 이 단의 NC3 다 — **정직한 부분 정보**와 **없음**을 가른다.

## 5. 게이트

| | 기준 |
|---|---|
| **R6-1** | DET 팔이 `phase=a209` 를 통과하고 `[A2-10][PRE] lane=DET` 에서 `line=r` |
| **R6-2** | 두 팔의 `q_set_hash`·`profile_id`·`profile_hash` **문자열 동일** |
| **R6-3** | N6-1~N6-4 기대대로 (특히 N6-3 이 FAIL 을 시연) |
| **R6-4** | 회귀: MC 팔 바이트-parity (결정론 발행 추가가 MC 를 건드리지 않는다) |
| **R6-5** | 적용범위 수치가 대장에 기재 — 전체 선 중 몇 %가 VALID 인가 |

## 6. 이 단이 열면

L1-3 의 세 최소조건 중 ③이 폐합된다(①은 R5·②는 R7 에서 이미).
그러면 **결정론 팔이 A2-10 까지 완주**하고, 두 팔 합류의 잣대(3열 지도)를 세울 수 있다.

⚠단, MC 팔의 실제 수송은 **사건 측도 단(E)** 이 열려야 산다.
R6 는 결정론 팔만 살린다 — 두 팔 비교는 E 이후다.
