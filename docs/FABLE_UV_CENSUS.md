# FABLE_UV_CENSUS — UV 과잉/형광 결손 사슬 전수조사

- 감사자: fable (독립 감사 레인). `docs/CODEX_UV_CENSUS.md`는 **열람하지 않았다** (독립성 유지).
- 모드: **읽기 전용**. 소스 수정 0, 신규 런 0, 커밋 0, GPU 제출 0.
- 감사 기준시각: 2026-08-02 (최신 문서 mtime `docs/CODEX_EMISS_E13_SUMMARY.md` 09:33).
- 1차 실측 대상 런: `/gpfs/kjhan/lumina_runner2/scratch/fluormat_capture_188828`
  (parity59 + `LUMINA_FLUOR_MATRIX_DUMP`, 설정 `docs/PARITY59_FLUORMAT.env`, `stdout.log` 38,095줄, iter 0–11).
- 표기: **[F]** = 본 감사에서 원자료(런 로그·CSV·소스)로 직접 실측/확인한 것.
  **[D]** = 기존 문서에서 인용(출처 명기). **[A]** = ARTIS 원본 소스 대조.

---

## 0. 한 줄 결론

> **UV 과잉은 "형광이 모자란" 문제가 아니라, 두 레인이 각자 다른 방식으로 UV를 자기 자신에게 되먹이고 있는 문제다.**
> 결정론 레인은 선 불투명도의 비열화분을 **주파수 결맞음 산란**으로 취급해 같은 빈에 가두고(`lumina_cmfgen.c:1073`),
> MC 레인은 매크로원자가 UV에서 UV로만 재분배한다(선방출의 93.8%가 λ<2900 Å). **[F]**
> 두 레인 모두 "재분배 연산자"가 잘못된 것이며, E10/E12/E13이 실패한 이유는
> **고장난 레인(MC)에서 측정한 연산자로 고장난 레인(결정론)을 고치려 했기 때문**이다.

---

## 1. 판정 요약 (과업 C 선행 제시)

| 순위 | 후보 | 크기 | 상태 | 단일-인자 판별시험 |
|---|---|---|---|---|
| **C1** | `chi_coherent = chi_es + (1−ε_l)·chi_line` — 비열화 선 불투명도를 **결맞음 산란**으로 취급 | UV 11.98× 과잉의 **99.9985%**를 대수적으로 폐합 (E8) | **확정(기전)·미확정(수리안)** | §5.1 |
| **C2** | MC 매크로원자가 UV→UV 순환기 — 선방출 93.8%가 UV, Fe III `p_iup`=0.886 | 재분배 행렬 자체가 UV 보존적 ⟹ E10/E12/E13 실패의 직접 원인 | **확정(현상)·미확정(씨앗)** | §5.2 |
| **C3** | 전 셸 **T_rad ≡ 10470.09 K 고정** (`TRAD_COLOR_FIX` + pure-CMFGEN 경로가 `solve_radiation_field` 미호출) | 미계량. UV 자체엔 작을 가능성 큼(§4.3 계산) but 잣대·τ·유도항 전반 오염 | **미해결(UNRESOLVED)** | §5.3 |

부수 확정 2건 (작지만 실물):
- **F-a** k-packet 자유-속박(fb) 배출은 **900 Å 미만**으로 떨어져 방출 센서스에 에너지 0으로 기록된다. **[F]**
- **F-b** 형광 행렬은 line 상호작용 **한 지점에서만** 기록된다(단일 호출부 `lumina_cuda.cu:6127-6132`). bf 활성화·k-packet 단독 활성화 입력은 연산자에 **없다**. **[F]**

---

## 2. 과업 A — 사슬 단계별 전수표

각 행: ① 가정 ② 검증 여부·근거 ③ **단독으로 UV 편중을 만들 수 있는가** ④ 판별시험(끄거나 항등화).

### A1. MC 수송 물리 (패킷 → 선 공명 교차)

① **가정** — Sobolev 근사(교차 = 순간·국소), 팽창 불투명도, 선목록은 진동수 정렬 이진탐색.
② **검증** — 부분. Stage-3.1 KA1/KA2/KA3는 **결정론 solver**를 검증했지 MC 커널을 검증하지 않았다. MC 추정자에 대한 오라클·왕복시험은 어느 문서에도 없다 **[D: Stage31 감사]**. 소스 상 무계수 절단 1건: `comov_nu <= nlte_nu_min`(λ>20000 Å) 패킷은 카운터 없이 버려진다(`src/lumina_cuda.cu:3484`) **[D]**.
③ **단독 UV 편중 가능?** — 아니오(직접적으로는). 단, 7D가 이 레인을 사실상 무죄로 만든 방식에 구조적 결함이 있다(A8 참조).
④ **판별** — 이미 있음: 7D의 `J_det/J_MC = 0.977181` (`docs/CODEX_STAGE31_BENCH7D.md:116-121`) **[D]**. **다만 그 시험은 §A8의 이유로 순환적이다.**

### A2. 이벤트 분류 (etype → 채널 태그)

① **가정** — `d_evch_from_etype`(`src/lumina_cuda.cu:4486-4497`)가 etype→채널을 유일하게 결정하고, etype 2/4는 호출부가 태그를 명시 제공. **[F]**
② **검증** — 채널→버킷 팬아웃(`d_census_accumulate`, `:4500-4545`)은 소스로 확인. 방출 센서스 `line` 열은 `EVCH_MA_RAD_DEEXC + EVCH_KPKT_COLLEXC_BB + EVCH_KPKT_COLLEXC` 3개를 합산(`:4509-4512`) **[F]**. 이벤트 로그 자체는 과거 cap 87% 드롭 전력 있음(**[D]** 사례 20).
③ **단독 UV 편중 가능?** — 아니오(계측 전용). **다만 잣대 결함은 만든다**: 센서스 파장창 `[CENSUS_EM_LAM_MIN, CENSUS_EM_LAM_MAX] = [900, 30000] Å` 밖 배출은 **에너지 0으로 보인다**.
④ **판별 (본 감사 실측)** — kpkt 배출 센서스는 fb 배출을 s0에서 1,445건 세는데(`lumina_census_kpkt_exit.csv`), 방출 센서스의 `fb` 열은 전 파장 합 **정확히 0.0**이다(`lumina_census_emission.csv`) **[F]**. ⟹ **fb 재결합 배출은 전량 900 Å 미만**. (기존 기록 "fb가 166–355 Å에서 발화" **[D]** BFNL:24 와 독립 정합.)

### A3. 매크로원자 분기 확률 조립 (`compute_transition_probabilities`)

① **가정** — Lucy-2002 에너지흐름 가중: 방출 ∝ `A_ul·β·hν_ij`, 내부점프(상·하) ∝ `rate·ε_lower`(중성바닥 기준 = 여기E + 누적 IP). 내부상향 = `(B_lu − B_ul·n_u/n_l)·β·J_blue`.
② **검증 — ARTIS 대조 [A]**:

| 항목 | ARTIS | Lumina | 판정 |
|---|---|---|---|
| 방출 가중 | `R·epsilon_trans`, `R=A_ul·β` (`macroatom.cc:101`, `macroatom.h:41-58`) | `A_ul·β·hν` (`lumina_plasma.c:4245,4564`) | **일치** |
| 내부하향 가중 | `(R+C)·epsilon_target` (`:103`) | `(A_ul·β + C)·ε_lower` (`:4285-4326,4568`) | **일치**(IDOWN_BETA=1 하에서) |
| 내부상향 가중 | `(R+C+NT)·epsilon_current` (`:132`) | `(coeff·β·J_blue + C_up)·ε_lower` (`:4502,4544,4568`) | **일치** |
| ε 기준점 | 중성바닥(`input.cc:403 (energyoffset_ev+levelenergy_ev)*EV`) | 중성바닥(`NEUTRAL_E=1`, `:4569-4574`) | **일치** |
| 상향률의 n_u/n_l | **실제 준위인구** `get_cellcache_levelpop` (`macroatom.cc:586`) | **희석-Boltzmann @ T_rad** (`:4491-4498`) | **불일치** |
| 상향률의 J | `radfield(nu_trans)` = 구간 희석-BB 모델값. `DETAILED_LINE_ESTIMATORS_ON=false` (`artisoptions.h:74`, classic도 `:68`) | **선별 MC blue-wing 추정자** `jblue_line` (85.7% 커버) | **불일치** |

③ **단독 UV 편중 가능?** — **예.** §4에서 정량화.
④ **판별** — `LUMINA_IUP_BINFIELD=1`이 이미 존재하며(`lumina_plasma.c:3350-3372`) 정확히 ARTIS의 `radfield(nu_trans)` 의미론을 재현한다. 현재 기본 OFF·parity59에 미설정. **단, 과거 parity28에서 역방향 결과** **[D]** APC:144.

### A4. 형광행렬 누적 (`d_fluor_matrix_record`)

① **가정** — "입력 = 선 흡수 직전 공변 (ν, E), 출력 = 캐스케이드 직후 공변 (ν, E)", 1000×1000 log-ν 격자.
② **검증** — 소스 확인: 인덱스 `ib = floor(log(ν_in/ν_min)/dlnν)`, `ob` 동형(`lumina_cuda.cu:4177-4183`), 누적 `matrix[ib*nbins+ob] += E_out`(`:4212`) **[F]**. 열폐쇄 2.045e-13, 총에너지 폐쇄 `Eemit/Eabs − 1 = 3.0443e-08` **[F: stdout.log:35886]**. E13이 양축 미러 반증 완료 **[D]**.
③ **단독 UV 편중 가능?** — **아니오, 그러나 결정적 구조 결손이 있다.**
   - **호출부가 단 하나**: `interaction_type == 1` (LINE) 분기 안 **[F: `lumina_cuda.cu:6102-6132`]**. bf 흡수로 활성화된 매크로원자, 순수 k-packet 경로의 입력은 **행렬에 존재하지 않는다.**
   - 행렬은 에너지 **정확 보존**(3.04e-8) ⟹ 파괴(열화) 자유도가 **구조적으로 0**. 순수 재분배 연산자다.
④ **판별** — 이미 실행됨(E12): 무편향 473k edge로 바꿔도 B0 8.29→26.43 악화 **[D]**. 본 감사의 해석은 §3.

### A5. 이진 덤프 (LCMFCE01 / LFMAT001)

① **가정** — 필드 단위 little-endian, 구조체 통짜 fwrite 금지, `dnu>0`, ν 내림차순.
② **검증** — 강함. 헤더 64B + `r_edge[nr+1]` + `nu[nnu]` + `dnu[nnu]` + 6×`[nr*nnu]` = 2,416,472 B가 실파일 크기와 일치 **[D+F: 파일 크기 실측 일치]**. sha256 사이드카, `eta_total == eta_fixed + eta_coherent` 정확 double 비교로 fail-closed.
③ **단독 UV 편중 가능?** — 아니오.
④ **판별** — 왕복 자체시험 존재(단 2셸×3빈 픽스처). **잔여 위험**: C 리더는 매니페스트 1행에서 `sha256=`를 파싱하는데 writer는 JSON `"sha256": "…"`을 쓴다 **[D]** — 벤치에서는 driver의 JSON 경로만 사용되어 노출 안 됨.

### A6. 판독 (`stage31_cmf_field_driver.c`, `emiss_e11_fluor_matrix.py`)

① **가정** — 계약 검사 `nr==50, nnu==1000, iteration==10, field_generation==10`, 3 플래그 전부.
② **검증** — fail-closed 확인 **[D]**.
③ **단독 UV 편중 가능?** — 아니오.
④ **판별** — 3회 SHA 동일성 확인됨.

### A7. 재분배 적용 (`emiss_e10_apply_redistribution.py`)

① **가정** — `η_j ← η_j − a_j/Δν_j + Σ_i R[j,i]·a_i/Δν_j`, `a_i = (1−ε_MC)·χ_line,proxy,i·J_i·Δν_i`.
② **검증** — 에너지 폐쇄 −6.77e-15 **[D]**.
③ **단독 UV 편중 가능?** — **예, 그리고 실제로 만들었다** (B0을 8.29→26.43으로).
④ **판별 (본 감사가 지목하는 것)** — **전역(all-shell) 연산자를 s8 단일 셸의 동결 선-반환 프록시에 적용**한다는 공변량 오정합. E12가 잔여 원인 #2로 기록 **[D]**. 여기에 §A4의 "입력 클래스가 선 흡수뿐"이 겹친다.

### A8. χ/η 재구성 + stage31 수송 ★

① **가정** — `input.eta_fixed = frozen.eta_total`, `chi_coherent = NULL`, `scatter_mode = NONE`.
② **검증** — KA1/KA2/KA3 통과.
③ **단독 UV 편중 가능?** — **아니오, 그러나 이 단계가 "수송 무죄" 판정을 구조적으로 예약한다.**
④ **판별 — 잣대 결함 2건 [D, Stage31 감사]:**
   - **(i) 순환성.** `eta_total = eta_fixed + chi_es·J_producer`이므로 결정론 solve는 **생산자 자신의 산란 방출률을 그대로 입력받는다.** "MC 장의 97.7% 재현"은 구조적으로 거의 보장된 결과다.
   - **(ii) 비교 대상 오정합.** 설계문서 `:499`는 `J_MC`를 **C2의 `J_raw`(생 MC 추정자)**로 규정했으나 구현은 payload의 `J_producer`(생산자 자신의 `cmfgen_solve_J` 후-감쇠 출력)를 쓴다(`scripts/stage31_cmf_field_bench.py:613`). 즉 **"97.7%"는 결정론 대 결정론**이다.
   - **(iii)** 7D PASS는 KA에서 스스로 적합한 절단허용치(`C=27.806`, 안전계수 1.0)에 의존하며, 생산 solve에서 인증-음수 90,761건·부호미정 931,122건을 허용한다. 동일 solver가 7C에서는 하드 FAIL했다.

### A9. 대역 집계

① **가정** — B0 600–1000 / B1 1000–1500 / B2 1500–2000 / B3 2000–2500 / B4 2500–3000 / BALL 600–3000 Å, 밴드값 = Δν-가중 평균강도.
② **검증** — 부분중첩 빈까지 정확 Δν 가중. **밴드 집계 함수 자체에 대한 단위시험·오라클은 없다** **[D]**.
③ **단독 UV 편중 가능?** — 아니오.
④ **판별** — CMFGEN 원격자→1000빈 적분보존 = 1.000000000000000 **[D]**.

### A10. CMFGEN 대조

① **가정** — `toy06_19.48d_jnu4`의 RVTJ/EDDFACTOR, 속도 log-J 보간(s8 가중 0.863582) 후 **적분보존** 평균.
② **검증** — 보존 1.0. 단 보간 가중치는 Γ 삼중대조 문서에서 상속(재유도 안 함).
③ **단독 UV 편중 가능?** — 아니오.
④ **판별** — s44–s49는 RVTJ 속도범위 밖 ⟹ 외삽 금지·UNRESOLVED 유지 중.

---

## 3. 왜 E10/E12/E13이 실패했는가 — 본 감사의 핵심 진단

E8이 정한 수리표적은 "다중선 재분배 연산자 `R`"이었다. E10(prefix)·E12(무편향)·E13(미러)이 모두 형상 FAIL했고, B0은 8.29 → 20.91 → 26.43 → 53.97로 **단조 악화**했다 **[D]**.

**본 감사의 진단: `R`을 MC 레인에서 측정했기 때문이다. MC 레인의 재분배는 UV 보존적이다.**

본 감사 실측 3건 (모두 동일 런 `fluormat_capture_188828`, iter 10/11):

**(1) MC 선방출의 93.8%가 UV다. [F]** `lumina_census_emission.csv` (에너지 가중, 공변, 창 900–30000 Å):

| 채널 | 전 파장 합 |
|---|---|
| `line` (MA_RAD_DEEXC + KPKT_COLLEXC[_BB]) | **2726** |
| ─ 그중 bin 0–7 (900 – 2896.5 Å) | **2511.0 = 92.1%** |
| ─ bin 0–8 포함 (900 – 3352.1 Å) | **2556.6 = 93.8%** |
| ─ bin 9 이상 (≥ 3352.1 Å) | 169.4 = 6.2% |
| `escatter` | 23.09 |
| `ff` | 6.30e-3 |
| `fb`, `bte`, `macap`, `bf_reemit`, `linetherm` | **정확히 0** |

빈별 최대는 bin 3 (1395–1615 Å) = 914.9, 다음이 bin 4 (1615–1869 Å) = 477.9.
`linetherm`이 0인 것은 `LUMINA_LINE_THERM=1`이 설정되어 있음에도 그렇다 — **미해결 배선 질문으로 등재**(§6).

**(2) 매크로원자는 사실상 열화하지 않는다. [F]** `lumina_census_ma_fate.csv` 전 셸 합:

| 종착 | 건수 | 분율 |
|---|---|---|
| `rad_deexc` | 485,105,515 | 0.99976 |
| `col_deexc` | **4,351** | **8.967e-06** |
| `rad_recomb` | 118,732 | 2.447e-04 |
| `internal_up_res` / `internal_down_res` | 0 / 0 | 0 |

터미널 ε 파괴(별도 경로)는 `[MA-LINE-DESTRUCT] it10: terminals=510,900,313 destroyed=555,876 (frac=0.0011)` **[F: stdout.log:35888]**.

**(3) k-packet은 열배출하지 않고 재주입한다. [F]** `lumina_census_kpkt_exit.csv` s0: `collexc 479,064 / ff 693 / fb 1,445 / total 481,202` ⟹ **collexc 99.56%**. 그리고 §A2에 따라 그 fb 1,445건의 에너지는 900 Å 미만으로 사라진다.

**(4) 형광 행렬은 에너지를 정확히 보존한다. [F]** `Eabs=3065.5031989625031, Eemit=3065.5032922856731, rel=3.0443e-08`.

⟹ **연산자 `R`은 "UV를 UV로 되돌리는 지도"를 충실히 인코딩한 것**이며, 그것을 결정론 레인의 결맞음 반환 항에 곱하면 UV가 **더** 갇힌다. E12의 `B2→B0 = 54.9%`와 E13의 "B2 입력 → 상향 53.58%, 평균 ν_out/ν_in = +0.3594%" **[D]** 는 같은 사실의 다른 표현이다.

---

## 4. `p_iup ≈ 88%`는 어느 변경의 산물인가

### 4.1 실측 재확인 [F]

E13이 인용한 값은 런타임 진단 `diag_macro_branch`(`src/lumina_plasma.c:5563-5760`)의 출력이며, 본 감사가 `stdout.log`에서 직접 재확인했다. iter 10, shell 3 (`stdout.log:35829-35845`):

```
[MA-BRANCH] Strong-line (tau > 1.0) branching at shell 3
  strong-line in UVblnk        |   472 |  p_emit 0.1080 | p_iup 0.6266 | p_idn 0.2639
  strong-line in NIR2          |  5090 |  p_emit 0.1481 | p_iup 0.0879 | p_idn 0.7606
  UV-strong levels' BB-emit destination: UVblnk 90.6%  ... NIR2 6.2%
  UV-strong-group: <p_iup>/<p_idn> = 2.37
   ion       n  | <p_iup> <p_idn> | UV%
  Z=26 III    83 | 0.8859  0.0966 | 69.2      <-- Fe III = 88.6%
  Z=27 III    85 | 0.8325  0.1447 | 66.3
  Z=28 III    57 | 0.7573  0.2055 | 85.8
  Z=27 II    149 | 0.4475  0.3733 | 91.5
  Z=26 II     29 | 0.3942  0.3801 | 98.8
```

셸 의존성이 결정적이다 (iter 10):

| 셸 | UV-strong `p_iup/p_idn` |
|---|---|
| s0 | 2.46 |
| s3 | 2.37 |
| s16 | **0.05** |

s16의 Fe III UV-strong 준위는 `p_iup=0.0406`이다. ⟹ **병리는 IGE 코어(내부 셸) 국한**이며 외곽은 정상이다.

**잣대 주의 [F]:** `diag_macro_branch`는 `sum_iup[b] / n_per_band[b]` — **준위 균등가중 평균**이다(`:5716-5719`). 매크로원자가 실제로 그 준위에서 활성화되는 빈도로 가중하지 **않는다**. 따라서 "88%"는 트래픽 가중 분기확률이 아니다. 트래픽 가중 관측치는 §3(2)의 MA-FATE 센서스이며 그쪽은 `internal_up_res = 0`(내부상향으로 미종결된 활성화 없음)만 말해준다.

### 4.2 계보 귀속 — 어느 변경도 단독으로 만들지 않았다

`p_iup`을 결정하는 항 5개와 각각의 도입 시점:

| 항 | 값/식 | 도입 | UV 계량 당시 존재? |
|---|---|---|---|
| 방출 = `A_ul·β·hν` | ARTIS 일치 | `MACROATOM_EWEIGHT`, 06-20 이전 | 있음(TIR:36) |
| 내부하향 = `(A_ul·β+C)·ε_low` | ARTIS 일치 | `IDOWN_BETA` 07-05 `6d08a35`계열, `IDOWN_COLL` 07-06 | **있음** — epay18이 MC 탈출 SED를 34.5/32.2/18.4/8.0/5.9로 변형 **[D]** FO:115 |
| 내부상향 계수 `(B_lu−B_ul n_u/n_l)` | **희석-Boltzmann @ 고정 T_rad** | `IUP_JBLUE`, 커밋 `192a2c3` 07-14 | **없음**(도입 시 무계량). 07-30 parity49에서 "byte-identical, 효과 0" **[D]** |
| 내부상향 `β` | `tau_sobolev`의 β | 동상 | 07-14 `iupb` A/B가 β 적용 시 스펙트럼 붕괴 **[D]** P1:86 |
| 내부상향 `J_blue` | **MC blue-wing 추정자** | 동상 | 없음 |
| 내부상향 `+C_up` | ARTIS M1 | parity 시대(미커밋) | 없음 |
| 에너지가중 `ε_low` (중성바닥) | Fe III ≈ 24.1 eV + 여기E | `NEUTRAL_E`, 06-20 이전 | 있음 |

**⟹ `p_iup≈88%`는 단일 변경의 산물이 아니다.** 모든 항이 개별적으로는 ARTIS 형태와 일치하거나(방출·하향·ε 기준) ARTIS와 어긋나되 그 어긋남이 작다(§4.3). 지배 인자는 **항이 아니라 그 항이 먹는 장 `J_blue`의 크기**다.

### 4.3 정량 — 무엇이 실제로 크고 무엇이 작은가 [F, 산술]

셸 3 실측: `W=0.1018, T_rad=10470.09 K, T_e=15667.6 K, n_e=1.618e9` (`lumina_plasma_state.csv`).

**(a) 자극방출 포화 결손 — 작다.** ARTIS는 `R_over_J = (B_lu − B_ul n_u/n_l)β`에서 n_u/n_l에 실제 인구를 쓴다(`macroatom.cc:586`). Lumina는 희석-Boltzmann을 쓴다(`plasma.c:4496-4498`). 대수적으로 Lumina/ARTIS = `stim_corr_dilute / stim_corr_true`.
E7이 s8 Fe III UV 상위준위 `b_u = 2.315/2.372/3.649`를 실측했으므로 **[D]**, 1500 Å(8.27 eV)·T_e=15.7 kK에서
`n_u g_l/(n_l g_u) ≈ b_u·exp(−hν/kT_e) ≈ 3.6 × 2.66e-3 ≈ 9.6e-3` ⟹ `stim_corr_true ≈ 0.990`.
희석 쪽은 `W·exp(−hν/kT_rad) ≈ 0.102 × 1.04e-4 ≈ 1.1e-5` ⟹ `stim_corr_dilute ≈ 1.000`.
**⟹ 과펌프 배율 ≈ 1.01. 무시 가능.** 단 `b_u/b_l ≳ 100`이면 발산하므로 **조건부 결함으로 등재**(§6).

**(b) 에너지가중 비대칭 — 중간(3–7×).** 내부점프는 `ε_low`(Fe III 중성바닥 기준 ≈ IP(Fe I)+IP(Fe II) = 24.1 eV + 여기E)로, 방출은 `hν`(1500 Å = 8.27 eV / 2500 Å = 4.96 eV)로 가중된다. 비 = 3–7배. **ARTIS도 동일하므로 divergence 아님** **[A: `macroatom.cc:132` vs `:101`, `input.cc:403`]**.

**(c) 장의 크기 — 이것이 지배한다.** 단일 상향/방출 쌍의 비는
`B_lu J ε_low /(A_ul hν) = (g_u/g_l)·(c²/2hν³)J·(ε_low/hν)`.
`J = W·B_ν(T_rad)`라면 셸3에서 `(c²/2hν³)W B = W/(e^{hν/kT_rad}−1) = 1.06e-5`, ×(g_u/g_l≈3)×(ε/hν≈3.6) = **1.1e-4** — `p_iup`은 사실상 0이어야 한다.
그런데 E7 실측은 `J_ours/J_CMFGEN(B0) = 33.8`, `J_CMFGEN/B(T_e) ≈ 0.8` **[D]** ⟹ `J ≈ 27·B(T_e)`. 그리고 `B(T_e=15668)/[W·B(T_rad=10470)]` = `exp(9.17−6.13)/0.102 ≈ 205`.
⟹ **`J_blue / (W·B(T_rad)) ≈ 5.5e3`**, 위 비는 1.1e-4 × 5.5e3 ≈ **0.64/쌍**. Fe III 저준위의 상향 채널 다중도(수십~수백) × 하향 채널 소수 ⟹ `p_iup/p_emit ≈ 50`(실측 0.886/0.018)에 도달한다.

**결론:** `p_iup≈88%`는 **원인이 아니라 결과**다. 재순환으로 부풀려진 `J_UV`가 상향률을 지배하고, 상향률이 매크로원자를 UV로 되돌려 다시 `J_UV`를 유지한다. §5.2가 이 고리의 절단 시험이다.

### 4.4 blue-wing 추정자 vs 구간장 — 실측 불일치 [F]

`[JBLUE-ANCHOR2] it10` (`stdout.log:35892`), `log10(J_blue/J_line)` 4버킷:

| 버킷 | n | <log10> | clamp −3 / +3 |
|---|---|---|---|
| thin-in (β>0.5, 1000–4000 Å) | 39,562,567 | **+0.011** | 620 / 0 |
| **thin-out (β>0.5, 창 밖)** | 70,860,746 | **+0.434** | 14,650 / **12,199,822** |
| thick-in (β<0.01) | 15,805 | −0.017 | 0 / 0 |
| thick-out | 41,533 | −0.040 | 76 / 87 |

**thin-out 표본의 17.2%가 +3 dex에서 포화**한다. 광학적으로 **얇은** 선은 in-line `(1−β)S` 포화가 없으므로 `J_blue ≈ J_line`이어야 하며 thin-in 버킷은 실제로 그렇다(+0.011). 창(1000–4000 Å) **밖** = EUV(λ<1000 Å) 또는 λ>4000 Å에서만 3자릿수 이상 벌어지고, 방향은 **한쪽뿐**(−3 dex 포화는 0.02%).
구간 폭이 `Δlnν = 5.298e-3`(0.53%)에 불과하므로 이 격차는 빈 내 대비로 설명되지 않는다.
**두 해석 모두 가능하며 본 감사는 방향을 확정하지 않는다:** (i) `J_blue` 과대, (ii) 창 밖 C1 구간장 기근(memory의 "s12+ FUV 기근"과 정합). ⟹ **UNRESOLVED** (§6), 판별시험 §5.2b.

---

## 5. 과업 C — 단일-인자 판별시험 (오프라인·기존자산 우선)

### 5.1 C1 — 결맞음 선 산란

**주장 [F, 소스 직접 확인]:**
```c
/* src/lumina_cmfgen.c:1073 */
cs->chi_es[idx] = chi_e + (chi_ln - chi_ln_th);
/* chi_ln_th = line_eps * chi_ln   (:1048) */
```
`cs->chi_es`는 이후 formal solver에서 **주파수 결맞음 산란분** `r = chi_es/chi_tot`로 소비된다(`:1534, :1592, :1708, :1834, :2209, :2475`). 즉 **비열화 선 불투명도 전량이 같은 빈에 되돌아온다.**
2-준위 원자에서만 타당한 형식이며, 다준위 원자에서 열파괴되지 않은 광자는 ν_l로 재방출되지 않고 **캐스케이드의 다른 선으로 나간다**(= 형광). CMFGEN은 이것을 다준위 source function으로 처리한다.

**크기 [D, E7/E8]:** s8 BALL `chi_coherent/chi_total = 97.7713%`, `eps_eff = 1.90567e-4`, 재순환 이득 `5247.49×` vs 관측된 과잉이 요구하는 이득 `5247.41×` — **0.0015% 일치**.

**단일-인자 판별시험 T1 (오프라인, 신규 런 불필요):**
동결 payload `emiss_ab_iter10.A`만으로 실행 가능.
- 조작: `chi_coherent`를 `chi_e`만으로 재정의(선 산란분을 결맞음에서 제거)하고, 제거한 `(1−ε_l)·χ_line·J`를 **λ-독립 등방 재분배**(즉 `R = 균등` — MC에서 측정한 R을 **쓰지 않는다**)로 되돌린다.
- 사전등록: BALL `J_det/CMFGEN`이 11.70 → O(1) 근방으로 떨어져야 한다(E9가 스칼라 ε 치환으로 이미 0.932를 얻었으므로 진폭 자유도는 검증됨).
- **핵심 차별점**: E10/E12는 `R = MC측정`을 썼다. T1은 `R = 균등`을 쓴다. 두 결과의 차이가 **"연산자의 형상이 문제냐, 결맞음 가정 자체가 문제냐"를 분리**한다.
- 부호 규칙: T1이 형상까지 개선하면 → 결함은 **결맞음 가정**. T1도 형상 FAIL이면 → 결함은 **χ,η 조립의 다른 곳**(EPAY 후보 상승).

### 5.2 C2 — MC 매크로원자 UV→UV 순환

**T2a (오프라인, 기존 덤프):** `fluor_matrix_iter10`은 `(ib,ob)` 만 있고 `(Z,ion,line)` 태그가 없다 **[D: E13이 이 결손을 UNRESOLVED로 기록]**. 그러나 `lumina_events_lines.bin`(20.7 MB)이 같은 런에 있다. 이벤트 로그의 (흡수선 → 방출선) 쌍을 **이온별**로 집계하면 "Fe III UV 흡수 → 어떤 이온·어느 밴드로 방출"의 이온분해 지도가 나온다. 이것이 E13의 "이론 Fe II 98.40% / Fe III 89.64% vs 측정 92.74%"를 **이온분해로 대조 가능**하게 만드는 유일한 기존자산이다.
- 사전등록: Fe III 조건부 UV-exit이 이론 89.64%에 맞으면 → 원자 분기 무죄, 장(`J_blue`)이 범인. 크게 초과하면 → 확률 조립.

**T2b (오프라인, 기존 덤프):** §4.4의 방향 확정. `lumina_c1_bins.csv`(880 KB, 셸×빈×`J_bin,W,T_R,mode`)와 `lumina_census_jnu_fine.csv`가 같은 런에 있다. λ<1000 Å 빈에서 `J_bin`이 (i) 물리적으로 기근인지 (ii) `mode`가 `fit` 실패/railed인지 읽으면 된다.
- 사전등록: 해당 빈의 `mode`가 railed이거나 `W`가 1e-8급이면 → **잣대(J_line) 결함**, `J_blue` 무죄. 정상 fit인데도 3 dex 벌어지면 → **`J_blue` 정규화 결함**.

**T2c (판정런 1회, 위 둘이 결론 못 낼 때만):** `LUMINA_IUP_BINFIELD=1` 단일 토글. 이미 구현·문서화되어 있고(`plasma.c:3350-3372`) ARTIS의 실제 소비자(`radfield(nu_trans)`, `DETAILED_LINE_ESTIMATORS_ON=false`)를 재현한다. **이것이 "IUP-JBLUE = ARTIS-exact"라는 서사를 직접 반증/확증하는 유일한 단일 인자다.**
- 주의 [D]: parity28에서 역방향 결과("오염의 민주화") 전력이 있다. 사전등록에 그 방향을 명시해야 한다.

### 5.3 C3 — 전 셸 T_rad 고정

**주장 [F, 3중 확인]:**
1. `lumina_plasma_state.csv` 50행 전부 `T_rad = 10470.093240` (uniq = 1). T_e는 21227.6 → 8769.8로 정상 변화.
2. `stdout.log:139` `[TRAD-COLOR-FIX] T_rad[s>=1] := T_rad[0]=10470 K (W unchanged)` — 적재 시각 1회. 소스 `src/lumina_atomic.c:377-382`.
3. `solve_radiation_field`(T_rad/W를 MC 추정자로 갱신)의 **유일한 호출부는 `src/lumina_cuda.cu:10447`**(고전 MC 루프)이며, 본 런은 `stdout.log:332`가 보여주듯 **PURE-CMFGEN 경로**(`:7734-9880`)를 탔다. 로그 어디에도 갱신 배너가 없다.

**⟹ 생산 설정에서 `T_rad`와 `W`는 전 반복 동안 모델파일 값에 동결되며, `T_rad`는 추가로 단일 상수로 평탄화된다.** 이 값이 먹이는 곳:
- `tau_sobolev`의 성운 준위인구(`plasma.c:2637-2679`) — NLTE 집합 밖 이온 전부
- 매크로원자 상향률의 `n_u/n_l`(`:4491-4498`)
- J cap/floor 비교자 `W·planck_bnu(T_rad,ν)`(`:4402-4411`)

**T3 (오프라인, 산술):** §4.3(a) 방식으로 `stim_corr` 민감도는 이미 ≈1.01로 계량됐다. 남은 계량은 **`tau_sobolev`** 쪽이다. `lumina_levelpop.csv`(69 MB, `b_k` 포함)와 `data/tardis_reference_toy06_19p48d/line_list.csv`를 조인해, NLTE 집합 **밖** 이온이 UV(1700–3000 Å) `tau>1` 선 1,093개 중 몇 개를 소유하는지 세면 된다(`[TAU-DIAG] stdout.log:33441`: `UV(1700-3000)=1093/343613 sum=227895.6`, `blue=5/282011`, `opt=0/325359` **[F]**).
- 사전등록: 그 소유분이 τ 합의 5% 미만이면 → C3는 UV에 대해 **무해**로 강등하고 잣대 결함으로만 등재. 20% 초과면 → 승격.

---

## 6. UNRESOLVED 등재 (정직 기록)

| # | 항목 | 왜 미확정인가 |
|---|---|---|
| U1 | `J_blue` 과대 vs C1 구간장 기근 (§4.4) | 한쪽 방향 12.2M건 +3 dex 포화는 실물이나, 분자·분모 어느 쪽 결함인지 본 감사로 분리 불가. T2b 필요 |
| U2 | `linetherm` 열이 전 파장 0 (`LUMINA_LINE_THERM=1`인데도) **[F]** | 채널이 죽었는지, `EVCH_HEAT_LINETHERM` 태그가 센서스에 도달 안 하는지 미분리 |
| U3 | fb 배출이 900 Å 미만 **[F]** | 물리인지(재결합 문턱이 실제로 EUV) 단일-대표-edge 잔재인지 미분리 |
| U4 | 형광 행렬 입력 클래스가 선 흡수 단독 **[F]** | bf-활성화·k-packet-단독 입력이 빠진 연산자의 편향 크기 미계량 |
| U5 | `diag_macro_branch`의 준위 균등가중 **[F]** | `p_iup=88%`의 트래픽 가중값 미측정 ⟹ 88%의 물리적 무게 미확정 |
| U6 | 자극방출 포화 결손(§4.3a)의 조건부 발산 | 현재 b_u≈2–4에서 1.01×로 무해하나 `b_u/b_l ≳ 100` 영역에서 발산. 그런 선의 존재 여부 미조사 |
| U7 | 7D "97.7%"의 순환성·비교대상 오정합 (§A8) | 설계 `:499`와 구현 `bench.py:613` 불일치가 문서로 계약 변경됐으나, **생 MC 추정자 대비 수치는 아직 없다** |
| U8 | EPAY 독립 장부 | payload에 직렬화되지 않음. E10/E12가 잔여 원인 #1로 지목했으나 계량 불가 |
| U9 | ARTIS 20.2% 재현 | 런디렉토리·커밋·에포크·UV 대역 정의 없음. 캠페인 전체가 이 한 숫자 위에 서 있다 |
| U10 | TF32 R_bf 레인(기본 ON, 유효 ~3자리)이 4자릿수 범위 J에 미치는 오차 | 등재만 되고 계량된 적 없음 **[D]** |

---

## 7. 과업 B — 변경 계보 요약표 (07-07 이후)

> **계보 조사의 1차 발견: 마지막 커밋이 `47bfa20` (2026-07-18)이다.** 이후 3주간의 전 변경 — Wave 1/2/3, Wave 3.2, ARTIS-parity 게이트군, Stage 3.1, E1–E13 — 은 **전부 미커밋 작업트리**다(`git diff --stat HEAD -- src/` = 14,662 삽입 / 689 삭제, 16 파일). 이분탐색·롤백·단일변수 증명의 기반이 git이 아니라 문서 mtime과 런디렉토리에 있다. **[F]**

| 변경 | 게이트 | 기본 | 시점 | UV 계량 | 판정 |
|---|---|---|---|---|---|
| k-packet 충돌채널 | `LUMINA_KPACKET` | OFF(생산 1) | 07-06 `ff58168` | **있음: UV 54.0 → 42.9%** | **완화(대), 그러나 blue −0.278→−0.397 회귀** |
| Rydberg 절단 | `MA_COLLISION_LIMIT_EV` | OFF | 07-06 | 있음: 42.9 → 43.8 | 무관(null) |
| EPAY 재형상 | `CMF_EPAY`(생산 2) | OFF | 07-05/06 | **전 있음(53.5%)·후 없음** | **미결 — E10/E12가 잔여원인 #1 지목, 장부 부재** |
| co-evolve 재배선 | `MC_COEVOLVE`(+CONSUME/INJECT) | OFF(생산 1/1/2) | 설계 07-07, 커밋 07-14 | **없음** | **미결 — OFF-중립성 UNTESTED(P0)** |
| **IUP-JBLUE** | `LUMINA_IUP_JBLUE` | OFF, **parity가 강제 ON** | 07-14 `192a2c3` | 도입 시 없음 / 07-30 parity49 **byte-identical** | **무관(효과 0) + ARTIS 서사 오류**: ARTIS는 `DETAILED_LINE_ESTIMATORS_ON=false` **[A]** |
| KPKT-FBUP | 없음(항상 ON) | — | 07-14 `192a2c3` | 없음 | 완화(미소): 해방된 채널이 kpkt 트래픽의 0.022% |
| BF_NLTE_POPS / FB_COOL_KT / KPKT_FB_MULTI / GPH_ALLLEVEL | 각 게이트 | 전부 OFF | 07-14 | **전부 없음** | 미결 |
| GPH_ALLLEVEL_NLTE | 〃 | OFF | 07-15 `0476c83` | narrow-corr 0.474→0.372 | **악화(스펙트럼)** |
| ALPHA-SPINGATE | 〃 | OFF(생산 1) | 07-15 `d57bc98` | 없음(이온화 전용) | 미결 |
| k-packet 재주입 수리 | `KPEMISS_REPAIR` 계열 | 전부 OFF | 설계 07-20 | **전 있음·후 미기록** | 미결 |
| super-level K=100 | `SUPER_CUTOFF`(생산 100) | 0 | 게이트 07-03, 값은 parity 시대 | **없음** — 광구 차단 78.4% 지배만 | **미결(위험 clamp 등재)** |
| **MA_LINE_DESTRUCT (및 그 철회)** | `MA_LINE_DESTRUCT` | OFF(생산 1 → ARTIS 레인 OFF) | 도입 parity 시대, 유죄 07-31 | **있음, 창 내 최상**: parity57에서 formal L −21.90%, 그 감소의 **63.15%를 2192–3235 Å가 운반** | **도입=악화(2p−p² 이중계수, ARTIS 부재 채널)** / **철회=완화** |
| JBAR_DAMP_UNIFY (N3) | `JBAR_DAMP_UNIFY` | OFF | 07-30 | 있음: formal 청 +4.6~8.9% / 적NIR −22% | 완화(소) |
| REC_SPINGATE (Y4) | `REC_SPINGATE` | **OFF (UV 때문에)** | 07-30 | 있음: FUV 1000–2000 Å **−10.0%** | **악화 — 기각** |
| Wave 1 (bf stim-recomb/neutral/spingate/multi-edge) | `FIX_BF_*` | 전부 OFF | 07-31 | **없음 (GPU 실행 금지 배치)** | **미결** |
| Wave 2 (continuum CDF, MA unclamp, no-line-therm) | `FIX_BF_CONTINUUM_EVENT` 외 | 전부 OFF | 07-31 | **없음 (동)** | **미결** |
| Wave 3 (element-wide SE) | `NLTE_ELEMENT_WIDE` | OFF | 07-31 | **없음 (동)**; s8 acceptance 후일 무효화 | **미결** |
| Wave 3.2 M_V 경계질량 | (EW 내부, 자체 게이트 없음) | — | 08-01 | **없음 (신규 런 0)** | 미결 |
| TF32 rate 레인 | 정밀도 하드코딩 / `NLTE_RATES_GEMM` | **레인 기본 ON** | 발견 08-01 | **없음 (오차상한 등재만)** | 미결(U10) |
| coarse-bin 1000빈 투영 | 게이트 없음 (`NLTE_N_FREQ_BINS=1000`) | 항상 | 구조적 | **있음(오프라인)**: 결맞음 파괴 95.1164%, χ_coh/χ_tot 97.77%, 이득 5247× | **도입(구조적) — C1의 필요조건** |

**계보의 총평 (본 감사 판정):**

> **2026-07-06 `ff58168` 이후 UV 지표는 42.9%에서 움직이지 않았다.** 그 사이 들어온 변경은 예외 없이 다음 넷 중 하나다:
> (i) **UV 대역 계량이 아예 없다** (co-evolve, Wave 1/2/3, Wave 3.2, super-level K=100, TF32, KPKT-FBUP — 즉 대다수),
> (ii) **계량했더니 효과 0** (IUP-JBLUE, parity49 byte-identical),
> (iii) **계량했더니 반대 방향** (REC_SPINGATE FUV −10%, GPH_ALLLEVEL_NLTE corr 0.474→0.372),
> (iv) **실물 UV를 재분배한 게 아니라 가짜 UV를 제거했다** (MA_LINE_DESTRUCT 철회: −21.9% 총광도, 그 63%가 근자외).
>
> **캠페인이 1년 가까이 수리에 실패한 구조적 이유는 물리 가설의 빈곤이 아니라, 변경의 대부분이 UV 잣대에 한 번도 닿지 않은 채 채택/보류되었다는 계측 부채다.** GPU 실행 금지 배치(Wave 1/2/3, Wave 3.2)만으로 게이트 12개 이상이 UV 계량 없이 트리에 들어왔다. 게이트 센서스가 500개 중 OFF-중립성 증거를 **7개**만 찾은 것 **[D]** 은 같은 부채의 다른 표현이다.

---

## 8. 권고 (실행 순서)

1. **T1(§5.1)을 먼저.** 오프라인, 기존 payload만 필요, 신규 런 0. `R=균등` vs `R=MC측정`의 차이가 E10/E12/E13 실패의 원인을 두 갈래로 분리한다. 이것이 본 감사가 제안하는 **유일한 최우선 항목**이다.
2. **T2b(§5.2) 병행.** `lumina_c1_bins.csv`만 읽으면 되고, §4.4의 12.2M건 +3 dex 포화의 방향을 확정한다 (U1 해소).
3. **T2a(§5.2)** — `lumina_events_lines.bin`으로 이온분해 형광 지도. E13의 UNRESOLVED(이온 태그 부재)를 기존 자산으로 우회하는 유일한 길.
4. **T3(§5.3)** — C3의 승격/강등. 산술만.
5. **판정런은 1회만**, 그리고 위 넷의 결과가 하나의 인자를 지목한 뒤에. 후보는 `LUMINA_IUP_BINFIELD=1` 단일 토글(T2c).
6. **계측 부채 상환**: 위 (i) 범주(UV 계량 없이 들어온 변경 12건 이상)에 대해, 채택 전 UV 밴드표를 사전등록 항목으로 강제할 것. 현 상태로는 어느 변경이 UV를 움직였는지 사후에도 알 수 없다.

---

## 부록 A — 본 감사가 직접 실측한 값 (재현 명령 포함)

전부 `RUN=/gpfs/kjhan/lumina_runner2/scratch/fluormat_capture_188828` 기준.

| 값 | 출처 |
|---|---|
| `line` 방출 전 파장 합 2726; bin 0–7(900–2896.5 Å) 2511.0 = **92.1%**, bin 0–8(≤3352.1 Å) 2556.6 = **93.8%**; `fb/bte/macap/bf_reemit/linetherm` = 0 | `$RUN/lumina_census_emission.csv` |
| MA 종착: rad_deexc 485,105,515 / col_deexc **4,351 (8.967e-6)** / rad_recomb 118,732 | `$RUN/lumina_census_ma_fate.csv` |
| kpkt 배출 s0: collexc 479,064 / ff 693 / fb 1,445 (collexc **99.56%**) | `$RUN/lumina_census_kpkt_exit.csv` |
| T_rad = 10470.093240, 50셸 uniq=1; T_e 21227.6→8769.8; W 0.2979→0.0108 | `$RUN/lumina_plasma_state.csv` |
| Fe III `p_iup` s3 = **0.8859**, `p_idn` = 0.0966, n=83; UV-strong 그룹 비 s0 2.46 / s3 2.37 / s16 **0.05** | `$RUN/stdout.log:35829-35845` |
| JBLUE-ANCHOR2 it10 thin-out: n=70,860,746, <log10>=+0.434, clamp+3 = **12,199,822** | `$RUN/stdout.log:35892` |
| IUP-JBLUE it10: J_blue 사용 110,666,997 / fallback 18,539,603 (**85.7%**) | `$RUN/stdout.log:35890` |
| MA-LINE-DESTRUCT it10: 555,876 / 510,900,313 = **0.0011** | `$RUN/stdout.log:35888` |
| FLUOR-MATRIX it10: events 509,203,774, nnz 473,045, kpacket 10,440,714 (2.05%), **Eemit/Eabs−1 = 3.0443e-08** | `$RUN/stdout.log:35886` |
| TAU-DIAG: UV(1700–3000) τ>1 = 1093/343613 (Στ=227,895.6); blue 5/282011 (22.6); opt 0/325359 (1.4) | `$RUN/stdout.log:33441` |

## 부록 B — 본 감사가 직접 확인한 소스/ARTIS 대조 좌표

| 좌표 | 내용 |
|---|---|
| `src/lumina_cmfgen.c:1073` | `cs->chi_es[idx] = chi_e + (chi_ln - chi_ln_th)` — **C1의 정확한 위치** |
| `src/lumina_cmfgen.c:1048` | `chi_ln_th = line_eps * chi_ln` |
| `src/lumina_cmfgen.c:1534,1592,1708,1834,2209,2475` | `r = chi_es/chi_tot` 결맞음 산란분 소비처 |
| `src/lumina_plasma.c:4483-4502` | 내부상향 계수 — 희석-Boltzmann `n_u/n_l` @ T_rad, maser clamp |
| `src/lumina_plasma.c:4562-4579` | Lucy 에너지가중 (방출 `hν` / 내부 `ε_low` 중성바닥) |
| `src/lumina_plasma.c:5563-5760` | `diag_macro_branch` — `p_iup` 산출부(준위 균등가중) |
| `src/lumina_plasma.c:3332-3337` | `g_ctp_jbar_min_ma` 기본 10 vs `LUMINA_JBAR_MIN=3` — 선언되지 않은 임계 분열 |
| `src/lumina_atomic.c:377-382` | `TRAD_COLOR_FIX` — 적재 시 T_rad 평탄화 |
| `src/lumina_cuda.cu:10447` | `solve_radiation_field` **유일 호출부**(고전 MC 루프; pure-CMFGEN 미도달) |
| `src/lumina_cuda.cu:6102-6132` | 형광행렬 **단일 호출부** (LINE 분기 전용) |
| `src/lumina_cuda.cu:4162-4218` | `d_fluor_matrix_record` — 빈 인덱스·누적 |
| `src/lumina_cuda.cu:4486-4545` | 이벤트 채널 매핑 + 방출 센서스 팬아웃 |
| `src/lumina_cuda.cu:3889-3900` | `jbar_line`/`jblue_line` — **동일 지점·동일 피가산식** |
| `artis-ref/macroatom.cc:585-586` | ARTIS 상향률: **실제 준위인구** `nnlevel_upper/nnlevel_lower` |
| `artis-ref/macroatom.cc:588-596` | `DETAILED_LINE_ESTIMATORS_ON` 분기 / `radfield(nu_trans)` |
| `artis-ref/artisoptions.h:74` | `constexpr bool DETAILED_LINE_ESTIMATORS_ON = false;` (classic `:68` 동일) |
| `artis-ref/macroatom.cc:101-103,132` | 방출/내부하향/내부상향 가중 |
| `artis-ref/input.cc:403` | `epsilon = (energyoffset_ev + levelenergy_ev)*EV` — 중성바닥 기준 |
