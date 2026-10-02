# 사전등록 — DET-OBS-WRITER: 최종 pass 원천 소비를 보존하는 관측자계 작성기 (2026-10-02, 기안)

## §0. 설계 판단

**단계 1은 작성기 구현·CPU fixture 검증·기존 출력 무변경 증명, 단계 2는 E′·R′ GPU 실행과 새 관측량 판독으로 분리한다. 단계 2는 단계 1 통과 뒤 user의 명시적 승인을 받아 착수한다.** 단계 2의 입력·관측량·분기·기대·주장 상한은 이 문서에서 먼저 고정한다.

입력 경로는 **런 안에서 최종 pass의 소비 상태가 살아 있는 동안 새 작성기를 호출하는 방식**으로 정한다. 역사적 E/R의 `work/`에는 완전한 최종 소비 배열 덤프가 없으므로, 공개 계수나 최종 인구로 누락된 배열을 추정하여 역사적 스펙트럼을 재구성하지 않는다. 단계 2에는 E 설정과 R 설정을 각각 새 바이너리로 실행하는 **GPU 런 두 번**이 필요하다.

새 작성기는 기존 ④를 대체하지 않는다. 기존 출력과 별도로 새 스펙트럼을 쓰고, 동일한 고정 상태에서 기존 작성기도 별도 파일로 호출하여 작성기 효과를 비교한다. 기존 ④의 최종 호출은 인구 갱신 뒤에 있으므로, 그 파일과 최종 pass 직후의 새 파일을 직접 빼서 순수한 작성기 효과라고 부르지 않는다.

기안자는 astra이며 개정25-r 및 이번 직접 발주에 따른 read-only 기안만 수행한다. 구현·직접시험·빌드·스테이징·실행·판독·판정은 운전자(25-w), 코드 감사는 `gpt-6-sol`(25-v) 담당이다. 기안자는 파일을 수정하거나 시험·계산·프리플라이트를 실행하지 않았다.

### user 결정과 별도 결정 필요 항목

확정된 결정은 다음과 같다.

- 작성기 계약 기안과 외곽 W2 오프라인 검토인 diag4를 병행한다.
- NRB는 연구 후보로 보존하고 생산 승격은 보류한다.
- 반복 연장은 보류한다.
- 기존 DET-NRB-E의 R2 판정과 기존 ④·원자료·저작 이력을 보존한다.

별도 user 결정이 필요한 항목은 다음과 같다.

- **단계 1 통과 자료, CPU 행진 비용 추정, 메모리 예상량을 제시받은 뒤 단계 2의 GPU 두 런을 승인할지 결정한다.**
- 실패·시간 초과 뒤 재제출, 추가 런, 반복 연장, 자원 상한 확대.
- NRB 생산 승격, 새 작성기의 기본값 승격, 물리 계약·관측량·판정 문턱 변경.

단계 1 내부의 구현·수리·직접시험·코드 감사 사이에 별도 user 승인 단계를 만들지 않는다. diag4 결과를 이 계약의 숨은 착수 조건이나 판정 문턱으로 추가하지 않는다.

### 경로와 기준선

```text
P  = /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
T0 = /gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source
B0 = /gpfs/kjhan/lumina/cmfgen_map/w5_20260930/cuda_build1
D  = /gpfs/kjhan/lumina/cmfgen_map/diag3_20261002
E  = /gpfs/kjhan/lumina/cmfgen_map/nteta_20261001/run1
R  = /gpfs/kjhan/lumina/cmfgen_map/nrb2_20261002/run1

N  = /gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002
T1 = /gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/source
B1 = /gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/cuda_build1
EP = /gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/Eprime/run1
RP = /gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/Rprime/run1
```

N以下は新設予定の作業先であり、既存物を上書きしない。  
N 아래 경로는 **신설 예정**이다. 스테이징 시 충돌하면 덮어쓰지 않고 경로만 집행 기록과 기계 선언에 일관되게 갱신한다.

T0와 P의 `src/`를 같은 소스 트리로 간주하지 않는다. 생산 변경은 **T0의 검증된 소스 사본 T1**에 적용하며 T0는 보존한다. B1은 변경 소스·헤더를 반영하여 새로 빌드한다. B0의 바이너리 해시를 새 바이너리에 요구하지 않는다.

선행 정본은 다음과 같다.

- D의 `DIAGNOSIS.md` §B.3·§D.
- D의 `FABLE_AUDIT_r4.md`, `DRIVER_DESIGN_AUDIT_r4.md`, `LOOP_CLOSED.md`.
- `docs/PREREG_DET_NRB_E_2026-10-02.md`.
- `docs/VERDICT_DET_NRB_E_2026-10-02.md`.
- W5 빌드 기록과 R의 스테이징·실행 기록.

저자가 직접 읽고 확인한 진단서 SHA-256은 다음과 같다.

```text
20646bd10eaaf1de81d97900e29569ce524078c6aa590cdbcedf32e2fe2bdf3f
```

B0·E·R의 기존 바이너리 신원은 선행 기록상 다음이다. 전체 소스·바이너리·입력 실물의 전건 재검증은 운전자의 G-ID에서 수행한다.

```text
7800484fb309addabfa5b2f6b879f3ab09b814db83d3c5db2f39debff3f1b390
```

## §1. 물리 계약 하나

**질문:** 동일한 최종 pass의 물질·복사장·경계를 고정했을 때, 조립기가 사용한 방출·산란 항을 누락하거나 중복하지 않는 관측자계 출력 경로를 만들 수 있는가? 그 작성기를 E′·R′에 동일하게 적용하면 등록된 대역 관측량은 어떻게 응답하는가?

작성기는 물질·복사장에 되먹임하지 않는다. NRB 율식, 인구·온도·불투명도 생산자, 반복 수와 솔버 허용오차를 변경하지 않는다.

### 1.1 원천별 소비 위치

`J_sc`는 **기존 산란 항이 실제 소비하는 해당 빈의 J**다. 선 원장의 continuum-only `J_c`로 대체하지 않는다.

| 성분 | 보존할 원천·계수 | 새 작성기의 단일 소비 위치 |
|---|---|---|
| 매핑 선 | 조립기가 실제 사용한 `S_l^used`, 매핑 계수 `el=1` | 해당 공명면의 선 점프 |
| 미매핑 선의 열 몫 | `ε_l Bν(T_e)` | 같은 선 점프의 열 항 |
| 미매핑 선의 산란 몫 | `(1−ε_l)J_sc` | 같은 선 점프의 산란 항 |
| 순수 전자 산란 | `χ_e J_sc` | 연속체 구간 수송 |
| bf·ff 연속체 | `cmf_cont_src_numerator`의 선 항을 제외한 실제 분자 | 연속체 구간 수송 |
| 침적 방출 | 해당 pass가 실제 추가한 `η_dep` | 연속체 방출 분자에 한 번 |
| 내경계 | 선택한 관측자계 레인의 고정 입사 강도 | 내경계 조건 |

양의 광학깊이에서 미매핑 선 원천은

\[
S_{u,l}=\epsilon_l B_\nu(T_e)+(1-\epsilon_l)J_{{\rm sc},b(l)}
\]

이며, 선 점프는

\[
I^+=I^-e^{-\tau_l}+D^3S_{u,l}(1-e^{-\tau_l})
\]

로 한다. **τ에 ε를 다시 곱하지 않는다.**

관측자계 연속체에는 순수 전자 산란만 남긴다. `cs->chi_es`에 들어 있는 미매핑 선 산란 \(\sum(1-\epsilon_l)w_l\)을 그대로 읽지 않는다. 선을 공명면에서 소비한 뒤 빈 기반 선 흡수·산란으로 다시 소비하지 않는다.

양의 τ 조립 범위에서 다음 원천 항을 대조한다.

\[
\eta_{\rm line,b}^{\rm frozen}
=
\sum_{\rm mapped}w_lS_l^{\rm used}
+\sum_{\rm unmapped}\epsilon_lw_lB_l
+\sum_{\rm unmapped}(1-\epsilon_l)w_lJ_{{\rm sc},b}.
\]

이 대조는 **원천 항의 보존**이다. 팽창 불투명도 빈 수송과 개별 Sobolev 점프의 전체 스펙트럼 동일성을 요구하지 않는다.

### 1.2 η_dep의 위치 확정

아래 행 번호는 T0 기준이다.

- `lumina_cmfgen.c:119–123`: `cmf_cont_src_numerator`의 ON 분자는 `eta_bf + chi_ff*B + eta_ln`이다. **η_dep는 포함하지 않는다.**
- `:2724–2726`: 위 분자를 `chi_t`로 나누어 `S_fixed`를 만든다.
- `:2731–2738`: W5 연속체 분자는 `eta_ln=0`으로 따로 저장한다. 여기도 η_dep는 없다.
- `:2775–2779`: 해당 조건에서 `kappa_dep*B/chi_t`를 `S_fixed`에 추가한다.
- `:2453` 부근의 EPAY 기본값은 0이며, E/R exports에는 `LUMINA_CMF_EPAY`가 없다. 이번 등록은 이 구성을 유지한다.

따라서 이번 레인의 새 작성기가 사용하는 비산란 연속체 분자는

\[
\eta_{\rm cont}^{\rm used}
=
\underbrace{\texttt{cmf\_cont\_src\_numerator}
(1,\chi_{\rm bf},\chi_{\rm ff},B,\eta_{\rm bf},0)}
_{\eta_{\rm bf/ff}^{\rm used}}
+
\underbrace{\eta_{\rm dep}^{\rm used}}_{\text{실제 추가 지점에서 확보}}
\]

로 확정한다.

`S_fixed*chi_tot`를 쓴 뒤 η_dep를 다시 더하지 않는다. `bf_get_eta` 내부의 ff 포함 규약을 임의로 고쳐 분자를 재조립하지 않는다. **기존 조립기가 사용한 bf/ff 분자 자체**를 소비한다.

기본 작성기는 `:3936`에서 `chi_abs*B+chi_es*J`를 사용하므로 Milne 분자뿐 아니라 η_dep도 직접 소비하지 않는다. 또한 `:3879–3880`의 “LINE_EPS off이면 chi_es가 순수 전자 산란”이라는 주석 전제를 E/R의 `LUMINA_CMF_FINE_UNMAPPED_EPS=1`이 만족하지 않는다. 두 사실은 새 계약의 근거이며, 그 효과 크기는 단계 2 전까지 미측정이다.

### 1.3 폴백·부호·영 극한

- 매핑 여부는 생산자의 권위 마스크로 정한다. `S_l>0` 여부로 매핑을 재분류하지 않는다.
- 기존 조립기가 명시적으로 사용한 B 폴백은 출처를 표시하여 보존할 수 있다. 이를 NLTE 인구 원천으로 표시하지 않는다.
- 결손·NaN·무한값·세대 불일치를 새 B 폴백으로 숨기지 않는다.
- coarse 조립기의 `τ>10⁻¹²` 선택과 signed 선 생산자의 범위를 분리하여 기록한다. 이 선택 밖의 선을 조용히 버리고 전체 생산기와 일치한다고 주장하지 않는다.
- signed·영 τ는 실제 생산자의 독립 방출·흡수 계수를 받아 안정적인 formal 식으로 처리한다. `η/χ`만 저장하여 χ=0에서 방출을 잃지 않는다.
- 생산자가 적용한 `SRCE_CHK`의 raw/effective 계수와 적용 여부를 보존한다. 새 작성기가 raw τ를 다시 적용하거나 별도 clipping을 하지 않는다.
- 생산 계수 연결을 구현·검증하지 못한 범위가 남으면 `UNSUPPORTED_SOURCE_DOMAIN`으로 표시한다. 부분 출력은 보존하되 전체 스펙트럼과 그 대역 관측량의 적격 판정에는 사용하지 않는다.

일정 계수 구간의 검증 식은

\[
I_{\rm out}
=I_{\rm in}e^{-\tau}+(\eta\ell)\phi(\tau),\qquad
\phi(\tau)=\frac{-\operatorname{expm1}(-\tau)}{\tau},\quad
\phi(0)=1
\]

이다. 큰 음의 τ에서 수치 표현 범위를 넘는 경우 성공값으로 잘라내지 않고 명시적으로 실패한다.

## §2. 설계

### 2.1 단계 분리와 입력 수명

| 단계 | 목표 산출 | 종료 조건 |
|---|---|---|
| 단계 1 | 실제 작성기에 연결된 fixture 결과, OFF 무변경 증거, 코드 감사 처분, 비용 추정 | §4의 단계 1 조건 충족. GPU 실행 권한은 발생하지 않음 |
| 단계 2 | E′·R′의 새 스펙트럼·동일 상태 기존 작성기 출력·등록 판독 | 단계 1 통과와 user 승인 후 실행. §5 분기와 §6 주장 상한으로 판정 |

새 입력 경로는 다음 순서로 구현한다.

1. 마지막 pass의 조립 과정에서 실제 사용한 선 원천·매핑·ε와 순수 전자 산란·bf/ff 분자·η_dep의 권위와 세대를 확보한다.
2. 그 pass의 복사장 계산 후 같은 물질 세대에 대응하는 `J_sc`를 연결한다.
3. signed·영 τ의 계수가 fine 생산자 내부의 임시 배열에만 존재하면, **해당 배열이 유효한 생산 지점에서 필요한 계수만 보존**한다. 다음 인구로 다시 계산하지 않는다.
4. 최종 radiation commit이 성공한 뒤, **다음 물질 갱신 전**에 고정 view를 검증하고 두 작성기를 호출한다.
5. 호출·I/O 성공 또는 실패와 무관하게 임시 상태를 정리하고 전역 상태를 복구한다. 새 작성기 실패는 명시적으로 전파하며 성공 manifest를 쓰지 않는다.

현재 구성의 기대 신원은 최종 call 7의 radiation/material/assembly 세대다. `physics_DET_iter0005`의 다음 인구 gen8을 이 view에 섞지 않는다. manifest에는 논리 반복, call, radiation·material·assembly 세대를 각각 기록한다. 서로 다른 종류의 카운터를 숫자가 같다는 이유만으로 같은 권위로 취급하지 않는다.

역사적 E/R의 `nlte_levels_final.csv`와 공개 `η_bb/χ_bb`는 새 작성기의 결손 입력 대용이 아니다.

### 2.2 기존 출력 보존과 동일 상태 짝 비교

새 키는 **`LUMINA_DET_OBS_WRITER` 하나**다.

- 부재 또는 문자열 `"0"`: OFF.
- 문자열 `"1"`: ON.
- 그 밖의 값: 명시적 설정 오류.
- 기본값은 OFF이며 `LUMINA_ENV_STRICT=1`의 등록 목록에도 추가한다.

ON일 때 각 새 런의 `work/` 아래에 다음 파일을 신설한다.

| 파일 | 의미 |
|---|---|
| `lumina_spectrum_det_obs.csv` | 새 작성기의 관측자계 출력 |
| `lumina_spectrum_det_obs_legacy.csv` | **같은 고정 view**를 받은 기존 작성기의 출력 |
| `det_obs_manifest.json` | 입력 신원·세대·지원 범위·출력 해시·호출 결과 |

기존 `lumina_spectrum.csv`와 `lumina_spectrum_comoving.csv`의 이름·기존 호출 위치·출력 형식은 유지한다. 새 파일이 이미 있으면 덮어쓰지 않고 실패한다.

새 행진은 명시적 context를 사용한다. 기존 작성기를 짝 비교용으로 호출하는 어댑터는 그 호출이 만지는 모든 `g_sob_*`와 관련 전역 상태를 저장·복구한다. `g_sob_sl_ptr`만 NULL로 되돌리는 방식으로 끝내지 않는다. 정상 종료뿐 아니라 할당 실패·파일 열기 실패·쓰기 실패에서도 복구해야 한다.

동일 view의 배열과 경계 신원은 두 호출 전후에 대조한다. 새 작성기는 J·인구·온도·불투명도·원장·난수 상태를 갱신하지 않는다.

### 2.3 관측자계 수송과 적분 규약

- 관측 주파수 격자와 출력 파장 범위는 기존 coarse 관측자계 출력의 격자를 유지한다.
- 원판 광선 수는 기존 기본값 256, 연속체 구간의 최대 속도 간격은 기존 기본값 30 km/s를 사용한다. 결과를 보고 격자를 조정하지 않는다.
- 선은 실제 공명 위치 순서로 소비한다. 선 앞·뒤 연속체 구간을 나누며, 구간 경계의 공명은 반열린 구간 소유 규칙으로 한 번만 배정한다.
- 흡수·방출·강도 변환은 동일한 Doppler 인자를 사용한다. 정적 fixture 외에 계수 변환과 경계 변환의 일관성도 직접시험한다.
- 내경계는 기존 관측자계 레인의 `INNER_BB_SCALE`, `T_inner`, 기하와 \(D^3B_\nu\) 처방을 고정한다. 기존 coarse/fine 내부 경계와의 완전 동치를 주장하지 않는다.
- 새 CSV의 값은 double 왕복 보존 정밀도로 쓴다. 기존 CSV의 출력 정밀도는 변경하지 않는다.

### 2.4 E′·R′ 설정

E′는 E의 봉인 초기 입력, R′는 R의 봉인 초기 입력에서 시작한다. 최종 인구 warm start를 사용하지 않는다. 양쪽 모두 같은 B1 바이너리를 사용한다.

각 선례와의 허용 변경은 새 바이너리 신원, 새 런 경로 이식, `LUMINA_DET_OBS_WRITER="1"`뿐이다. 경로 이식은 선행 계약의 다음 키와 동일한 상대 대상에 한정한다.

```text
LUMINA_MODEL_DIR
LUMINA_CMFGEN_SIGMA_BF
LUMINA_DEPOSITION_FILE
LUMINA_FIXED_TE_PROFILE
LUMINA_IONIZATION_REFERENCE_FILE
LUMINA_TOPION_LEVELS_FILE
LUMINA_PHYSICS_COMPARISON_DIR
```

중복 키를 거부하고 부재·빈 문자열·`"0"`을 구분한다. 임의 문자열 치환으로 다른 설정 차이를 숨기지 않는다.

E′와 R′ 사이에는 선행 E/R의 차이를 유지한다.

| 키 | E′ | R′ |
|---|---|---|
| `LUMINA_BB_NRB` | 부재 | `"1"` |
| `LUMINA_A210_SPRODUCER_CAPTURE` | `"0"` | `"1"` |
| `LUMINA_DET_OBS_WRITER` | `"1"` | `"1"` |

새 작성기는 E′에서 기존 NRB 캡처 키를 켜야만 작동하도록 구현하지 않는다. 필요한 최소 입력 확보는 새 키가 소유한다.

고정 유지 항목은 6회 반복, FIXED_TE, `NT_FULL=0`, `XSHELL_RESPONSE=0`, W5 연속 원천 ON, 미매핑 ε ON, GPU 두 장과 선행 솔버 설정이다. exports 전체 비교가 이 목록을 보완한다.

### 2.5 구성상 도달 가능성 자기 점검

| 점검 대상 | 요구하지 않는 불가능하거나 부당한 조건 | 이번 처리 |
|---|---|---|
| 역사적 입력 | E/R에 없던 최종 소비 덤프 | E′·R′의 런 안 입력 경로 사용 |
| 바이너리 | B1 해시가 B0와 같음 | 기준 소스와 등록 patch·새 빌드 신원 연결 |
| 세대 | call 7 재료가 최종 인구 gen8과 같음 | 갱신 전 view에서 짝 비교 |
| 미매핑 선 | 모든 `line_source_S`가 양수 | 권위 마스크와 열·산란 분할 사용 |
| signed·영 τ | 모든 τ가 양수, 모든 `SRCE_CHK=0` | 실제 유효 계수와 지원 상태 검사 |
| E′ | NRB armed/captured 태그 | NRB 비활성이 정상 |
| INIT | 최종 Sobolev view와 새 출력 | 새 작성기는 최종 pass에서만 호출 |
| XSHELL | XSHELL 상수·출력·G-RATE·G-DIR | 이번 관측량과 무관하므로 요구하지 않음 |
| 기존·새 스펙트럼 | ON 출력이 기존 출력과 동일 | OFF 보존과 ON의 동일 상태 비교를 분리 |
| 에너지 | FIXED_TE에서 강제 전역 에너지 폐합 | 적분 광도 보고만 수행 |
| 물리 응답 | FUV 하강·CMFGEN 접근만 정상 | 모든 부호·혼합 결과를 등록 |
| 실행 완주 | CPU fixture 통과가 GPU 완주 보증 | 실행 게이트를 별도로 평가 |

**자기 점검 결론:** 확인된 구성상 모순은 없다. 특히 “signed 입력이 없을 것”을 성공의 전제로 두지 않는다. 생산 계수 연결이나 비용이 구현 단계에서 충족되지 않으면 단계 1 미통과로 보고하며, 조건을 낮춰 GPU 실행으로 넘기지 않는다.

## §3. 변경집합·산출물·자원

### 변경집합 — DET-OBS-WRITER (표)

아래 T1 파일은 T0에서 복사한 뒤 수정하는 **신설 작업 트리의 파일**이다. 신규 심볼 접두는 `det_obs_`로 통일한다.

| 경로 | 함수·항목 | 변경 범위 |
|---|---|---|
| `/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/source/src/lumina_cmfgen.c` | 기존 `cmfgen_assemble_impl`, `cmfgen_fine_jbar`, `cmfgen_run`, `cmfgen_free`; 신규 `det_obs_capture_*`, `det_obs_write`, `det_obs_march`, `det_obs_legacy_pair`, `det_obs_restore` | 기본 OFF 입력 확보, 실제 계수 연결, 별도 작성기와 마지막 pass 호출, 복구·오류 처리 |
| `/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/source/src/lumina_cmfgen.h` | `CMFGENState`의 `det_obs_` context·수명 선언 | 최소 입력 view·소유 버퍼·세대. 구조체 사용 translation unit 재빌드 |
| `/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/source/src/env_universe.h` | `LUMINA_DET_OBS_WRITER` | 기존 목록 보존, 새 키 등록·COUNT 정합 |
| `/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/source/tests/det_obs_writer_selftest.c` | `det_obs_` fixture·어댑터 | 신설. 실제 생산 함수 호출, 해석해·결함 주입·OFF 및 상태 보존 |
| `scripts/det_obs_readout.py` | `det_obs_` 접두의 `selftest`, `preflight`, `readout`; `REQUIRED_CASES` | 신설. CPU 시험 실행·증거 검사, 신원·대역 적분·분기 |

### 변경집합 끝

`lumina_plasma.c`, `line_net_rate.c`, CUDA 커널, 기존 작성기 계산식, 기존 판독기와 원자자료는 변경 대상이 아니다. 필요한 signed 계수는 기존 생산 경로에서 읽는다. 이 범위 안에서 연결할 수 없다면 변경집합 개정 사유로 보고한다.

기존 작성기 함수의 이름·위치를 참조하는 것은 수정 허가와 다르다. 짝 비교는 어댑터로 호출하며 기존 함수의 계산·출력 형식 변경을 포함하지 않는다.

### 3.1 산출물과 보존

| 위치 | 소비 목적 |
|---|---|
| T1·B1, 기준 소스 manifest·patch·빌드 manifest | 등록 변경과 실제 실행 바이너리 연결 |
| `N/stage1/` | fixture·결함 주입·OFF 바이트 대조·복구·비용 추정 |
| `N/evidence/` | 프리플라이트·코드 감사 처분·user 승인·제출·회계 기록 |
| EP·RP | 봉인 입력 사본·exports·runner·기존 출력·새 출력·footer |
| `N/readout/observables.csv` | 모든 등록 대역 광도·색비·작성기 차이·E′/R′ 응답 |
| `N/readout/result.json` | 게이트·유효성·분기·사유 |
| `N/readout/evidence_manifest.json` | 실제 소비 파일·판독기·의존 파일 신원 |

최종 판정문은 `docs/VERDICT_DET_OBS_WRITER_2026-10-02.md`로 신설한다.

T0·B0·E·R·diag3·기존 ④·기존 R2 판정은 보존한다. 입력 사본은 쓰기를 공유하는 하드링크로 만들지 않는다. 실패 산출과 부분 출력도 삭제하지 않는다. 재판독은 새 출력 디렉터리를 사용한다.

입력 배열 전체를 매 반복 덤프하지 않는다. 선택한 경로는 런 안 호출이며, manifest는 신원을 증명한다. **해시만으로 저장하지 않은 배열의 사후 CPU 재실행이 가능하다고 주장하지 않는다.**

### 3.2 단계 1 자원

CPU fixture·판독·기계 프리플라이트는 grammar-debug에서 실행한다. 로그인 노드에서 계산하지 않는다.

fixture는 단일 CPU·소형 배열을 사용하고, 컴파일은 병렬도 4 이하로 제한한다. 대규모 E/R 수송이나 새 GPU 사전 런으로 확대하지 않는다.

B1은 W5 `cuda_build1`의 실제 빌드 명령·컴파일러·모듈 기록을 따라 만든다. 새 구조체 정의를 사용하는 CPU·CUDA 객체를 모두 재빌드하며 오래된 객체를 혼용하지 않는다.

### 3.3 단계 2 자원과 시간 근거

각 런의 자원은 다음으로 고정한다.

```text
--partition=h100,a100
--nodes=1
--ntasks=1
--gres=gpu:2
--cpus-per-task=16
--mem=64G
--time=30:00:00
--no-requeue
--exclude=syn102
```

H200은 금지한다. 선행 runner의 H100/A100 80 GB급·동종 GPU 두 장 검사를 유지한다. 모듈은 runner 안에서 `module purge` 후 선행 빌드와 일치하게 로드한다.

R의 Slurm 실측 경과는 **11:36:36**이며 모델 footer의 41,788초와 구분한다. 두 런의 선례 기반 계산 시간은 합계 약 23.2시간, GPU 비용은 약 **46.4 GPU·시간에 새 CPU 작성기 실행 중의 GPU 할당 시간이 더해지는 값**이다. 등록 상한은 런당 60 GPU·시간, 합계 **120 GPU·시간**이다.

새 행진 시간은 단계 1에서 다음 방식으로 추정하고 user 승인 자료에 싣는다.

1. 실제 컴파일된 새 작성기와 짝 비교용 기존 작성기를 작은 합성 격자에서 실행한다.
2. 주파수·광선·셸·선 밀도를 달리하여 구간 처리량, 공명 검색·소비량, I/O 시간을 측정한다.
3. E/R의 격자 크기·선 수·광선 수로 작업량을 환산한다. 전체 선 목록을 매 구간 순회하는 구현이면 그 비용도 포함한다.
4. 정상 대조와 고밀도 사례에서 얻은 보수적 추정 중 큰 값을 사용하고, 두 작성기·입력 확보·검사의 합에 2배 여유를 둔 \(T_{\rm add}\)를 제시한다.
5. 제출 가능 조건은 \(2T_R+T_{\rm add}\le30\)시간이다. 2배 \(T_R\)는 A100 실측이 아니라 장치·실행 변동에 대한 계획 여유다.

이 조건을 넘으면 광선 수·대역·물리 설정을 몰래 줄이지 않고 자원 재결정 대상으로 보고한다. 시간 초과 시 재큐잉·자동 연장·자동 재제출하지 않는다.

메모리는 기존 R의 회계 자료와 새 소유 버퍼의 실제 크기로 추정한다. \(N_{\rm line}N_{\rm shell}=129{,}439{,}900\)이라면 double 배열 하나만 약 0.96 GiB이므로 전체 선×셸 사본을 무제한 추가하지 않는다. 유효 기간 안의 읽기 전용 배열은 빌려 쓰고, 임시 생산 계수 중 필요한 것만 보존한다. 64G 안의 여유를 입증하지 못하면 단계 2를 제출하지 않는다.

## §4. 게이트

| 게이트 | 막는 실제 실패 | 기준과 소비 |
|---|---|---|
| **G-ID** | 다른 소스·입력·exports·세대·판독기 혼용 | §4.1. 등록 개입과 동일 상태 비교의 성립 |
| **G-IMPL** | 누락·중복·잘못된 계수·상태 누출·기존 파일 변경 | §4.2. 단계 1 작성기 수용 |
| **G-READOUT** | 결손을 성공 처리, 잘못된 적분·분기, 시험 누락 | §4.3. 새 관측량 판독 |
| **G-RUN** | 미완주·작성기 미호출·지원 불가·불완전 출력 | §4.4. 단계 2 관측량 적격성 |

별도 G-FIRE·G-GENERATION·G-BYTE를 만들지 않고 위 게이트에 합친다.

### 4.1 G-ID

제출 전에는 다음을 확인한다.

- B0의 build source manifest와 T0 실물, T1의 기준 사본·등록 patch·변경집합을 연결한다.
- B1의 바이너리·소스·빌드 명령·모듈을 봉인하고 EP/RP 입력 실물 및 runner 기록에 연결한다.
- E→EP, R→RP의 초기 입력·온도·침적·원자자료는 내용이 같아야 한다. exports 차이는 §2.4와 일치해야 한다.
- 판독기·fixture·어댑터·동결 의존 파일의 해시와 시험 기록을 연결한다.
- 단계 1 통과 기록과 단계 2 user 승인, §3.3의 시간·메모리 추정이 존재해야 한다.

실행 후에는 footer·실제 Slurm 자원·재시작 횟수·실제 출력 manifest로 연결을 완성한다. 실행 전 아직 없는 footer를 요구하지 않는다.

### 4.2 G-IMPL — fixture 세 묶음과 OFF 보존

검사 대상은 실제 생산 작성기·행진·입력 어댑터다. 해석해끼리의 비교나 시험 전용 별도 수송 함수만으로 PASS를 만들지 않는다.

정적 double fixture의 허용오차는 사전에

\[
|I_{\rm got}-I_{\rm exact}|
\le10^{-12}\max(1,|I_{\rm exact}|)
\]

로 고정한다. 이는 해석해 구현 오차의 기준이며 스펙트럼의 물리적 허용오차가 아니다. 원천 계수 합산 대조는 합산 절댓값 스케일과 double 누적 오차를 사용하며 근사 스펙트럼에 맞춰 완화하지 않는다.

| 시험 묶음 | 정상 대조·독립 기대값 | 필수 결함 주입 |
|---|---|---|
| 미매핑 열·산란 소비 | `I_in=1`, `τ=ln2`, `B=2`, `J_sc=5`. ε=0 → 3, ε=1 → 1.5, ε=0.25 → 2.625. 실제 입력 어댑터를 거쳐 선·연속체를 함께 행진 | 산란 누락, 연속체에 선 산란 재가산, τ에 ε 재승산 |
| 독립 방출·부호·영 극한 | `τ=−ln2`, `ηℓ=ln2` → 3; `τ=0`, `ηℓ=0.25` → 1.25. 작은 ±τ의 연속 극한. Milne 분자를 χB와 다르게 지정. 침적만 바꾼 해석해. 생산자의 effective 계수 연결 | Kirchhoff 재대체, 부호 clipping, 영 τ 방출 누락, η_dep 누락·중복 |
| 소비·신원·경계·보존 | 두 구간 경계의 선 소비 횟수 1. 같은 view의 짝 출력. 다른 세대 거부. 정상·오류 경로 전역 상태 복구. 기존 파일 보존 | 경계 중복·누락, stale view, 기존 파일 충돌, 할당·열기·쓰기 실패, `g_sob_faithful` 누출 |

정적 예제의 정답은 실제 출력 숫자로 기록하고, 의도적 결함은 지정 사유로 실패해야 한다.

**OFF 바이트 동일 증명**은 다음 피연산자로 수행한다.

- T0로 빌드한 기준 CPU 어댑터와 T1로 빌드한 CPU 어댑터.
- 동일 compiler·flags·fixture·기본 환경에서 새 키 부재 및 `"0"`.
- 양쪽의 기존 `lumina_spectrum.csv`, `lumina_spectrum_comoving.csv`, 기존 작성기에 전달되는 상태 배열.
- T1의 ON 호출 전후 기존 파일과 상태, 이어서 호출한 기존 작성기의 출력.

비교 파일은 `N/stage1/base/`, `N/stage1/off_unset/`, `N/stage1/off_zero/`, `N/stage1/on_restore/`로 격리하고 정확한 파일 목록·해시·`cmp` 결과를 기록한다.

이 시험은 **동일 CPU fixture에서 기본 경로의 바이트 보존**을 증명한다. 과거 GPU 전체 런을 다시 실행하지 않고 역사적 ④와 새 GPU 런의 바이트 동일성을 증명했다고 쓰지 않는다. 기존 작성기 계산식의 무변경과 OFF 제어 흐름은 patch·코드 감사로 함께 확인한다.

코드 감사의 차단 사항은 수리와 해당 직접시험으로 해소한다. 운전자의 구현 자체시험을 독립 코드 감사로 표시하지 않는다.

### 4.3 G-READOUT

`scripts/nrb_e_readout.py`의 G-ID·G-RUN·NC·`REQUIRED_CASES` 구조를 선례로 사용하되, 기존 NRB 분기와 “바이너리 불변” 조건을 복사하지 않는다. 기존 판독기는 수정하지 않는다.

새 `REQUIRED_CASES`에는 다음 실패·경계 범주를 빠짐없이 명시한다.

- 성한 E′/R′ 및 같은 입력 대 같은 입력의 정상 대조.
- spectrum·manifest·footer 결손, 잘못된 헤더, 중복 파장, NaN·무한값.
- 지원하지 않는 대역, 0 또는 음의 적분값과 로그 미정의.
- exports 중복·미등록 차이, 잘못된 바이너리·입력·view·세대.
- 기존 ④를 동일 상태 짝 파일로 잘못 넣음.
- 작성기 실패·부분 파일·지원 불가 범위·출력 해시 불일치.
- §5 분기의 경계값 ±0.30, 불변, 같은 방향, 반대 방향.
- 필수 시험 누락 및 다른 판독기 해시로 작성된 시험 요약.

시험 요약에 필수 사례가 하나라도 없으면 거부한다. 예외 종료를 유효한 물리 분기로 바꾸지 않는다.

### 4.4 G-RUN과 유효성

단계 2의 적격성은 다음을 모두 요구한다.

- Slurm COMPLETED, 재시작 0, `model_rc=0`, 등록 runner의 성공 footer.
- 선행 생산 계산의 성공 조건 유지.
- 최종 pass의 새 작성기와 동일 상태 기존 작성기 호출 성공.
- 같은 view 신원, 지원 범위 확인, 상태 복구 성공, 완전한 별도 출력과 manifest.
- E′는 NRB 비활성, R′는 선행 계약과 같은 NRB 활성 증인.
- G-ID·G-IMPL·G-READOUT 통과.

유효성은 `VALID` 또는 `X`로 구분한다. `X`에는 구체적 사유를 붙인다. 완료된 일부 관측값은 보고할 수 있으나 X를 FUV 상승·하강·불변 결과로 대신 판정하지 않는다.

**단계 1 통과는 G-ID의 단계 1 부분, G-IMPL, G-READOUT, 코드 감사 차단 해소를 뜻한다. 단계 2 결과나 승인을 뜻하지 않는다.**

## §5. 기대치·판정량·분기

### 5.1 새 관측량과 권위

새 관측량은 기존 ④와 구별하여 **⑤**로 부른다.

각 상태 \(X\in\{E',R'\}\), 작성기 \(w\in\{N,L\}\)에 대해 N은 새 작성기, L은 동일 view의 기존 작성기다.

| 대역 | 범위 Å | 용도 |
|---|---:|---|
| FUV0 | 912–1200 | 보고 |
| FUVa | 1200–1500 | FUV 분기 |
| FUVb | 1500–2000 | FUV 분기 |
| NUV | 2500–4000 | 독립 보고 |
| OPT | 4000–6000 | 색비 분모 |
| 출력 전 범위 | 실제 파일 양 끝 | 총 출력 적분 보고 |

파장을 오름차순 정렬하고 중복을 거부한다. 대역 경계는 인접 점의 선형 보간으로 삽입하고 사다리꼴 적분한다. 대역을 파일이 덮지 않으면 외삽하지 않는다.

\[
L_{X,w}(a,b)=\int_a^bL_{\lambda,X,w}\,d\lambda ,
\qquad
C_{X,w,j}=\frac{L_{X,w,j}}{L_{X,w,\rm OPT}}.
\]

보고할 값은 다음과 같다.

1. 각 런·작성기의 대역 광도와 색비.
2. 동일 상태 작성기 효과:
   \[
   W_{X,j}=\log_{10}\frac{C_{X,N,j}}{C_{X,L,j}}.
   \]
3. 같은 새 작성기로 측정한 E′→R′ 응답:
   \[
   \Delta_{N,j}=\log_{10}\frac{C_{R',N,j}}{C_{E',N,j}}.
   \]
4. 동일 view 기존 작성기의 응답 \(\Delta_{L,j}\) 및
   \[
   \Delta_{N,j}-\Delta_{L,j}=W_{R',j}-W_{E',j}.
   \]

마지막 식은 판독의 대수 확인이며 독립 물리 증거가 아니다. 색비와 함께 분자·분모 광도 변화를 반드시 보고한다.

### 5.2 문턱과 분기

주 분기는 **새 작성기의 FUVa·FUVb 응답 \(\Delta_{N,a},\Delta_{N,b}\)**에만 적용한다.

\[
{\rm cat}(x)=
\begin{cases}
{\rm down},&x\le-0.30\\
{\rm flat},&-0.30<x<0.30\\
{\rm up},&x\ge0.30.
\end{cases}
\]

0.30 dex는 이번 유한 응답의 보고 범주를 나누는 사전 문턱이다. 작성기 정확도·CMFGEN 일치·물리적 무변화의 문턱이 아니다.

| 분기 | 배타적 조건 | 해석 |
|---|---|---|
| **X** | 필수 게이트·관측량 부적격 | 새 FUV 판정 불가 |
| **O-MIX** | 한 대역 down, 다른 대역 up | 대역별 반대 응답 |
| **O-DOWN** | down이 있고 up은 없음 | 하나 이상 등록 문턱 이상의 FUV 색비 하강 |
| **O-UP** | up이 있고 down은 없음 | 하나 이상 등록 문턱 이상의 FUV 색비 상승 |
| **O-FLAT** | 두 대역 모두 flat | 등록 문턱 이상의 FUV 색비 변화 없음 |

X를 먼저 분리한다. 유효한 반대 방향은 실패가 아니라 O-MIX다.

NUV·FUV0·전 범위 광도와 \(W_{E'},W_{R'}\)는 수치·부호·같은 문턱 범주를 모두 보고하되 주 분기를 바꾸지 않는다. NUV가 불리하다는 이유로 대역을 삭제하거나 FUV 판정을 바꾸지 않는다.

### 5.3 사전 기대

- 단계 1에서는 §4.2의 해석해·누락 및 중복 검출·OFF 보존을 기대한다.
- 기존 작성기가 직접 읽지 않던 매핑 선 원천·Milne 분자·η_dep의 소비와 선 산란 중복 제거로 ON 출력이 달라질 가능성이 있다.
- 이 변화들의 합에는 상쇄가 가능하므로 **전체 FUV·NUV 변화의 부호나 최소 크기는 합격 조건으로 예측하지 않는다.**
- NRB의 내부 선 원천 하강이 새 FUV 색비 하강으로 반드시 이어진다고 가정하지 않는다. O-DOWN·O-UP·O-FLAT·O-MIX가 모두 가능한 결과다.
- 새 작성기의 차이가 작아도 fixture가 통과하고 실제 입력 연결이 확인되면 작성기 계약 실패가 아니다.

H2의 근사 재현 및 ±0.2 dex, 공개 물질량의 0.3 dex 일치를 정확해·합격·귀속 문턱으로 사용하지 않는다.

## §6. 주장 상한

1. 단계 1 통과는 등록된 소비·해석해·보존 계약의 구현 검증이다. CMFGEN 스펙트럼 재현이나 다준위·재분배·MC 수송 전체의 정확성 증명이 아니다.
2. \(W_X\)는 **동일하게 고정한 물질·복사장·경계에 조건부인 작성기 효과**다. 선·연속체·침적 각각의 독점적 기여율은 측정하지 않는다.
3. \(\Delta_N\)는 E′·R′의 등록된 6회 반복 구성 사이의 유한 응답이다. 수렴값·장기 안정성·NRB 생산 적격성을 뜻하지 않는다.
4. 새 런의 기존 ④와 역사적 E/R의 기존 ④ 비교는 재실행 차이를 살펴보는 보조 자료다. 이를 동일 상태 작성기 효과와 합치지 않는다.
5. 새로운 결과로 기존 `docs/VERDICT_DET_NRB_E_2026-10-02.md`의 통계·R2 판정을 사후 교체하지 않는다.
6. H2 또는 공개 물질량의 근접성은 전체 CMFGEN 차이의 원인을 작성기에 귀속시키는 근거가 아니다.
7. 비양의 τ·영 τ·경계·세대의 지원 불가를 숨긴 스펙트럼으로 전체 출력의 적격성을 주장하지 않는다.
8. 어느 분기도 W2·NT의 인과적 배제, 반복 연장, 새 기본값 또는 NRB 생산 승격을 자동 승인하지 않는다.

## §7. Q-GOAL·Q-LEAN 자기답·집행 기록

### Q-GOAL 자기답

**PASS — 기안자의 자기답이며 독립 감사 판정이 아니다.**

① 최종목적은 Monte-Carlo 수송으로 CMFGEN 수준의 SN Ia 스펙트럼을 재현하는 것이다. 이번 작업은 FUV·NUV 어긋남을 읽는 출력 경로를 정비한다. 어긋남을 몇 dex 줄일지는 아직 예측할 수 없다.

② 단계 1의 산출은 **관측량 정비**다. 단계 2에서 비로소 정비된 관측량으로 작성기 효과와 E′→R′ 유한 응답을 측정한다. 작성기 fixture를 새로운 물리 수리 성과로 세지 않는다.

③ 외곽 W2 오프라인 검토가 대안 표적으로 열려 있고 user 결정에 따라 병행한다. 이 계약이 W2의 종료를 기다리거나 그 검토를 대체하지 않는다.

④ 이 단계를 건너뛰면 실제 원천을 충분히 읽지 못하고 선 산란을 중복 소비할 수 있는 출력으로 후속 물리 수리를 평가하게 된다. 무엇이 내부 상태 효과이고 무엇이 출력 경로 효과인지 분리할 기준을 잃는다.

### Q-LEAN 자기답

**APPROPRIATE — 기안자의 자기답이며 독립 감사 판정이 아니다.**

| 항목 | 실제 소비 지점 |
|---|---|
| 원천·세대 view | §1의 단일 소비와 §5의 동일 상태 짝 비교 |
| 새 스펙트럼 | ⑤의 광도·색비·E′→R′ 응답 |
| 동일 상태 기존 스펙트럼 | \(W_X\). 최종 ④와의 세대 혼동 방지 |
| 최소 manifest·해시 | G-ID·G-RUN의 입력·출력 연결 |
| 소스 범위·폴백·signed 상태 | 전체 스펙트럼 적격 여부와 주장 상한 |
| fixture 세 묶음 | G-IMPL의 실제 누락·중복·상태 누출 검출 |
| OFF 바이트 대조 | 기존 작성기·④ 경로 보존 |
| 판독 NC·필수 목록 | G-READOUT의 결손·적분·분기 오류 검출 |
| 시간·메모리 측정 | user의 단계 2 비용 결정 |
| Slurm·footer | 완주·자원·재시작 여부 확인 |

삭제·통합한 항목은 매 반복 소비 배열 덤프, 새 GPU 사전 시험, H2 적합 문턱, 공개 물질량 문턱, 별도 수렴 캠페인, 원천별 GPU 소거 실험, 독립 G-FIRE·G-GENERATION·G-BYTE다.

단계 1은 하나의 CPU 검증 묶음이며 내부 시험마다 별도 단계 종료를 만들지 않는다. 단계 2는 필요한 상태를 얻는 E′·R′ 두 런으로 제한한다. 동일 상태 짝 출력은 그 런 안의 CPU 호출이므로 추가 GPU 런을 요구하지 않는다.

코드량은 구현 전 미측정이다. 단계 1 기록에 실제 변경 행 수와 시험·판독 코드량을 함께 보고하되 이를 합격 문턱으로 만들지 않는다.

### 집행 기록

| 항목 | 담당 | 현재 상태 |
|---|---|---|
| 계약 기안 | astra | 이 본문 작성, read-only |
| 파일화·기계 프리플라이트 | 운전자 | 미실행 |
| 구현·직접시험·빌드 | 운전자, 25-w | 미실행 |
| 코드 감사 | `gpt-6-sol`, 25-v | 미실행 |
| 단계 1 판정·비용 제시 | 운전자 | 미실행 |
| 단계 2 승인 | user | **미승인** |
| 스테이징·GPU 두 런·판독·판정 | 운전자 | 미실행 |
| 종료감사 | 현행 배분에 따라 별도 기록 | 미실행 |

운전자가 구현자와 판정자를 겸한다는 점을 남긴다. diag3의 종료를 이 작성기의 구현 검증이나 단계 2 승인으로 승계하지 않는다.

### 운전자 프리플라이트·재현

이 본문을 `docs/PREREG_DET_OBS_WRITER_2026-10-02.md`로 **신설**한 뒤, P를 작업 디렉터리로 하여 grammar-debug에서 실행한다.

```bash
timeout 5m python3 -B scripts/check_prereg_preflight.py \
  docs/PREREG_DET_OBS_WRITER_2026-10-02.md --root .

timeout 5m python3 -B scripts/check_gate_referents.py \
  docs/PREREG_DET_OBS_WRITER_2026-10-02.md --root .

timeout 5m python3 -B scripts/check_doc_canonical.py \
  docs/PREREG_DET_OBS_WRITER_2026-10-02.md \
  docs/PREREG_DET_NRB_E_2026-10-02.md \
  docs/VERDICT_DET_NRB_E_2026-10-02.md \
  CLAUDE.md \
  /gpfs/kjhan/lumina/cmfgen_map/diag3_20261002/DIAGNOSIS.md \
  /gpfs/kjhan/lumina/cmfgen_map/diag3_20261002/FABLE_AUDIT_r4.md \
  /gpfs/kjhan/lumina/cmfgen_map/diag3_20261002/DRIVER_DESIGN_AUDIT_r4.md \
  /gpfs/kjhan/lumina/cmfgen_map/diag3_20261002/LOOP_CLOSED.md
```

새 판독기의 구현할 인터페이스는 다음이다.

```bash
python3 -B scripts/det_obs_readout.py selftest \
  --baseline-source "$T0" --source "$T1" \
  --out "$N/stage1"

python3 -B scripts/det_obs_readout.py preflight \
  --base-e "$E" --base-r "$R" \
  --eprime "$EP" --rprime "$RP" \
  --source "$T1" --build "$B1" \
  --stage1 "$N/stage1" --evidence "$N/evidence" \
  --out "$N/evidence/preflight"

python3 -B scripts/det_obs_readout.py readout \
  --eprime "$EP" --rprime "$RP" \
  --stage1 "$N/stage1" --evidence "$N/evidence" \
  --out "$N/readout"
```

제출은 단계 2 승인 후 각 런에 한 번씩만 수행한다.

```bash
sbatch \
  --partition=h100,a100 --nodes=1 --ntasks=1 \
  --gres=gpu:2 --cpus-per-task=16 --mem=64G \
  --time=30:00:00 --no-requeue --exclude=syn102 \
  --output="$EP/slurm-%j.out" --error="$EP/slurm-%j.err" \
  "$EP/job.slurm" "$EP"

sbatch \
  --partition=h100,a100 --nodes=1 --ntasks=1 \
  --gres=gpu:2 --cpus-per-task=16 --mem=64G \
  --time=30:00:00 --no-requeue --exclude=syn102 \
  --output="$RP/slurm-%j.out" --error="$RP/slurm-%j.err" \
  "$RP/job.slurm" "$RP"
```

### 선언 블록의 범위

PF-1은 명시된 변경 경로와 `det_obs_` 심볼의 정합을 검사한다. 신설 예정 파일은 `expected_extra`로 선언한다. 이것이 미구현 파일의 존재·시험 통과를 증명하지 않으며, 구현 후 실제 patch 검사는 G-ID가 맡는다.

분기 입력 `v`는 판독기가 만든 **FUVa·FUVb 범주 두 값**이다. `−1/0/1`은 down/flat/up이며, 부적격은 `9` 또는 불완전 입력으로 표현한다. 아래 선언은 배타성과 전체 덮음을 검사한다. 실제 대역 적분·문턱 경계·게이트 실패의 X 변환은 G-READOUT이 검사한다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-OBS-WRITER (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/source/(?:src|tests)/[A-Za-z0-9_/.]+\\.(?:c|h)|scripts/det_obs_readout\\.py)",
    "symbol": "det_obs_",
    "roots": [
      "/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/source/src/lumina_cmfgen.c",
      "/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/source/src/lumina_cmfgen.h",
      "/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/source/src/env_universe.h",
      "/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/source/tests/det_obs_writer_selftest.c",
      "scripts/det_obs_readout.py"
    ],
    "expected_extra": [
      "/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/source/src/lumina_cmfgen.c",
      "/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/source/src/lumina_cmfgen.h",
      "/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/source/src/env_universe.h",
      "/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002/source/tests/det_obs_writer_selftest.c",
      "scripts/det_obs_readout.py"
    ]
  },
  "branches": {
    "regimes": [
      [-1.0, -1.0],
      [0.0, 0.0],
      [1.0, 1.0],
      [9.0, 9.0]
    ],
    "metrics": {
      "valid": "len(v) == 2 and all(x in (-1,0,1) for x in v)",
      "down": "sum(1 for x in v if x == -1)",
      "up": "sum(1 for x in v if x == 1)"
    },
    "rules": [
      {"name": "X", "predicate": "not valid"},
      {"name": "O-MIX", "predicate": "valid and down > 0 and up > 0"},
      {"name": "O-DOWN", "predicate": "valid and down > 0 and up == 0"},
      {"name": "O-UP", "predicate": "valid and up > 0 and down == 0"},
      {"name": "O-FLAT", "predicate": "valid and down == 0 and up == 0"}
    ],
    "adversarial_fixtures": [
      {"name": "down-down", "mix": [[-1,2]]},
      {"name": "down-flat", "mix": [[-1,1],[0,1]]},
      {"name": "flat-down", "mix": [[0,1],[-1,1]]},
      {"name": "up-up", "mix": [[1,2]]},
      {"name": "up-flat", "mix": [[1,1],[0,1]]},
      {"name": "flat-up", "mix": [[0,1],[1,1]]},
      {"name": "down-up", "mix": [[-1,1],[1,1]]},
      {"name": "up-down", "mix": [[1,1],[-1,1]]},
      {"name": "flat-flat", "mix": [[0,2]]},
      {"name": "invalid", "mix": [[9,1],[0,1]]},
      {"name": "missing-band", "mix": [[0,1]]},
      {"name": "extra-band", "mix": [[0,3]]},
      {"name": "empty", "mix": []}
    ]
  },
  "references": [
    {"path": "CLAUDE.md"},
    {"path": "docs/PREREG_DET_NRB_E_2026-10-02.md"},
    {"path": "docs/VERDICT_DET_NRB_E_2026-10-02.md"},
    {"path": "scripts/check_prereg_preflight.py"},
    {"path": "scripts/check_gate_referents.py"},
    {"path": "scripts/check_doc_canonical.py"},
    {
      "path": "scripts/nrb_e_readout.py",
      "flags_existing": ["REQUIRED_CASES", "def gid(", "def grun(", "def fuv("]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag3_20261002/DIAGNOSIS.md",
      "flags_existing": ["J_sc", "expm1", "IMPLEMENTATION_NOT_TESTED"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag3_20261002/DRIVER_DESIGN_AUDIT_r4.md",
      "flags_existing": ["K-1", "K-2", "K-3"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag3_20261002/FABLE_AUDIT_r4.md",
      "flags_existing": ["η_dep", "3879"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag3_20261002/LOOP_CLOSED.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/lumina_cmfgen.c",
      "flags_existing": [
        "cmf_cont_src_numerator",
        "g_chi_line_sc_unmapped",
        "cmfgen_write_spectrum_obs",
        "cmfgen_fine_jbar",
        "g_sob_faithful",
        "kappa_dep * B / chi_t"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/lumina_cmfgen.h",
      "flags_existing": ["CMFGENState"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/env_universe.h",
      "flags_existing": ["LUMINA_ENV_UNIVERSE_COUNT 534"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/line_net_rate.c",
      "flags_existing": [
        "line_net_sobolev_material",
        "effective_integrated_opacity",
        "emission_per_sr",
        "srce_chk_applied"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/cuda_build1/source_sha_at_build.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/w5_20260930/cuda_build1/binary.sha256",
      "flags_existing": [
        "7800484fb309addabfa5b2f6b879f3ab09b814db83d3c5db2f39debff3f1b390"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/nteta_20261001/run1/input/resolved_lumina.exports",
      "flags_existing": [
        "LUMINA_A210_SPRODUCER_CAPTURE=\"0\"",
        "LUMINA_PURE_CMFGEN_ITER=\"6\""
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/nrb2_20261002/run1/input/resolved_lumina.exports",
      "flags_existing": [
        "LUMINA_A210_SPRODUCER_CAPTURE=\"1\"",
        "LUMINA_BB_NRB=\"1\"",
        "LUMINA_CMF_FINE_UNMAPPED_EPS=\"1\"",
        "LUMINA_CMF_DEP_SOURCE=\"1\""
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/nrb2_20261002/stage_run1.sh"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/nrb2_20261002/run1/RUN_FOOTER.txt",
      "flags_existing": ["elapsed_seconds=41788", "terminal_state=SUCCESS"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/nrb2_20261002/evidence/slurm_states.json",
      "flags_existing": ["COMPLETED", "RESTARTS=0", "REQMEM=64G"]
    }
  ]
}
```
