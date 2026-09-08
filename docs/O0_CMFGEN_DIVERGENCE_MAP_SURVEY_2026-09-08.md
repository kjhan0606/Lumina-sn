# O-0 실측 — CMFGEN 어긋남 지도(쌍둥이 T_e 레인)의 실물 확인 (2026-09-08)

저자 = 운전자. **계측·판독이지 계약이 아니다**(개정23: 계획·계약 기안 = astra).
전건 **런 0회**, 로그인 노드 파일 조회·판독 범위. 모든 행 `[저자 실행]`.
경위 = Q-GOAL 사전감사(`/gpfs/kjhan/lumina/ne_thresh/qgoal/ASTRA_QGOAL.md`)가 DET-NE-THRESH 를
`부분부합·차단 1` 로 막고 대안 ③ 을 지목했고, **user 가 2026-09-08 「1번으로 갑니다」로 표적을 교체**했다.

## 1. 잣대 — 공개 StaNdaRT CMFGEN 자료 (실재)

| 항목 | 실측 |
|---|---|
| 경로 | `data/standart_data1/toy06/{phys,ionfrac_<el>}_toy06_cmfgen.txt` |
| 원소 (ionfrac, CMFGEN) | **ca · co · fe · ni · si · s** (6종) |
| 시각 | `#NTIMES: 57`, **19.480 d 블록 실재** |
| 속도 구간 | `#NVEL: 100`, `vel_mid` **1025.0 – 40325.0 km/s** |
| 단계 | `#NSTAGES: 6` (0–5) |
| phys 열 | `vel_mid[km/s] temp[K] rho[gcc] ne[/cm^3] natom[/cm^3]` |
| 19.48 d T 범위 | **8,849 – 25,690 K** |
| 19.48 d n_e 범위 | 3.185e4 – 1.848e10 cm⁻³ |

⚠**자체 CMFGEN 런은 이 잣대가 아니다** — 대장이 물리 오라클 부적격을 기록했다. 공개 자료만 쓴다.

## 2. 비교 대상 — 봉인 base 런 (실재)

`/gpfs/kjhan/lumina/superlevel_b_a100x2_k36/slb_20260906T234024Z_dc1_ne_closure_on`

| 항목 | 실측 (`input/model/config.json`·`geometry.csv`) |
|---|---|
| `source_model` | **`StaNdaRT toy06 (snia_toy06_1h_lowres)`** — 잣대와 **같은 모델** |
| `target_epoch_d` | **19.48** — 잣대와 **같은 시각** |
| `n_shells` | **50** |
| 속도 | `v_inner_min` 3.90e8 cm/s · `v_outer_max` 4.03e9 cm/s = **3,900 – 40,300 km/s** |
| `T_inner_K` | 10,020 |
| 진단 레인 T_e | `input/requested_diag_te_K.txt` = **10020 (전 셸 균일)** |
| 반복 | `LUMINA_PURE_CMFGEN=1`·`LUMINA_PURE_CMFGEN_ITER=2` |

## 3. 격자 겹침 — 대조 가능성 (실측)

Lumina 3,900–40,300 km/s 안에 드는 CMFGEN 속도구간 = **77 / 100**, 겹침 **4,182.9 – 40,267.3 km/s**.
⟹ Lumina 정의역 거의 전체가 잣대에 덮인다. 보간은 CMFGEN(100) → Lumina 셸(50) 방향이 자연스럽다.
경계 셸(v < 4,183 km/s)은 잣대 밖 외삽 구간이므로 **별도 표기 대상**이다.

## 4. 교란 변수 — T_e 불일치가 지배한다 (실측)

CMFGEN 19.48 d 는 8,849–25,690 K 인데 base 는 **전 셸 10,020 K 고정**이다.
CMFGEN Fe 분율 실측(19.48 d): v=3,886.8 에서 III 2.19e-4·**IV 0.972**, v=7,926.1 에서 **III 0.743**·IV 0.251,
v=11,928.7 에서 II 1.13e-2·**III 0.967**. 기존 설계가 기록한 Fe II 7자릿수 격차는 **이 T 핀이 만든 것**이며
배선 격차의 크기가 아니다(`docs/DESIGN_DC_DD_CMFGEN_CONTRAST_2026-09-05.md` §2-1).
⟹ **쌍둥이 T_e 레인 없이 만든 지도는 잣대 사고다.**

## 5. 레인·산출 노브 — 배선 실측 (코드 신설 불요)

| 노브 | 배선 위치 | 실측 |
|---|---|---|
| `LUMINA_FIXED_TE_PROFILE` | `src/lumina_plasma.c:16188` | 형식 `<shell_id> <T_e>` 행별; **전 셸 피복 강제**(`seen[]`), 유한·양수 검사, 중복·범위 초과 거부, `A210_PRODUCTION_TE_MIN/MAX_K` 검증, **sha256 공표**(`[A2-10][TE_PUBLICATION] te_profile_sha256=…`), 실패는 이름 붙은 fail-closed(`RADEQ_FIXED_T_*`). `te_lane=FIXED_T` 로 전환 |
| `LUMINA_ION_POP_DUMP_ITER` | `src/lumina_cuda.cu:8648` | 반복별 이온 단계 밀도 덤프 |
| `LUMINA_ION_POP_DUMP` | `src/lumina_cuda.cu:10280,11298` | PURE-CMFGEN 경로 덤프 |
| `LUMINA_NLTE_LEVEL_DUMP` | `src/lumina_cuda.cu:1971`, `src/lumina_plasma.c:20760` | NLTE 준위 population |

★**운전자 정정(기재)**: 첫 조회를 `src/*.c src/*.h` 로만 돌려 `ION_POP_DUMP_ITER` 을 `getenv=0`,
즉 **죽은 노브로 잘못 읽었다**. 배선은 `.cu` 에 있다. 계약에 들어가기 전에 잡았으나, 「이름이 아니라
한 번의 탐침」 규칙의 **검색 범위** 판이다 — 이 프로젝트의 소스는 `.c`·`.h`·`.cu` 셋이다.

## 6. 이 문서가 말하지 않는 것

어떤 계약을 걸어야 하는지, 몇 런이 필요한지, 무엇을 기대치로 등록할지. 그것은 astra 의 계획 몫이다.
쌍둥이 T_e 레인이 **작동한다**는 것도 여기서는 미시연이다 — 배선의 실존과 형식만 확인했다.
