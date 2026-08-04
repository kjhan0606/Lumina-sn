## 판정

Wave 2 검증은 **PASS(허용 범위 내)** 입니다. CUDA 재빌드, OFF 불변성, CPU production CDF sampler, 게이트 단독/조합을 통과했습니다. GPU kernel 실행은 금지 조건 때문에 판정 범위 밖입니다.

### 빌드·실행 exit

| 항목 | exit |
|---|---:|
| `make -B cuda` | 0 |
| `make -B bench_frozen_oracle` | 0 |
| OFF-unset / explicit-zero | 0 / 0 |
| D1 / stim / D1+stim | 0 / 0 / 0 |
| multi-edge / MA-J / no-line-therm | 0 / 0 / 0 |
| MA 두 게이트 조합 / all-5 | 0 / 0 |
| 수정된 CPU CDF probe 컴파일 | 0 |
| CDF probe D1 / D1+stim | 0 / 0 |

CUDA 빌드는 기존 `g_fgemm_nulo` unused 경고만 발생했습니다. 임시 probe 첫 컴파일은 libc `stat` 이름 충돌로 exit 1이었고, 함수명을 바꾼 최종 probe는 exit 0입니다.

### 3셀 OFF 불변 SHA-256

`eligible`은 header와 `status=available` 행입니다.

| 셀 | eligible SHA-256 | full CSV SHA-256 | unset vs `=0` |
|---|---|---|---:|
| s0 | `beaac19b21bd5b9c0d8c7c81903a1c8c13c8f139ba05cf2e01c414f193678cfa` | `4789f13c89a3bb613e89cb23e836242285aae31bee6065b2631d61324eee1952` | cmp 0 |
| s8 | `54f9fafad8da44602a419562a2ef37c9f0c726fdad6780c72e99df436e87d05f` | `a4f1a146a313501a3eaf56232d2d7d3cd4f798425ebd8f426067292edb1538e2` | cmp 0 |
| s43 | `b971a0381d4d6c8246979c3bb8d013290d65deac6985898795bee94894380804` | `c48d2619f160191d4a91e37334cf165d2fc312d2263635a281112523e70b72aa` | cmp 0 |

### 게이트별 oracle ON 효과

변화 수의 분모는 셀별 available 행 `s0/s8/s43 = 121/159/131`입니다.

| 게이트 | 변화 행 s0/s8/s43 | BF Γ·α rate | 판정 |
|---|---:|---:|---|
| `FIX_BF_CONTINUUM_EVENT` | 0 / 0 / 0 | 각 셀 0/12 | packet fate는 oracle 밖; OFF와 full CSV 동일 |
| `FIX_BF_STIM_RECOMB` | 12 / 28 / 16 | 각 셀 0/12 | χ·η만 예상 방향으로 감소 |
| `FIX_BF_MULTI_EDGE` | 0 / 0 / 0 | 0/12 | GPU FB 방출 전용이라 oracle 부적격 |
| `FIX_MA_J_UNCLAMP` | 0 / 0 / 0 | 0/12 | MA 확률 소비점은 oracle 밖 |
| `FIX_MA_NO_LINE_THERM` | 0 / 0 / 0 | 0/12 | CUDA line emission 전용 |
| all-5 | stim 단독과 동일 | 0/12 | 예상 밖 교차효과 없음 |

Stim 단독의 available 합산 변화:

| 셀 | Δχ 1000 Å | Δχ 5000 Å | Δη 1000 Å | Δη 5000 Å |
|---|---:|---:|---:|---:|
| s0 | −0.577995% | −10.733806% | −0.000728% | −0.000335% |
| s8 | −0.006301% | −78.787176% | −0.001097% | −0.000401% |
| s43 | −0.000256% | −0.017506% | −0.001187% | −0.000399% |

Thermal ledger도 모든 arm에서 셀당 0/8 변화입니다.

### D1 CDF 및 `nu_edge/nu`

Production CPU sampler로 각 점당 20만 회 추첨했습니다. `Neff=1/Σp²`입니다.

| 셀·파장 | 유효 route D1→D1+stim | CDF endpoint Δ | Neff 변화 | 기대 MA 확률 변화 |
|---|---:|---:|---:|---:|
| s0 1000 Å | 10404→5805 | −0.022235% | 36.806→36.791 | 0.921135→0.921137 |
| s0 5000 Å | 4894→1527 | −97.020121% | 382.076→275.995 | 0.779423→0.830649 |
| s8 1000 Å | 11393→11244 | −0.002852% | 15.260→15.259 | 0.924610→0.924610 |
| s8 5000 Å | 5313→21 | −86.635998% | 41.218→1.000 | 0.834020→0.971599 |
| s43 1000 Å | 989→840 | −0.000254% | 9.469→9.469 | 0.873578→0.873577 |
| s43 5000 Å | 419→292 | −0.017664% | 1.499→1.498 | 0.947456→0.947481 |

추첨 검증:

| 검사 | 결과 |
|---|---:|
| 12개 CDF endpoint 재합산 상대오차 | 정확히 0 |
| sampler 실패 | 0 / 2,400,000 |
| `nu_edge/nu && target>=0` branch 최대 \|z\| | 1.670 |
| 최상위 route 빈도 최대 \|z\| | 1.309 |
| 20-bucket CDF 최대 χ² | 25.086 / df 19 |
| bucket 최대 절대 빈도차 | 0.001194 |

D1+stim에서 특히 s8/5000 Å CDF가 사실상 단일 route로 집중되는 효과가 확인됐습니다. CDF endpoint는 전체 원자 route의 bin-center 값이고 oracle χ 표는 선택된 8개 이온이므로 두 수치를 직접 동일시하면 안 됩니다.

### 조합 상호작용

| 비교 | full CSV cmp | 해석 |
|---|---:|---|
| D1 vs OFF | 0 | oracle 수량 불변 |
| D1+stim vs stim | 0 | oracle 밖 event CDF에만 stim corrfactor 반영 |
| MA-J + no-line-therm vs OFF | 0 | oracle 범위 내 교차효과 없음 |
| all-5 vs stim | 0 | 나머지 네 게이트의 예상 밖 rate/thermal 영향 없음 |

CPU/GPU CDF 공식과 독립적인 두 번째 난수 분기도 정적으로 대조했습니다: [CPU sampler](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6536), [GPU mirror](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:3487), [CPU branch](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_transport.c:567), [GPU branch](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5728).

소스 수정, GPU 실행, `git` 명령은 없었습니다.