## 결론

CMFGEN toy06 19.48d는 **6원소의 27개 명시적 다준위 이온**을 구성해 계산했습니다: **Si, S, Ca, Fe, Co, Ni**. 활성 원자모형 합계는 **FL 20,749개, SL 1,637개**입니다.

단, 이 런은 정상 수렴·정상 종료한 과학적 해가 아닙니다. 마지막 `OUTGEN`은 iteration 65에서 광도가 `NaN`이고, 실행은 출력 포맷 오류로 `CMFGEN_EXIT=2`였습니다: [OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/OUTGEN:4576), [batch.log](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/batch.log:1).

## A–F 판정

- **A — 지지.** 실제 활성 원소는 Si·S·Ca·Fe·Co·Ni 6개뿐입니다. [MOD_SUM](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:9)이 이 6원소만 열거하고, `SN_HYDRO_DATA`도 mass fraction 6개만 가지며 그 이름이 SIL·SUL·CAL·IRON·COB·NICK입니다: [SN_HYDRO_DATA](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/SN_HYDRO_DATA:5).

- **B — 지지(정의 필요).** `XzV_PRES=.TRUE.`인 명시적 다준위 이온은 **27개**입니다. 다만 CMFGEN 코드는 원소마다 다음 한 단계의 단일준위 closure ion을 자동 추가하므로, 이온화 방정식에 들어가는 ionic stage는 **33개**입니다: [cmfgen.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen.f:395), [cmfgen.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen.f:422).

- **C — 반증.** `[X_ISF]`의 정확한 의미는 **`NV, NS, NF`**, 즉 important-variable 수, superlevel 수, full-level 수입니다. 파서가 `RD_STORE_3INT(NV,NS,NF,...)`로 읽습니다: [cmfgen.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen.f:374). 이 런에서는 우연히 모든 행이 `NV=NS`라서 `NS,NS,NF`처럼 보일 뿐입니다.

- **D — 지지.** `VADAT`의 C/O/Mg 등 값은 이 런의 유효 조성 목록이 아닙니다. 실행 결과는 `NUM_SPECIES=28 6`으로 범용 종족표 28개 중 hydro에서 읽은 종족이 6개임을 기록합니다: [OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/OUTGEN:74). 유효 hydro 조성은 Si·S·Ca·Fe·Co·Ni뿐입니다.

- **E — 지지.** 공통 27이온 중 정확히 **10개**에서 우리 덱이 CMFGEN 활성 잘림보다 FL와 SL을 모두 더 많이 사용합니다. Fe II는 2698/135 대 2599/131, S III는 380/127 대 256/79가 맞습니다.

- **F — 지지(정확화·제한 있음).** level-bearing 집합 차이는 **32이온**, 원소 집합 차이는 **9원소**입니다. 다만 32개 전부가 새 9원소 소속은 아닙니다. 새 9원소가 26이온이고, 나머지 6개는 Si I·S I·Ca I·Fe I·Co I·Ni I입니다. 또한 abundance는 shell 0–29에서만 양수입니다. 덱은 50 shell인데 `abundances.csv`에는 30 shell 열만 있어, 로더상 shell 30–49는 0으로 남습니다: [config.json](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv_ftos/config.json:5), [abundances.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv_ftos/abundances.csv:1), [lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:817).

## 이온별 CMFGEN 활성 정본표

`MOD_SUM` 표기는 `FL/SL`입니다. `Sk=Si`, `Nk=Ni`입니다.

| Z | ion0 | 분광표기 | 활성 FL | 활성 SL | 정본 |
|---:|---:|---|---:|---:|---|
| 14 | 1 | Si II | 125 | 69 | [MOD_SUM:9](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:9) |
| 14 | 2 | Si III | 147 | 99 | [MOD_SUM:9](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:9) |
| 14 | 3 | Si IV | 61 | 50 | [MOD_SUM:9](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:9) |
| 14 | 4 | Si V | 203 | 52 | [MOD_SUM:10](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:10) |
| 16 | 1 | S II | 322 | 55 | [MOD_SUM:11](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:11) |
| 16 | 2 | S III | 256 | 79 | [MOD_SUM:11](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:11) |
| 16 | 3 | S IV | 176 | 65 | [MOD_SUM:11](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:11) |
| 16 | 4 | S V | 163 | 39 | [MOD_SUM:12](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:12) |
| 20 | 1 | Ca II | 62 | 40 | [MOD_SUM:13](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:13) |
| 20 | 2 | Ca III | 232 | 44 | [MOD_SUM:13](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:13) |
| 20 | 3 | Ca IV | 375 | 41 | [MOD_SUM:13](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:13) |
| 20 | 4 | Ca V | 528 | 70 | [MOD_SUM:14](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:14) |
| 26 | 1 | Fe II | 2599 | 131 | [MOD_SUM:15](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:15) |
| 26 | 2 | Fe III | 1500 | 105 | [MOD_SUM:15](/gpfs/kjhan/cmfgen_runs/toy06_19p48d/MOD_SUM:15) |
| 26 | 3 | Fe IV | 1000 | 63 | [MOD_SUM:15](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:15) |
| 26 | 4 | Fe V | 1000 | 45 | [MOD_SUM:16](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:16) |
| 26 | 5 | Fe VI | 2000 | 67 | [MOD_SUM:16](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:16) |
| 27 | 1 | Co II | 1000 | 55 | [MOD_SUM:17](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:17) |
| 27 | 2 | Co III | 1000 | 52 | [MOD_SUM:17](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:17) |
| 27 | 3 | Co IV | 1000 | 56 | [MOD_SUM:17](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:17) |
| 27 | 4 | Co V | 1000 | 43 | [MOD_SUM:18](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:18) |
| 27 | 5 | Co VI | 1000 | 41 | [MOD_SUM:18](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:18) |
| 28 | 1 | Ni II | 1000 | 59 | [MOD_SUM:19](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:19) |
| 28 | 2 | Ni III | 1000 | 47 | [MOD_SUM:19](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:19) |
| 28 | 3 | Ni IV | 1000 | 54 | [MOD_SUM:19](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:19) |
| 28 | 4 | Ni V | 1000 | 54 | [MOD_SUM:20](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:20) |
| 28 | 5 | Ni VI | 1000 | 62 | [MOD_SUM:20](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/MOD_SUM:20) |

암묵적 closure stage는 Si VI `(14,5)`, S VI `(16,5)`, Ca VI `(20,5)`, Fe VII `(26,6)`, Co VII `(27,6)`, Ni VII `(28,6)`입니다. 이들은 다준위 원자파일이 없는 단일 상태이므로 위 FL/SL 표의 27개에는 포함하지 않았습니다.

## 우리 덱과의 불일치 전건

`levels.csv`를 `(atomic_number, ion_number)`로 묶어 FL 수를 세고, `super_level` 고유값을 세어 SL 수를 얻었습니다.

| 이온 | CMFGEN FL/SL | 우리 덱 FL/SL | 차이 FL/SL |
|---|---:|---:|---:|
| Si II `(14,1)` | 125/69 | 157/79 | +32/+10 |
| Si IV `(14,3)` | 61/50 | 66/55 | +5/+5 |
| S II `(16,1)` | 322/55 | 324/56 | +2/+1 |
| S III `(16,2)` | 256/79 | 380/127 | +124/+48 |
| S IV `(16,3)` | 176/65 | 194/69 | +18/+4 |
| S V `(16,4)` | 163/39 | 216/50 | +53/+11 |
| Ca II `(20,1)` | 62/40 | 77/43 | +15/+3 |
| Ca IV `(20,3)` | 375/41 | 378/43 | +3/+2 |
| Ca V `(20,4)` | 528/70 | 613/73 | +85/+3 |
| Fe II `(26,1)` | 2599/131 | 2698/135 | +99/+4 |

나머지 **17/27은 정확히 일치**합니다: Si III, Si V, Ca III, Fe III–VI, Co II–VI, Ni II–VI.

추가 32 level-bearing 이온은 다음과 같습니다.

- 새 9원소의 26이온: C I–III, O I–III, Mg I–III, Al I–IV, Sc I–III, Ti II–IV, V I, Cr I–IV, Mn II–III.
- 기존 6원소의 추가 중성 이온: Si I, S I, Ca I, Fe I, Co I, Ni I.

## 파일 권위 판정

- **실제 실행 roster와 활성 FL/SL의 최상위 정본: `MOD_SUM`.** 실행이 사용한 27이온과 `FL/SL`을 직접 출력합니다. 현재 최종 시각은 2026-07-18 14:47:54입니다.

- **`MODEL_SPEC`: 유효 입력 지시서.** 활성 이온 선택과 `NV/NS/NF` 상한을 지정합니다. 실제 런 파일과 저장소 스냅샷은 byte-identical입니다. 다만 실행 여부를 스스로 증명하지 않으므로 `MOD_SUM`보다 한 단계 아래입니다.

- **`atomic_links.txt`: 경로 정본.** 어느 osc/`f_to_s` 파일을 읽을지만 결정하며 활성 FL/SL 수는 결정하지 않습니다: [atomic_links.txt](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/atomic_links.txt:1).

- **osc: 원자파일의 전체 FL 용량 정본.** 10이온에서 전체 용량이 `MODEL_SPEC` 활성 NF보다 큽니다. 이는 실제 활성 수가 아니라 선택 가능한 원본 수입니다.

- **`f_to_s`: 전체 FL→SL 매핑 정본.** 역시 전체 mapping이며, 실제 활성 NS는 `MODEL_SPEC`가 잘라 결정합니다. 위 10이온에서 원본 mapping과 활성 NS가 다릅니다.

- **`*OUT`·`POP*`: 실행 FL 교차확인 정본.** 예를 들어 `POPSIL`은 Si II 125, Si III 147, Si IV 61, Si V 203을 각각 기록합니다: [POPSIL](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/POPSIL:17). 단 이 파일들은 03:11 시점의 이전 stint입니다.

- **`RVTJ`: roster 정본 아님.** 03:11 시점이며 Sk2와 0-density H/He population만 기록해 전체 종족 census로 사용할 수 없습니다: [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19.48d/RVTJ:273).

- **`MEANOPAC`, `EDDFACTOR`: roster/FL/SL 정본 아님.** 전달·평균불투명도 상태 파일이며 종족 목록 필드가 없습니다.

## UNRESOLVED·주의사항

- 현재 디렉터리는 단일 원자적 스냅샷이 아닙니다. `RVTJ`·`*OUT`·`POP*`는 03:11, `MEANOPAC`·`EDDFACTOR`·`OUTGEN`·`MOD_SUM`은 14:47입니다. roster와 FL은 양쪽에서 일치하지만 플라스마 상태를 서로 혼합해 해석하면 안 됩니다.
- CMFGEN 활성 종족/FL/SL은 확정됐지만, 수렴한 최종 복사전달 해는 없습니다. 마지막 광도는 `NaN`이며 종료코드는 2입니다.
- 우리 덱이 실제 완료된 Lumina 모델 런에 투입됐다는 실행 산출물은 확인되지 않았습니다. 현재 `verification.log`도 실패 한 줄뿐입니다: [verification.log](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv_ftos/verification.log:1). 다만 코드상 로더는 15원소와 abundance를 읽고 모든 원소에 대해 ion population/electron-density 계산을 수행합니다: [lumina_atomic.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:808), [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:2452).