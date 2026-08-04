읽기 전용 조건 때문에 `docs/CODEX_INPUT_ATOMIC.md`는 생성하지 않았다. 아래가 해당 보고서의 완성 본문이며, 파일·commit·모델·GPU 실행은 없었다.

# CODEX_INPUT_ATOMIC

## -o 요약

| 항목 | 판정 | 핵심 근거 |
|---|---|---|
| I1 충돌강도 Υ | **잔류** | Lumina Co IV 표 4,455전이는 Fe III 표의 정확한 부분집합이다. CMFGEN 런의 Co IV 입력은 tabulated 전이 0개다. |
| I2 `A_ul` | **잔류** | 엄격히 결합된 880,406선 중 75,075선 불일치. 전체 중앙 상대차 0, 최대 89.0103. |
| I3 σ(ν) | **잔류** | 비교 가능한 3,953,894 주파수점 중 1,233,529점 불일치. 중앙 상대차 \(9.22\times10^{-15}\), 최대 \(4.461\times10^7\). |
| I4 슈퍼레벨 | **잔류** | 공통 21개 이온 모두 Lumina `min(level,100)` 분할과 CMFGEN `F_TO_S` 분할의 SL 수가 다르다. |
| I5 재결합·DR | **잔류** | Lumina에는 생산 설정에서 Co IV→Co III DR이 남아 있지만 CMFGEN 런은 `[DIE_CoIV]=F,F`. RR 전수 계수 대조는 별도 결판 요건이 남는다. |

## 공통 실행 입력

실제 캡처는 다음을 사용한다.

- 모델: `data/tardis_reference_toy06_19p48d_sivcaiv` — [환경 92행](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/PARITY59_INSTR.env:92)
- σ binary 명시 — [환경 28행](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/PARITY59_INSTR.env:28)
- CMFGEN Υ 표 사용 — [환경 109행](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/PARITY59_INSTR.env:109)
- 슈퍼레벨 `K=100`, 활성 — [환경 126행](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/PARITY59_INSTR.env:126)
- 실소비 확인: 2,584,132선, 26,592레벨, σ 26,087레벨, Υ 표 40개 — [stdout 156행](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:156), [166행](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:166), [255행](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:255)

---

## I1 — 충돌강도 Υ

### Lumina 입력과 변환

- manifest의 모든 `status=OK` binary를 읽는다 — [lumina_atomic.c:1451](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:1451).
- binary 필드는 `(Z, ion0, n_trans, n_temp, n_levels_ref, T_grid, level pair, omega[])` — [lumina_atomic.c:1358](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:1358).
- 선과 표는 `(Z, ion, level_lower, level_upper)`로 연결된다 — [lumina_plasma.c:780](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:780).
- Υ는 온도에 선형 보간되고 양 끝 온도로 clamp된다 — [lumina_plasma.c:15532](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15532).
- 표가 없으면 `f_lu>10^-5`는 van Regemorter, 그 이하는 `OMEGA_SET=0.1` — [lumina_plasma.c:688](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:688).
- 전체 선 census: tabulated 29,840, van Regemorter 1,742,025, `0.1` 812,267 — [stdout 259행](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:259).

### Co IV 대용 확정

Lumina Co IV 원본은 [COB/IV/19apr23/col_data](/gpfs/kjhan/cmfgen_21jun23/atomic/COB/IV/19apr23/col_data:18)의 4,455전이×20온도다. Fe III는 [FE/III/19apr23/col_data](/gpfs/kjhan/cmfgen_21jun23/atomic/FE/III/19apr23/col_data:28)의 22,139전이×20온도다.

전수 수치 대조 결과:

- Co IV 4,455/4,455 Υ 벡터가 Fe III에 정확히 한 번씩 존재
- 최대 절대차: 0
- 4,357개는 레벨명 쌍까지 동일
- 98개는 이름 한 부분만 다르나 20개 Υ 값은 전부 동일
- 예: Co IV 1730행 `3d6_1D1e[2]` ↔ Fe III 1902행 `3d6_1De[2]`
- 따라서 대용 범위는 Co IV collision table의 **전 전이 4,455개**다.

다만 실제 Co IV line list 4,041선에서 이 표가 연결되는 범위는:

| 분기 | 선 수 | 파장 범위 |
|---|---:|---:|
| Fe III 복제표 사용 | 376 | 1,067 Å–294 μm |
| van Regemorter | 2,642 | 460–18,179 Å |
| `OMEGA_SET=0.1` | 1,023 | 459–57,741 Å |

즉 “Co IV 모든 선”이 아니라 **Co IV 표의 전 전이**, 실제 선에서는 376/4,041선이다.

CMFGEN 런의 대응 입력은 [atomic_links.txt:77](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/atomic_links.txt:77)의 `CoIV_COL_DATA`이며, 연결된 [col_guess.dat:18](/gpfs/kjhan/cmfgen_21jun23/atomic/COB/IV/18oct00/col_guess.dat:18)은 tabulated 전이 0개, `f=0` 기본값 0.1이다.

따라서 Lumina/CMFGEN 차이는 **tabulated 행 4,455 대 0**이다. 한쪽에 대응 행이 없으므로 상대차 중앙값·최댓값은 정의할 수 없다.

### 대용 전수 목록

40개 Lumina collision table, 총 114,952전이를 전수 교차검사했다.

| 관계 | 결과 |
|---|---|
| **Fe III → Co IV** | Co IV 4,455/4,455 벡터가 Fe III와 완전 동일. 대용 확정. |
| **Si I ↔ S I** | 양쪽 11/11 벡터 완전 동일. 어느 쪽이 donor인지 데이터만으로 결판 불가. 두 이온 모두 CMFGEN 런 대응물 없음. |
| Sc I ↔ Ni I | Sc I 20개 곡선 형상은 Ni I 일부와 비례하지만 배율이 전이별로 다름. 대용 확정 근거로 사용하지 않음. |

그 밖에 전체 표가 정확히 재사용된 이온쌍은 없었다.

**판정: 잔류.**

---

## I2 — `A_ul`

### Lumina 입력과 변환

- 실제 소비 필드: `line_list.csv:A_ul` — [lumina_atomic.c:649](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:649).
- 생성 시 CMFGEN `osc_data` 전이의 `A` 필드를 가져온다 — [expand_atomic_data_cmfgen.py:522](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:522).
- CSV에는 `.6e`로 기록된다 — [expand_atomic_data_cmfgen.py:590](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:590).
- 런타임에는 선택적 `AUL_SCALE`이 있으나 생산 환경에 해당 변수가 없으므로 배율 변환은 없다 — [lumina_atomic.c:666](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:666).

CMFGEN 대응 입력은 각 `*_F_OSCDAT`의 전이 `A` 필드다. 실제 연결은 [atomic_links.txt:1](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/atomic_links.txt:1)–[108행](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/atomic_links.txt:108)에 있다.

### 직접 대조

공통 21개 이온의 Lumina 선 2,121,277개를 훑었다. 구성명·통계중량·준위에너지가 모두 같은 레벨만 결합했다.

- 직접 결합: 880,406선
- 상대차 \(>10^{-6}\): 75,075선
- 전체 결합선 중앙 상대차: 0
- 최대 상대차: 89.0103, S III
- 파장대별 불일치: `<1000 Å` 14,020, `1000–3000 Å` 26,006, `3000–10000 Å` 21,186, `≥10000 Å` 13,863

집중 이온:

| 이온 | 불일치 |
|---|---:|
| Ni III | 21,314 |
| Ni II | 16,411 |
| Co III | 13,081 |
| Ca IV | 8,485 |
| S III | 5,583 |
| Ni IV | 3,658 |
| S IV | 3,316 |
| S V | 1,889 |
| Co IV | 1,223 |
| Fe V / Fe IV | 113 / 2 |

공통 이온 안에서도 레벨을 엄격히 결합할 수 없었던 1,240,871선과 CMFGEN 런에 이온 자체가 없는 선은 제거 근거가 아니다.

**판정: 잔류.**

---

## I3 — 광이온 단면적 σ(ν)

### Lumina 입력과 변환

실제 입력은 [cmfgen_sigma_bf.bin](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/cmfgen_sigma_bf_superlev_ionfix_ddc15strat_sivcaiv.bin)이다.

- 26,592레벨×1,000 주파수 bin, cm² 단위 double — [lumina_atomic.c:1007](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:1007).
- 로더는 단위변환·보간·clip 없이 배열을 그대로 읽는다 — [lumina_atomic.c:1054](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:1054).
- coverage: 26,087/26,592레벨; 나머지 505레벨은 런타임 Kramers fallback — [stdout 166행](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:166).
- fallback은 이온별 `sigma_0`, 없으면 \(7.91\times10^{-18}/Z_\mathrm{eff}^2\) — [lumina_plasma.c:6777](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6777), 이후 \((ν_\mathrm{th}/ν)^3\) — [lumina_plasma.c:15794](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15794).

CMFGEN 대응 입력은 각 `PHOT*_A` 파일이다. 필드는 configuration, cross-section type, energy ratio/fit parameter, `sigma_Mb` — [cmfgen_parser.py:245](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmfgen_parser.py:245), [381행](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmfgen_parser.py:381). 실제 run 연결은 [atomic_links.txt:3](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/atomic_links.txt:3) 이하의 `PHOT..._A` 필드다.

### 직접 대조

CMFGEN 입력의 검증 가능한 type 1, 7, 20–22를 Lumina 1,000개 중심주파수에 평가하여 binary 실제값과 대조했다.

- 직접 비교 레벨: 7,418
- CMFGEN σ>0 비교점: 3,953,894
- 상대차 \(>10^{-6}\): 1,233,529점, 31.20%
- 전체 중앙 상대차: \(9.22\times10^{-15}\)
- 최대 상대차: \(4.461\times10^7\), Ni II
- CMFGEN 양수/Lumina 0: 197,231점
- 집중: Ni II 410,393, Co III 334,365, S III 161,165, S IV 72,036, S V 70,585, Fe III 62,782, Co IV 46,827점

CMFGEN hydrogenic 계열 type 2/3/8의 2,084개 대응 레벨은 같은 방법으로 결판내지 않았다. 결판에는 해당 CMFGEN 평가기를 동일 주파수점에 적용한 값이 필요하다. binary에는 bake 당시 소스 revision·옵션이 저장되어 있지 않으므로 bake provenance 자체도 복원 불가다.

**판정: 잔류.**

---

## I4 — 슈퍼레벨 분할

### Lumina 입력과 변환

- 원래 `levels.csv:super_level`을 읽는다 — [lumina_atomic.c:733](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:733).
- 생산 설정 `K=100`이 이를 전부 `super=min(level_number,100)`으로 덮어쓴다 — [lumina_atomic.c:761](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:761).
- `LUMINA_SUPER_LEVELS=1`로 실제 projection에 사용된다 — [lumina_plasma.c:14277](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14277).

CMFGEN 대응 입력은 `*_F_TO_S`의 full-level→SL 필드다 — [cmfgen_parser.py:485](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmfgen_parser.py:485).

### 직접 대조

표의 값은 `Lumina SL 수 / CMFGEN F_TO_S SL 수`다.

| 이온 | SL 수 | 이온 | SL 수 |
|---|---:|---|---:|
| Si II | 101 / 79 | Si III | 101 / 99 |
| Si IV | 66 / 50 | S II | 101 / 56 |
| S III | 101 / 127 | S IV | 101 / 69 |
| S V | 101 / 50 | Ca II | 77 / 43 |
| Ca III | 101 / 44 | Ca IV | 101 / 43 |
| Ca V | 101 / 73 | Fe II | 101 / 135 |
| Fe III | 101 / 105 | Fe IV | 101 / 63 |
| Fe V | 101 / 45 | Co II | 101 / 55 |
| Co III | 101 / 52 | Co IV | 101 / 56 |
| Ni II | 101 / 59 | Ni III | 101 / 47 |
| Ni IV | 101 / 54 |  |  |

21/21 이온에서 그룹 수부터 다르므로 분할은 동일할 수 없다.

추가로 run [MODEL_SPEC](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MODEL_SPEC:10)의 선언과 `F_TO_S` 파일 사이에도 S II `55 대 56`, Ca V `70 대 73` 불일치가 있다. 어느 값을 적용해도 Lumina의 101과는 다르다.

**판정: 잔류.**

---

## I5 — 재결합·DR

### Lumina 입력과 변환

독립적인 RR 파일은 없다. RR은 I3의 σ binary로부터 Milne 적분된다.

- 각 lower level의 σ 행을 사용 — [lumina_plasma.c:15738](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15738).
- spontaneous+stimulated Milne 적분 — [lumina_plasma.c:15745](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15745).
- `ALPHA_SPINGATE=1`이면 금지된 daughter level의 재결합 적분을 0으로 만든다 — [lumina_plasma.c:15855](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:15855).

DR은 코드 내 `DR_TABLE`을 사용하고 \(T^{-3/2}\sum c_i e^{-E_i/T}\)로 평가된다 — [lumina_plasma.c:8052](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:8052). 생산 설정은 Badnell/NORAD/Mazzotta/AUTOSTRUCT를 0으로 만들지만 CMFGEN source는 0으로 만들지 않는다 — [환경 52행](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/PARITY59_INSTR.env:52).

따라서 남는 항목은 Co IV→Co III:

- 계수 — [lumina_plasma.c:8014](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:8014)
- \(\alpha_\mathrm{DR}(10^4\mathrm{K})=1.35895\times10^{-11}\ \mathrm{cm^3\,s^{-1}}\)
- NLTE 행렬에는 `FROZENIN_DR`과 무관하게 추가됨 — [lumina_plasma.c:16224](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16224).

CMFGEN 런의 대응 입력:

- `[DIE_CoIV]=F,F` — [VADAT:702](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/VADAT:702)
- run 디렉터리에 Co DIE 입력 링크 없음

따라서 이 채널은 Lumina \(1.35895\times10^{-11}\), CMFGEN 0이다. CMFGEN 분모가 0이므로 상대차는 정의되지 않는다.

RR 계수 전수 비교는 **UNRESOLVED**다. CMFGEN에는 독립 RR 입력이 없고 `PHOT` 입력으로부터 산출되므로, 결판에는 동일 \(T_e,J_\nu\), 동일 상태별 target mapping에서 양 코드가 산출한 level-resolved RR 계수가 필요하다. 그러나 I5 전체는 위 DR 불일치만으로 제거할 수 없다.

**판정: 잔류.**

## CMFGEN 런 대응물이 없는 Lumina 이온

다음 이온 블록은 CMFGEN 런에 대응 입력이 없어 제거 대상이 아니다:

C I–III, O I–III, Mg I–III, Al I–IV, Si I, S I, Ca I, Sc I–III, Ti II–IV, V I, Cr I–IV, Mn II–III, Fe I, Co I, Ni I.