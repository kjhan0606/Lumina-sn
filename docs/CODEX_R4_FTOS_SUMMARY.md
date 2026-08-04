R4 스크립트 준비와 CPU 자기검사를 완료했습니다. GPU·모델·덱 생성·commit은 실행하지 않았고, R4에서 `src/`도 수정하지 않았습니다.

## 1단계 — 포맷 전수표

공통 경로 prefix: `/gpfs/kjhan/cmfgen_21jun23/atomic/`

SL 수는 전 행의 link ID가 `1..N` 연속임을 확인해 실측했습니다.

| 이온 | 파일 | 판본 | 형식 | FL→SL |
|---|---|---|---|---:|
| Si II | `SIL/II/19apr23/f_to_s_79` | 19apr23 | explicit FL ID | 157→79 |
| Si III | `SIL/III/19apr23/f_to_s_ls` | 19apr23 | explicit | 147→99 |
| Si IV | `SIL/IV/5dec96/f_to_s_split.dat` | 5dec96 | implicit row order | 66→55 |
| Si V | `SIL/V/19apr23/f_to_s_52` | 19apr23 | explicit | 203→52 |
| S II | `SUL/II/19apr23/f_to_s_56` | 19apr23 | explicit | 324→56 |
| S III | `SUL/III/3oct00/f_to_s_127.dat` | 3oct00 | explicit | 380→127 |
| S IV | `SUL/IV/3oct00/f_to_s_69.dat` | 3oct00 | explicit | 194→69 |
| S V | `SUL/V/3oct00/f_to_s_50.dat` | 3oct00 | explicit | 216→50 |
| Ca II | `CA/II/19apr23/f_to_s_43` | 19apr23 | explicit | 77→43 |
| Ca III | `CA/III/10apr99/f_to_s.dat` | 10apr99 | explicit | 232→44 |
| Ca IV | `CA/IV/10apr99/f_to_s.dat` | 10apr99 | explicit | 378→43 |
| Ca V | `CA/V/10apr99/f_to_s.dat` | 10apr99 | explicit | 613→73 |
| Fe II | `FE/II/19apr23/f_to_s_135` | 19apr23 | explicit | 2698→135 |
| Fe III | `FE/III/19apr23/f_to_s_105` | 19apr23 | explicit | 1500→105 |
| Fe IV | `FE/IV/18oct00/f_to_s_63.dat` | 18oct00 | explicit | 1000→63 |
| Fe V | `FE/V/18oct00/f_to_s_45.dat` | 18oct00 | explicit | 1000→45 |
| Fe VI | `FE/VI/18oct00/f_to_s_67.dat` | 18oct00 | explicit | 2000→67 |
| Co II | `COB/II/18oct00/f_to_s_55.dat` | 18oct00 | explicit | 1000→55 |
| Co III | `COB/III/18oct00/f_to_s_52.dat` | 18oct00 | explicit | 1000→52 |
| Co IV | `COB/IV/18oct00/f_to_s_56.dat` | 18oct00 | explicit | 1000→56 |
| Co V | `COB/V/18oct00/f_to_s_43.dat` | 18oct00 | explicit | 1000→43 |
| Co VI | `COB/VI/18oct00/f_to_s_41.dat` | 18oct00 | explicit | 1000→41 |
| Ni II | `NICK/II/18oct00/f_to_s_59.dat` | 18oct00 | explicit | 1000→59 |
| Ni III | `NICK/III/18oct00/f_to_s_47.dat` | 18oct00 | explicit | 1000→47 |
| Ni IV | `NICK/IV/18oct00/f_to_s_54.dat` | 18oct00 | explicit | 1000→54 |
| Ni V | `NICK/V/18oct00/f_to_s_54.dat` | 18oct00 | explicit | 1000→54 |
| Ni VI | `NICK/VI/18oct00/f_to_s_62.dat` | 18oct00 | explicit | 1000→62 |

형식은 총 2종: explicit 26개, implicit row-order 1개입니다. 27개 모두 linked osc와 f_to_s의 선언 FL 수가 정확히 일치했습니다.

## 파서·음성대조

엄격 파서는 [cmfgen_parser.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmfgen_parser.py:501)에 구현했습니다. 행 수, FL 정확히 한 번, SL 연속성, 형식 판별 근거를 강제합니다.

실행 결과:

```text
POSITIVE explicit Si II: 157 FL -> 79 SL PASS
POSITIVE implicit Si IV: 66 FL -> 55 SL PASS
NEGATIVE prior bug: Si IV final zero misread as explicit FL ID: FAIL caught
NEGATIVE explicit FL column shifted final->penultimate: FAIL caught
NEGATIVE mapped FL count != declared FL count: FAIL caught
NEGATIVE FL ID not exactly once: FAIL caught
NEGATIVE SL numbering hole: FAIL caught
R4 f_to_s fixture SELF-CHECK PASS
R4 generator wiring SELF-CHECK PASS: ions=27, levels=21185, every linked FL->SL exact
```

Fixture는 [r4_ftos_fixture.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/r4_ftos_fixture.py)와 [r4_ftos_wiring_fixture.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/r4_ftos_wiring_fixture.py)에 있습니다.

## 자원 게이트

[audit_r4_ftos.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/audit_r4_ftos.py) 실행 결과:

- 논리 SL 행렬 최대 `N=1600`
- `d_matrices = 50 × 1600² × 8 = 1,024,000,000 byte` (`976.562 MiB`)
- 현 CUDA 코드는 FL 기준 `N=4198`을 과대 할당: `7,049,281,600 byte` (`6,722.719 MiB`)
- `_links` 대비 증가: 101 levels, 4,716 lines, 14,148 macro edges
- 식별 GPU 증가: `18,733,528 byte`
- 보수 총량: `110,587,891,696 byte` = `105,464.832 MiB`
- H200 여유: `40,166,928,400 byte` = `38,306.168 MiB`

판정: **H200 FIT — 구현 진행 게이트 PASS**.

## 구현·검증 위치

- 생성 gate `CMFGEN_LINK_FTOS`: [expand_atomic_data_cmfgen.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/expand_atomic_data_cmfgen.py:173)
- 새 덱 driver: [deck_regen_r4_ftos_driver.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/deck_regen_r4_ftos_driver.py:14)
- OFF 대조 driver: [deck_regen_r4_offcontrol_driver.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/deck_regen_r4_offcontrol_driver.py)
- 최종 검증기: [verify_deck_r4_ftos.py](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/verify_deck_r4_ftos.py:76)
- sbatch: [sbatch_deck_r4_ftos.sh](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/sbatch_deck_r4_ftos.sh)
- 전체 기록: [CODEX_R4_FTOS_SUMMARY.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_R4_FTOS_SUMMARY.md)

최종 검증기는 27개 count/membership, R1 전체 게이트, 임시 gate-OFF 덱과 `_links`의 모든 파일 byte identity를 전수 검사합니다. sbatch의 마지막 명령이며 종료 코드가 job 결과입니다.

## 운전석 명령

```bash
cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
sbatch --export=ALL,REPO_ROOT="$PWD" scripts/sbatch_deck_r4_ftos.sh
```

## UNRESOLVED

- 새 덱 생성 및 최종 검증 gate: **NOT RUN**
- 실제 MaxRSS·경과시간: **NOT RUN**
- `$SLURM_TMPDIR`에 OFF 대조 덱용 약 7 GiB 이상 여유 필요
- 목표 `data/tardis_reference_toy06_19p48d_sivcaiv_ftos/`는 현재 생성되지 않았습니다.