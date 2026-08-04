# CODEX R3A `g` / R3B E 동등성 재측정 및 처분

작성일: 2026-08-03  
대상: `docs/ATOMIC_EQUIV_PLAN.md`의 R3a·R3b  
선행: R1 job 399943, `_sivcaiv_links` 4-gate PASS

## 결론

R1 판본 정렬이 R3a와 R3b를 함께 닫았다.

- `g` 불일치: **211 → 0준위**. 전수 재대조 0이므로 R3a 완료다.
- E 불일치: **1,119 → 0준위**. 신규 덱 20,749개 활성·존재 준위의 최대
  `|ΔE|=4.033×10⁻⁷ cm⁻¹`로 기존 임계 `1×10⁻⁶ cm⁻¹` 안이다.
- 매핑 비항등: **2 → 0이온**.
- 따라서 R3a/R3b용 추가 덱 생성·sbatch는 만들지 않았다. 이미 존재하는 R1 덱을 다시
  생성하는 것은 금지된 덱 생성의 중복이며, E 반올림 잔차를 고치는 것도 물리적 이득이 없다.
- 부착 커버리지는 별개다. Υ 부재는 `9,860 → 19,962`, σ 부재는 `147 → 148`이다.
  이는 `g`/E 수리를 되돌릴 이유가 아니라 R3c 및 σ 커버리지의 남은 항목이다.

GPU·모델 런·덱 생성·commit은 실행하지 않았다. `src/`, 기존 덱 세 개,
`validation/regression_ledger/`, `scripts/regression_ledger.py`를 수정하지 않았다.

## 고정 판정 기준과 정본

데이터를 읽기 전에 아래 기준을 `scripts/audit_r3_level_equivalence.py:4-10,33-40`에
고정했다. 구·신 덱에 같은 함수 한 번씩을 적용했으며 결과를 보고 임계값을 바꾸지 않았다.

- 기준 CMFGEN 준위: `MODEL_SPEC`의 NF와 `atomic_links.txt`의 `F_OSCDAT`, rank는
  `abs(ID)-1`.
- E: `|E_Lumina-E_CMFGEN| > 1×10⁻⁶ cm⁻¹`이면 불일치. 상대차 분모는
  `max(|E_CMFGEN|, 1 cm⁻¹)`.
- `g`: 정수 정확 일치.
- 매핑: 정확 E+`g` → 정규화 configuration+`g` → rank 보충의 기존 순서. 기존 보고와
  동일하게 부재 suffix는 커버리지이지 매핑 비항등으로 세지 않는다
  (`audit_r3_level_equivalence.py:213-266,352-368`).
- σ: binary `has_cmfgen != 0`이면서 1,000-bin 행에 양의 유한값이 실제로 있어야 한다.
  flag/grid 불일치는 fail-closed다 (`:269-297`).
- Υ: `coldata_cmfgen_manifest.csv`의 `status==OK` binary 및 Fe III 전용 binary에서 해당
  `level_number`가 적어도 한 전이 끝점에 나와야 한다 (`:300-336`).

정본은 두 덱의 `levels.csv`, sigma/collision binary와 manifest, 그리고 CMFGEN run의
`MODEL_SPEC`·`atomic_links.txt`·링크된 osc 파일뿐이다. 과거 문서의 수치는 판정 입력으로
쓰지 않고, 구 덱 재현값과 일치하는지에만 사용했다.

## 1단계 — 구 덱 대 R1 새 덱 재측정

CMFGEN 활성 범위는 두 측정 모두 20,749준위다. 구 덱은 그중 10,986준위가 존재했고,
새 덱은 20,749준위 전부 존재한다.

| 지표 | 구 `_sivcaiv` | 신 `_sivcaiv_links` | R1이 흡수한 구 문제 | 순변화 |
|---|---:|---:|---:|---:|
| `g` 불일치 | 211 | **0** | **211** | −211 |
| E 불일치 | 1,119 | **0** | **1,119** | −1,119 |
| 매핑 비항등 이온 | 2 | **0** | **2** | −2 |
| 존재분 Υ 부재 | 9,860 / 10,986 (89.75%) | **19,962 / 20,749 (96.21%)** | 기존 문제 11 해소 | +10,102 |
| 존재분 σ 부재 | 147 / 10,986 (1.34%) | **148 / 20,749 (0.713%)** | 기존 문제 0 해소 | +1 |

Υ·σ는 분모 확대 때문에 단순 순변화만으로 판본 효과를 말할 수 없다. CMFGEN
`(Z, ion0, rank)`가 구·신 모두 존재하는 공통 준위와 R1 신규 편입 준위를 분리하면 다음과 같다.

| 부착 | 구 문제 해소 | 구 문제 지속 | 공통 준위 신규 부재 | R1 신규 편입 준위 중 부재 | 신 합계 |
|---|---:|---:|---:|---:|---:|
| Υ | 11 | 9,849 | 430 | 9,683 | 19,962 |
| σ | 0 | 147 | 0 | 1 (Si V rank 0) | 148 |

즉 R1은 `g`·E·매핑을 전량 흡수했지만 Υ/σ 커버리지 수리는 아니었다. 특히 링크 판본의
충돌표가 최신 자동선택 판본보다 희소해 공통 준위 Υ 부재가 430개 늘었고, 새로 편입한
9,763준위 중 9,683개에도 tabulated Υ 끝점이 없다. 대체 처방이 CMFGEN과 같은지는 R3c의
질문이므로 여기서 대체값이나 clamp를 넣지 않았다.

## 2단계 — R3a `g` 불일치

### 잔여 전건 목록

**없음 — 0건.** 따라서 이온(0-기반+분광표기), CMFGEN rank, 에너지, 양쪽 `g`, 양쪽
파일:행을 채울 잔여 행 자체가 없다. 측정기는 잔여가 생기면 이 여섯 필드를 전건 출력한다
(`audit_r3_level_equivalence.py:390-400,500-513`). 표본 추출 경로는 없다.

### 에너지 분포와 저준위 판정

신규 덱에는 불일치가 없으므로 잔여 분포는 공집합이며, 저준위도 **0건**이다. R1 전의
211건은 전부 Co II `(Z=27, ion0=1, 분광 Co II)`였고 에너지는 다음과 같았다.

| 통계 | 바닥에서의 E (cm⁻¹) |
|---|---:|
| 최소 | 40,695.600 |
| p50 | 113,870.787 |
| p90 | 121,682.837 |
| p99 | 124,960.883 |
| 최대 | 125,002.392 |

`E ≤ kT=14,753 cm⁻¹`인 구 불일치는 **0/211**, 고준위는 **211/211**이었다. 가장 낮은
것도 `E/kT=2.76`이므로 Boltzmann 가중치가 약 0.063이고, 중앙값은 약 `4.5×10⁻⁴`까지
눌린다. 따라서 구 `g` 결함은 정수 배수 오류라는 성격은 분명하지만 저준위 population을
직격한 사례는 없었다. 현재는 그 고준위 결함까지 모두 없어졌다.

### 원인 3분류

| 원인 | 구 211건 | 신 잔여 | 판정 근거 |
|---|---:|---:|---|
| 파싱 오류 | 0 | 0 | 링크 osc의 순차 ID·E·`g`를 직접 파싱한 새 덱이 전 rank 정확 일치 |
| 준위 병합 규칙 차이 | 0 | 0 | 새 덱 매핑 비항등 0이온, configuration까지 R1 gate 2 항등 |
| 잔여 판본 차이 | **211** | 0 | 구 Co II는 19apr23, CMFGEN 링크와 신 덱은 18oct00 |

파일 증거는 다음과 같다.

- CMFGEN 정본: `toy06_19.48d_jnu4/atomic_links.txt:69`이
  `COB/II/18oct00/coii_osc.dat`를 선택한다.
- 구 덱 provenance: `_sivcaiv/coldata_cmfgen_manifest.csv:48`은
  `COB/II/19apr23/osc_data`다.
- 신 덱 provenance: `_sivcaiv_links/atomic_vintage_manifest.csv:50`은
  `selection_source=links`, `osc_vintage=18oct00`이다.

구 `g` 211건은 구 E 불일치에도 **211/211 같은 준위로 포함**됐다. 다른 이온이나
저준위로 흩어진 파싱 결함이 아니라 Co II 판본/준위 정렬 문제였고, R1이 이를 제거했다.

### 수리 후 전수 확인

CMFGEN 활성 27이온·20,749준위 전수 결과:

```text
mapping nonidentity ions = 0
g mismatch levels        = 0
```

R3a의 완료 조건인 “전수 재대조 0건”을 만족한다.

## 3단계 — R3b E 불일치

### 수리 전 크기 판정

구 1,119개 불일치만의 분포는 다음과 같았다.

| 통계 | `|ΔE|` (cm⁻¹) | 상대차 |
|---|---:|---:|
| p50 | 101.361 | 9.365×10⁻⁴ |
| p90 | 469.100 | 4.440×10⁻³ |
| p99 | 2,804.026 | 2.281×10⁻² |
| 최대 | 33,673.934 | 4.635×10⁻¹ |

- Fe V `(26, ion0=4)` 150건: `|ΔE|=0.0100–0.8000 cm⁻¹`. 물리적으로 작지만 기존
  `1e-6` 기준에는 실패했다.
- Co II `(27, ion0=1)` 969건: 판본/매핑 차이이며, 이 중 **26건이
  `|ΔE|≥1000 cm⁻¹`**였다. ranks는
  `126, 149, 262, 321, 380, 424, 476, 563, 611, 635, 683, 684, 687, 714, 720,
  732, 895, 918, 919, 929, 939, 942, 943, 948, 969, 976`이다. 최대 사례는 rank 149,
  `72,654.320 → 106,328.254 cm⁻¹`, `|ΔE|=33,673.934 cm⁻¹`로 같은 준위의 수치
  반올림이 아니라 다른 준위를 짝지은 경우다.

구 E 불일치는 Fe V·Co II 두 이온에만 몰렸고, 구 `g` 불일치 211건은 전부 Co II의
같은 rank E 불일치와 겹쳤다. 이는 R1 판본 원인과 독립된 수치 정밀도 문제가 아님을
뒷받침한다.

### R1 후 전수 분위수와 물리 판정

신규 덱은 임계 실패가 0이므로 “불일치 집합”의 분위수는 정의되지 않는다. 대신 요청한
물리적 크기 판정을 위해 **전 20,749개 매핑 준위의 남은 저장 반올림차**를 제시한다.

| 통계 | `|ΔE|` (cm⁻¹) | 상대차 |
|---|---:|---:|
| p50 | 2.033×10⁻⁷ | 6.486×10⁻¹³ |
| p90 | 3.631×10⁻⁷ | 2.449×10⁻¹² |
| p99 | 3.998×10⁻⁷ | 7.277×10⁻¹² |
| 최대 | **4.033×10⁻⁷** | **1.359×10⁻⁹** |

- `|ΔE|≥1000 cm⁻¹`: **0건**.
- 최대 Boltzmann 인자 상대변화 상한은 `|ΔE|/(kT/hc)=2.73×10⁻¹¹`.
- 1500 Å에서 최대 `|Δλ|≈9.1×10⁻⁹ Å`; 8 Å bin의 약 `1.1×10⁻⁹`이다.

따라서 잔차는 **무해**하며, 기존 엄격 임계에도 이미 PASS다. `levels.csv`의 10자리 eV
직렬화로 생긴 서브-마이크로 cm⁻¹ 반올림을 더 줄이는 수리는 하지 않는다.

## 구현·자기검사

추가한 `scripts/audit_r3_level_equivalence.py`는 덱 읽기 전용이며, 명시적으로 `--json`이나
`--markdown`을 줄 때만 그 산출 경로를 쓴다. 덱 경로에는 쓰기 코드가 없다.

실행한 전수 명령:

```bash
python3 scripts/audit_r3_level_equivalence.py \
  --json /tmp/r3_audit.json --markdown /tmp/r3_audit.md
```

구 덱 출력이 과거의 `211/1,119/2/9,860/147`을 정확히 재현했고 종료코드 0이었다.

fixture는 정확 rank, 물리 비항등 mapping, configuration fallback+E 실패, 임계 경계의
양성·음성 판별을 검사한다. 실제 출력은 다음과 같았다.

```text
POSITIVE exact rank identity: PASS
POSITIVE physical nonidentity mapping: PASS
POSITIVE normalized-configuration fallback + E mismatch: PASS
NEGATIVE frozen energy threshold discrimination: PASS
FIXTURE VERDICT: PASS
```

`py_compile`도 두 스크립트 모두 PASS했다.

## 수리안과 운전석 명령

R3a/R3b에는 **추가 수리안이 없다**. R1 링크 정본 덱이 이미 `g`·E·mapping을 0으로
만들었고, R3b 잔차는 임계 이하다. 따라서 새 덱 디렉터리나 생성 sbatch를 준비하지 않았다.

운전석이 독립 재현할 명령은 GPU·모델·덱 생성이 아닌 읽기 전용 감사뿐이다. 프로젝트
상설 규약에 따라 grammar-debug에서 실행한다.

```bash
ssh grammar "ssh grammar-debug 'cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn && \
python3 scripts/audit_r3_level_equivalence.py \
--json /gpfs/kjhan/lumina_runner2/slurm/r3_level_audit.json \
> /gpfs/kjhan/lumina_runner2/slurm/r3_level_audit.log'"
```

fixture만 다시 돌릴 때:

```bash
ssh grammar "ssh grammar-debug 'cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn && \
python3 scripts/r3_level_equivalence_fixture.py'"
```

## 남은 UNRESOLVED

1. **R3a `g`: 없음.** 0건으로 완료.
2. **R3b E: 없음.** 임계 실패 0, 남은 반올림은 물리적으로 무해하므로 무수리.
3. **R3c Υ: 19,962/20,749 존재 준위 부재.** CMFGEN의 tabulated-부재 대체 처방과
   Lumina 처방이 같은지 확인하기 전에는 결함/수리 대상으로 단정하지 않는다.
4. **σ: 148/20,749 존재 준위 부재.** 구 147건은 그대로이고 R1 신규 Si V rank 0 한
   건이 더해졌다. R3a/R3b와 독립된 커버리지 항목이다.
5. R2 Co IV Υ 대용 문제는 이 작업 범위 밖이며 변하지 않았다.

