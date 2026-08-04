# 발주서 L1-A-R1 — 층 1 판정 가능화 수리

발주일: 2026-08-04  
발주서 저작: Codex  
검수·실행·대장 반영: 운전석  
구현: Codex  
상태: 구현 전, read-only 저작 산출물

---

## 1. 계약

### 1.1 목적

현 L1-A 계측기를 다음 상태로 만든다.

> 선택한 Lumina 덱과 실제 CMFGEN 런 권위를 같은 역할·단계·좌표에서 대조하고, 양자화·계보·golden·음성 대조 게이트를 모두 통과한 레코드만 층 1 판정 후보로 출력한다.

과학적 `MATCH`를 강제하지 않는다. 정당한 `DIFFER`, `NO-COUNTERPART`, `UNVERIFIABLE`도 유효한 결과다. 실패해야 하는 것은 불일치 자체가 아니라 권위·분모·정밀도·계보·검증 계약의 위반이다.

### 1.2 수리 단위

이 발주서 1건 안의 아래 일곱 계약은 각각 독립 수리·독립 검수한다. 앞 계약의 PASS가 뒤 계약의 과학적 판정으로 합산되어서는 안 된다.

1. 계약 A — authority·비교축·semantic key
2. 계약 B — `lines`: I2 계열·I4·I12·I17
3. 계약 C — `sigma`: I3 계열·I17
4. 계약 D — `collision`: I1·I19
5. 계약 E — C04·양자화·임계
6. 계약 F — golden·완결성·출력 가시성
7. 계약 G — 음성 대조·사전등록 게이트

### 1.3 절대 규율

- `src/` 편집 금지.
- 덱·CMFGEN 입력·기존 수치 수정 금지.
- clamp/floor로 차이를 숨기지 않는다.
- 구 수치와 새 수치가 다르면 양쪽을 함께 기록하고 원인을 보고한다.
- 커밋·푸시·PR·대장 갱신 금지.
- 실행은 운전석만 한다.
- 허용 쓰기 범위는 `scripts/l1a_*.py`, 필요 시 `docs/L1_GOLDEN_MANIFEST.json`이다.
- 과학적 mismatch 때문에 프로세스를 실패시키지 않는다. 계약·사전등록 위반 때문에만 비영 종료한다.

---

## 2. 실측 근거

아래 full-run 결과는 운전석 제공 증거이며, Codex가 이 발주 과정에서 다시 실행한 값이 아니다.

### 2.1 재현 명령형

두 실행은 다음 인자 조합에 해당한다.

```bash
python3 scripts/l1a_instrument.py \
  --deck data/tardis_reference_toy06_19p48d_sivcaiv \
  --cmfgen-tree data/atomic/cmfgen \
  --cmfgen-run /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4 \
  --engine all \
  --chunk-points 1048576 \
  --threshold-mode rel
```

```bash
python3 scripts/l1a_instrument.py \
  --deck data/tardis_reference_toy06_19p48d_sivcaiv_ftos \
  --cmfgen-tree data/atomic/cmfgen \
  --cmfgen-run /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4 \
  --engine all \
  --chunk-points 1048576 \
  --threshold-mode rel
```

관측:

```text
엔진      left.authority        right.authority             두 덱에서
lines     selected Lumina deck  CMFGEN run atomic links     값이 바뀜
sigma     legacy deck           current deck                동일
I4·I12    legacy deck           current deck                동일
collision legacy deck           current CMFGEN-linked deck  동일
```

```text
              구 덱 _sivcaiv          _ftos
I2   A_ul     DIFFER   881,085        MATCH  1,703,064
I2a  Fe IV    DIFFER     4,336        MATCH     72,223
I2b  Ni IV    DIFFER     4,085        MATCH     72,898
I2c  Co IV    DIFFER     3,663        MATCH     69,425
I2d  Fe III   MATCH    136,263        MATCH    136,263
I17(lines)    DIFFER 3,406,111        DIFFER 2,220,953
I3 계열       DIFFER (두 덱 완전 동일)
```

```text
L1A WARNING: C04_THRESHOLD_BELOW_QUANTIZATION
  collision/I1
  lines/I2
  lines/I2a
  lines/I2b
  lines/I2c
  lines/I2d
  lines/I17
```

자원:

```text
peak RSS <= 596 MiB
덱당 wall 52–55초
```

### 2.2 음성 대조 실패

실행된 명령:

```bash
python3 scripts/l1a_fixture.py --negative all
```

출력:

```text
FileNotFoundError: '.../all'
```

근인은 `scripts/l1a_fixture.py:216-224`이다. `--negative`는 enum이 아니라 기존 fixture 루트의 `Path`를 받는다. `all`이라는 디렉터리를 찾으려 했으므로 실패했다.

현 인터페이스의 올바른 사용법은 다음과 같다.

```bash
L1A_FIXTURE_ROOT="$(mktemp -d /tmp/l1a_fixture.XXXXXX)"
python3 scripts/l1a_fixture.py --generate "$L1A_FIXTURE_ROOT"
python3 scripts/l1a_fixture.py --negative "$L1A_FIXTURE_ROOT"
```

`--generate` 없이 임의의 빈 디렉터리를 `--negative`에 주어도 안 된다. 음성 대조는 생성된 다음 네 하위 입력을 전제로 한다.

1. `l1a_fixture`
2. `l1a_fixture_ftos`
3. `cmfgen_tree`
4. `cmfgen_run`

### 2.3 원래 I3 비교축 확정

`docs/CODEX_INPUT_ATOMIC_SUMMARY.md:123-150`에 기록된 원래 I3는 다음 대조다.

- 왼쪽: Lumina가 실제 소비한 `cmfgen_sigma_bf.bin`
- 오른쪽: CMFGEN 런 `atomic_links.txt`가 선택한 `PHOT*_A`
- 좌표: Lumina 1,000개 주파수점
- 직접 비교 가능 CMFGEN 유형: 1, 7, 20–22
- 비교 분모: CMFGEN σ가 양수인 3,953,894점
- 불일치: 1,233,529점
- 유형 2·3·8의 2,084레벨: 별도 `unsupported`

따라서 현재 `sigma`의 구 덱↔현 덱 비교는 원래 I3를 재현하지 않는다. 이는 추정이 아니라 정본 문서와 구현의 직접 충돌이다.

반면 I19는 `docs/OUTSIDE_LOOP_POOL.md:560-576`에서 명시적으로 다음 두 축을 요구한다.

1. CMFGEN 현재 branch에 대한 identity
2. 구→현 `Υ_eff(T)`·`q_ij(T)` physics change

그러므로 collision의 epoch 대조 자체는 옳다. 잘못은 I1과 I19를 분리하지 않고 모든 collision 레코드를 고정 구↔현 대조로 만든 것이다.

---

## 3. 계약 A — authority·비교축·semantic key

### A-1. 근인

- `scripts/l1a_instrument.py:31-37`은 파일명 규칙으로 peer를 자동 추정한다.
- `scripts/l1a_instrument.py:83`은 모든 엔진에 그 peer를 공급한다.
- `scripts/l1a_sigma.py:151`은 `cmfgen_tree`, `cmfgen_run`을 즉시 삭제한다.
- `scripts/l1a_sigma.py:152-167`은 항상 legacy/current 바이너리를 고정한다.
- `scripts/l1a_collision.py:285-295`도 legacy/current를 고정한다.
- `scripts/l1a_lines.py:190-191`, `267-326`은 I4·I12만 legacy/current로 고정한다.
- `scripts/l1a_common.py:399-408`의 이른바 `semantic_level_map`은 실제로 `(Z, ion, level_number)`만 사용한다.
- `scripts/l1a_lines.py:20`, `30-45`, `108-115`도 선 결합을 양쪽의 원시 level rank로 수행한다.

마지막 항목은 구 덱 I2 분모가 과거 엄격결합 880,406이 아니라 881,085로 나온 유력 근인이다. 이는 아직 실행으로 폐합되지 않았으므로 **코드 근거가 있는 추정**으로 둔다. 원래 감사의 level 결합은 configuration·통계중량·에너지 동일성을 요구했다.

### A-2. 수리

모든 레코드에 `comparison_axis`를 추가하고 다음 값만 허용한다.

- `selected_vs_cmfgen`
- `legacy_vs_current`
- `coverage_vs_cmfgen`
- `validator_observation`

`--deck`은 `selected_vs_cmfgen`과 `coverage_vs_cmfgen`의 Lumina 왼쪽을 반드시 결정한다. 두 덱 값이 우연히 같아도 허용하지만, `left.consumed_path`, `left.sha256`, 분모 계산은 선택 덱에서 다시 수행되어야 한다.

epoch 비교가 필요한 I19에만 peer를 사용한다. 파일명 추정 fallback은 제거하고 다음 인자를 명시적으로 받는다.

```text
--epoch-peer <deck>
--super-cutoff 100
```

semantic level key는 다음 네 성분으로 구성한다.

1. 원자번호
2. 이온번호
3. 정규화한 configuration
4. 통계중량과 에너지

에너지는 무근거 반올림으로 결합하지 않는다. 양쪽 원문 정밀도에서 산출한 구간이 겹칠 때만 후보로 삼고, 후보가 둘 이상이면 임의 선택하지 않고 `unsupported` 또는 `ambiguous` 세부값으로 기록한다. 기존 4상태를 유지해야 한다면 `unsupported` 안에 `unsupported_reason=AMBIGUOUS_JOIN`을 둔다.

### A-3. 합격 조건

- selected 축 레코드의 왼쪽은 실제 `--deck`.
- epoch 축 레코드는 I19의 세 metric에만 허용.
- 같은 semantic key의 중복·다중결합은 비영 종료 또는 명시적 unsupported.
- 두 덱 실행에서 경로만 바뀌고 계산을 재사용하는 캐시 금지.
- 정상 데이터가 우연히 같은 값을 낸다는 이유로 실패시키지 않는다.

---

## 4. 계약 B — `lines` 수리

### B-1. I2·I2a–I2d·I17(lines)

현재 실제 CMFGEN 입력을 읽는 큰 방향은 맞다. `scripts/l1a_lines.py:201-218`은 실런이면 `MODEL_SPEC`로 제한한 osc 링크를 오른쪽에 둔다.

결함은 rank를 semantic identity로 오인한 것이다.

수리:

- 선택 덱의 level rank를 configuration·g·energy semantic key로 변환한다.
- CMFGEN `MODEL_SPEC`의 `NF` cap을 적용한 linked osc level에도 같은 key를 만든다.
- 양쪽 semantic level 결합이 확정된 전이에 대해서만 `A_ul`을 비교한다.
- `I17(lines)`는 raw rank union이 아니라 semantic transition union으로 계산한다.
- 과거 880,406과 새 881,085를 어느 쪽도 수정하지 말고 함께 출력한다.
- `golden`에는 기존 방식의 결과와 새 semantic 방식의 결과를 구분해 기록한다.

### B-2. I4

근인:

- `scripts/l1a_lines.py:267-288`은 구·현 `levels.csv:super_level`만 비교한다.
- 원래 I4는 `docs/CODEX_INPUT_ATOMIC_SUMMARY.md:154-186`의 Lumina 런타임 `min(level,100)` 대 CMFGEN linked `F_TO_S`였다.
- 현재 코드는 선택 덱도, `--cmfgen-run`의 `F_TO_S`도 I4 비교량에 사용하지 않는다.

수리:

- 왼쪽은 선택 덱의 semantic level에 `--super-cutoff 100`을 적용한 실제 Lumina 런타임 membership.
- 오른쪽은 `atomic_links.txt`가 선택한 `*_F_TO_S`이며 `MODEL_SPEC NF`로 제한.
- `--super-cutoff`는 하드코딩하지 않고 evidence command와 authority에 기록.
- I4는 예상상 `DESIGN/ACCEPT`일 수 있으나 결과를 코드에 고정하지 않는다.

### B-3. I12

근인:

- `scripts/l1a_lines.py:291-296`은 검증기 명령 문자열만 만든다.
- `scripts/l1a_lines.py:298-300`은 검증기를 실행하지 않고 `exit_code=0`을 기록한다.
- `scripts/l1a_lines.py:319`도 `r1_confirmed_exit=0`을 하드코딩한다.
- `scripts/l1a_lines.py:302-325`은 실제 line-bit 검사를 수행하지 않으면서 metric 이름에 line-bit identity를 포함한다.

이는 무증상 허위 PASS 경로다.

수리:

- I12를 다음 두 레코드로 분리한다.

  1. `I12: level/rank identity (partial)`
  2. `I12: line-bit identity (partial)`

- 실제 선택 덱을 대상으로 R1 검증기의 관련 검사를 실행하거나 같은 알고리즘을 직접 실행한다.
- 외부 검증기를 호출한다면 실제 argv, 실제 stdout SHA, 실제 exit code를 기록한다.
- 검증기 실패를 `exit_code=0`으로 덮지 않는다.
- macro-atom topology는 계속 제외한다.
- `_ftos/verification.log`의 과거 실패를 새 실행 PASS로 바꾸어 쓰지 않는다. 과거는 `STALE`, 새 실행은 별도 evidence다.

---

## 5. 계약 C — `sigma` 수리

### C-1. 근인

- `scripts/l1a_sigma.py:151`의 `del cmfgen_tree, cmfgen_run`이 직접 근인이다.
- `scripts/l1a_sigma.py:152-167`은 구·현 σ 바이너리만 연다.
- `scripts/l1a_sigma.py:157-160`은 raw rank 교집합과 항상 current deck의 HDF5로 unsupported를 분류한다.
- `scripts/l1a_sigma.py:173`은 분모를 `공통 epoch level × 1,000`으로 정의한다. 원래 I3의 `CMFGEN σ>0 비교점` 분모와 다르다.
- `scripts/l1a_sigma.py:226-251`의 I17도 구·현 coverage이지 선택 덱 대 CMFGEN coverage가 아니다.

### C-2. 수리

- 왼쪽: 선택 덱 `cmfgen_sigma_bf.bin`.
- 오른쪽: 실런 `atomic_links.txt`가 선택한 `PHOT*_A`를 `MODEL_SPEC NF`로 제한해 평가한 값.
- CMFGEN tree는 링크 target이 직접 읽히지 않을 때만 byte-identical mirror로 사용한다.
- `scripts/cmfgen_parser.py:242-395`의 PHOT parser를 이용할 수 있다.
- 평가기는 `scripts/expand_atomic_data_cmfgen.py:912-1574`를 참고하되, 생성기 전체를 import해 숨은 전역·환경변수에 의존하지 않는다. 필요한 평가식을 L1-A 쪽의 독립 함수로 고정하고 producer SHA를 기록한다.
- type 1, 7, 20, 21, 22를 지원한다.
- type 2, 3, 8의 2,084레벨은 이번 계약에서 계속 `unsupported`로 둔다. 0으로 대체하지 않는다.
- level 결합은 계약 A의 semantic key를 사용한다.
- I3 분모는 원래 정의대로 “지원 evaluator가 있고 CMFGEN 평가값이 양수인 점”을 별도 기록한다. 전체 선택점, 양쪽 0, missing, unsupported도 함께 보존한다.
- I17(σ)는 선택 덱의 addressable/present level과 CMFGEN linked PHOT coverage의 union으로 계산한다.

### C-3. 표본 의미론

구 바이너리 헤더에는 point sample인지 bin average인지 기록이 없다. 현재 생성기에는 bin-average 구현이 있으나, 이것만으로 구 바이너리 계보를 증명할 수 없다.

따라서 두 대안을 모두 산출한다.

1. geometric bin center 평가
2. bin-average 평가

판정 규칙:

- build attestation이나 독립 재현으로 한 의미론이 확정되면 그것을 primary로 사용.
- 확정되지 않았지만 두 대안 모두 같은 `MATCH/DIFFER`를 내면 결과는 `PROVENANCE` 경고를 가진 강건한 판정 후보.
- 두 대안의 결과가 다르면 `posedness=UNVERIFIABLE`, `disposition=DEFINE`.
- 어느 경우에도 더 잘 맞는 대안을 사후 선택하지 않는다.

---

## 6. 계약 D — `collision` 수리

### D-1. 설계 결론

- I1은 선택 덱 대 CMFGEN 권위 대조가 맞다.
- I19 physics-change는 구 덱 대 현 덱 epoch 대조가 맞다.
- I19 identity는 구 덱과 현 덱을 각각 CMFGEN 현재 권위에 대조해야 한다.

따라서 “collision 전체를 CMFGEN 대조로 바꾼다”도 틀리고, “전체를 epoch 대조로 유지한다”도 틀리다.

### D-2. 근인

- `scripts/l1a_collision.py:285-295`는 모든 레코드 endpoint를 legacy/current로 고정한다.
- `scripts/l1a_collision.py:300`은 선택 덱과 무관하게 current line list만 census한다.
- `scripts/l1a_collision.py:56-87`은 7개 lost ion의 manifest suffix만 링크와 비교한다. 실제 linked col 파일의 내용·SHA를 metric authority로 사용하지 않는다.
- `scripts/l1a_collision.py:82-87`의 `cmfgen_tree` 검사는 디렉터리 존재만 확인한다.
- `scripts/l1a_collision.py:170-180`은 자체 fallback 식을 CMFGEN 실행식처럼 사용하지만 실행 바이너리 attestation은 없다.
- `scripts/l1a_collision.py:356-357`은 `identity_distance_current=0`을 계산하지 않고 고정한다.
- `scripts/l1a_collision.py:369-371`은 retention을 고정 `None`으로 둔다.

### D-3. 수리

collision은 다음 다섯 metric으로 분리한다.

1. `I1: selected branch census versus CMFGEN`
2. `I1: selected Upsilon_eff(T) and q_ij(T) versus CMFGEN`
3. `I19: legacy branch identity versus CMFGEN`
4. `I19: current branch identity versus CMFGEN`
5. `I19: legacy-to-current physics change`

CMFGEN authority는 실제 `atomic_links.txt`의 `*_COL_DATA`와 같은 이온의 linked osc 파일이다.

- col 파일의 tabulated 전이 수와 `OMEGA_SET` 값을 직접 파싱한다.
- 실제 link target과 local mirror가 모두 존재하면 SHA가 같아야 한다. 다르면 실패한다.
- 선택 덱의 manifest가 실런 link와 같다고 가정하지 않고 실제 파일별로 대조한다.
- current distance, legacy distance, retention은 모두 계산값이어야 한다.
- CMFGEN 실행 바이너리 계보가 없으므로 fallback의 **입력 처방 identity**와 **실행 당시 runtime identity**를 구분한다.
- 바이너리 attestation이 없으면 runtime identity는 `UNVERIFIABLE/PROVENANCE`; 입력 처방 비교는 계속 가능하다.
- I19 physics-change는 구→현 대조를 유지하되 semantic transition mapping을 사용한다.

---

## 7. 계약 E — C04·양자화·임계

### E-1. 근인

- 전역 임계는 `docs/L1_GOLDEN_MANIFEST.json:4-9`의 `rel=1e-6` 하나다.
- `scripts/l1a_instrument.py:87`이 이를 모든 metric에 공급한다.
- `scripts/l1a_lines.py:233`, `254-255`는 실측 없이 digits를 5로 고정한다.
- `scripts/l1a_collision.py:307`도 categorical census에 digits=5와 수치 임계를 붙인다.
- `scripts/l1a_common.py:243-250`의 C04는 경고만 내므로 판정 무효 레코드도 정상 종료한다.

### E-2. 일곱 경고 metric의 처분

1. `collision/I1 branch census`  
   수치 양자화 대상이 아니다. branch enum과 정수 count를 exact 비교한다. `quantization.applicable=false`.

2. `lines/I2 A_ul`
3. `lines/I2a Fe IV A_ul`
4. `lines/I2b Ni IV A_ul`
5. `lines/I2c Co IV A_ul`
6. `lines/I2d Fe III A_ul`  
   다섯 metric은 같은 규칙으로 하되 각 선택 universe에서 따로 실측한다.

7. `lines/I17 coverage`  
   집합 membership과 정수 count의 exact 비교다. 상대 임계가 적용되어서는 안 된다. `quantization.applicable=false`.

### E-3. A_ul 양자화 산출법

float로 변환하기 전 원문 숫자 token을 보존한다.

각 token에 대해 다음을 기록한다.

- 표기법
- 유효숫자 수
- exponent
- 마지막 표기 자리의 절대 간격
- IEEE-754 변환 ULP

한 값의 허용구간은 “원문 반올림 반 간격 + float 변환 반 ULP”로 구성한다. 두 값의 허용구간이 겹치면 양자화만으로 설명 가능한 차이다. 겹치지 않으면 두 구간 사이의 실제 간격을 오차로 기록한다.

이는 물리값 clamp/floor가 아니다. 원값은 바꾸지 않고 측정 불확실성 구간만 비교한다.

반드시 별도 산출할 값:

- 왼쪽 token 유효숫자 histogram
- 오른쪽 token 유효숫자 histogram
- pair별 허용구간 overlap 수
- non-overlap 수
- 최대 절대 구간 간격
- 최대 상대 구간 간격
- 기존 `rel>1e-6` mismatch 수

새 판정은 quantization-aware 결과를 사용하고, 기존 `rel>1e-6` 수치는 historical diagnostic으로만 남긴다.

### E-4. C04 게이트

C04는 다음 중 하나가 아니면 판정 자격을 차단한다.

- exact categorical/set metric이며 `quantization.applicable=false`와 사유가 있음
- ULP metric이며 dtype·endianness·distance rule이 완비됨
- abs/rel metric이며 위 방법으로 실측한 quantization rule과 수치가 기록됨

C04 실패 시 해당 레코드는 `judgment_eligible=false`이고 전체 실행은 비영 종료한다. 그러나 정확해가 0차이를 내는 것 자체를 실패시키지 않는다.

---

## 8. 계약 F — golden·완결성·출력

### F-1. 근인

- `scripts/l1a_instrument.py:115`에서 `compare_golden`을 호출하지만 요약표 `:125-135`에는 결과 열이 없다.
- `scripts/l1a_common.py:334-355`는 결과를 `measurements.golden` 깊숙이 넣을 뿐 실패나 요약을 만들지 않는다.
- golden key는 `id:metric`뿐이어서 덱·비교축·authority를 구분하지 않는다.
- `scripts/l1a_common.py:323-330`의 checksum은 `expected_denominator`와 `expected_mismatch`만 묶고 manifest의 command·version·authority를 묶지 않는다.
- golden 미등록 metric은 `scripts/l1a_common.py:337-339`에서 조용히 통과한다.
- 한 엔진 실패 뒤 다른 엔진 레코드는 출력될 수 있어, exit code를 무시하는 소비자가 부분 출력을 PASS로 오인할 수 있다.

### F-2. 수리

golden key는 최소 다음을 묶는다.

```text
specimen_id
comparison_axis
id
metric
authority manifest SHA
threshold/quantization rule version
```

checksum은 다음 필드를 모두 포함한다.

- 정규화 command
- instrument version
- specimen
- comparison axis
- authority SHA
- expected denominator 또는 denominator 산출 규칙
- historical expected mismatch가 있는 경우 그 값
- quantization rule version

모든 판정 레코드에 최상위 `golden`을 둔다.

```json
{
  "status": "MATCH|DIFFER|NOT_REGISTERED|NOT_APPLICABLE",
  "expected_denominator": 0,
  "observed_denominator": 0,
  "expected_historical_mismatch": null,
  "observed_historical_mismatch": null,
  "manifest_key": "",
  "checksum": ""
}
```

stderr 요약에도 `GOLDEN`과 `ELIGIBLE` 열을 추가한다.

규칙:

- 구 덱 I2/I3의 기존 880,406·75,075·3,953,894·1,233,529는 historical diagnostic으로 보존한다.
- 새 semantic/quantization 판정값으로 기존 값을 덮어쓰지 않는다.
- current 덱에는 과학적 mismatch 정답을 사전등록하지 않는다. authority·분모·방법만 사전등록한다.
- historical golden 차이는 양쪽 수치를 출력하고 비영 종료한다.
- 과학적 `DIFFER`는 정상 종료할 수 있다.
- 엔진 하나라도 실패하면 출력된 모든 레코드에 `run_complete=false`, `judgment_eligible=false`.
- 성공한 `--engine all`은 아래 20개 metric key를 정확히 한 번씩 출력해야 한다.

### F-3. 성공 레코드 20개

Collision 5개:

1. I1 selected branch census
2. I1 selected Υ/q
3. I19 legacy identity
4. I19 current identity
5. I19 legacy-to-current physics change

Lines 9개:

6. I2
7. I2a
8. I2b
9. I2c
10. I2d
11. I17 line coverage
12. I4
13. I12 level/rank partial
14. I12 line-bit partial

Sigma 6개:

15. I3
16. I3a
17. I3b
18. I3c
19. I3 evaluator-support census
20. I17 sigma coverage

---

## 9. 계약 G — 사전등록 게이트와 음성 대조

### G-1. 필수 게이트 10개

| 게이트 | 정상 조건 | 결함 주입 |
|---|---|---|
| P01 authority resolution | 실제 link target·mirror·SHA 일관 | collision/PHOT link 하나를 다른 fixture 파일로 교체 |
| P02 comparison axis | selected 축은 `--deck`, epoch 축은 I19만 | selected record endpoint를 peer로 교체 |
| P03 semantic identity | config·g·energy 결합 유일 | 동일 rank에 다른 config를 주입 |
| P04 state exhaustion | present+missing+zero+unsupported=denominator | sigma state 1점 누락 |
| P05 duplicate policy | 다중결합을 임의 선택하지 않음 | 중복 semantic level/line 주입 |
| P06 quantization | metric별 실측 규칙 존재 | A_ul 원문 정밀도보다 엄격한 고정 `1e-6` 주입 |
| P07 golden binding | command·specimen·axis·SHA checksum 일치 | denominator 또는 command 한 글자 변경 |
| P08 validator evidence | 실제 exit/stdout SHA 기록 | R1 fixture를 실패시키고 exit 0 하드코딩 시도 |
| P09 run completeness | 기대 metric set 완비 | sigma 엔진 강제 실패 뒤 lines만 출력 |
| P10 resource | peak RSS ≤ 2³⁰ | fixture record에 2³⁰+1 byte 주입 |

각 주입의 자식 실행은 비영 종료해야 한다. 음성 대조 wrapper 자체는 모든 예상 실패를 관측했을 때만 0으로 끝난다.

### G-2. 기존 C01–C12

기존 열두 제약도 모두 실행한다. C04는 위 계약대로 판정 자격 게이트로 강화한다. C07 epoch mismatch는 다음처럼 구분한다.

- `selected_vs_cmfgen`에서 우발 epoch 혼합: FAIL
- 사전등록된 `legacy_vs_current`: 정상이며 `EPOCH_MIXED`가 아니라 의도된 축으로 기록

음성 대조 출력은 최소 다음 형태로 개별 case를 보여야 한다.

```text
EXPECTED_FAIL P01 authority_resolution child_exit=1
EXPECTED_FAIL P02 comparison_axis child_exit=1
...
EXPECTED_FAIL P10 resource_limit child_exit=1
EXPECTED_FAIL C01 ...
...
EXPECTED_FAIL C12 ...
NEGATIVE lines damaged-semantic-key PASS
NEGATIVE sigma states-denominator PASS
NEGATIVE collision unsupported-not-missing PASS
NEGATIVE SUITE PASS
```

`12/12 PASS` 한 줄만 출력하는 것은 불충분하다. 어느 주입이 실제로 실패했는지 열거해야 한다.

---

## 10. 범위 밖

이번 발주에서 고치지 않는 항목은 다음 열두 건이다.

1. `src/`의 원자 로더·솔버·물리 코드
2. 원자데이터 자체의 수리
3. 구 덱 또는 `_ftos` 재생성
4. CMFGEN 재빌드·재실행
5. I5 DR/RR·Milne
6. I6 모델 덱·공간 범위
7. I7 수송/continuum 격자
8. I8 광도·경계조건 정의
9. I9 clamp·반복·damping
10. I12 macro-atom topology
11. I15 build attestation 복원
12. PHOT type 2·3·8 evaluator 구현

이유는 각각 계측기 authority 수리와 다른 계약이거나, 원장·솔버·빌드 계보를 건드리기 때문이다. 특히 type 2·3·8은 0이나 근사식으로 채우지 않고 `unsupported`로 남기는 편이 정확하다.

---

## 11. 운전석 실행 명령

저장소 루트에서 실행한다.

### 11.1 문법

```bash
python3 -m py_compile \
  scripts/l1a_common.py \
  scripts/l1a_instrument.py \
  scripts/l1a_lines.py \
  scripts/l1a_sigma.py \
  scripts/l1a_collision.py \
  scripts/l1a_fixture.py
```

### 11.2 fixture 생성·음성 대조

```bash
L1A_FIXTURE_ROOT="$(mktemp -d /tmp/l1a_fixture.XXXXXX)"
python3 scripts/l1a_fixture.py --generate "$L1A_FIXTURE_ROOT"
python3 scripts/l1a_fixture.py --negative "$L1A_FIXTURE_ROOT"
```

합격:

- 두 명령 모두 최종 exit 0.
- negative 내부에 P01–P10과 C01–C12의 `EXPECTED_FAIL`이 개별 열거됨.
- lines·sigma·collision 세 엔진 음성 대조가 각각 표시됨.

### 11.3 구 덱 full-run

```bash
L1A_OUT="$(mktemp -d /tmp/l1a_layer1.XXXXXX)"
L1A_CMF_TREE="data/atomic/cmfgen"
L1A_CMF_RUN="/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4"
L1A_OLD="data/tardis_reference_toy06_19p48d_sivcaiv"
L1A_NEW="data/tardis_reference_toy06_19p48d_sivcaiv_ftos"

python3 scripts/l1a_instrument.py \
  --deck "$L1A_OLD" \
  --epoch-peer "$L1A_NEW" \
  --cmfgen-tree "$L1A_CMF_TREE" \
  --cmfgen-run "$L1A_CMF_RUN" \
  --super-cutoff 100 \
  --engine all \
  --chunk-points 1048576 \
  --threshold-mode rel \
  >"$L1A_OUT/old.jsonl" \
  2>"$L1A_OUT/old.stderr"
```

### 11.4 `_ftos` full-run

```bash
python3 scripts/l1a_instrument.py \
  --deck "$L1A_NEW" \
  --epoch-peer "$L1A_OLD" \
  --cmfgen-tree "$L1A_CMF_TREE" \
  --cmfgen-run "$L1A_CMF_RUN" \
  --super-cutoff 100 \
  --engine all \
  --chunk-points 1048576 \
  --threshold-mode rel \
  >"$L1A_OUT/ftos.jsonl" \
  2>"$L1A_OUT/ftos.stderr"
```

### 11.5 실제 R1 검증기 독립 대조

```bash
python3 scripts/verify_deck_r1_vintage.py \
  --base "$L1A_OLD" \
  --fullcov data/tardis_reference_toy06_19p48d_sivcaiv_fullcov \
  --new "$L1A_NEW" \
  --cmf-run "$L1A_CMF_RUN"
```

### 11.6 R4 독립 대조

```bash
python3 scripts/verify_deck_r4_ftos.py \
  --new "$L1A_NEW" \
  --links "$L1A_CMF_RUN/atomic_links.txt" \
  --cmf-run "$L1A_CMF_RUN" \
  --links-deck data/tardis_reference_toy06_19p48d_sivcaiv_links \
  --off-control /gpfs/kjhan/lumina_runner2/scratch/r4tmp/r4_ftos_offcontrol
```

### 11.7 출력 완결성 검사

```bash
python3 - "$L1A_OUT/old.jsonl" "$L1A_OUT/ftos.jsonl" <<'PY'
import json
import sys

for name in sys.argv[1:]:
    rows = [json.loads(line) for line in open(name) if line.strip()]
    assert len(rows) == 20, (name, len(rows))
    keys = [(r["id"], r["metric"]) for r in rows]
    assert len(set(keys)) == 20, (name, "duplicate metric key")
    assert all(r["run_complete"] for r in rows), name
    assert all(r["golden"]["status"] != "NOT_REGISTERED" for r in rows), name
    assert all(r["judgment_eligible"] for r in rows), name
    assert not any(
        "C04_THRESHOLD_BELOW_QUANTIZATION" in r.get("schema_flags", [])
        for r in rows
    ), name
    assert max(r["resources"]["peak_rss_bytes"] for r in rows) <= 2**30, name
    print(name, "PASS", len(rows))
PY
```

관측 mismatch 수가 기존과 다르다는 이유만으로 파일을 고치지 않는다. 위 검사에서 `judgment_eligible`이 거짓이면 그 사유와 양쪽 수치를 보고한다.

---

## 12. 운전석 검수 항목

운전석은 다음을 반박 대상으로 삼는다.

1. 원래 I3가 정말 덱↔CMFGEN PHOT였는지 `docs/CODEX_INPUT_ATOMIC_SUMMARY.md:123-150` 외의 원 산출물로 재확인할 것.
2. I19만 epoch 비교라는 결론이 `docs/OUTSIDE_LOOP_POOL.md:560-576`과 일치하는지 확인할 것.
3. I2의 880,406↔881,085 차이가 rank 결합 때문이라는 추정을 실제 결합 census로 폐합할 것.
4. I4 왼쪽이 저장된 `super_level`이 아니라 런타임 `min(level,K)`여야 한다는 근거를 실행환경과 대조할 것.
5. I12가 실제 검증기를 실행하며, 하드코딩 exit 0이 완전히 제거됐는지 확인할 것.
6. sigma primary sampling을 “더 잘 맞는 쪽”으로 사후 선택하지 않았는지 확인할 것.
7. collision이 실제 linked col/osc를 읽는지, manifest suffix만 확인하지 않는지 확인할 것.
8. categorical·coverage metric의 C04 면제가 단순 우회가 아니라 exact set/count 비교인지 확인할 것.
9. current golden에 과학적 mismatch 정답을 몰래 사전등록하지 않았는지 확인할 것.
10. 한 엔진 실패 시 남은 부분 JSONL이 판정 가능 상태로 남지 않는지 확인할 것.
11. 정상덱의 과학적 `DIFFER`를 프로세스 실패로 만드는 게이트가 없는지 확인할 것.
12. 수정 파일이 허용 범위 안이며 `src/`, 덱, CMFGEN 런을 건드리지 않았는지 확인할 것.

### 놓칠 수 있는 무증상 실패 경로 12개

1. `atomic_links.txt` 판독 뒤 실제 symlink가 바뀌는 TOCTOU.
2. 절대 link target과 `cmfgen-tree` mirror가 서로 다른데 한쪽만 hash하는 경로.
3. `MODEL_SPEC NF` cap을 osc에는 적용하고 PHOT·F_TO_S·collision에는 누락하는 경로.
4. configuration 정규화가 서로 다른 level을 같은 것으로 합치는 경로.
5. 동일 config·g·energy의 다중 후보를 첫 행으로 조용히 선택하는 경로.
6. PHOT의 여러 route·final state 중 첫 route만 사용하는 경로.
7. point sample과 bin average 중 작은 mismatch 쪽을 자동 선택하는 경로.
8. 구 σ 바이너리의 생성 계보가 없는데 현재 생성기 의미론을 소급 적용하는 경로.
9. collision fallback 재구현이 CMFGEN 실행 바이너리와 달라도 input identity를 runtime identity로 보고하는 경로.
10. A_ul 원문 token을 float로 먼저 바꿔 유효숫자·후행 0 정보를 잃는 경로.
11. 부분 엔진 출력만 읽고 shell exit code를 무시하는 소비 경로.
12. golden key가 specimen·axis·authority를 묶지 않아 반대 덱 결과와 우연히 맞는 경로.

이 열두 경로 중 하나라도 미폐합이면 운전석은 발주 이행을 반려해야 한다.

---

## 13. 최종 인수 조건

다음이 모두 성립해야 L1-A를 “층 1 판정을 낼 수 있는 상태”로 인수한다.

- P01–P10 결함 주입이 각각 내부 비영 종료를 시연한다.
- C01–C12가 개별적으로 시연된다.
- 올바른 `--negative <생성된 fixture root>` 명령이 최종 exit 0이다.
- 구 덱과 `_ftos` 실행이 각각 20개 metric을 중복 없이 출력한다.
- selected 축 authority가 실제 `--deck`에 반응한다.
- I19만 명시적 epoch 축을 유지한다.
- I2/I3 historical golden 결과가 레코드와 요약에 보인다.
- C04 경고 7개가 사라지되 수치 clamp/floor는 없다.
- 모든 판정 레코드가 `run_complete=true`, `judgment_eligible=true`.
- peak RSS가 `2^30` bytes 이하이다.
- 기존 수치와 다른 값은 수정하지 않고 병기된다.
- `src/`, 덱, CMFGEN 입력은 무변경이다.
- 인수 뒤 실제 층 1 판정과 대장 반영은 운전석이 별도 수행한다.
---

## 14. 운전석 검수 결과 (2026-08-04)

**판정: 조건부 수용.** §11 실행 명령 1건 반박, 나머지 실측 확인.

| 검수 항목 | 판정 | 실측 |
|---|---|---|
| B-3 I12 허위 PASS | **확인** | `l1a_lines.py` 가 `r1_command` 문자열만 만들고 `evidence(..., exit_code=0)`·`"r1_confirmed_exit": 0` 기록. 검증기 미실행, line-bit 검사 부재 |
| D-2 I19 identity 공허 | **확인** | `identity_distance_current = 0` 리터럴 → `outcome="MATCH" if ... == 0` 이므로 **구성상 항상 MATCH**. `authoritative_tabulated_retention: None` 고정 |
| 2.3 원래 I3 비교축 | **확인** | `CODEX_INPUT_ATOMIC_SUMMARY.md` I3 절: Lumina `cmfgen_sigma_bf.bin` 대 CMFGEN `PHOT*_A` 를 Lumina 1,000 중심주파수에 평가. "CMFGEN σ>0 비교점 3,953,894 / 상대차>1e-6 1,233,529점". **덱-대-덱 아님. 문서-구현 충돌 확정** |
| 2.2 fixture 사용법 | **확인** | `--negative` 는 `Path` + `resolve(strict=True)`. `--generate` 후 같은 root 를 넘겨야 함 |
| 음성대조 실물성 | **확인(추가)** | `negative()` 는 실제로 레코드를 변조하고 `ContractError` 코드 접두사를 검사하며 "픽스처가 결함을 표현하는지" 사전 확인도 있다. **단 `validate_record` 만 시험한다** — 합성 레코드에 가드를 걸 뿐 실제 엔진을 결함 덱에 돌리지 않는다. §G-1 P01–P10 이 이 구멍을 메우는 것이 맞다 |
| 11.6 R4 off-control 경로 | **확인** | `/gpfs/kjhan/lumina_runner2/scratch/r4tmp/r4_ftos_offcontrol` 실재 (mtime 08-03 18:23) |
| 11.5 R1 인자 | **정정 불요** | `--base --fullcov --new --cmf-run` 전부 default 보유. 발주서의 4인자 명시형이 운전석의 2인자형보다 낫다 |
| metric 20개 | **확인** | 열거: collision 5(I1 census · I1 Υ/q · I19 legacy id · I19 current id · I19 physics) + lines 9(I2·I2a·I2b·I2c·I2d·I17line·I4·I12 rank·I12 bit) + sigma 6(I3·I3a·I3b·I3c·I3 evaluator·I17 sigma) = **20** |
| 게이트 P01–P10 | **확인** | §G-1 표 10행 |
| **§11.3·11.4 `/usr/bin/time -v -o`** | **★반박** | grammar-debug 실측: `ls: cannot access '/usr/bin/time': No such file or directory`. 셸 builtin `time` 은 `-v -o` 미지원. **실행 즉시 실패** |

### 운전석 정정 (실행은 운전석 담당이므로 발주서를 반려하지 않고 직접 수정)

`/usr/bin/time -v -o` 를 쓰지 않는다. 자원은 계측기 자신의 `resources` 필드
(`peak_rss_bytes`·`wall_seconds`)와 `resource` 모듈로 측정한다. 러너는
운전석이 `~/.lumina_scratch/` 에 작성해 grammar-debug 에서 실행한다.

### 운전석 추가 지적 (결함 아님, 정직성 요구)

§10 이 **I15 build attestation 복원**을 범위 밖으로 두는데, §C-3(표본 의미론)과
§D-3(collision runtime identity)의 판정이 그것에 의존한다
("build attestation이나 독립 재현으로 한 의미론이 확정되면 그것을 primary").
⟹ 해당 metric 은 `posedness=UNVERIFIABLE`, `kind=PROVENANCE` 로 착지할 것이
**예상된다. 이를 실패가 아니라 예상 결과로 사전등록하라.**
