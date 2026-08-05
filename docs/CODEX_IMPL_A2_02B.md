# A2-02B fine dump 집계 builder 구현 보고

## 1. 결과

산출물 격차를 `scripts/a2_02_prepare_fine_dump.py` 하나로 메웠다. 이 builder는 새
Lumina/GPU run을 만들지 않고 다음의 기존 자료만 읽는다.

- `/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/EDDFACTOR`와 같은 디렉터리의
  `EDDFACTOR_INFO`, `RVTJ`
- `/gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10`와
  기존 checksum sidecar
- `data/tardis_reference_toy06_19p48d_sivcaiv_ftos/`의 `geometry.csv`, `levels.csv`,
  `ionization_energies.csv`, `cmfgen_sigma_bf.bin`, `line_list.csv`
- `docs/A2_02_FREQUENCY_UNION.json`과 기존 입력 template

출력 NPZ의 key는 template의 `npz_contract`에 적힌 다음 12개로 정확히 고정했다.

```text
nu_edges_hz shell_id j_nu j_state chi_nu eta_nu
bf_kernel bf_shell_id bf_id line_profile line_shell_id line_id
```

`src/`, 덱, `/gpfs`를 변경하지 않았고 commit/push하지 않았다. 이 세션에서는 지시대로
대용량 EDDFACTOR와 실제 CHIETA 캡처를 열지 않았다. 따라서 production NPZ와 manifest는
아래 lageunha 운전석 실행 전까지 `PENDING_DRIVER_EXECUTION`이다.

## 2. 판독과 보존 집계

### 2.1 EDDFACTOR와 셸 매핑

direct-access 판독은 새로 만들지 않았다. builder가
`validation/chain_replay_parity59/common.py`의 `read_info`, `read_eddfactor`,
`parse_rvtj_block`을 import해 사용한다. 그 앞뒤에 A2-00에서 확립한 다음 fail-closed
조건을 붙였다.

```text
ND=90, RECL=728, WORD=8, little-endian
14 header records + NCF 196185 = 196199 records
196199 * 728 = 142832872 bytes
record 5 FINISH = 1.0
유효 frequency payload = (196185, 90)
```

`FL`은 기존 판독 정의대로 `1e15 Hz` 단위이므로 Hz로 바꾸고 오름차순으로 정렬한다.
각 Lumina 셸의 속도는 CHIETA의 `r_edge/t_exp`와 덱 `geometry.csv`의
`(v_inner+v_outer)/2`가 일치하는지 확인한 뒤, 그 속도를 RVTJ depth 속도축에 놓는다.
실제 유효 셸 집합이 정확히 `s0..s43`이 아니면 rc=2로 중단한다. `s44..s49`에는
외삽하지 않는다.

주파수 방향은 EDDFACTOR의 이웃 표본 사이를 선형 함수로 보고 trapezoid 적분한다.
fine edge에는 EDDFACTOR의 모든 합집합 내부 주파수를 포함하므로 각 출력 bin은 native
구간을 건너지 않으며, 출력값은 point sample이 아니라 그 구간의 보존 bin average다.

### 2.2 CHIETA

포맷은 기존 작동 소비기 `scripts/cmf_chieta_check.py`로 확립했다.

```text
magic LCMFCE01, little-endian v1, 64-byte header
nr=50, nnu=1000, iteration=10, field_generation=10
post-damp + coherent-frozen + frequency-descending
arrays: r_edge, nu, dnu, chi_total, chi_coherent,
        eta_fixed, eta_coherent, eta_total, J_producer
```

builder는 sidecar hash와 eta 분해 bitwise identity까지 기존 checker로 먼저 검증하고,
`chi_total`과 `eta_total`을 사용한다. `nu,dnu`에서 log-grid edge를 복원해 center와 width
round trip을 검사한 뒤 piecewise-constant overlap integral로 fine grid에 옮긴다.

CHIETA의 실제 support는 `1.5e14..3e16 Hz`인데 NPZ의 공통 edge는 전 소비자 합집합을
덮어야 한다. `chi_nu/eta_nu`에는 validity 배열이 없으므로 capture support 밖은 0으로
패딩한다. 이것은 측정된 0이라는 주장이 아니며 manifest와 §6 위험에 명시한다.

### 2.3 공통 fine grid

다음 edge의 정렬된 합집합을 사용한다.

1. 소비자 합집합 양 끝점
2. 합집합 안 EDDFACTOR native 주파수 전부
3. CHIETA 1000-bin edge
4. CMFD 1000-bin edge
5. 선택한 BF level threshold

따라서 `F >= 16000`을 기계적으로 검사하면서 EDDFACTOR의 fine 구조, CHIETA/CMFD의
원래 bin 경계, threshold 불연속을 동시에 보존한다. 정확한 F는 운전석 입력을 읽은 뒤
manifest의 `provenance.builder.fine_bins`에 기록된다.

## 3. `j_state` 판정 규칙

숫자 payload가 0이어도 state를 보지 않고 의미를 판단해서는 안 된다.

| code | 이름 | 판정 |
|---:|---|---|
| 1 | `MEASURED` | 셸 속도가 RVTJ 안이고 출력 bin 전체가 EDDFACTOR 주파수 support 안이며, 필요한 두 native frequency endpoint가 모두 유효하다. depth bracket의 J가 양쪽 모두 양수이면 기존 방식대로 velocity-log-J 보간한다. bin average가 양수이거나 frequency 방향에서 exact-zero와 양수를 잇는 유효 구간이면 이 상태다. |
| 2 | `EXACT_ZERO` | 위와 같은 완전 유효 bin이고, 양 끝의 EDDFACTOR 유도값이 모두 bit-exact `0.0`이라 보존 적분도 정확히 0일 때만 사용한다. |
| 3 | `UNSAMPLED` | 셸은 RVTJ 안이지만 bin 일부가 EDDFACTOR 주파수 support 밖이거나, 한 depth endpoint만 0이고 다른 쪽은 양수이거나, 음수/비유한 endpoint 때문에 log-J 보간이 정의되지 않는 경우다. 값을 발명하지 않고 숫자 payload는 0, 의미는 state 3으로 둔다. |
| 4 | `OUT_OF_RANGE` | 셸 midpoint 속도가 RVTJ 속도 범위 밖이다. 해당 셸의 전 주파수 bin에 적용하며 숫자 payload는 0이다. 이번 계약에서는 정확히 `s44..s49`다. |

즉 `EXACT_ZERO`는 EDDFACTOR의 양쪽 depth와 양쪽 frequency endpoint에서 증명된 경우만
나오며, support 부재나 정의되지 않은 log 보간을 0으로 분류하지 않는다.

## 4. BF kernel과 line 표본

### 4.1 BF

`cmfgen_sigma_bf.bin`의 기존 CMFD/v1 layout과 파일 크기를 확인하고
`levels.csv` global row와 1:1로 묶는다. threshold는 같은 `(Z, ion)`의
`ionization_energy_eV - level_energy_eV`를 Hz로 환산한다. 양의 CMFD 값이 threshold
위에 실제로 남는 level만 후보로 인정한다.

기본 표본은 후보 threshold 범위에서 log-Hz로 균등한 24개 anchor에 가장 가까운 서로
다른 level 24개다. 각 level을 `s0,s8,s9,s43`에 기록하므로 기본 BF row는 96개다.
`s0,s8`은 A2-02 판정 가능 영역의 양 끝, `s9`는 판정/기록 경계, `s43`은 RVTJ 유효
영역의 마지막 셸이다. kernel은 각 fine bin에서 다음 양의 bin average이며 threshold
아래는 정확히 0이다.

```text
bf_kernel = bin-average [ 4*pi*sigma(nu)/(h*nu) ]
```

CMFD가 주는 것은 이미 1000-bin 평균 sigma이므로 각 원 bin에서는 sigma를 상수로
보존하고 `1/nu`는 `log(nu_hi/nu_lo)`로 정확 적분한다. ID는
`bf:s{shell}:Z{Z}:i{ion}:l{level}:g{global_row}`로 고정한다.

### 4.2 line

220만여 선을 NPZ에 복제하지 않는다. 기본 표본은 다음 두 층을 합쳐 중복 제거한다.

- 전 line frequency 합집합을 log-Hz로 16등분한 anchor 각각에 가장 가까운 선
- frequency-descending `line_list.csv`의 행 rank를 16분위로 나눈 anchor 선

첫 층은 약 10 decade인 희박한 합집합 끝단까지 덮고, 둘째 층은 UV/optical line forest의
밀집도를 반영한다. 기본 unique line 수는 중복 정도에 따라 16..32개이고, 같은 네 셸에
복제하므로 64..128 row다. 두 번의 streaming pass에서 전체 row 수, 유한 양의 주파수,
descending 순서와 합집합 포함 여부를 확인한다.

현재 A2-02 `Jbar` 소비자는 line-center 평균을 요구하지만 물리 Doppler 폭 입력은 NPZ
계약에 없다. 따라서 각 선택 선은 line center를 포함하는 fine cell에 적분 1인 delta
top-hat으로 보존 투영한다. `sum(line_profile*dnu)==1`을 전 row에서 검사한다. ID는
`Z/ion/lower/upper/catalog line_id/global row/shell`을 모두 포함한다. 이 선택은
line-center 해상도 검사용이며 실제 thermal/turbulent profile 대체물은 아니다.

## 5. manifest

builder는 `docs/A2_02_RESOLUTION_INPUT_TEMPLATE.json`을 읽고 schema와 frequency-ledger
hash를 확인한 뒤 실물 manifest를 쓴다. 기존 template 필드는 유지하며 다음을 채운다.

- `fine_dump.path`, `fine_dump.sha256`
- `provenance.existing_dump_ids`: 요구된 두 dump 절대경로
- `provenance.existing_dump_sha256`: 두 dump 각각의 SHA-256
- `provenance.ancillary_inputs`: EDDFACTOR_INFO, RVTJ, CHIETA sidecar와 실제 소비한 덱
  파일들의 SHA-256
- builder hash, F, 표본 수, profile shell, 4개 `j_state` count와 50개 RVTJ bracket

NPZ와 manifest는 각각 같은 디렉터리의 임시 파일에 완성한 뒤 rename한다. 기존 산출물이
있으면 기본은 실패하고, 운전석 명령처럼 `--force`를 명시한 경우에만 두 지정 산출물을
원자적으로 교체한다.

## 6. 남은 위험

1. **CHIETA 외부 support:** `chi/eta` validity state가 NPZ 계약에 없다. 0 패딩은 특히
   등록 대역 20000–25000 Å 중 capture 밖인 20000–25000 Å를 측정하지 못한다. 격자 간
   비교에서는 같은 0이 재빈되지만 물리적 opacity/emissivity coverage 증명은 아니다.
2. **line profile:** 적분 1 delta top-hat은 line-center Jbar 해상도 표본이다. 실제 셸별
   `T_e`, 원자질량, microturbulence를 포함한 Doppler profile 사다리는 별도 입력 계약이
   생기기 전까지 미해결이다.
3. **표본 통계:** 24 BF threshold와 최대 32 line anchor는 합집합과 밀집 forest를
   결정론적으로 층화하지만 최악의 220만 번째 선/모든 level을 보증하지 않는다.
4. **EDD frequency 의미:** 기존 검증 경로와 같이 EDDFACTOR J를 frequency 표본 사이
   piecewise-linear로 적분한다. direct-access 파일 자체는 별도의 frequency-bin edge를
   선언하지 않는다.
5. **고정 oracle:** record 수와 크기는 A2-00의 정확한 jnu4 oracle에 고정했다. 다른
   정상 CMFGEN run을 `--edd`로 넘겨 범용 변환하는 도구가 아니며 의도적으로 rc=2다.
6. **운전석 자원:** fine grid는 약 196k native frequency를 포함하고 dense float64
   BF/line row를 만든다. 압축 전 working set과 NPZ 작성 시간이 작지 않다. lageunha에서만
   실행하며 로그인 노드에서 시도하지 않는다.

## 7. 운전석 실행 명령

아래 블록은 로그인 노드에서 lageunha로 넘겨 그대로 실행할 수 있다. `/usr/bin/time`을
사용하지 않으며 `/gpfs`와 덱에는 쓰지 않는다.

```bash
ssh lageunha 'bash -s' <<'EOF'
set -euo pipefail
REPO=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
cd "$REPO"

PYTHONPYCACHEPREFIX=/tmp/a2_02_pycache \
  python3 -m py_compile scripts/a2_02_prepare_fine_dump.py
PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/a2_02_prepare_fine_dump.py --self-test

PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02_prepare_fine_dump.py \
  --edd /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/EDDFACTOR \
  --rvtj /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ \
  --chieta /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10 \
  --deck data/tardis_reference_toy06_19p48d_sivcaiv_ftos \
  --ledger docs/A2_02_FREQUENCY_UNION.json \
  --template docs/A2_02_RESOLUTION_INPUT_TEMPLATE.json \
  --output validation/a2_02/a2_02_fine_bin_averages.npz \
  --manifest validation/a2_02/A2_02_RESOLUTION_INPUT.json \
  --force

python3 - <<'PY'
import hashlib
import json
import numpy as np

npz = 'validation/a2_02/a2_02_fine_bin_averages.npz'
manifest_path = 'validation/a2_02/A2_02_RESOLUTION_INPUT.json'
with np.load(npz, allow_pickle=False) as data:
    expected = {
        'nu_edges_hz', 'shell_id', 'j_nu', 'j_state', 'chi_nu', 'eta_nu',
        'bf_kernel', 'bf_shell_id', 'bf_id', 'line_profile',
        'line_shell_id', 'line_id',
    }
    assert set(data.files) == expected
    assert data['j_nu'].shape == data['j_state'].shape
    assert data['j_nu'].shape == data['chi_nu'].shape == data['eta_nu'].shape
    assert data['j_nu'].shape[0] == 50
    assert set(np.unique(data['j_state'])) <= {1, 2, 3, 4}
    assert np.all(data['j_state'][44:] == 4)
hasher = hashlib.sha256()
with open(npz, 'rb') as stream:
    for block in iter(lambda: stream.read(4 << 20), b''):
        hasher.update(block)
digest = hasher.hexdigest()
manifest = json.load(open(manifest_path))
assert manifest['fine_dump']['sha256'] == digest
assert len(manifest['provenance']['existing_dump_sha256']) == 2
print('A2_02_PREPARE_POSTCHECK PASS', digest)
PY
EOF
```

기대 종료코드는 전체 블록 **0**이다. 주요 정상 marker는 다음 세 줄이다.

```text
A2_02_PREPARE_SELFTEST PASS conservative=1 states=4 point_sample=0
A2_02_PREPARE PASS rc=0 ... npz_sha256=<64 hex> ...
A2_02_PREPARE_POSTCHECK PASS <같은 64 hex>
```

실물 산출 경로는 다음과 같다.

```text
/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/a2_02/a2_02_fine_bin_averages.npz
/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/a2_02/A2_02_RESOLUTION_INPUT.json
```

builder 입력/schema/packing 실패의 기대 rc는 **2**이며 이때
`A2_02_PREPARE_FAIL <원인>`을 stderr에 남긴다. 정상 manifest는 바로 다음 사다리
명령의 `--manifest` 입력이다.

```bash
python3 scripts/a2_02_resolution_ladder.py run \
  --manifest validation/a2_02/A2_02_RESOLUTION_INPUT.json \
  --output validation/a2_02/a2_02_resolution_result.json
```

이 마지막 명령의 rc는 측정 결과에 따라 선택 성공이면 0, 최종 8000→16000까지 실패하면
계약대로 3(`BLOCKED`)이며 builder 성공 여부와는 구분한다.

## 8. 이 세션에서 수행한 검증

대용량 production 입력을 열지 않는 범위에서 다음을 실행했고 모두 rc=0이었다.

```text
PYTHONPYCACHEPREFIX=/tmp/a2_02_pycache python3 -m py_compile scripts/a2_02_prepare_fine_dump.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02_prepare_fine_dump.py --self-test
PYTHONDONTWRITEBYTECODE=1 python3 scripts/a2_02_resolution_ladder.py self-test
```

self-test는 CHIETA log edge round trip, piecewise-constant 적분 보존, EDD형
piecewise-linear J 평균, `EXACT_ZERO`와 양쪽 `UNSAMPLED` 분리를 검사한다.
