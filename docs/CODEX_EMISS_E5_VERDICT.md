# E5 정식화 관문 최종 판정 — A/B/B2 3-lane

판정일: 2026-08-02 (Asia/Seoul)  
입력: `/gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10.{A,B,B2}`  
판정기: `scripts/emiss_ab_insitu_e4.py` (E5 3-lane consumer)  
출력: `validation/emiss_e5/`

## 0. 최종 판정

**관문 상태: UNRESOLVED (fail-closed).**

- 입력 인증과 B2 통제 정책은 **PASS**다.
- A deterministic solve는 **PASS**했고, 사전등록 기준 UV 과잉은
  `BALL(600–3000 Å) = 11.70379136 × CMFGEN`으로 재현됐다.
- B는 정확히 `1208.743248 Å`에서 기존 certified-negative 트립을 재현했다.
- B2도 같은 frequency/ray/segment에서 사실상 같은 certified-negative 값으로
  트립했다. 따라서 B2의 6대역 `J_det`와 두 이온의 Gamma는 생성되지 않았다.
- 그러므로 사전등록한 **붕괴 / 유지 / 부분 붕괴** 중 어느 분기도 실행할 수 없다.
  정식화가 UV 과잉의 진범이라고 확정할 수도, 정식화 좌표를 기각하고
  profile/EPAY로 이동할 수도 없다.
- 단, 별도 하위 진단은 명확하다. 미정의분을 A값으로 유지한 B2에서도 B와 같은 트립이
  남았으므로, **“미정의분 삭제가 1208.7 Å 트립 원인” 가설은 반증**된다. 트립은 B와
  B2에 공통인 **covered 집합의 `A_ul*n_u` 정식화 교체**에 묶인다. 이는 트립 원인
  판정이며, B2 UV 대역 평균의 붕괴/유지 판정을 대신하지 않는다.

clamp는 추가하지 않았고 신규 LUMINA/CMFGEN 런도 수행하지 않았다. 어떤 명령도 1분
제한에 걸리지 않았다. 커밋하지 않았다.

## 1. 입력 인증

### 1.1 파일과 SHA-256

세 payload는 모두 정확히 2,416,472 bytes이며 각 실제 SHA-256은 manifest의
`sha256`과 일치한다.

| lane | payload SHA-256 | manifest SHA-256 |
|---|---|---|
| A | `ac62eae5bec6d6beaf06d513c2cef38386b365f6b7826cff61c4edc0f2a34011` | `63e935fb6367a5114574b9a9e1e327985cf6b75240f875142b684f5ed099fe40` |
| B | `95b87203673b69e6f3ef756361e11d8f9fb5d0a5e174be7d1762f9f31d184582` | `5fe7a4b832779c15270a2db02e48d5b96f3b6f3934e04f31f2228ffe5c5d55fe` |
| B2 | `775c9e844b4000551caba7061be79ea8a7b4e25173139910ad447fc4b6689f5a` | `00d2d10666ce08198585530d1c55c68c0f01bba1bcda0e3bd6c1c8f0f437d88b` |

세 `undefined.csv`도 byte-identical하며 SHA-256은 모두
`7995daceb31543a8e9f6e0759a62bef5dde5d5a1175e1bc7253cd6f3d4d06a95`다.
각 CSV는 header 외 40,708 rows다.

### 1.2 epoch/state 동일성

| 항목 | A | B | B2 |
|---|---:|---:|---:|
| iteration | 10 | 10 | 10 |
| field_generation | 10 | 10 | 10 |
| post_damping | true | true | true |
| seeded defect hits | 0 | 0 | 0 |
| common assembly-state SHA-256 | `302a64e2...bf4b044` | 동일 | 동일 |

공통 state의 전체 값은
`302a64e2e394d74a9bbbb92d26a0eb009dacda88145b10e42d3f0b280bf4b044`다.
판정기의 bitwise 검사에서 `r_edge`, `nu`, `dnu`, `chi_total`, `chi_es`,
`J_producer` 및 header epoch/dimension/flags가 동일했다. 세 payload 모두 독립
`cmf_chieta_check.py` 검사에서 다음과 같이 PASS했다.

```text
PASS: iteration=10 field_generation=10 post_damp=1 bytes=2416472
PASS: iteration=10 field_generation=10 post_damp=1 bytes=2416472
PASS: iteration=10 field_generation=10 post_damp=1 bytes=2416472
```

### 1.3 B2 통제 정책과 커버리지

B는 `zero-undefined-fail-closed`, B2는
`retain-production-A-explicit-controlled`이고 B2의 `controlled_retention=true`다.
B2 manifest에 유지 전이 수, 유지 line-shell 수, 유지 A-reference power와 기여율이
모두 명시되어 있다.

| 장부 | 수/값 | 비율 |
|---|---:|---:|
| active transitions | 1,681,176 | 100% |
| defined transitions | 1,640,468 | 97.57859974% |
| undefined transitions | 40,708 | **2.42140026%** |
| B2 retained transitions | 40,708 | undefined와 동일 |
| active line-shell cells | 19,246,925 | 100% |
| undefined line-shell cells | 501,462 | 2.60541359% |
| B2 retained line-shell cells | 501,462 | undefined와 동일 |
| A-reference line power | `1.1328513709898634e-3` | 100% |
| undefined/retained A-reference power | `6.7288871480912374e-5` | **5.93977932%** |
| defined A-reference contribution | `1.0655624995089480e-3` | 94.06022068% |

`undefined_a_reference_diagnostic`의 epoch은 `pre-EPAY`다. `by_band[1000]`와
`by_shell[50]`의 합은 각각 위 undefined power에 `1e-12` 상대 허용오차 안에서 닫힌다.

**판정 유효 범위:** B2는 정의된 97.5786% 전이, 즉 A-reference line power의
94.0602%에서만 정식화를 바꾼다. 미정의 2.4214% 전이/5.9398% 기여는 A 생산값을
유지한 통제 집합이므로 새 정식화의 유효성 주장을 받지 않는다. 따라서 B2가 완주했다면
covered 정식화 좌표를 이 범위에서 판정할 수 있었지만, 실제로는 solve guard가 먼저
중단되어 `J_det` 수준의 판정 유효 범위는 A lane에 한정된다.

참고로 pre-EPAY undefined A power의 대역별 투영은 다음과 같다. BALL 밖에도
49.9040%가 있으므로 BALL 열은 전체 undefined power의 절반만 포함한다.

| 대역 [Å] | undefined A power | 전체 undefined 중 | 전체 A line power 중 |
|---|---:|---:|---:|
| 600–1000 | `2.7296535893866156e-5` | 40.56619660% | 2.40954256% |
| 1000–1500 | `6.3499410198788312e-6` | 9.43683685% | 0.56052728% |
| 1500–2000 | `1.5637333458024861e-8` | 0.02323911% | 0.00138035% |
| 2000–2500 | `6.0400462733359145e-9` | 0.00897629% | 0.00053317% |
| 2500–3000 | `4.0848810409069913e-8` | 0.06070664% | 0.00360584% |
| 600–3000 | `3.3709003103885420e-5` | 50.09595549% | 2.97558921% |

## 2. 3-lane deterministic 판정

판정 조건은 shell 8, `nmu=16`, `T_inner=10020 K`, `bb_scale=1`이다. A는 clamp,
solution-negative-excess, sign-uncertain, nonfinite가 모두 0이고 transport residual은
`9.4203400445093635e-7`이었다.

### 2.1 600–3000 Å 6대역표

| 대역 [Å] | A/CMFGEN | B2/CMFGEN | B2/A |
|---|---:|---:|---:|
| 600–1000 | 33.74128011 | UNRESOLVED | UNRESOLVED |
| 1000–1500 | 32.19137871 | UNRESOLVED | UNRESOLVED |
| 1500–2000 | 7.38976221 | UNRESOLVED | UNRESOLVED |
| 2000–2500 | 6.86442991 | UNRESOLVED | UNRESOLVED |
| 2500–3000 | 15.58062427 | UNRESOLVED | UNRESOLVED |
| **600–3000** | **11.70379136** | **UNRESOLVED** | **UNRESOLVED** |

B도 같은 guard 중단 때문에 전 대역이 UNRESOLVED다. `UNRESOLVED`는 0이나 결측값을
물리 수치로 해석한 것이 아니라, 해당 lane에 완성된 `J_det`가 없다는 뜻이다.

### 2.2 Gamma 3자 비교

| target | matrix index | Gamma_A | Gamma_B2 | Gamma_CMFGEN | A/CMFGEN | B2/CMFGEN | B2/A |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fe III C48 lump | 201 | 685.69281641 | UNRESOLVED | 28.07174861 | 24.42643762 | UNRESOLVED | UNRESOLVED |
| S II SL4 | 4 | 17.25864971 | UNRESOLVED | 0.4748076950 | 36.34871527 | UNRESOLVED | UNRESOLVED |

B Gamma도 두 target 모두 UNRESOLVED다.

### 2.3 B/B2 negative trip

frequency index 470은 `nu=2.4801996493363745e15 Hz`, 즉
`lambda=1208.743248069627 Å`다.

```text
B : radial=0 frequency=470 ray=9 segment=44 substep=0
    value=-0.906355779526826
    interval=[-0.90635577952687785,-0.90635577952677415]
    scale=23.467292540518883 h=0.02 B_trunc=0.26101653388714929

B2: radial=0 frequency=470 ray=9 segment=44 substep=0
    value=-0.90635577952681956
    interval=[-0.90635577952687141,-0.90635577952676771]
    scale=23.467292540518883 h=0.02 B_trunc=0.26101653388714929
```

두 값의 차이는 약 `6.4e-15`에 불과하다. 이 빈에서 post-EPAY `eta_fixed`의 B2−B
최대 절대차는 `2.3621650945937978e-26`인 반면 A−B의 최대 절대차는
`4.7095022617963364e-12`다. 또한 이 빈의 pre-EPAY undefined A power는
`3.7821044624646972e-10`, 전체 A line power의 `0.000033385708%`뿐이다.

전체 600–3000 Å post-EPAY `eta_fixed*dnu` 장부도 B2/B=`1.0000000147`인 반면
B2/A=`603824.712632`다. 이는 transport 결과가 아니라 입력 source 진단이므로
`J_det` 대용으로 쓰지 않는다. 다만 B2의 통제 유지량이 트립 빈에서 B를 의미 있게
변경하지 못했고, 두 lane 공통의 covered 정식화 변화가 트립을 지배한다는 진단과
일치한다.

## 3. 사전등록 판독

사전등록 분기는 B2의 완성된 BALL `J_det`가 필요하다.

| 사전등록 분기 | 필요한 관측 | 이번 상태 |
|---|---|---|
| 정식화가 진범 확정 | B2의 11.70x 과잉이 실질 붕괴 | 판정 불가 |
| 정식화 좌표 기각 | B2에서 과잉 유지 | 판정 불가 |
| 대역·이온별 부분 판정 | B2에서 부분 붕괴 | 판정 불가 |

따라서 정식화 관문의 최종 과학 판정은 **UNRESOLVED**다. B2가 clamp 없이 완주하지
않는 한 이 표를 억지로 채울 수 없다.

한편 별도 트립 원인 판정은 다음과 같다.

1. B의 1208.743248 Å 트립은 재현됐다.
2. 미정의분을 유지한 B2도 같은 트립을 냈다.
3. 따라서 “미정의분 삭제가 트립 원인”은 확증이 아니라 **반증**이다.
4. 트립을 발생시키는 intervention은 covered 전이에 공통인 정식화 교체다.

## 4. 잔여 과잉과 다음 표적

“붕괴 확정 시” 조건이 성립하지 않았으므로 B2 잔여 과잉과 잔여 이온 표적은 지정하지
않는다. A 기준에서 큰 과잉이 남은 위치는 B0=33.7413x, B1=32.1914x,
B4=15.5806x이고 Fe III idx201/S II SL4 Gamma가 각각 24.4264x/36.3487x지만,
이 값들은 B2 개입 후의 잔여가 아니라 A baseline이다. 이를 profile/EPAY 이동의 근거로
오인해서는 안 된다.

## 5. 판독 grammar 사건

첫 판정 시도는 3.1초 안에 A sidecar grammar에서 실패했다.

```text
sidecar failed closed contract validation: ...emiss_ab_iter10.A.manifest.json
frozen load failed: LCMF_ESCHEMA
```

원인은 payload나 manifest 내용이 아니라 E4 driver의 JSON 크기 상한이었다. E5
manifest는 pre-EPAY `by_band`/`by_shell` 배열 때문에 약 23.6 KiB인데 driver 상한이
16,384 bytes였다. `scripts/stage31_cmf_field_driver.c`의 sidecar 상한만 1,048,576 bytes로
확장했다. payload, 물리식, guard, clamp에는 변경이 없다. 재실행은 13.5초에 partial
verdict를 정상 기록했다. 어느 단계도 60초 timeout에 닿지 않았다.

## 6. 재현 명령

### 6.1 파일 실재, 크기, SHA-256

```bash
find /gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766 \
  -maxdepth 1 -type f -printf '%f %s bytes\n' | sort

timeout 60s sha256sum \
  /gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10.A \
  /gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10.B \
  /gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10.B2 \
  /gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10.A.undefined.csv \
  /gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10.B.undefined.csv \
  /gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10.B2.undefined.csv

timeout 60s sha256sum \
  /gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10.A.manifest.json \
  /gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10.B.manifest.json \
  /gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10.B2.manifest.json
```

### 6.2 manifest와 독립 payload 검사

```bash
for lane in A B B2; do
  jq -r '[.emiss_ab_lane,.iteration,.field_generation,
    .common_assembly_state_sha256,.sha256,.controlled_retention,
    .undefined_transition_policy,.coverage.undefined_transition_count,
    .coverage.retained_transition_count,.coverage.undefined_line_shell_count,
    .coverage.retained_line_shell_count,
    .coverage.a_reference_undefined_contribution_fraction,
    .coverage.a_reference_retained_contribution_fraction] | @tsv' \
    /gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10.${lane}.manifest.json
done

timeout 60s bash -c '
for lane in A B B2; do
  python3 scripts/cmf_chieta_check.py \
    /gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10.${lane}
done'
```

### 6.3 3-lane 판정

최초 grammar 실패와 수정 후 재실행에 사용한 명령은 동일하다.

```bash
timeout 60s python3 scripts/emiss_ab_insitu_e4.py \
  /gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10 \
  --out-dir validation/emiss_e5
```

판정기가 내부에서 컴파일하는 driver 명령과 lane별 실행 형태는 다음과 같다.

```bash
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -Werror \
  -D_POSIX_C_SOURCE=200809L -Isrc \
  scripts/stage31_cmf_field_driver.c src/lumina_cmf_field.c \
  -lm -o /tmp/stage31_cmf_field_driver_e4

/tmp/stage31_cmf_field_driver_e4 PAYLOAD PAYLOAD.manifest.json \
  8 16 10020.0 1.0 OUTPUT.tsv
```

### 6.4 index/커버리지/source 진단

```bash
timeout 60s python3 - <<'PY'
from pathlib import Path
import csv, sys
import numpy as np
sys.path.insert(0, 'scripts')
from cmf_chieta_check import check_artifact

base = Path('/gpfs/kjhan/lumina_runner2/scratch/emiss_ab2_capture_188766/emiss_ab_iter10')
arts = {k: check_artifact(Path(str(base) + '.' + k)) for k in ('A','B','B2')}
a = arts['A']
nu = np.asarray(a.arrays[1])
dnu = np.asarray(a.arrays[2])
lam = 2.99792458e18 / nu
coverage = a.manifest['coverage']
diag = np.asarray(a.manifest['undefined_a_reference_diagnostic']['by_band'])

print('bin470', nu[470], lam[470])
print('undefined_transition_pct',
      100*coverage['undefined_transition_count']/coverage['active_transition_count'])
print('undefined_cell_pct',
      100*coverage['undefined_line_shell_count']/coverage['active_line_shell_count'])
print('undefined_power_pct',
      100*diag.sum()/coverage['a_reference_line_power'])

bands = [('B0',600,1000), ('B1',1000,1500), ('B2',1500,2000),
         ('B3',2000,2500), ('B4',2500,3000), ('BALL',600,3000)]
for name, lo, hi in bands:
    mask = (lam >= lo) & (lam < hi if hi < 3000 else lam <= hi)
    power = diag[mask].sum()
    print(name, power, 100*power/diag.sum(),
          100*power/coverage['a_reference_line_power'])

for arr_i, name in ((5, 'eta_fixed'), (7, 'eta_total')):
    values = {k: np.asarray(x.arrays[arr_i]).reshape(50,1000)
              for k, x in arts.items()}
    for left, right in (('A','B'), ('A','B2'), ('B','B2')):
        delta = values[left] - values[right]
        print(name, left, right,
              'neq', np.count_nonzero(delta),
              'maxabs', np.max(np.abs(delta)),
              'bin470max', np.max(np.abs(delta[:,470])))

mask = (lam >= 600) & (lam <= 3000)
integrals = {}
for lane, artifact in arts.items():
    eta = np.asarray(artifact.arrays[5]).reshape(50,1000)
    integrals[lane] = float(np.sum(eta[:,mask] * dnu[mask]))
print('BALL_eta_fixed', integrals,
      'B2/A', integrals['B2']/integrals['A'],
      'B2/B', integrals['B2']/integrals['B'])
PY
```

### 6.5 결과 artifact ledger

```bash
sha256sum validation/emiss_e5/verdict.json \
  validation/emiss_e5/band_table.csv \
  validation/emiss_e5/gamma_table.csv \
  validation/emiss_e5/jdet_A.tsv
```

결과 SHA-256은 차례로 다음과 같다.

```text
4c4f6ef3d7ceb3cf10a3d052a1b4c6dfa43aef3c880f609b50559cc65a8693ae  verdict.json
b22f0ccb08e7822a3556c8c14cc8ea66f0808675c0ef8ac8ceb1b8429e32ed2f  band_table.csv
4a3aa1f71465673c364ffc6f95b61b8662857ad7fb2e925c6ca22f5e71cb0f18  gamma_table.csv
c9d96d85be18266e71691b1f86dc7b0502e76cc7c6e26a34ebf4dd345b4342e4  jdet_A.tsv
```

`jdet_B.tsv`와 `jdet_B2.tsv`는 guard가 output table 완성 전에 중단했으므로 존재하지
않는다. 실패 상태와 전문은 `validation/emiss_e5/verdict.json`의 `stage31.B`와
`stage31.B2`에 보존되어 있다.
