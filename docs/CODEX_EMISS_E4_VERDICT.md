# E4 정식화 관문 판정 — in-situ A/B 이중 조립 payload

작성일: 2026-08-01 (KST)  
입력: `/gpfs/kjhan/lumina_runner2/scratch/emiss_ab_capture_188747/`  
판정기: `scripts/emiss_ab_insitu_e4.py`

## 0. 최종 판정

**벤치 판정: `UNRESOLVED-SOLVER-GUARD` (RC 2).**

입력 인증과 A-lane deterministic solve는 통과했다. 그러나 B-lane은 stage31의
certified-negative guard에서 중단됐다. 동일 B 명령 3회의 실패 문구와 stderr SHA-256이
모두 같았다. clamp, floor, acceptance 완화 없이 fail-closed했으므로 B의 `J_det`, 여섯
대역 `B/CMFGEN`·`B/A`, 두 target의 `Gamma_B`는 존재하지 않는다.

따라서 사전등록된 세 물리 분기인

1. UV 과잉 붕괴 = 정식화 진범 확정,
2. UV 과잉 유지 = 이 좌표 기각 후 profile/EPAY 이동,
3. 대역·이온별 부분 반응,

중 어느 것도 이번 payload만으로 판독할 수 없다. B 정식화가 deterministic transfer에서
강한 음수 해를 유발했다는 사실은 별도의 hard defect 신호이지만, 이를 UV band mean이나
Gamma의 붕괴로 바꾸어 읽지 않았다.

신규 모델/GPU run, clamp/floor 추가, 입력 수정, commit은 모두 0이다. 모든 실행 단계는
60초 timeout 안에 끝났으며 timeout에 걸린 단계는 없다.

## 1. 입력 인증

### 1.1 manifest와 payload

| 항목 | A-lane | B-lane | 판정 |
|---|---|---|---|
| lane | `A-production` | `B-Aul-nu` | PASS |
| iteration / field generation | 10 / 10 | 10 / 10 | PASS |
| post damping / dimensions | true / 50x1000 | true / 50x1000 | PASS |
| common assembly-state SHA-256 | `64519a9e5dd36edaf3c9388913ce41aea2218bdcf8c1309cf9da864fe6a577f8` | 동일 | PASS |
| seeded defect | line −1, hits 0 | line −1, hits 0 | clean capture |
| payload SHA-256 | `8ab6668b74b65f9ad0127d3f4e24781635a8555ca48dbc2dcde971af73320430` | `461762fd8d22012e66c781eabeb69bcbc9a0b6f2c00eaabce0cfa0638fa06e4a` | sidecar와 재계산 일치 |

판정기의 pair validator가 `r_edge`, `nu`, `dnu`, `chi_total`, `chi_es`,
`J_producer`의 bitwise 항등과 header epoch/flag 항등을 먼저 통과했다. 따라서 허용된
payload 차이는 emissivity 배열뿐이다.

### 1.2 A-lane과 같은-run chieta

`chieta_iter10`의 재계산 SHA-256은
`8ab6668b74b65f9ad0127d3f4e24781635a8555ca48dbc2dcde971af73320430`이고
`emiss_ab_iter10.A`와 `cmp` RC 0, 즉 payload 전체가 byte-identical이다. 두 artifact의
sidecar JSON 자체는 같지 않다. A sidecar에는 E4 lane/state/coverage 필드가 추가되기
때문이다. 물리 payload가 완전히 같으므로 “같은 run·같은 iter에서 chi_total이 같아야
한다”는 관문보다 더 강한 조건을 충족한다.

세 payload 모두 독립 `cmf_chieta_check.py`에서 iteration=10, generation=10,
post-damping=1, 2,416,472 bytes로 PASS했다.

## 2. B 미정의 전이 커버리지

### 2.1 census 폐합

| 단위 | active | defined | undefined | defined 비율 | undefined 비율 |
|---|---:|---:|---:|---:|---:|
| unique transition | 1,681,176 | 1,640,468 | 40,708 | 97.578600% | 2.421400% |
| active line-shell cell | 19,246,925 | 18,745,463 | 501,462 | 97.394586% | 2.605414% |

두 lane의 `undefined.csv`는 byte-identical이며 SHA-256은
`7995daceb31543a8e9f6e0759a62bef5dde5d5a1175e1bc7253cd6f3d4d06a95`이다.
40,708행 전부 reason mask 1, `population_not_tracked`이고, 모든 행의 `A_ul`은 양수다.
`undefined_shell_cells`의 합은 sidecar의 501,462와 정확히 닫힌다.

sidecar의 pre-EPAY A-reference line-power 장부는 다음과 같다.

| 양 | 값 |
|---|---:|
| 전체 A-reference line power | `1.1328513709450933e-3` |
| B-defined 전이가 차지하는 A-reference power | `1.0655624994655650e-3` |
| B-undefined 전이가 차지하는 차 | `6.7288871479528282e-5` |
| 기여 coverage | **94.0602206781%** |
| 미정의 기여 | **5.9397793219%** |

이는 `eta_A * dnu`를 line-shell마다 더한 **pre-EPAY A-reference 장부**이지 최종
emergent luminosity가 아니다.

### 2.2 600–3000 Å 대역별 A_ul 가중 장부

`undefined.csv.line_id`를 같은 model data의 `line_list.csv`에 one-to-one join했다.
capture A_ul과 model A_ul의 최대 상대차는 CSV 유효숫자 범위의 `7.752e-13`이다.
미정의 population 자체가 없으므로 `h nu A_ul n_u`라는 물리 emissivity는 계산할 수
없다. 아래 “A_ul 가중 기여”는 명시적으로

```text
W_undef(B) = sum_{undefined line l in band B} A_ul(l) * N_undefined_shell(l)
```

인 population-free proxy이고, 괄호 안 백분율은 40,708개 미정의 전이 전체에서의
몫이다. active-defined 전이의 대역별 A_ul 장부는 capture에 기록되지 않았으므로 이를
전체 active 대비 물리 기여율로 가장하지 않았다.

| band [Å] | 미정의 선 수 (% of undef) | 미정의 cell (% of undef) | `sum A_ul*Ncell` [s^-1 cell] (% of undef) |
|---|---:|---:|---:|
| B0 600–1000 | 2,981 (7.322885%) | 44,443 (8.862686%) | `2.559047434842e13` (20.125944%) |
| B1 1000–1500 | 3,407 (8.369362%) | 45,000 (8.973761%) | `3.392284638832e12` (2.667904%) |
| B2 1500–2000 | 4,306 (10.577773%) | 57,744 (11.515130%) | `2.962574894941e12` (2.329954%) |
| B3 2000–2500 | 2,583 (6.345190%) | 28,066 (5.596835%) | `5.523964140566e11` (0.434439%) |
| B4 2500–3000 | 1,603 (3.937801%) | 16,540 (3.298356%) | `1.992213520510e11` (0.156680%) |
| **BALL 600–3000** | **14,880 (36.553012%)** | **191,793 (38.246766%)** | **`3.269695164830e13` (25.714921%)** |
| outside 600–3000 | 25,828 (63.446988%) | 309,669 (61.753234%) | `9.445471914130e13` (74.285079%) |
| all undefined | 40,708 | 501,462 | `1.271516707896e14` |

### 2.3 미정의분 처리의 소스 판독

무단 A-lane/Planck fallback은 없다. `src/lumina_cmfgen.c:801-813`에서 reason이 있으면
통계만 올리고 `eta_line`을 더하지 않는다. `h nu A_ul n_u/(4 pi dnu)` 가산은 `else`
분기에서 defined 전이에만 실행된다. 따라서 미정의 전이의 **직접 B line emissivity는
pre-EPAY에서 정확히 0**이다.

다만 이 런은 `LUMINA_CMF_EPAY=2`다. 후속 EPAY가 thin/hot 영역에서 paid-power
rate-shape로 `S_fixed`를 재작성할 수 있으므로 “최종 B payload의 해당 coarse bin 전체가
항상 0”이라는 뜻은 아니다. 정확한 진술은 **미정의 전이의 A식 직접 대체는 없고, 그
전이의 직접 B 가산만 0**이라는 것이다.

유효 범위도 이에 따라 제한된다. 이 A/B는 B 공식을 정의할 수 있었던 pre-EPAY
A-reference line power 94.0602%에 대한 정식화 교체와, 나머지 5.9398% line power의
zeroing이 합쳐진 intervention이다. 후자가 작지 않으므로 B solve가 성공했더라도 순수한
“공식만의 효과”로 100% 외삽할 수는 없다.

## 3. 판별 벤치

### 3.1 A solve

A-lane은 PASS했다.

| guard/metric | 값 |
|---|---:|
| transport residual | `9.420346789444974e-7` |
| source residual | 0 |
| clamp | 0 |
| solution_negative_excess | 0 |
| sign_uncertain | 0 |
| nonfinite | 0 |
| nonfatal `bdf_eta_negative` counter | 177,814 |

A 산출물 `validation/emiss_e4/jdet_A.tsv`는 87,777 bytes, SHA-256
`a6961690e99a47b53150ebd9e0c70292648b7bce5d8157ac548e6766afa248dd`이다.

### 3.2 B solve hard stop

B-lane은 다음 guard에서 RC 1을 반환했다.

```text
LCMF_ENEGATIVE: certified negative solution exceeds truncation bound
radial=0 frequency=470 ray=9 segment=44 substep=0
value=-0.90635577952717661
interval=[-0.90635577952722846,-0.90635577952712476]
scale=23.467292540527662 h=0.02 B_trunc=0.26101653388724694
```

frequency bin 470의 중심은 1208.743248 Å로 B1에 속한다. 동일 driver 명령 3회가 모두
RC 1이며 stderr SHA-256
`8226dfa24348fe75bee874cd18a9d57785024d0463cfac14166ca44d43d563a3`로 동일했다.
판정기는 B solve 직후 fail-closed하므로 `jdet_B.tsv`, `verdict.json`,
`band_table.csv`, `gamma_table.csv`를 쓰지 않았다.

### 3.3 §7.2 여섯 대역

A 수치는 성공한 A `J_det`에 CMFGEN의 s8 integral-preserving bin average를 적용해
독립 회수했다. 참고 열의 `J_producer/CMFGEN`이 차터가 말한 기존 11.98x 기준이다.

| band [Å] | A/CMFGEN | B/CMFGEN | B/A | 참고: J_producer/CMFGEN |
|---|---:|---:|---:|---:|
| B0 600–1000 | 33.7412801 | UNRESOLVED | UNRESOLVED | 33.7640286 |
| B1 1000–1500 | 32.1913787 | UNRESOLVED | UNRESOLVED | 32.3230579 |
| B2 1500–2000 | 7.38976221 | UNRESOLVED | UNRESOLVED | 7.37578857 |
| B3 2000–2500 | 6.86442991 | UNRESOLVED | UNRESOLVED | 6.91222871 |
| B4 2500–3000 | 15.5806243 | UNRESOLVED | UNRESOLVED | 16.2921675 |
| **BALL 600–3000** | **11.7037914** | **UNRESOLVED** | **UNRESOLVED** | **11.9770975** |

A-lane deterministic transfer만으로는 기존 UV 과잉이 유지된다. 그러나 사전등록 판독은
B/A 변화가 핵심이므로 이 사실만으로 B 좌표를 채택하거나 기각하지 않는다.

### 3.4 Gamma 3자 대조

성공한 A field에 대해서는 같은 sigma/threshold/route/within-SL quadrature로 Gamma를
재계산했다. B field가 없으므로 B 열과 모든 B 비율은 UNRESOLVED다.

| target | Gamma_A [s^-1] | Gamma_B [s^-1] | Gamma_CMFGEN [s^-1] | A/CMFGEN | B/CMFGEN | B/A |
|---|---:|---:|---:|---:|---:|---:|
| Fe III C48 lump, idx 201 | `6.856928164117e2` | UNRESOLVED | `2.807174861437e1` | 24.4264376 | UNRESOLVED | UNRESOLVED |
| S II SL4, idx 4 | `1.725864971413e1` | UNRESOLVED | `4.748076950322e-1` | 36.3487153 | UNRESOLVED | UNRESOLVED |

identity 재검증은 Fe III 1,400 members/1,400 routes, S II 1 member/1 route로
통과했다.

## 4. 사전등록 판독과 다음 관문

이번 결과는 “유지”, “붕괴”, “부분”이 아니라 **측정 불성립**이다.

- A와 frozen `J_producer`는 각각 BALL에서 11.7038x, 11.9771x로 UV 과잉을 유지한다.
- B는 1208.74 Å가 속한 B1의 ray solution 단계에서 certified-negative로 중단됐다.
- 따라서 정식화가 진범인지, 이 좌표가 기각되는지, profile/EPAY로 이동해야 하는지를
  수치로 결정할 B band mean과 B Gamma가 없다.
- 이 guard를 풀기 위한 clamp, floor, tolerance/acceptance 변경은 금지 규율상 하지 않았다.
  후속은 운전석의 grammar-debug/solver-debug 관문이며, 현재 명령과 정확한 failure tuple을
  그대로 재현 입력으로 넘긴다.

## 5. 전 수치 재현 명령

모든 명령은 repository root에서 실행한다. 각 장시간 가능 단계에는 60초 hard timeout을
명시했다.

### 5.1 payload/manifest 인증

```bash
BASE=/gpfs/kjhan/lumina_runner2/scratch/emiss_ab_capture_188747
sha256sum "$BASE"/emiss_ab_iter10.A "$BASE"/emiss_ab_iter10.B \
  "$BASE"/chieta_iter10
cmp -s "$BASE"/emiss_ab_iter10.A "$BASE"/chieta_iter10
echo "A_vs_chieta_cmp_rc=$?"
python3 scripts/cmf_chieta_check.py "$BASE"/chieta_iter10
python3 scripts/cmf_chieta_check.py "$BASE"/emiss_ab_iter10.A
python3 scripts/cmf_chieta_check.py "$BASE"/emiss_ab_iter10.B
jq '{iteration,field_generation,emiss_ab_lane,common_assembly_state_sha256,
     sha256,coverage,seeded_defect}' "$BASE"/emiss_ab_iter10.{A,B}.manifest.json
```

### 5.2 coverage와 A_ul 대역 장부

```bash
timeout 60s python3 - <<'PY'
import pandas as pd
from pathlib import Path

base = Path('/gpfs/kjhan/lumina_runner2/scratch/emiss_ab_capture_188747')
model = Path('data/tardis_reference_toy06_19p48d_sivcaiv')
u = pd.read_csv(base / 'emiss_ab_iter10.B.undefined.csv')
lines = pd.read_csv(model / 'line_list.csv',
                    usecols=['line_id', 'wavelength', 'A_ul'])
x = u.merge(lines, on='line_id', how='left', validate='one_to_one',
            suffixes=('_capture', '_model'))
x['Aul_cells'] = x['A_ul_s-1'] * x['undefined_shell_cells']
tot_c = x['undefined_shell_cells'].sum()
tot_a = x['A_ul_s-1'].sum()
tot_ac = x['Aul_cells'].sum()
bands = [('B0',600.,1000.), ('B1',1000.,1500.),
         ('B2',1500.,2000.), ('B3',2000.,2500.),
         ('B4',2500.,3000.), ('BALL',600.,3000.)]
print('reason counts', x['reason'].value_counts().to_dict())
print('max relative Aul join error',
      ((x['A_ul_s-1']/x['A_ul'] - 1).abs()).max())
print('all', len(x), tot_c, tot_a, tot_ac)
for name, lo, hi in bands:
    upper = x.wavelength <= hi if name in ('B4','BALL') else x.wavelength < hi
    y = x[(x.wavelength >= lo) & upper]
    print(name, len(y), 100*len(y)/len(x),
          y.undefined_shell_cells.sum(),
          100*y.undefined_shell_cells.sum()/tot_c,
          y['A_ul_s-1'].sum(), 100*y['A_ul_s-1'].sum()/tot_a,
          y.Aul_cells.sum(), 100*y.Aul_cells.sum()/tot_ac)
outside = x[(x.wavelength < 600.) | (x.wavelength > 3000.)]
print('outside', len(outside), 100*len(outside)/len(x),
      outside.undefined_shell_cells.sum(),
      100*outside.undefined_shell_cells.sum()/tot_c,
      outside['A_ul_s-1'].sum(),
      100*outside['A_ul_s-1'].sum()/tot_a,
      outside.Aul_cells.sum(), 100*outside.Aul_cells.sum()/tot_ac)
PY
```

### 5.3 정식 A/B 판정기 — 실제 RC 2 명령

```bash
timeout 60s python3 scripts/emiss_ab_insitu_e4.py \
  /gpfs/kjhan/lumina_runner2/scratch/emiss_ab_capture_188747/emiss_ab_iter10 \
  --out-dir validation/emiss_e4 \
  --capture-dir /gpfs/kjhan/lumina_runner2/scratch/emiss_ab_capture_188747
```

실측 wall time은 8.7초였다. A solve 뒤 B의 `LCMF_ENEGATIVE`로 판정기 RC 2가 났다.

### 5.4 B 실패 3회 결정론 확인

```bash
timeout 60s python3 - <<'PY'
import hashlib, subprocess
cmd = [
 '/tmp/stage31_cmf_field_driver_e4',
 '/gpfs/kjhan/lumina_runner2/scratch/emiss_ab_capture_188747/emiss_ab_iter10.B',
 '/gpfs/kjhan/lumina_runner2/scratch/emiss_ab_capture_188747/emiss_ab_iter10.B.manifest.json',
 '8', '16', '10020.0', '1.0', '/tmp/e4_b_repeat.tsv']
for i in range(3):
    p = subprocess.run(cmd, text=True, capture_output=True)
    print(i+1, p.returncode, hashlib.sha256(p.stderr.encode()).hexdigest(),
          p.stderr.strip())
PY
```

### 5.5 성공한 A 대역과 Gamma 회수

이 명령은 B 값을 만들지 않으며, 판정기가 이미 만든 A table만 읽어 A/CMFGEN 및
Gamma_A를 회수한다.

```bash
timeout 60s python3 - <<'PY'
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, 'scripts')
import stage31_cmf_field_bench as b
import w3_gamma_triple_compare as g
from cmf_chieta_check import check_artifact

cap = Path('/gpfs/kjhan/lumina_runner2/scratch/emiss_ab_capture_188747')
a = check_artifact(cap / 'emiss_ab_iter10.A')
edges, _, _ = b.canonical_grid()
_, table = b.parse_driver_table(Path('validation/emiss_e4/jdet_A.tsv'))
ja = table['J_det'][::-1]
jp = np.asarray(a.arrays[8]).reshape(50, 1000)[8][::-1]
g.CMF_RUN = b.DEFAULT_CMF.resolve()
g.EW_DIR = Path('/tmp/w31_on_a.JuCpDY').resolve()
ctx, rates = b.load_gamma_context(cap, edges, ja)
for row in b.make_band_rows(edges, ja, jp, ctx['cmf']['J']):
    print(row['band'], row['J_det_over_J_CMFGEN'],
          row['J_MC_over_J_CMFGEN'])
for row in rates:
    print(row['target'], row['Gamma_det_D'], row['Gamma_CMFGEN_C'],
          row['Gamma_det_over_CMFGEN'])
PY
```

## 6. 생성 산출물

- `validation/emiss_e4/jdet_A.tsv`: 존재, A solve PASS.
- `validation/emiss_e4/jdet_B.tsv`: 미생성, B solve hard stop.
- `validation/emiss_e4/verdict.json`, `band_table.csv`, `gamma_table.csv`: 미생성,
  fail-closed 계약에 따른 정상 동작.
