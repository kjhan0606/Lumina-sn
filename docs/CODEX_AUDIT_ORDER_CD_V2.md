# 발주서 C·D·E v2 읽기 전용 감사

감사 중 구현·파일 수정·산출물 직접 쓰기는 하지 않았다.

| 항목 | 판정 | 실측 근거(명령과 출력) | 발주서 수정 요구 |
|---|---|---|---|
| 1. §2.5 권위 사슬 | 확인 | 아래 E1. `mk_sn_hydro.py:23-29`에 IGE·56Ni `1e-10` floor와 6원소 재규격화가 실제 존재한다. 원본은 859행 중 데이터 zone 807개, 각 21열이다. 속도 절단 결과 700개 zone이 `SN_HYDRO_DATA`의 700 data points가 된다. | §2.5에 “859행=파일 행 수, 807=실제 zone 수, 700=속도 절단 후 남은 원본 zone/data point 수이며 Lumina 50셸과 무관”을 명시한다. |
| 2. D1–D10 등급·전수검사 | 반박 | 아래 E2. 기존 D ID의 FATAL/WARN 배정 자체로 정상 덱이 죽는 사례는 발견하지 못했다. 그러나 `D1–D4,D7–D10`은 **9종이 아니라 8종**이다. G8의 `9/9`는 정의상 불가능하다. D7의 음수·NaN·Inf를 각각 검사하면 최소 주입 사례 수는 10개다. 또한 `data/`의 “110덱”에는 다른 형식인 `data/model/abundances.csv`가 포함된다. 정상 Lumina 형식 분모는 `109−결함 3=106`, D6 정상 WARN 대상은 43/106이다. | 전 문서의 “FATAL 9종/9·9”를 “FATAL 8개 ID/8·8”로 고친다. 픽스처는 D7 3종을 분리하여 최소 10건으로 적는다. “43/110덱”을 “정상 Lumina 형식 43/106”으로 고친다. |
| 3. G4 `1e-12` | 확인 | 아래 E3. 정본의 double 순차합 최대 편차는 `2.220446049250313e-16`, 십진 입력값 자체의 최대 편차는 `1.822e-16`; `1e-12` 위반은 0개다. | 임계 수정 불필요. 합산 방식이 double 순차합인지 명시하면 재현성이 좋아진다. |
| 4. G10 기대 경고 | 확인 | 아래 E3. 정본은 D2/D3/D4/D6/D7/D8/D9/D10에 걸리지 않고, D5 누락 목록만 `[12,13,21,22,23,24,25]`이다. D6 최대 편차는 `2.22e-16 < 1e-6`이다. | “D5 경고 1건”을 “집계 경고 이벤트 1건, 목록에 Z 7개”로 명확히 한다. Z마다 한 줄을 찍는 구현과 혼동하지 않게 한다. |
| 5. §6 발주 E 실행 가능성 | 반박 | 아래 E4. 원 생성기는 `scripts/build_toy06_epoch.py`로 특정 가능하다. 생성기 mtime이 정본보다 약 11초 앞서며, 기본 출력 경로가 정본이고, 메모리 재현 결과 2,808바이트가 완전히 동일하다. 다만 직접 입력은 CMFGEN의 19.48일 807-zone 파일이 아니라 저장소의 1시간 202-zone 파일이며 자체 Bateman 붕괴·중심점 선형보간을 사용한다. 또한 CMFGEN 원천은 36,000 km/s까지만 있어 Lumina 50셸 중 44개 완전, 1개 부분, 5개 무피복이다. §6.1의 “저장소로 가져와 provenance를 남긴다”는 §1의 E=`편집 없음`과 모순이다. | E의 직접 출처를 `snia_toy06_1h_lowres.dat → build_toy06_epoch.py → 정본`으로 사전등록한다. 외부 19.48일 파일은 읽기 전용 해시·경로 기록으로 대체하거나 E를 데이터/provenance 편집 발주로 재분류한다. 비교는 44개 완전 피복 셸, 부분 셸 44, 미정의 셸 45–49로 나눠야 한다. 영값에서 상대차가 정의되지 않으므로 절대차 또는 명시적 분모 규칙도 추가한다. |
| 6. C2 드라이버 목록 | 반박 | 아래 E5. atomic expander에 `OUT_DIR`을 지정하는 경로는 4개가 아니라 5개이며 누락 파일은 `scripts/deck_quarantine_driver.py`다. 이 파일은 `_ftos`의 abundance를 필터링해 `_active`에 다시 쓰므로 현재 30열 결함도 승계할 수 있다. 별도로 정본과 toy06 후보를 생성하는 abundance writer 3개도 존재한다. 138개 `slurm_*.sh` writer 중 toy06 명시 writer는 0개다. | C2에 `deck_quarantine_driver.py`를 추가하고 복사 후 형상 게이트를 건다. 생산경로 대장에는 `build_toy06_epoch.py`, `build_toy06_cmfgencomp_deck.py`, `build_toy06_standart_deck.py`와 각 sbatch wrapper도 별도 계보로 기록한다. |
| 7. 남은 무증상 실패 경로 | 반박 | 아래 E6. 헤더 순서를 역전해도 D2/D3은 통과하며 현재 판독 결과가 동일하다. 헤더만 있고 데이터 행이 0개인 파일도 기존 D1–D10 중 FATAL이 없고 D5/D6 WARN만 남는다. `atom_masses.csv` 중복 Z는 D4/D5 집합 비교와 abundance 측 D9를 모두 통과한다. 현재 정상 파일에서는 이 조건들이 0건이므로 새 FATAL이 정상 덱을 죽이지 않는다. | 정확한 헤더 스키마 `atomic_number,0,…,n_shells-1`를 FATAL로 추가한다. 인식된 abundance 행 0개 또는 셸 합이 비유한/`≤0`이면 FATAL로 분리한다. `atom_masses.csv`의 중복 Z·비정수 Z·비양수/비유한 mass와 행 수 불일치를 FATAL로 검사한다. 검사는 `n_elements`행에서 멈추지 말고 EOF까지 수행해야 한다. NUL 바이트와 Z 토큰의 후행 쓰레기도 D8 범위에 명시한다. |

## E1. 권위 사슬 실측

```text
$ nl -ba /gpfs/kjhan/cmfgen_runs/toy06_19.48d/mk_sn_hydro.py | sed -n '1,30p'
2  SRC='/gpfs/kjhan/cmfgen_runs/toy06_19.48d/snia_toy06_19.48d.dat'
3  OUT='/gpfs/kjhan/cmfgen_runs/toy06_19.48d/SN_HYDRO_DATA'
14 vel=d[:,1]; rad=d[:,9]; dens=d[:,10]; temp=d[:,11]
15 X_Ni=d[:,13]; X_Co=d[:,14]; X_Fe=d[:,15]; X_Ca=d[:,16]; X_S=d[:,17]; X_Si=d[:,18]
16 X_56Ni=d[:,12]
17 m=(vel>=VLO)&(vel<=VHI)
23 floor=1.0e-10
24 X_Ni=np.maximum(...); X_Co=np.maximum(...); X_Fe=np.maximum(...); X_56Ni=np.maximum(...)
27 MF={'SIL':X_Si,'SUL':X_S,'CAL':X_Ca,'IRON':X_Fe,'COB':X_Co,'NICK':X_Ni}
28 tot=sum(MF.values())
29 for k in MF: MF[k]=MF[k]/tot
```

```text
$ python3 <read-only source-shape census>
source_total_lines=859
source_data_rows=807
field_count_distribution={21: 807}
velocity_all_minmax=25.0 40325.0
below_1000=20 selected=700 above_36000=87
selected_centres=1025.0 35975.0
```

열 배치는 파일 자기선언대로 다음과 같다.

```text
1 zone index
2 velocity
3 zone mass
4 Lagrangian mass coordinate
5 stable IGE at t=0
6 56Ni at t=0
7 IME
8 Ti
9 unburnt C+O
10 radius
11 density
12 temperature
13 X_56Ni
14 X_Ni
15 X_Co
16 X_Fe
17 X_Ca
18 X_S
19 X_Si
20 X_O
21 X_C
```

```text
$ sha256sum toy06_19.48d/SN_HYDRO_DATA toy06_19.48d_jnu4/SN_HYDRO_DATA
fba01d5b4aed2e48abd00f765f08ef770d11ebe3367720403d8d42cb28cbd9ec  ...
fba01d5b4aed2e48abd00f765f08ef770d11ebe3367720403d8d42cb28cbd9ec  ...

$ sed -n '1,7p' toy06_19.48d/SN_HYDRO_DATA
Number of data points:        700
Number of mass fractions:          6
Number of isotopes:          3
Time(days) since explosion:     19.4800000
```

## E2. D1–D10 전수 결과

검사 범위는 저장소 아래 실제 파일·유효 심링크 전부이다.

```text
$ find . \( -type f -o -type l \) -name abundances.csv | wc -l
921
$ find . \( -type f -o -type l \) -name atom_masses.csv | wc -l
923
```

CSV를 strict numeric parsing하여 헤더/행 필드 수, 같은 디렉터리의 geometry·mass 원소 집합, 비유한/음수, 중복 Z, 셸 합, 실제 바이트 행 길이를 검사한 출력:

```text
FILES abundances=921 atom_masses=923
paired=920 abundance_without_masses=1 masses_without_abundance=3

BY_TOP abundances {'data': 110, 'logs': 811}
HEADER_FORMS {('atomic_number', True): 920, ('shell_id', False): 1}

D2 files=3
  ..._sivcaiv_ftos/abundances.csv     header=30 geometry=50
  ..._sivcaiv_fullcov/abundances.csv  header=30 geometry=50
  ..._sivcaiv_links/abundances.csv    header=30 geometry=50

D3_runtime events=45 files=3
D3_header events=0 files=0
D4 events=0 files=0
D5 events=44 files=44
D7 events=0 files=0
D8 events=0 files=0
D9 events=0 files=0
D10 events=0 files=0
D6 shells_checked=37580 violating_shells=25506 violating_files=533
MAX_PHYSICAL_LINE bytes_including_ending=1499
```

D1 후보 3개는 모두 `logs/.../ref`의 불완전한 실패 산출물이다. 실행 로그도 geometry 부재로 이미 실패했다.

```text
D1 candidate dirs:
 logs/...ab_super_161346/ref
 logs/...ab_trunc_161345/ref
 logs/...ionfix_super_wcap_161728/ref

stderr.log:
ERROR: Cannot open .../ref/geometry.csv
```

따라서 정상 실행 덱을 D1 FATAL이 새로 죽이는 사례로 보지 않는다.

`data/` 분모를 다시 분리한 결과:

```text
data abundances files=110
Lumina-format=109
known defective=3
normal Lumina-format=106
D6 all data Lumina-format=46
D6 known defective=3
D6 normal=43
```

`atom_masses.csv` 자체도 전수 확인했다.

```text
atom_masses_files=923
header events=0
field-count events=0
invalid-Z events=0
duplicate-Z events=0
nonpositive/nonfinite-mass events=0
```

FATAL 개수 산술:

```text
$ python3 - <<'PY'
ids=['D1','D2','D3','D4','D7','D8','D9','D10']
print(ids, len(ids))
PY
['D1', 'D2', 'D3', 'D4', 'D7', 'D8', 'D9', 'D10'] 8
```

## E3. G4·G10

```text
$ python3 <canonical-deck census>
shape elements=8 shells=50
float_sum_min=0.9999999999999999
float_sum_max=1.0000000000000002
max_abs_dev=2.220446049250313e-16
violations_gt_1e-12=0
violations_gt_1e-6=0

decimal_sum_min=0.9999999999999998178
decimal_sum_max=1.000000000000000135
decimal_max_abs_dev=1.822E-16

D4_extra=[]
D5_missing=[12, 13, 21, 22, 23, 24, 25]
D7_bad=0
D9_dups=[]
header_order_ok=True
data_field_counts=[50]
max_line_bytes=401
```

## E4. 정본 생성기와 E의 격자 한계

```text
$ stat -c 'mtime=%y size=%s %n' \
    scripts/build_toy06_epoch.py \
    data/tardis_reference_toy06_19p48d/abundances.csv

mtime=2026-06-29 14:53:58.758408000 +0900 size=10764 scripts/build_toy06_epoch.py
mtime=2026-06-29 14:54:10.080646766 +0900 size=2808 .../abundances.csv
```

원 스크립트의 알고리즘을 쓰기 없이 메모리에서 재생한 결과:

```text
script=scripts/build_toy06_epoch.py
model=data/standart_data1/input_models/snia_toy06_1h_lowres.dat
source_shape=(202, 21)
source_tend_days=0.041667
target_days=19.48
i_phot=19
v_inner_kms=3900.0
v_outer_kms=40300.0
shells=50
generated_text_bytes=2808
actual_bytes=2808
byte_equal=True
array_max_abs_diff=0.0
```

실행 명령 기록 자체는 없으므로 “이 스크립트가 실제 실행됐다”는 mtime·기본 출력 경로·바이트 동일성에 근거한 강한 추론이다. 그러나 생성 알고리즘과 직접 입력의 특정에는 충분하다.

CMFGEN 700점과 Lumina 50셸의 피복:

```text
source_edges_kms=1000.0 36000.0
target_edges_kms=3900.0 40300.0
full_coverage_shells=44 0 43
partial_coverage=[(44, 0.09171277848584788)]
zero_coverage_shells=[45, 46, 47, 48, 49]
finite_abundance_shells=45
```

## E5. toy06 생산·변조 경로

동일 atomic expander에 출력 디렉터리를 지정하는 모든 Python 경로:

```text
$ rg -l '(module|expand)\.OUT_DIR\s*=' scripts/*.py | sort
scripts/deck_quarantine_driver.py
scripts/deck_regen_fullcov_driver.py
scripts/deck_regen_r1_vintage_driver.py
scripts/deck_regen_r4_ftos_driver.py
scripts/deck_regen_r4_offcontrol_driver.py
```

추가 abundance writer:

```text
scripts/build_toy06_epoch.py:190
    ).to_csv(out / "abundances.csv", index=False)

scripts/deck_quarantine_driver.py:25-26
SOURCE = ..._sivcaiv_ftos
TARGET = ..._sivcaiv_active
scripts/deck_quarantine_driver.py:141-150
def write_filtered_elements(...):
    for name in ("atom_masses.csv", "abundances.csv"):
        ...
        writer.writerows(rows)

scripts/build_toy06_cmfgencomp_deck.py:124
write_abundances(args.output / "abundances.csv", ...)

scripts/build_toy06_standart_deck.py:96
write_abundances(args.output / "abundances.csv", ...)
```

```text
$ python3 <slurm writer census>
slurm_abundance_writers=138
toy06_named_among_writers=0
```

## E6. 빠진 fail-closed 조건의 반례

현재 로더는 `src/lumina_atomic.c:823`에서 헤더를 버리고 값 행만 읽는다. 같은 50개 값에 정상 헤더와 역순 헤더를 붙여 현재 동작을 재현했다.

```text
header_field_counts=50 50
canonical_vs_reversed_loaded_equal=True
header_sequences_equal=False
```

헤더만 존재하는 빈 body:

```text
HEADER_ONLY
D2_count_pass=True
data_rows=0
D5_warn_count=1
D6_warn_shells=50
fatal_IDs_triggered=[]
```

이 경우 전 조성이 0인데도 비영 종료 조건이 없다. 이는 D1의 근거인 “0 조성은 어떤 경우에도 정당하지 않다”와 충돌한다.

중복 `atom_masses` Z 반례:

```text
mass_rows=[6,6]
abundance_rows=[6]
D4=[]
D5=[]
D9_abundance=False
mass_duplicate=True
```

현재 920개 Lumina 형식 파일에는 새 조건이 걸리지 않는다.

```text
ordered abundance headers=920/920
minimum existing shell sum=0.69191185161
nonpositive existing shell sums=0
duplicate atom-mass Z files=0/923
```

**반려**