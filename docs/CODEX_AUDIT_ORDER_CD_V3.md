# 발주서 C·D·E v3 읽기 전용 감사

감사 중 구현·파일 수정·산출물 직접 쓰기는 하지 않았다. v3 변경분만 검사했다.

| 항목 | 판정 | 실측 근거(명령과 출력) | 발주서 수정 요구 |
|---|---|---|---|
| 1. D12·D13·D14 정상 덱 영향 및 실제 판독 경로 | 확인 | 아래 E1. `logs/` 811개는 전부 실제 Lumina 판독 형식이고 헤더 순서가 정상이다. 정확한 D12를 geometry와 대조하면 알려진 결함 3개만 실패한다. D13 위반은 0개, D14 위반도 923개 중 0개다. `data/model`은 atomic reference가 아니며 저장소의 실제 Lumina 호출 경로에 없다. | §4a의 `920/920 정상`을 “순서 자체 920/920, geometry까지 포함한 완전 D12는 도달 가능한 909개 중 906개 정상·알려진 결함 3개 실패”로 정밀화한다. |
| 2. EOF까지 읽는 루프 변경 | 확인 | 아래 E2. 정본은 abundance 8행, atom masses 15행으로 같지 않지만 **부족한 쪽**이다. 전체 paired 파일은 `lt=44, eq=876, gt=0`; 초과 abundance 행은 없다. 따라서 현재 정본·기존 덱의 적재값을 바꾸지 않고, 향후 초과행·후반 중복을 검출하게 된다. | 수정 없음. 다만 “정본 행 수 == 원소 수”가 아니라 `8 < 15`임을 §4d에 명시하면 구현자가 equality 검사를 추가하는 오류를 막을 수 있다. |
| 3. E3 실행 가능성 | 반박 | 아래 E3. Bateman 함수는 별도 함수지만 `main()`은 입력을 상수로 고정하고 조성 구성→전자밀도→광구 탐색→재격자를 한 경로에 결합한다. 입력/붕괴 생략 옵션도 없다. 19.48일판으로 같은 경로를 재생하면 광구가 3,900이 아니라 4,025 km/s가 되어 정본과 속도 셸도 달라진다. | 기존 `main()`을 그대로 실행하지 않는다. 19.48일판의 총 Ni·Co·Fe 및 Ca/S/Si/O/C 열을 읽고, **정본 geometry의 고정된 50개 중심점에 조성만** 보간하는 별도 read-only 분석 경로를 사전등록한다. 출력은 정본과 다른 전용 경로로 고정한다. |
| 4. E5 공통 속도격자 | 반박 | 아래 E4. 202-zone 중심은 `100+200k`, 807-zone 중심은 `25+50k`라 중심점 교집합이 0개다. 무관한 격자는 아니고 factor-4 staggered refinement지만, 최외곽 50 km/s도 일치하지 않는다. “공통 속도격자”만으로는 보간·보존량·외곽 처리 규칙이 정해지지 않는다. | E5를 둘로 나눈다. 붕괴식 단독 검증은 19.48일판 자체의 t=0 열에 Bateman을 적용해 같은 807점의 19.48일 열과 직접 비교한다. 1h 저해상도와의 비교는 high-res를 zone-mass 가중 보존형으로 201개 완전피복 저해상도 셀에 축약하고 마지막 부분피복 셀을 별도 표기한다. 후자는 붕괴식뿐 아니라 해상도 차이도 포함한다고 명시한다. |
| 5. C3를 E 앞으로 옮긴 순서 | 반박 | 아래 E5. 문서는 E가 기존 파일을 변조하지 않는다고 하지만, E3가 재사용하라는 스크립트의 기본 출력은 바로 현 정본 `data/tardis_reference_toy06_19p48d`다. 출력 인자를 빠뜨리면 E가 C3의 복사 원본을 덮어쓰고 G1의 기준도 바뀐다. | E3 전용 출력 경로를 발주서에 절대적으로 지정하고, 정본 경로와 같으면 비영 종료하는 게이트를 둔다. G1은 E 완료 후에도 다시 실행한다. 이 조치 뒤에만 C3와 E가 독립이다. |
| 6. “계보 일관성이지 CMFGEN 동일성이 아니다” | 확인 | 아래 E6. C3는 현 Lumina 정본의 바이트 복사다. 반면 CMFGEN 입력 변환기는 Ni/Co/Fe에 `1e-10` floor 후 재규격화를 수행한다. 같은 13,000 km/s에서 Lumina는 Ni/Co/Fe가 모두 정확히 0이지만 CMFGEN 경로는 각각 약 `1e-10`이다. 과소·과대주장이 아니다. | 수정 없음. “계보 일관성”을 “현 Lumina 정본과의 artifact/byte 일치”로 병기하면 의미가 더 명확하다. |
| 7. 남은 무증상 실패 경로 | 반박 | 아래 E7. (a) FATAL 표에는 11개가 아니라 **13개 ID**가 있다. G8 `11/11`은 다시 산술적으로 불가능하다. (b) 인식된 한 행이 전부 0이면 D13은 통과하고 D6 WARN만 내며 종료 0이다. (c) `X=1.2`도 FATAL 없이 WARN만 남는다. (d) `strtod("1e-9999")`는 ERANGE와 함께 유한한 0으로, `strtol("4294967302")` 후 int cast는 Z=6으로 변해 기존 조건을 통과할 수 있다. | G8을 우선 `13/13`으로 고치고 실제 주입 사례를 열거한다. 셸별 `ΣX > 0`과 개별 `X ≤ 1`을 FATAL로 추가한다. 모든 `strtod/strtol`에서 `errno==ERANGE`를 FATAL로 하고, Z는 양수이면서 대상 정수형에 표현 가능해야 한다. D14에도 Z 양수·범위 조건을 명시한다. |

## E1. D12·D13·D14 전수 및 판독 경로

```text
$ python3 <read-only CSV census>
FILES abundances=921 atom_masses=923
ABUND_BY_TOP {'data': 110, 'logs': 811}

HEADER first_atomic_number_and_ordered=920/920
model_schema=['data/model/abundances.csv']

logs files=811 atomic_schema=811 ordered=811
data files=110 atomic_schema=109 ordered=109

D12_vs_geometry ok=906 bad=3 no_geometry_preempts=11
D12_bad=[
 ('..._sivcaiv_fullcov/abundances.csv', 30, 50),
 ('..._sivcaiv_ftos/abundances.csv', 30, 50),
 ('..._sivcaiv_links/abundances.csv', 30, 50)
]

D13 zero_body=0 zero_recognized=0

D14 atom_mass_files=923
 bad_header=0 bad_fields=0 bad_Z=0 bad_mass=0
 row_count_mismatch=0 duplicate_Z=0 zero_rows=0
```

`logs/`의 `ref` 디렉터리는 실제 argv로 판독기에 전달된다.

```text
$ rg -n 'argv:.*logs/.*/ref' logs -g 'stdout*.log' | head
.../stdout.log:119:
  argv: .../lumina_cuda .../logs/.../ref 100000 8 spectrum nlte
.../stdout.log:121:
  argv: .../lumina_cuda .../logs/.../ref 100000 8 spectrum nlte
```

반면 `data/model`은 세 파일뿐이며 atomic reference 필수 파일이 없다.

```text
$ find data/model -maxdepth 1 \( -type f -o -type l \) | sort
data/model/abundances.csv
data/model/geometry.csv
data/model/thermodynamics.csv

$ rg -n 'data/model|model/abundances.csv' scripts src \
    -g '!scripts/regression_ledger.py'
(no matches)

$ rg -n 'model = repo / "data/model"|argv:.*data/model' \
    scripts/regression_ledger.py
1257:    model = repo / "data/model"
1299:        f"  argv: {binary} data/model 1 1 spectrum nlte\n"
```

마지막 두 줄은 실제 Lumina 실행이 아니라 `regression_ledger.py`가 만드는 fixture 문자열이다. 실제 바이너리는 argv 디렉터리에서 먼저 reference data를 읽은 뒤 같은 디렉터리를 `load_atomic_data()`에 전달한다.

```text
src/lumina_main.c:111: if (argc > 1) ref_dir = argv[1];
src/lumina_main.c:113: load_tardis_reference_data(ref_dir, ...)
src/lumina_main.c:119: load_atomic_data(&atom_data, ref_dir, geo.n_shells)
```

따라서 `data/model/abundances.csv`는 저장소의 실제 이 판독기 경로를 타지 않는다.

## E2. EOF 루프의 기존 파일 영향

```text
$ python3 <abundance rows versus atom-mass rows census>
ROW_REL abundance_vs_mass={'lt': 44, 'eq': 876, 'gt': 0}
excess_files=[]

CANONICAL abundance_rows,mass_rows,recognized=(8,15,8)
```

현재 D4 위반도 0이므로 기존 abundance 행은 모두 인식 가능한 Z다. 초과행이 없어 EOF 전환으로 새로 적재되는 기존 값은 없다.

## E3. E3 코드 결합과 격자 변화

```text
$ nl -ba scripts/build_toy06_epoch.py | sed -n '38p;74,115p;142,166p;224,230p'
38  MODEL = Path("data/.../snia_toy06_1h_lowres.dat")
74  def bateman(...):
88  def main(...):
89      d = np.loadtxt(MODEL)
104     # --- decayed composition ---
106     Ni, Co, Fe = bateman(...)
112     # electron density ... photosphere search
142     # --- build shell grid ---
146     v_edge = np.linspace(v_inner, v_max, n_shells + 1)
150     Xs = {Z: np.interp(v_cen, v, Xel[Z]) ...}
164     # matrix, renormalized
225 keeper = ...
226 out = ... else "data/tardis_reference_toy06_19p48d"
230 main(keeper, out, tgt, tau_p, nsh)
```

19.48일판의 기존 조성 열을 사용해 89–166행을 메모리 재생한 결과:

```text
direct19 i_phot=80
v_inner_kms=4025.0
tau=0.6680009479678789
next_tau=0.6582877828407996
n_above=727

E3_candidate_edges=4025.0 40325.0
centres=4388.0 39962.0
dv=726.0

canonical_edges=3900.0 40300.0
centres=4264.0 39936.0
dv=728.0

edge_array_equal=False
max_edge_diff=125.0
```

따라서 Bateman 함수 자체는 분리되어 있어도 E3에 필요한 “입력 조성만 교체한 동일 격자 산출”은 현재 코드 경로로 실행할 수 없다.

## E4. 202-zone과 807-zone 격자 관계

```text
$ python3 <velocity-grid census>
center_intersection=0

lowres_centers=100.0 40300.0
step=[200.0]
n=202

highres_centers=25.0 40325.0
step=[50.0]
n=807

lowres_edges=0.0 40400.0
highres_edges=0.0 40350.0
lowres_edges_in_highres_edges=202/203
missing=[40400.0]

nearest_center_offset_kms=[25.0]
```

부분집합도 무관한 격자도 아니다. 807-zone 격자는 202-zone 셀의 staggered factor-4 refinement이며 마지막 저해상도 셀만 완전 피복되지 않는다.

## E5. C3와 E의 공유 출력 경로

```text
$ nl -ba docs/ORDER_CD_COMPOSITION_IDENTITY.md | sed -n '298,323p'
300 - 세 덱 abundances.csv를 현 정본에서 복사
314 E는 편집 없는 감사가 아니다
315 반입은 신규 파일 추가이며 기존 파일 변조는 없다
321 E3 build_toy06_epoch.py의 재격자 로직을 그대로 사용
322 E4 정본과 E3 산출물을 셸별·원소별 대조

$ nl -ba scripts/build_toy06_epoch.py | sed -n '224,230p'
224 if __name__ == "__main__":
225     keeper = ...
226     out = sys.argv[2] if len(sys.argv) > 2 \
            else "data/tardis_reference_toy06_19p48d"
230     main(keeper, out, tgt, tau_p, nsh)
```

E3 산출 경로가 발주서에 지정되지 않았으므로 “기존 파일 변조 없음”은 현재 닫힌 계약이 아니다.

## E6. C3가 CMFGEN 동일성을 만들지 못한다는 실측

```text
$ nl -ba /gpfs/kjhan/cmfgen_runs/toy06_19.48d/mk_sn_hydro.py \
    | sed -n '22,29p'
22 # floor IGE ...
23 floor=1.0e-10
24 X_Ni=np.maximum(...); X_Co=np.maximum(...);
   X_Fe=np.maximum(...)
26 # elemental mass fractions (6 species), renormalized to sum=1
27 MF={...}
28 tot=sum(MF.values())
29 for k in MF: MF[k]=MF[k]/tot
```

동일 속도에서 비교:

```text
$ python3 <same-velocity composition check>
velocity_kms=13000.0
Lumina_NiCoFe=[0.0,0.0,0.0]
CMFGEN_postfloor_linear_at_same_v=[
 9.999999997e-11,
 9.999999997e-11,
 9.999999997e-11
]
exact_equal=False
```

C3는 현 Lumina 정본과 결함 덱 사이의 계보를 복원하지만 CMFGEN 입력 조성과의 동일성을 만들지 않는다.

## E7. 남은 fail-open 반례

FATAL ID 산술:

```text
$ python3 - <<'PY'
ids=['D1','D2','D3','D4','D7a','D7b','D7c',
     'D8','D9','D10','D12','D13','D14']
print(ids)
print(len(ids))
PY
['D1','D2','D3','D4','D7a','D7b','D7c',
 'D8','D9','D10','D12','D13','D14']
13
```

문서는 여전히 다음과 같이 적혀 있다.

```text
§4f: FATAL 11개 ID 각각 결함 주입
G8 : FATAL 11개 ID ... 11/11
```

값 범위와 0 조성 반례:

```text
$ python3 <corpus bounds and injected predicate evaluation>
CORPUS max_X=(1.0, ...)
min_shell_sum=(0.69191185161, ...)
nonpositive_shell_sums=0
nonpositive_abundance_Z=0

ONE_RECOGNIZED_ALL_ZERO
 fatal_D2_3_4_7_8_9_10_12_13_14=[]
 D6=WARN
 exit=0

ONE_RECOGNIZED_X_1p2
 fatal_D2_3_4_7_8_9_10_12_13_14=[]
 D6=WARN
 exit=0
```

수치 파서 범위 반례:

```text
$ python3 <libc strtod/strtol boundary reproduction>
strtod 1e-9999 -> 0.0
errno=34
finite=True
negative=False
trailing=b''

strtod 1.2 -> 1.2
errno=0
finite=True
negative=False
trailing=b''

strtol 4294967302 -> long 4294967302
cast_int=6
errno=0
trailing=b''
aliases_Z6=True
```

현재 저장소의 atom-mass Z 범위는 새 양수·표현범위 조건에 안전하다.

```text
atom_mass_Z_minmax=6 28
nonpositive=0
above_INT_MAX=0
```

**반려**