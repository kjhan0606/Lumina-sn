# ★E13 — 빈 색인 관례 감사와 macro-atom 분기 대조

작성일: 2026-08-02 KST  
배경: `docs/CODEX_EMISS_E12.md`  
범위: 기존 iter-10 capture의 오프라인 감사, counterfactual 양축 mirror, CPU stage31
formal solve, 원자 CSV 기반 결정론 cascade. 생산 코드 수정, 신규 모델/GPU run,
clamp/floor, 커밋은 하지 않았다.

## 0. 판정 요약

**색인 관례 불일치 가설은 기각한다.** `LFMAT001`의 bin 0은 저주파/장파장이고
bin 번호가 커질수록 `nu`가 증가한다. 반면 `LCMFCE01` payload는 파일상
`nu` 내림차순이다. E10/E12 적용기는 payload를 읽을 때 `[:, ::-1]`로 작업 배열을
오름차순으로 바꾸고, 행렬의 `(input_bin, output_bin)`을 그 배열에 그대로 적용한 뒤
다시 내림차순으로 직렬화한다. 변환은 필요하고 정확히 한 번 존재한다.

양축 mirror `i'=999-i, o'=999-o`는 E12 실패를 고치지 않았다. 오히려 B2가 B0
유입에서 차지하는 비율은 **54.9245% -> 64.8169%**, exact stage31 B0
`J_det/CMFGEN`은 **26.4325 -> 53.9739**로 악화했다. 따라서 mirror가 물리적으로
타당한 장파장 형광 그림을 만든다는 판정 조건도 실패했다.

물리 sanity에는 한정된 이상 신호가 남는다. native 행렬 전체의 직접 관측 가능한
*출력-energy 가중* 평균은 `nu_out/nu_in=1.00359416`, 즉 `+0.3594%` 상향이다.
그러나 UV 입력만 보면 lower-nu가 51.6213%, higher-nu가 45.5159%로 하향이 우세하고,
B2 입력은 반대로 higher-nu 53.5772%, lower-nu 43.2807%이다. 행렬은 edge별 packet
count와 k-packet 표지를 보존하지 않으므로 (a) 흡수 광자당 photon-count 가중 평균,
(b) k-packet을 제외한 평균 이동은 둘 다 **UNRESOLVED**이다. k-packet 흡수 에너지
점유율은 2.04205%이다.

색인이 정상이므로 2단계도 수행했다. 같은 atomic CSV와 shell-8 producer J로 만든
radiative-only Lucy 단일-activation terminal UV-exit은 **Fe II 98.4018%, Fe III
89.6351%**이고, 모든 이온/모든 생산 물리를 합친 측정 행렬은 **92.7362%**로 둘
사이에 있다. 따라서 원자 A값과 Lucy 에너지 가중만으로 분기 결함을 입증하지
못했다. 다만 LFMAT001에는 Z/ion/line edge 태그가 없어 Fe별 동일 모집단 비교는
**UNRESOLVED**이다. 생산 로그의 strong-UV 조건부 분기는 Fe III에서
`p_iup=0.8819--0.8859`, `p_idown=0.0966--0.0972`로 internal-up이 압도한다.
이론 proxy와의 차이가 생기는 좌표는 원자 line list가 아니라 **Sobolev beta,
실제 line/bin J 및 stimulated correction, collision, bound-free ion jump, k-packet,
level-population/activation weighting, damping**을 포함하는 생산 확률 조립이다.

ARTIS 소스는 동일한 Lucy 에너지 가중 계약을 재확인한다. 다만 저장소에서 발견한
“ARTIS algo + Lumina data 20.2% vs native 42.9%”는 역사 보고서의 단일 문장이고,
그 20.2%에 연결된 정확한 output, epoch/band 정의, 실행/적분 명령은 찾지 못했다.
현재 `artis-ref/tests/toy06_*` spectrum은 2026-07-08 이후 자산이다. 따라서
**20.2%의 독립 재현은 UNRESOLVED**이며 여기서 새 ARTIS 모델을 실행하지 않았다.

## 1. 1단계 — 색인 관례 감사

### 1.1 LFMAT001: 오름차순 nu, 내림차순 lambda

생산 코드의 계약은 다음 경로로 닫힌다.

1. `src/lumina_cuda.cu:4160-4183`은 양의 `d_fluor_dlognu`에 대해
   `floor(log(nu/numin)/dlognu)`로 input/output bin을 각각 계산한다. 그러므로
   작은 index가 작은 `nu`이고 큰 index가 큰 `nu`이다.
2. `src/lumina_cuda.cu:6094-6106`은 line scatter 전에 comoving 흡수 주파수와
   에너지를 저장한다. `src/lumina_cuda.cu:6127-6131`은 cascade 뒤의 comoving
   방출 주파수와 에너지를 recorder에 넘긴다. 즉 첫 축은 입력, 둘째 축은 출력이다.
3. `src/lumina_cuda.cu:4212-4216`은 실제로
   `matrix[input_bin * nbins + output_bin]`에 output energy를 누적한다.
4. `src/lumina_cuda.cu:7372-7383`은 `NLTE_NU_MIN`, `NLTE_NU_MAX`,
   `log(max/min)/N`으로 grid를 초기화한다.
5. `src/lumina_cuda.cu:4376-4402`은 `nu_min`, `nu_max`, `dlognu`와 `(i,o,E)`를
   index 변환 없이 기록한다.

실제 header로 재구성한 1000개 geometric center의 양 끝은 다음과 같다.

| index | nu center (Hz) | lambda center (A) |
|---:|---:|---:|
| 0 | 1.5039790062e14 | 19933.2874 |
| 999 | 2.9920630417e16 | 100.195903 |

따라서 **LFMAT001 index는 nu 오름차순, lambda 내림차순**이다.

### 1.2 LCMFCE01: 파일 payload는 nu 내림차순

R7 설계 계약은 `docs/CODEX_STAGE3_1_CMF_FIELD_DESIGN_2026-08-01.md:352-368`에서
`frequency_descending`과 descending `nu[nnu]`를 명시한다. writer도 이 계약과
일치한다.

- `src/lumina_cmfgen.c:265-269`: 내부 `CMFGENState.nu`는 엄격한 오름차순이어야 한다.
- `src/lumina_cmfgen.c:327-335`: descending flag를 세우고 `b=n_bins-1..0` 순서로
  `nu`, `dnu`를 쓴다.
- `src/lumina_cmfgen.c:336-353`: chi, eta, J를 모두 같은 역순으로 쓴다.
- `src/lumina_cmfgen.c:375-381`: sidecar에도 `frequency_descending: true`를 쓴다.
- `scripts/cmf_chieta_check.py:65-71,95-97`: reader가 flag와 실제 엄격 내림차순을
  독립적으로 거부/검증한다.

### 1.3 E10/E12 적용기의 정합

`scripts/emiss_e10_apply_redistribution.py`는 다음 순서다.

- `:124-130,134`: payload의 모든 shell-frequency 배열을 `[:, ::-1]`로 읽어
  **nu 오름차순 작업 배열**로 만든다.
- `:158-166`: formal LFMAT header의 min/max/dlog를 canonical ascending grid와
  비교한다.
- `:167-188`: formal `(input_bin, output_bin)`을 뒤집지 않고 보존한다.
- `:202-229`: ascending `dnu`와 행렬 index를 직접 사용해 input별 return을 빼고
  output별 power를 더한다.
- `:241-264`: 같은 ascending 배열에서 source를 구성하며 음수/nonfinite를 실패시킨다.
- `:277-279`: 결과를 `[:, ::-1]`로 되돌려 LCMFCE01 descending payload로 쓴다.

결론은 `LFMAT ascending -> ascending work array -> LCMFCE01 descending`이다.
누락된 변환도 이중 변환도 없다.

## 2. 판정 실험 — 양축 mirror

원 행렬의 모든 1-D input ledger를 뒤집고, sparse edge에
`(i,o)->(999-i,999-o)`를 적용했다. energy 값, shell ledger, header grid는
변경하지 않았다. 이는 “행렬 index 0이 사실 high-nu였다”는 가설의 직접
counterfactual이다.

### 2.1 행렬/source 단계

| 지표 | native | 양축 mirror | mirror 판정 |
|---|---:|---:|---|
| B2 terminal -> B0 | 8.67522% | 15.9895% | 악화 |
| B2 -> higher nu | 53.5772% | 79.3044% | 강한 상향 |
| UV terminal -> B0 | 9.56326% | 19.0686% | 악화 |
| source-weighted B0 유입 | 6.04844e-4 | 1.13293e-3 | 1.873배 |
| 그중 B2 점유율 | 54.9245% | 64.8169% | +9.8924%p |
| B0 source/E9 | 2.76313 | 5.17171 | 악화 |
| B2 source/E9 | 0.717305 | 0.244510 | 과도한 제거 |

mirror application의 operator column closure 최대 오차는 `2.04503e-13`, 전체
적용 energy closure는 `1.62093e-14`이다. 즉 악화는 비보존 산술 때문이 아니다.

### 2.2 exact CPU stage31

| band | E9/CMFGEN | E12 native/CMFGEN | mirror/CMFGEN | mirror/E9 |
|---|---:|---:|---:|---:|
| B0 600--1000 A | 8.290551 | 26.432495 | **53.973904** | 6.51029 |
| B1 1000--1500 A | 4.916143 | 5.658865 | **15.492634** | 3.15138 |
| B2 1500--2000 A | 1.839881 | 1.691298 | **0.535280** | 0.290932 |
| BALL 600--3000 A | 0.932288 | 1.528677 | **1.653875** | 1.77400 |
| optical 3000--10000 A | 6.921039 | 6.374160 | **6.390260** | 0.923309 |

판정에 쓰인 native B0=26.4324946009와 mirror 전 band 수치는 각 JSON에 보존되어
있다. mirror formal solve는 3회 byte-identical이며 Jdet SHA256은
`4a5f7abd2426e4921ef68bcbb14b8a01736b5f5d3b75325aebfdf7da1fb5c21b`이다.

**판정:** mirror는 B2->B0 지배도 없애지 못하고 B0 형상을 2.04배 더 악화한다.
관례 불일치는 E12 실패 원인이 아니다.

## 3. 평균 진동수 이동 sanity

LFMAT edge가 보존하는 양은 edge별 output energy이므로 다음은 그 energy로 가중한
직접 산출값이다. outside-grid terminal은 output `nu`가 없어 평균에서 제외했고,
on-grid energy는 3064.58045687이다.

| 모집단 | 지표 | native | mirror |
|---|---|---:|---:|
| 전체 on-grid | mean input nu (Hz) | 1.9400971939e15 | 2.8287553745e15 |
| 전체 on-grid | mean output nu (Hz) | 1.9470702169e15 | 2.8201894043e15 |
| 전체 on-grid | output-input (Hz) | **+6.97302e12** | -8.56597e12 |
| 전체 on-grid | lower / same / higher | 48.5625 / 2.76392 / 48.6736% | 48.6736 / 2.76392 / 48.5625% |
| UV 입력 | lower / higher | **51.6213 / 45.5159%** | 46.4046 / **50.7285%** |
| B2 입력 | lower / higher | 43.2807 / **53.5772%** | 17.2553 / **79.3044%** |

mirror의 전체 평균 부호만 하향으로 바뀌지만, 실제 문제인 UV/B2 조건부 흐름은 더
상향이 되고 B0 유입도 더 커진다. 그러므로 전체 평균 부호 하나를 mirror 채택 근거로
쓸 수 없다.

엄격한 물리 명제 “k-packet 경유를 제외한 흡수 광자당 평균 방출 nu <= 흡수 nu”는
현재 schema로 판정할 수 없다.

- `src/lumina_cuda.cu:4196-4201`은 k-packet을 전체/shell scalar로만 센다.
- `src/lumina_cuda.cu:4212-4216`의 sparse edge에는 k-packet bit나 event count가 없다.
- 따라서 총 2.04205%의 k-packet absorbed energy를 어느 edge에서 뺄지 알 수 없다.
- edge가 output energy만 저장하므로 photon-count 가중 `mean(nu_out-nu_in)`도 복원할
  수 없다.

두 값은 **UNRESOLVED**이다. 이것은 index 판정을 뒤집지 않지만 다음 계기 schema에는
edge별 `{event_count, non-k energy, k energy}`가 필요함을 뜻한다. 본 차터에서는 생산
수정을 금지했으므로 제안만 남긴다.

## 4. 2단계 — atomic-data radiative 분기

### 4.1 계산 계약

`scripts/emiss_e13_index_audit.py`는
`data/tardis_reference_toy06_19p48d/{levels,line_list,ionization_energies}.csv`와
iter-10 `chieta`의 shell-8 producer J를 읽는다. UV line activation proxy는
`B_lu J h nu`이고 lower population은 CSV에 없으므로 제외했다. 한 activation이
내부 jump를 반복해 radiative emission으로 끝날 때까지 다음 Lucy weights로
결정론적으로 전파한다.

```text
emit(u->l)          = (A_ul + B_ul J) h nu
internal-down(u->l) = (A_ul + B_ul J) E_l,neutral
internal-up(l->u)   = B_lu J E_l,neutral
```

neutral-ground energy는 해당 ion 아래의 ionization potential을 누적한다. 이 구조는
Lumina의 설명/구현 `src/lumina_plasma.c:3908-3945,4548-4578` 및 ARTIS
`macroatom.cc:73-135`와 같다.

이 proxy가 의도적으로 제외한 것은 Sobolev beta, stimulated-opacity population
correction, collisions/nonthermal, k-packet, bound-free ion change, lower-level
population activation weight, production probability damping이다. J grid 밖은 0으로
두었고 100--20000 A 안의 결과를 보고한다. 이 계산에는 clamp/floor가 없다.

### 4.2 결과

| ion | levels / lines | 첫 action emit | internal-down | internal-up | terminal UV | terminal B0 |
|---|---:|---:|---:|---:|---:|---:|
| Fe II | 2698 / 531662 | 25.8187% | 67.7395% | 6.44179% | **98.4018%** | 2.39840% |
| Fe III | 1500 / 136263 | 20.2793% | 77.9730% | 1.74768% | **89.6351%** | 24.9997% |
| LFMAT native, 모든 이온/물리 | -- | -- | -- | -- | **92.7362%** | 9.56326% |

Fe II 결과는 2026-07-07의 기존 `scripts/cascade_walk_fe2.py` k=1 결과 98.4%와
재현 수준으로 일치한다(`docs/FLUOR_ATTACK_DESIGN_2026-07-06.md:63-68`). 같은 문서는
Fe II 531662 line이 ARTIS와 bit-identical이었다고 기록한다(`:70-72`).

측정 전체값이 Fe II/III 이론값 사이에 있다는 사실은 “원자데이터에서 계산하면
장파장 cascade인데 측정기만 UV로 뒤집혔다”는 증거가 아니다. 반대로 Fe별 measured
terminal을 뽑을 수 없으므로 정량 동등성도 주장하지 않는다: **ion-resolved measured
vs theory = UNRESOLVED**.

### 4.3 차이가 위치하는 생산 좌표

동일 capture의 생산 로그는 strong-line, strongest-line-in-UV 조건에서 다음을 쓴다.

| shell | ion / n | p_iup | p_idown | BB emit 중 UVblank | NIR2 |
|---:|---|---:|---:|---:|---:|
| 0 | Fe III / 100 | 0.8819 | 0.0972 | 73.4% | 22.6% |
| 0 | Fe II / 31 | 0.6062 | 0.2492 | 96.6% | 2.0% |
| 3 | Fe III / 83 | 0.8859 | 0.0966 | 69.2% | 19.3% |
| 3 | Fe II / 29 | 0.3942 | 0.3801 | 98.8% | 0.0% |

원문 위치는 capture `stdout.log:34765-34766,35842-35845`이다. 이는 level 수에
가중한 즉시 branch이고 LFMAT의 activation/energy 가중 terminal 통계와 같지 않다.
따라서 수치 차이를 결함으로 직접 빼지는 않는다. 다만 어디서 radiative-only proxy와
갈라지는지는 소스로 특정할 수 있다.

- emission `A_ul beta`: `src/lumina_plasma.c:4239-4245`
- internal-down의 beta/legacy 선택과 collision: `:4273-4327`
- internal-up J source, stimulated correction, beta: `:4341-4505`
- Lucy `h nu` / neutral-ground `E_lower` weighting: `:4548-4578`
- k-packet collision energy/fair-draw 좌표: `:4633-4653`
- bound-free photo/collisional ion-up: `:4825-4890`
- normalization/damping: `:4897-4905`

즉 현 증거가 지목하는 것은 **internal-up 확률을 크게 만드는 beta와 실제 J/level-pop
조합**, 그리고 그 뒤의 collision/BF/k-packet 경로이다. 어느 하나가 E12 형상 실패의
단일 원인인지는 이 감사만으로 **UNRESOLVED**이다.

## 5. ARTIS 등가 대조

로컬 ARTIS checkout은
`/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref`에서 확인했다.

- `macroatom.cc:77-110`: radiative deexcitation `R*epsilon_trans`, collision
  `C*epsilon_trans`, internal-down `(R+C)*epsilon_target`.
- `macroatom.cc:112-135`: internal-up `(R+C+NT)*epsilon_current`.
- `macroatom.cc:165-185`: photo/collisional ion-up `(...)*epsilon_current`.
- `macroatom.cc:569-603`: `B_lu-B_ul n_u/n_l`, beta, detailed Jblue 또는 binned
  `radfield(nu)`의 excitation rate.

따라서 Lumina와 ARTIS가 대조해야 할 공통 좌표는 위 energy weights 자체가 아니라
`R`, `C`, `J`, population, ion-changing channel과 k-packet 선택이다.

두 역사 수치는 구분해야 한다.

1. `docs/FLUOR_ATTACK_DESIGN_2026-07-06.md:70-75`는 동일 Fe II line data에서
   ARTIS emergent UV 14.6%, radiative single-cycle 98%, offline multicycle 29.8%를
   기록한다.
2. `validation/cmfgen_toy06_19p48d/analysis/criminal_record/CRIMINAL_RECORD.md:87`은
   “ARTIS algo + Lumina data 20.2% vs native 42.9%”를 기록한다.

하지만 20.2% 문장에는 실행 디렉터리/commit, epoch, UV band, `spec.out` column,
`F_nu dnu` 적분 명령이 결부되어 있지 않다. 발견한 toy06 ARTIS spectra는 7월 8일
이후라 7월 7일 자산이라고 확정할 수 없다. 임의 정의로 새 숫자를 만드는 대신
**same-data ARTIS 20.2%의 독립 재현 = UNRESOLVED**로 둔다.

## 6. 재현 명령

아래 명령은 신규 모델/GPU transport를 실행하지 않는다. 첫 명령은 기존 capture를
읽어 mirror와 이론 JSON을 만들고, 둘째는 기존 E10 적용기를 mirror에 적용한다.
셋째는 기존 frozen payload에 대한 CPU formal solver 컴파일/실행이다.

```bash
mkdir -p validation/emiss_e13

python3 scripts/emiss_e13_index_audit.py \
  --matrix /gpfs/kjhan/lumina_runner2/scratch/fluormat_capture_188828/fluor_matrix_iter10 \
  --chieta /gpfs/kjhan/lumina_runner2/scratch/fluormat_capture_188828/chieta_iter10 \
  --e9-payload validation/emiss_e12/e9_same_capture/emiss_e9_effective_iter10 \
  --source-payload /gpfs/kjhan/lumina_runner2/scratch/fluormat_capture_188828/emiss_ab_iter10.A \
  --preregistration validation/emiss_e12/preregistration.json \
  --out-dir validation/emiss_e13 \
  > validation/emiss_e13/index_branch_audit.stdout

python3 scripts/emiss_e10_apply_redistribution.py \
  --e9-payload validation/emiss_e12/e9_same_capture/emiss_e9_effective_iter10 \
  --source-payload /gpfs/kjhan/lumina_runner2/scratch/fluormat_capture_188828/emiss_ab_iter10.A \
  --matrix validation/emiss_e13/fluor_matrix_iter10_mirror_both_axes \
  --matrix-format formal \
  --column-closure-tolerance 2e-12 \
  --preregistration validation/emiss_e13/mirror_application_contract.json \
  --out-dir validation/emiss_e13/mirror_apply \
  > validation/emiss_e13/mirror_apply.stdout

gcc -O2 -std=c11 -Wall -Wextra -Werror \
  -D_POSIX_C_SOURCE=200809L -Isrc scripts/stage31_cmf_field_driver.c \
  src/lumina_cmf_field.c -lm -o /tmp/stage31_cmf_field_driver_e13

/tmp/stage31_cmf_field_driver_e13 \
  validation/emiss_e13/mirror_apply/emiss_e10_redistributed_iter10 \
  validation/emiss_e13/mirror_apply/emiss_e10_redistributed_iter10.manifest.json \
  8 16 10020 1 validation/emiss_e13/jdet_mirror_s8.tsv \
  > validation/emiss_e13/stage31_mirror.stdout

python3 scripts/emiss_e10_jdet_measure.py \
  --payload validation/emiss_e13/mirror_apply/emiss_e10_redistributed_iter10 \
  --jdet validation/emiss_e13/jdet_mirror_s8.tsv \
  --e9-jdet validation/emiss_e12/jdet_e9_same_capture_s8.tsv \
  --preregistration validation/emiss_e13/mirror_application_contract.json \
  --source-measurement validation/emiss_e13/mirror_apply/source_band_measurement.csv \
  --cmf-run /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4 \
  --out-dir validation/emiss_e13/mirror_measure \
  > validation/emiss_e13/mirror_measure.stdout

python3 -m py_compile scripts/emiss_e13_index_audit.py
python3 scripts/emiss_e13_index_audit.py --help
```

3회 byte-identical 검증은 마지막 두 명령을 동일 input으로 3회 반복하고 생성 Jdet의
SHA256을 비교했다. E12와 같은 방식이며 summary의 `repeat_count=3`,
`repeat_hashes_identical=true`에 기록되어 있다.

## 7. 산출물과 무결성

| 산출물 | SHA256 |
|---|---|
| `scripts/emiss_e13_index_audit.py` | `7757ff3b8e51d8b10f800b100a6ba3797b5a2cb68495865a53d00f03d5b18c75` |
| `validation/emiss_e13/index_branch_audit.json` | `be674197bb535a65ae024baef2bfd19db81a378a8c996979b031199e6a8017b2` |
| mirrored LFMAT001 | `e0e76d5114678be6e9eb47e586ce915ae9e0a5b1fdfd56446bb5f6f29c5decad` |
| mirror application contract | `f47271c00697cfcc7020cf9b67d4a8b4f609486569614f89ae6fc58fdfcfc6ac` |
| mirror LCMFCE01 payload | `6bc7441384cb686ed5e2a287286f4b77cd548ad7323b36c6447e711679726c91` |
| mirror stage31 summary | `7b438d684efd74023af992e71a042d1f12cbc82df8378ab87918556ea4e9002f` |
| mirror Jdet | `4a5f7abd2426e4921ef68bcbb14b8a01736b5f5d3b75325aebfdf7da1fb5c21b` |

E13이 새로 만든 것은 오프라인 감사 script, 본 보고서, `validation/emiss_e13/`
산출물뿐이다. 작업 시작 전부터 존재한 dirty worktree의 생산 코드 변경은 건드리지
않았다. E13 명의의 production source 수정이나 commit은 없다.

## 8. 최종 결론

1. **REFUTED — 색인 불일치:** LFMAT001 ascending nu와 LCMFCE01 descending nu는
   적용기에서 올바르게 정합된다. mirror는 B2->B0와 B0 악화를 모두 키운다.
2. **OBSERVED — 국소 물리 sanity 문제:** native 전체 energy-weighted 평균은
   +0.3594% 상향이고 B2도 상향 우세다. 다만 UV 전체는 하향 우세다.
3. **UNRESOLVED — 엄격한 광자/k-packet sanity:** 현재 schema로 edge-level k 제거와
   photon-count 평균은 계산할 수 없다.
4. **NOT DEMONSTRATED — atomic radiative branch 결함:** Fe II/III direct radiative
   terminal UV fractions가 measured global을 bracket한다. 이온별 measured edge가 없어
   일대일 판정은 불가능하다.
5. **LOCATED, NOT CONVICTED — 다음 물리 좌표:** strong-UV Fe III의 생산 분기는
   internal-up 약 88%다. beta/J/population과 collision/BF/k-packet을 포함한 확률 조립이
   후속 white-box 대조 좌표다.
6. **UNRESOLVED — ARTIS 20.2% 재현:** 역사 기록은 확인했지만 정확한 자산/recipe가
   없어 독립 수치로 승격하지 않는다.
