# 발주 3 — 상준위별 펌프 구동 proxy (전 선목록 스트리밍, 런 0회) — 2026-09-05

작성 = 운전석(Fable). 구현 = gpt-5.6-sol. 판독기이지 판정기 아님. **런루트·덱에 쓰기 0**, 산출은 이 디렉토리에만.

## 0. 목적
관측 ROW 선(상준위 98개, Z=26/27/28 ion 1, shell 0)은 상위 90% 방출선이라 **각 상준위로 들어오는 펌프 선 전체**를 담지 않는다. 전 선목록에서 각 상준위 u 로 들어오는 모든 선 l→u 를 모아 반복 0→1 의 펌프 구동 변화 `Δlog P_u` 를 만들고, 판독 1 의 `Δlog N_SL`(`out_331050/sl_groups.csv` 의 `Y_s`, `pairs.csv` 로 upper_level 매핑) 과의 SL 수준 공변동을 잰다.

## 1. 입력 (읽기 전용)
- 런루트 `R=/gpfs/kjhan/lumina/superlevel_b_h200x2_k36/slb_20260903T080407Z_dproxy_provenance_k42_ion1_h200`
- `R/input/model/line_list.csv` (363 MB; 열: atomic_number,ion_number,level_number_lower,level_number_upper,line_id,wavelength,f_ul,f_lu,nu,B_lu,B_ul,A_ul,wavelength_cm) — **한 번만 스트리밍**, `pandas` 금지(부동소수 재해석 함정), `csv` 모듈로 문자열→float.
- `R/input/model/levels.csv` (energy_eV, g, metastable)
- `R/work/physics_comparison/physics_DET_iter000{0,1}.spectral.csv` — shell_id=0 행만, bin `[nu_lo,nu_hi)` 와 `J_nu`.
- `out_331050/pairs.csv`, `out_331050/sl_groups.csv` (판독 1 산출) — 상준위(Z, ion, upper_level, runtime_sl_id) 목록과 `Y_s`.

## 2. 계산
1. 대상 상준위 집합 U = pairs.csv 의 (Z,1,upper_level) 전부(럼프 멤버 포함). 선목록에서 `atomic_number∈{26,27,28}, ion_number=1, level_number_upper∈U` 인 선 전부 수집. 선 수·상준위별 선 수 보고.
2. 각 선 l→u 에 대해 shell 0 의 bin J 를 두 반복에서 조회: `J_k(ν)` (bin 밖이면 제외 계수). 가중치는 **population 무관·덱 고정**: `w_lu = f_lu · g_l · exp(−E_l/kT)`, T=10020 K (하준위 Boltzmann 가중; 이온 분배함수는 공통이라 생략). 
3. `P_u,k = Σ_l w_lu · J_k(ν_lu)` (k=0,1), `Δlog P_u = ln(P_u,1/P_u,0)`. 함께 `Δlog P_u^{≥4eV}`(ν 가 4 eV 이상인 선만), `Δlog P_u^{<4eV}` 도 낸다.
4. SL 단위: runtime SL 의 Y_s 대 `X_u = Δlog P_u`(럼프는 멤버 u 의 P 합). 블록 (Z) 별 midrank Spearman 과 블록 중심화 pooled ρ, 상이값 수. 추가로 하준위 계급별 분해: 바닥(E_l<0.1 eV)·준안정(metastable=1)·기타 상태에서 오는 펌프의 Δlog 를 따로.
5. 산출: `pump_drive_per_upper.csv`, `pump_drive_sl.csv`, `pump_drive_summary.json`(선 수·제외·ρ·분포). p-value·분기 없음.

## 3. 금지·보고
발주 1 §4 와 같다. selftest 는 합성 선목록·합성 spectral 로 양성 1 + 음성 3(bin 밖 ν·U 밖 상준위·비유한 J). 실행은 운전석이 grammar-debug 에서 하므로 샌드박스에서 ssh 를 시도하지 마라. 완료 보고 10행 이내.
