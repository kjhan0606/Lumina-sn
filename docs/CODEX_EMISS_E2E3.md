# 방출률 캠페인 E2/E3 — 조립기 복제 항등 감사

작성일: 2026-08-01 (KST)  
최종 상태: **E2 UNRESOLVED / E3 NOT RUN (관문 차단)**

## 0. 결론

E1 오프라인 재조립을 권위 `chieta_iter10`과 다시 대조한 결과는 다음과 같다. 잣대는
E1과 같은 권위장 대비 전역 relative L1이다.

| E2 관문장 | relative L1 | 요구치 | 판정 |
|---|---:|---:|---|
| `chi_total` | 8.226059670973736e-2 | ≤ 1e-10 | FAIL |
| `eta_fixed` | 3.517015818466757e-1 | ≤ 1e-10 | FAIL |

잔차는 각각 허용치의 약 8.23e8배, 3.52e9배다. 부동소수 연산 순서로 설명할 수 있는
범위가 아니다. 따라서 **E2 복제 항등은 UNRESOLVED**다.

직접 원인은 캡처와 E1 CSV의 **세대 불일치**다. 권위 payload는 반복 10의
`compute_bf_opacity -> cmfgen_assemble -> cmfgen_solve_J -> J damping` 직후에 기록됐다.
그러나 E1이 상태 복원에 사용한 `lumina_levelpop.csv`, `lumina_ion_pops.csv`,
`lumina_plasma_state.csv`는 반복 11 종료 시점에 기록됐다. level-pop dump는 부가
final-resolve 직전, plasma/ion dump는 resolve 뒤 원래 상태를 복원한 다음 기록됐다. 정확한
반복-10 입력인 per-line `tau_sobolev`, fine-level population, `T_e/n_e`, BF 중간장과
EPAY 장부는 남아 있지 않다.

`LCMFCE01`에는 합쳐진 `chi_total`, `chi_es`, `eta_fixed`, `eta_coherent`,
`eta_total`, `J`만 있다. 이 집계장으로는 `chi_abs`와 `chi_line_th`, pre-EPAY
continuum/line eta, per-shell EPAY scale/shape를 유일하게 역산할 수 없다. 권위
`chi_total`과 `eta_fixed`를 그대로 복사하는 것은 재조립 항등이 아니라 정답장
pass-through이므로 채택하지 않았다.

E3는 “E2 항등 후에만”이라는 명시 관문에 따라 **실행하지 않았다**. 따라서
`A_ul*n_u` 후보 chieta, stage31 `J_det`, §7.2 대역표 및 사전등록 세 분기 판정은
생성하지 않았다.

## 1. 권위와 감사 자산

- 권위 payload:
  `/gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10`
- SHA-256:
  `94d75988034454f55fb6b130f04521f01c56f875cb22ef3a711850d7382ffa2f`
- 권위 epoch: iteration/generation 10/10, post-damping, 50 shell × 1000 bin.
- E1 replay SHA-256:
  `fa13326d50fff3d84d89bed0041a7b85828bd91952fcddb134d9942f38f625e1`
- 신규 감사기: `scripts/emiss_assembly_identity_e2.py`.
- 기계 판정: `validation/emiss_e2e3/assembly_identity_audit.json`.
- 대역별 항 감사: `validation/emiss_e2e3/term_gap_by_band.csv`.

감사기는 관문 실패 시 의도적으로 RC 2를 반환한다. 이를 성공으로 바꾸는 보정,
floor, clamp 또는 권위장 대입은 없다.

## 2. 실행 경로와 세대 감사

생산 순서는 `src/lumina_cuda.cu:7529-7538`의 BF 재계산과 조립/solve,
`:7546-7561`의 J damping, `:7563-7594`의 권위 dump다. 같은 반복의 plasma/NLTE
갱신과 `tau_sobolev` 갱신은 그 뒤에 수행되고, 반복 요약은 `:7708-7713`에서
출력된다. 따라서 iteration-10 payload를 조립한 입력은 iteration-9 끝에서 준비된
상태다.

권위 stdout의 실제 순서는 다음과 같다.

| 사건 | stdout line |
|---|---:|
| iteration 10 완료 요약 | 33485 |
| iteration 11 완료 요약 | 36005 |
| `lumina_levelpop.csv` 기록 | 37975 |
| `lumina_plasma_state.csv` 기록 | 38074 |
| `lumina_ion_pops.csv` 기록 | 38075 |

즉 E1의 세 CSV는 권위 조립 뒤 최소 두 plasma/NLTE 갱신을 지난 상태다. stdout에서
복원한 iteration-10 입력의 반올림 `T_e`와 최종 plasma CSV의 차이만으로도 relative
L1 1.3108957%, 최대 셀 차이 8.01366%(s11)다.

| shell | capture 입력 `T_e` (stdout, 1 K 정밀도) | E1 CSV `T_e` [K] |
|---:|---:|---:|
| 0 | 21081 | 21227.639444 |
| 11 | 10646 | 11499.134290 |
| 25 | 8319 | 8475.677865 |
| 49 | 12971 | 13052.111389 |

인구와 `tau_sobolev`는 iteration-10 입력 세대 파일이 아예 없다. 최종 CSV로
재구성한 E1 선 tau와 BF projection은 같은 물리 모델의 **다른 epoch**이지 권위
조립 입력의 동결본이 아니다.

## 3. 고정된 좌표

다음 좌표는 원인에서 제외된다.

| 좌표 | 감사 결과 |
|---|---|
| `r_edge` | 51/51 double bitwise equal |
| `nu`, `dnu` | 각각 1000/1000 double bitwise equal |
| 캡처 후 주입한 `J` | 50000/50000 double bitwise equal |
| payload eta 폐합 | `eta_fixed + eta_coherent == eta_total`, 50000셀 bitwise equal |
| `cmfgen_assemble` 소스 | 캡처/현재 SHA-256 동일 (`0f0e9adb...`) |
| 호출부 `lumina_cuda.cu` | 캡처/현재 SHA-256 동일 (`0fb49408...`) |
| atomic loader | 캡처/현재 SHA-256 동일 (`65493618...`) |
| 물리 환경 | 권위 RESOLVED CONFIG 122개 중 119개를 stdout 값 그대로 import |

제외한 세 환경은 실행 수단뿐이다: `LUMINA_BIN`은 offline CPU driver,
`OMP_NUM_THREADS=1`, `LUMINA_CMF_SOLVE_GPU=0`이다. 조립 물리 게이트는 권위값을
그대로 사용한다. 주요 값은 `LINE_EPS_PHYS=1`, `CMF_EPAY=2`,
`CMF_BF_MILNE=2`, `CMF_DEP_SOURCE=1`, `EPAY_HOTF=0`, `EPAY_SMIN=5`,
`EPAY_TAUBIN=10`이다. `SRC_NLTE`는 권위에서 unset이므로 양쪽 모두 기본 OFF다.

캡처 인증 뒤 현재 `lumina_plasma.c`와 `lumina.h`의 해시는 바뀌었다. E1 재생을 다시
실행하면 기존 E1 payload와 같은 SHA-256이 나오므로 현재 E1 결과 자체는 재현되지만,
이 두 파일의 캡처 시점 사본도 exact historical replay에는 필요하다. 이 드리프트를
무시하고 `cmfgen.c`만 같다는 이유로 전체 생산 경로가 같다고 주장하지 않는다.

## 4. 저장된 채널 기준 항별 분해

`LCMFCE01`이 실제 저장한 채널의 권위 대비 결과다.

| 저장/유도 항 | 의미 | relative L1 |
|---|---|---:|
| `chi_total` | 전체 opacity | 0.0822605967 |
| `chi_es` | electron + line coherent 몫 | 0.0676076219 |
| `chi_total-chi_es` | continuum absorption + thermal line 몫 | 0.0822986272 |
| `eta_fixed` | EPAY 이후 고정 emissivity | 0.3517015818 |
| `eta_coherent` | `chi_es*J` | 0.1297308016 |
| `eta_total` | fixed + coherent | 0.1299475834 |
| `J` | 캡처장 재주입 | 0 (bitwise) |

세부 대역에서도 %급 이상이다.

| band [A] | `chi_total` rel L1 | `eta_fixed` rel L1 |
|---|---:|---:|
| 600–1000 | 0.0763176 | 0.6599765 |
| 1000–1500 | 0.0897143 | 0.2103720 |
| 1500–2000 | 0.0761681 | 0.1979979 |
| 2000–2500 | 0.1600853 | 0.1342318 |
| 2500–3000 | 0.2080164 | 0.0626515 |

### 4.1 선 기여

생산식은 `lumina_cmfgen.c:523-566`에서 per-line `tau_sobolev`를 coarse bin의
`w=(1-exp(-tau))*nu/(c*t*dnu)`로 투영하고, `LINE_EPS_PHYS=1`이면
`chi_line_th += w*eps_l`, `eta_line += w*eps_l*B(T_e)`를 만든다. `eps_l`도
`n_e`, `T_e`, `tau`의 함수다.

E1은 최종 CSV population으로 `tau`를 다시 만들었다. 권위 iteration-10 입력의
per-line tau/인구가 저장되지 않았으므로 캡처의 정확한 `chi_line`,
`chi_line_th`, `eta_line`과 그 차이는 산출 불가능하다. `chi_es` 차이 6.7608%는
line-coherent 몫이 달라졌다는 직접 증거지만, electron scattering과의 정확한 분리는
캡처 `n_e` 전 셸이 없어서 불가능하다.

### 4.2 연속항

생산식은 `:661-700`에서 당시 인구로 계산한 BF, 당시 `n_e/T_e`의 FF 및 electron
scattering을 합친다. E1은 최종 population/plasma로 `compute_bf_opacity`를 다시
호출했다. 캡처는 `chi_abs` 또는 BF/FF 각각을 저장하지 않았으므로
`chi_total-chi_es`의 8.2299% 차이를 continuum과 thermal-line 사이에 유일하게
배분할 수 없다.

### 4.3 EPAY와 deposition

EPAY=2는 `:739-829`에서 `(chi_abs+chi_line_th)*J`, BF Milne eta,
`chi_line_th*B(T_e)`, deposition, thick-bin 분기를 다시 사용해 셸별 source 모양을
덮어쓴다. 따라서 선/연속 입력 세대 차이가 `eta_fixed`에서 비선형으로 증폭된다.

권위 iteration-10 조립의 stdout scale은 s25/s38/s49 =
`1.850e0 / 7.954e-1 / 1.682e-2`; 동일 E1 재생 명령은
`1.599e0 / 7.630e-1 / 1.649e-2`를 출력했다. EPAY gate 자체가 다른 것이 아니라
그 장부 입력이 다른 것이다. 캡처가 pre-EPAY eta와 scale 배열을 저장하지 않아
35.1702% `eta_fixed` 갭을 line/continuum/deposition/shape로 더 쪼갤 수 없다.

### 4.4 게이트와 빈 경계

환경은 권위 stdout에서 복원됐고 grid 세 배열은 bitwise 항등이다. 선 중심의 bin
계산도 동일 `nu_min`, `d_log_nu`, `floor(log(...))`를 사용한다. 따라서 이번 갭의
원인으로 게이트/빈 경계 차이는 **기각**한다. 다만 다른 epoch의 tau가 같은 경계에
투영되므로 각 bin의 합 자체는 달라진다.

## 5. 왜 현재 자산으로 E2를 더 닫을 수 없는가

한 셀에서 캡처가 주는 opacity 정보는 `chi_total`과 `chi_es` 두 값뿐이다. 생산 입력은
적어도 electron scattering, BF, FF, thermal-line, coherent-line의 다섯 합성분이다.
emissivity도 저장된 것은 EPAY 이후의 한 `eta_fixed`뿐이며, 그 전 line eta,
continuum eta, deposition과 EPAY scale/branch는 없다. 미지수가 관측 집계보다 많아
역문제가 비유일하다.

E2를 실제로 닫으려면 같은 iteration/generation에서 최소 다음을 함께 동결해야 한다.

- exact `T_e`, `n_e`, ion population, full/fine-level population 및 super-level member
  fraction;
- per-line `tau_sobolev`, `eps_l`, line-to-bin index;
- BF `chi/eta`, FF, electron, line thermal/coherent의 조립 전 배열;
- deposition 배열, thick-bin mask, EPAY `acc_abs/acc_emit/acc_w/acc_dep`와 최종 scale;
- 해당 binary의 `lumina.h`, `lumina_plasma.c`를 포함한 전체 source manifest.

현재 규율은 신규 모델/GPU run을 금지하며 기존 run에는 이 snapshot이 없다. 따라서
추가 실행으로 메우지 않고 정직하게 UNRESOLVED로 종료한다.

## 6. E3 관문 처리

E2가 FAIL이므로 다음 작업은 모두 0건이다.

- `eps_l*B(T_e) -> (h*nu/4pi)*A_ul*n_u` 단일 인자 교체;
- super-level member 분배를 포함한 capture-epoch `n_u` 소비;
- 후보 chieta 생성;
- stage31 driver 실행과 `J_det` 생성;
- §7.2 `J_det(A_ul*n_u)/CMFGEN` 대역표 및 “붕괴/유지/부분” 판정.

최종 E3 상태는 물리 좌표의 기각도 채택도 아닌 **NOT RUN / UNRESOLVED**다. E1의
11.98배와 비교하는 새 수치를 만들지 않았다.

## 7. 재현 명령

workspace root에서 실행한다.

```bash
python3 -m py_compile scripts/emiss_assembly_identity_e2.py

# 감사기는 E2 FAIL을 뜻하는 RC=2를 반환한다.
python3 scripts/emiss_assembly_identity_e2.py
test $? -eq 2

python3 scripts/cmf_chieta_check.py \
  /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605/chieta_iter10
python3 scripts/cmf_chieta_check.py validation/emiss_e1/chieta_A_replay

# E1 재조립 및 EPAY scale/동일 SHA 재현(약 50초, 출력은 /tmp만 사용).
gcc -O2 -w -std=gnu11 -D_GNU_SOURCE -DLUMINA_FROZEN_ORACLE \
  -ffunction-sections -fdata-sections -Isrc -Wl,--gc-sections \
  -o /tmp/emiss_e2_probe scripts/emiss_population_swap_e1_driver.c \
  src/lumina_plasma.c src/lumina_element_wide.c \
  src/lumina_atomic.c src/lumina_cmfgen.c -lm
/tmp/emiss_e2_probe \
  /gpfs/kjhan/lumina_runner2/scratch/chieta_capture_parity59_188605 \
  data/tardis_reference_toy06_19p48d_sivcaiv \
  validation/emiss_e1/cmfgen_b_populations.bin /tmp/emiss_e2_probe_out
sha256sum /tmp/emiss_e2_probe_out/chieta_A_replay
```

마지막 SHA는
`fa13326d50fff3d84d89bed0041a7b85828bd91952fcddb134d9942f38f625e1`이어야 한다.
신규 모델/GPU run, `src` 수정, clamp/floor, stage31 E3 실행, 커밋은 없었다.
