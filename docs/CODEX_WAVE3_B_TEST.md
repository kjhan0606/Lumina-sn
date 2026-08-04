## 최종 판정

Wave 3 Stage 2A는 채택 불가입니다. 판정은 **무개선**입니다. 단, `improvement <= 0`을 측정한 결과가 아니라 선행 topology gate 실패로 유효한 `p_elem`이 생성되지 않아 개선을 입증하지 못했다는 gate 처분입니다.

동시에 다음 실패가 확인됐습니다.

- `FAIL-TOPOLOGY`: S/Fe target coverage 및 I/V boundary gate 실패
- `FAIL-NUMERICS`: incomplete matrix 강제진단에서 `κ₂ > 1e12`
- `FAIL-OFF-INVARIANCE`: default/OFF 자체는 불변이지만, s8 ON-shadow가 대상 밖 s0·s43 oracle까지 변경

### 1. 재빌드·OFF 불변

CPU `bench_frozen_oracle` 재빌드 성공, GPU 실행과 git 명령은 사용하지 않았습니다. 검증 전후 source/header/Makefile/spec SHA manifest는 `cmp=0`입니다.

parity59 권위 frozen subset으로 미설정과 명시 `EW=0`을 독립 실행했고 3셀 모두 byte-identical이며 저장 oracle과도 일치했습니다.

| 셀 | SHA-256 |
|---|---|
| s0 | `7a79f4f345d5c8500b48530a7f859bbe05a57771f43e52f983639ef9a0839381` |
| s8 | `2ee175c6be4a0ab6ae07034722180c6312f0a6e55e1009fcdda23b72310693eb` |
| s43 | `f75b84a314e85831825aea3e2ef64d9bdbc1729c50e83132b6e02c2151b5cd8e` |

다만 s8 ON-shadow는 대상 밖 oracle도 변경했습니다.

| 셀 | OFF vs ON | 변경된 공통 행 |
|---|---:|---:|
| s0 | `cmp=1` | 31 |
| s8 | `cmp=1` | 49 |
| s43 | `cmp=1` | 36 |

원인은 ON이 전역 NLTE layout과 `sigma_bf`/target 자료를 바꾸기 때문입니다: [bench_frozen_oracle.c:654](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/bench_frozen_oracle.c:654), [lumina_atomic.c:985](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:985), [lumina_plasma.c:13969](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:13969).

### 2. s8 acceptance

두 shadow run의 EW dump 16개와 oracle 3개가 전부 byte-identical했습니다.

| 항목 | S II–IV | Fe II–IV |
|---|---:|---:|
| N / raw rank | 303 / 302 | 303 / 302 |
| target coverage | **574/581 실패** | **4076/4198 실패** |
| assembled route fail | 14 | 244 |
| 최초 결손 lower | global 4036, S II level 117 | global 11965, Fe II level 35 |
| boundary fraction | `1.958e-5` 실패 | `7.923e-5` 실패 |
| solve attempted | 0 | 0 |
| verdict | `EW_FAIL_SHADOW` | `EW_FAIL_SHADOW` |

따라서 명세상 solve 금지이며:

| acceptance | S | Fe |
|---|---:|---:|
| ion-fraction improvement% | N/A | N/A |
| 25% 문턱 | 미판정 | 미판정 |
| `b_k` median log-error improvement | N/A | N/A |
| absolute provisional median/p95 | N/A | N/A |
| gate 처분 | **무개선** | **무개선** |

후보 solution dump의 0 population을 이용한 비율 산정은 하지 않았습니다.

### 3. 조건수·잔차·pivot 위생

아래는 coverage 실패 matrix를 독립적으로 강제 equilibration/LU한 진단값이며 acceptance 해가 아닙니다.

| 항목 | S | Fe | 문턱 |
|---|---:|---:|---:|
| max channel column-sum | `2.808e-16` | `2.843e-16` | `<=1e-12` |
| equilibrated rank | 303 | 302 | 303 |
| `κ₂` | `3.892e12` | `1.760e13` | `<=1e12` |
| rcond | `2.570e-13` | `5.681e-14` | — |
| pivot growth | `0.9999998` | `1.0000000` | `<=1e8` |
| scaled SE residual | `9.40e-16` | `1.53e-15` | `<=1e-10` |
| conservation error | `4.15e-16` | `1.68e-16` | `<=1e-12` |
| permutation max ion-fraction Δ | `2.82e-11` | `3.63e-14` | `<=1e-10` |
| negative / nonfinite | 0 / 0 | 0 / 0 | 0 / 0 |

두 원소 모두 `1e12 < κ₂ <= 1e14`이므로 명세상 `CONDITIONING FAIL`입니다.

### 4. 역방향 축

- s0 Fe: target `4076/4198`, boundary fraction `1.3749e-2`; solve 미시도. II/III/IV별 `d_k`와 aggregate improvement는 N/A → **무개선**.
- s20 S: parity59 및 parity50 archive에 frozen J/C1/C2가 없어 `s20: frozen input load failed`, exit 1, EW dump 0개 → **무개선**.

따라서 역방향 축 회복과 production 확장은 주장할 수 없습니다.

### 5. ARTIS 대조

Lumina 내부 support/sign 위생은 확인됐습니다.

| plane | S offdiag | Fe offdiag |
|---|---:|---:|
| rad_bb | 7151 | 7736 |
| coll_bb | 7180 | 7736 |
| rad_bf | 400 | 402 |
| coll_bf | 404 | 402 |
| nt_bf | 2 | 2 |
| nt_bb / autoion | 0 / 0 | 0 / 0 |

모든 활성 plane의 off-diagonal sign mismatch는 0이고 보존행은 1개, charge row는 0개입니다. 그러나 동일 identity/checksum/frozen state의 ARTIS matrix dump가 저장소에 없어 rate dex, support equality, residual-vector, solution 비교는 수행 불가입니다. 따라서 §4.2 ARTIS PASS도 성립하지 않습니다.

주요 산출물: [S diagnostics](/tmp/w3b_p59s8a.oa23NN/lumina_ew_iter0011_z16_s008_diagnostics.csv), [Fe diagnostics](/tmp/w3b_p59s8a.oa23NN/lumina_ew_iter0011_z26_s008_diagnostics.csv), [s0 Fe diagnostics](/tmp/w3b_p59s0fe.0IMrr1/lumina_ew_iter0011_z26_s000_diagnostics.csv).