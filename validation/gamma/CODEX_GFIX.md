## 1. 진단

- [실측] `heating_rate`의 정상적인 `0.0`이 온도 계약의 `te[i] <= 0.0` 조건에 걸려 모든 덱에서 발행이 실패했습니다.
- [설계] `erg/s/cm³` 데이터를 `"A2-07:T_e:K:..."` 도메인으로 해시해 도메인과 불변식 모두 잘못 상속했습니다.
- [설계] `nonthermal_ioniz_rate`에는 독립 매니페스트가 없어 소비자 측 변조 검사도 불완전했습니다.

수정 후 두 배열은 각각 다음 계약을 사용합니다.

- `GAMMA:q_dep:erg/s/cm3:IEEE754:shell-order:v1`
- `GAMMA:nonthermal_ioniz_rate:ionizations/s/cm3:IEEE754:shell-order:v1`
- 공통 유효성: 유한하며 `>= 0`; 전 셸 `0` 허용
- 발행과 `gamma_deposition_require()` 모두 두 매니페스트를 동일하게 계산·검사

`compute_gamma_deposition` 계산식은 한 글자도 변경하지 않았습니다.

## 2. 패치

생략 없는 unified diff:

[gamma_deposition_owner_nc3.patch](/tmp/claude-10396/codex_gfix/gamma_deposition_owner_nc3.patch:1)

SHA-256:

```text
907259d20ec37ac08dc140da719db948c4522700e596630611e71a1d45e74468
```

필수 적용 검사를 정확히 수행했고 통과했습니다.

```sh
cd ./scratch
git apply --check ../gamma_deposition_owner_nc3.patch
# PASS (exit 0)
```

검증 결과:

- 변경된 C 파일 5개 `gcc -c`: PASS
- `lumina_cuda.cu` `nvcc -dc`: PASS
- 전 셸 0 발행, 도메인 분리, 비열 배열 변조, 음수·무한값 거부 하네스: PASS
- `git diff --check`: PASS
- 새 `getenv`, clamp, floor: 없음
- `population_te_manifest_sha256`: 수정하지 않음

CPU 컴파일에는 기존 코드의 경고 두 건(`coupled_photoion_rate_jnu` 정적 선언, `setenv` 선언)이 있었지만 객체 생성은 성공했습니다.

## 3. 재발방지

패치 전체의 매니페스트·계약 재사용을 다시 훑었습니다.

- 감마 배열에서 빌려온 온도 계약 호출은 모두 제거했습니다.
- 남은 `population_te_manifest_sha256` 호출은 실제 `T_e` 또는 `c.T_e`만 해시하므로 양수 불변식이 올바릅니다.
- `heating_rate`와 `nonthermal_ioniz_rate`는 SHA-256 구현만 공유하고, 각각 독립된 도메인과 저장된 도장을 가집니다.
- 이 패치 안에서 다른 물리량이 타 물리량의 도메인·유효성 계약을 빌려 쓰는 사례는 더 발견되지 않았습니다.