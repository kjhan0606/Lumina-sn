# Codex A-S31 round 5 — MPFR 인증 재생 및 KA3 최종 재판정

상태: **rung8 UNRESOLVED / STOP; rung9--11 NOT RUN**  
정본: `docs/CODEX_STAGE31_DESIGN_REV4.md`, 원설계 `docs/CODEX_STAGE3_1_CMF_FIELD_DESIGN_2026-08-01.md` §6  
판정일: 2026-08-01

## 1. 결론

이 환경에는 MPFR 런타임 `libmpfr.so.6`과 RPM `mpfr-4.1.0-7.el9.x86_64`은 있으나, 인증 재생을 컴파일하는 데 필요한 `mpfr.h`, `mpfr.pc`, `mpfr-devel`이 없다. 저장소와 `/usr`, `/usr/local`, `/opt`, `/home/kjhan`에서도 헤더를 찾지 못했다. 따라서 REV4가 고정한 MPFR directed-rounding certificate replay를 구현·컴파일·실행할 수 없으며 rung8은 **UNRESOLVED**다.

`mpmath`, long double, 수동 MPFR ABI 선언, binary64 radius 미세조정으로 대체하지 않았다. 이들은 각각 MPFR directed-rounding C 계약을 충족하지 않거나 감사 가능한 개발 인터페이스를 우회한다. 기존 double 생산 중심 계산과 기존 `src/`는 건드리지 않았다.

rung8 인증 gate가 닫히지 않았고 차터가 신규 런을 금지했으므로 `(1024,4096)` 계산을 실행하지 않았다. 이에 따라 새 최상위 triple의 KA3 전 항 PASS를 선언할 근거가 없고 rung9는 **NOT RUN / UNRESOLVED**다. KA3 전 항 PASS를 전제로 한 rung10 KA2와 rung11 무가속 coherent scattering에는 진입하지 않았다.

| rung | 내용 | 판정 |
|---:|---|---|
| 8 | 고정 정밀도 MPFR directed-rounding certificate replay | **UNRESOLVED / STOP** |
| 9 | `(1024,4096)` 추가 및 최상위 triple KA3 재판정 | **NOT RUN / UNRESOLVED** |
| 10 | KA2 Nyström oracle | **NOT IMPLEMENTED / NOT RUN** |
| 11 | 무가속 coherent-scattering iteration | **NOT IMPLEMENTED / NOT RUN** |

## 2. rung8 MPFR 가용성 판정

읽기 전용 환경 probe 결과는 다음과 같다.

```text
pkg-config --modversion mpfr
  Package 'mpfr' not found

rpm -q mpfr mpfr-devel
  mpfr-4.1.0-7.el9.x86_64
  package mpfr-devel is not installed

ldconfig -p | rg libmpfr
  libmpfr.so.6 => /lib64/libmpfr.so.6

find ... -name mpfr.h -o -name mpfr.pc
  결과 0건
```

런타임 shared object만으로는 `mpfr_t`, 정밀도/지수 형식과 함수 prototype을 제공하는 공개 개발 계약을 컴파일할 수 없다. 헤더 구조를 역으로 선언해 `dlopen`하는 방식은 감사 가능한 MPFR 소스 인터페이스가 아니므로 사용하지 않았다. `gmpy2`도 설치되어 있지 않으며, 설치된 `mpmath`는 요구된 MPFR directed rounding의 대안으로 사용하지 않았다.

따라서 사전등록 필드는 값을 발명하지 않고 다음처럼 남긴다.

| grid | 요구 bits | certified sign-uncertain | certified non-finite | certified min lower | certified max width | 최초 unresolved 좌표 | 판정 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 128x512 | 2048 | 측정 없음 | 측정 없음 | 측정 없음 | 측정 없음 | 측정 없음 | **UNRESOLVED** |
| 256x1024 | 2048 | 측정 없음 | 측정 없음 | 측정 없음 | 측정 없음 | 측정 없음 | **UNRESOLVED** |
| 512x2048 | 2048 | 측정 없음 | 측정 없음 | 측정 없음 | 측정 없음 | 측정 없음 | **UNRESOLVED** |

사전등록 요구값은 전 격자 `certified_sign_uncertain_count=0`, `certified_nonfinite_count=0`이다. 여기서 “측정 없음”은 0이 아니며 PASS로 해석할 수 없다.

### 2.1 기존 scalar enclosure 수치의 지위

아래 값은 round 4의 legacy binary64 독립-radius 진단이다. MPFR certificate로 바꾸어 부르지 않으며 rung8 acceptance에 사용하지 않는다.

| grid | legacy sign-uncertain | legacy non-finite |
|---:|---:|---:|
| 128x512 | 51,036 | 0 |
| 256x1024 | 242,152 | 0 |
| 512x2048 | 1,019,773 | 28 |

## 3. rung9 KA3 최종 재판정

### 3.1 기존 실측과 신규 사전등록의 분리

기존 두 격자의 중심 수치는 `docs/s31_results/ka3_rev3.json`에서 그대로 옮겼다. `(1024,4096)` 열은 REV4 사전등록일 뿐 실측이 아니다.

| grid | 값 종류 | profile L1 | profile L2 | centroid error | area error | residual |
|---:|---|---:|---:|---:|---:|---:|
| 256x1024 | 기존 실측 | `4.2297852449e-4` | `3.8350100080e-4` | `1.3306738988e-8` | `1.3314833138e-8` | `3.2192980898e-11` |
| 512x2048 | 기존 실측 | `1.0565893300e-4` | `9.5801490261e-5` | `3.3380977843e-9` | `3.3390294475e-9` | `2.9878041237e-11` |
| 1024x4096 | 사전등록만 | 중심 `2.64e-5`, 창 `[2.50,2.80]e-5` | 측정 없음 | 측정 없음 | 측정 없음 | 측정 없음 |

새 공식 triple은 `(256,1024)/(512,2048)/(1024,4096)`이어야 한다. 그러나 fine level이 없으므로 다음 항목은 모두 최종 판정 불가다.

| 항목 | 문턱/사전등록 | round 5 측정 | 판정 |
|---|---:|---:|---|
| finest profile L1 | `<=1e-4`; 등록 창 `[2.50,2.80]e-5` | 없음 | **UNRESOLVED** |
| finest profile L2 | `<=1e-4` | 없음 | **UNRESOLVED** |
| finest centroid error | `<=1e-4` | 없음 | **UNRESOLVED** |
| finest invariant-area error | `<=1e-4` | 없음 | **UNRESOLVED** |
| official triple `p_obs(L2)` | `[1.8,2.2]` | 없음 | **UNRESOLVED** |
| finest transport residual | `<=1e-4` | 없음 | **UNRESOLVED** |
| boundary fractions | 각각 `<1e-12` | 없음 | **UNRESOLVED** |
| clamp / solution-negative | 공식 triple 각각 0 | fine 없음 | **UNRESOLVED** |
| certified sign-uncertain / non-finite | 공식 triple 각각 0 | 인증 없음 | **UNRESOLVED** |

기존 `256->512` 차수 `2.0011103392368392`를 새 공식 triple의 차수로 재사용하지 않았다. 새 공식 값은 `L2(512)/L2(1024)`이므로 fine 실측 없이는 계산할 수 없다. 외삽값도 실측값으로 승격하지 않았다.

### 3.2 최종 KA3 verdict

KA3는 PASS도 FAIL도 아니다. **필수 인증과 fine-grid 실측이 없어 UNRESOLVED**다. acceptance 문턱, 제외 cell, tail, tolerance는 변경하지 않았다.

## 4. rung10 KA2 및 rung11 산란

차터는 “KA3 전 항 PASS 시”에만 rung10과 rung11 진입을 허용한다. 그 전제가 성립하지 않았으므로 두 구현 patch와 실행 산출물을 만들지 않았다. 빈 patch 또는 NOT-RUN marker를 적용 가능한 구현 patch로 위장하지 않는다.

| 요청 patch | 상태 | 이유 |
|---|---|---|
| `patches/s31_rung8.patch` | 생성 안 함 | MPFR 개발 인터페이스 부재로 인증 구현을 검증할 수 없음 |
| `patches/s31_rung9.patch` | 생성 안 함 | rung8 미해결 및 신규 런 금지로 최종 KA3 산출 불가 |
| `patches/s31_rung10.patch` | 생성 안 함 | KA3 전 항 PASS prerequisite 미충족 |
| `patches/s31_rung11.patch` | 생성 안 함 | KA3 전 항 PASS prerequisite 미충족 |

## 5. 규율 및 산출물

- acceptance 완화 0, clamp/floor/tail 제외 0.
- 기존 double 생산 계산 수정 0; 기존 `src/` 수정 0.
- KA/model/GPU 신규 실행 0. 수행한 명령은 파일/패키지 가용성 probe와 기존 JSON 읽기뿐이다.
- 커밋 0.
- 수치표: `docs/s31_results/round5_verdict_table.csv`.
- 작업 시작 전부터 존재하던 dirty worktree의 타 변경은 건드리지 않았다.

입력 provenance:

| 산출물 | SHA-256 |
|---|---|
| `docs/CODEX_STAGE31_DESIGN_REV4.md` | `2f1fc5bc1a6eaf9be04fdb0fda600679e801da7b00f161bedf58c649b8a9ee7a` |
| `docs/CODEX_STAGE31_IMPL4.md` | `a75e798aa0b9403b7d9328fe281ea8e3f38bc2809b1ba01468283119ff84a05c` |
| `docs/s31_results/ka3_rev3.json` | `f4f6e8a7813ef956fb3f650f23b05802eda1aece796c474932b480a97186b922` |
| `patches/s31_rung7.patch` | `5171612278e008db3303099f8d14f1d48ae6989c545ff1bec63cc1a0b9e80bae` |

해제 조건은 MPFR 개발 인터페이스를 제공한 뒤 정본의 고정 정밀도(`2048/4096 bit`)로 rung8을 재생하고, 별도 승인된 KA 실행으로 `(1024,4096)` 실측을 얻는 것이다. 두 결과가 모두 0-counter 및 기존 수치 문턱을 통과하기 전에는 KA3 PASS나 rung10/11 진입을 선언할 수 없다.
