# 과업 T2/N9 — 인구-native 단일인자 시험과 EPAY 폐기 국지화 (1단: 스크립트+사전등록)

## 0. 이 단계에서 할 일과 하지 않을 일

**할 일**: 분석 스크립트 1개와 사전등록 문서 1개를 작성한다.
**하지 않을 일**: 무거운 연산을 실행하지 않는다. 로그인 노드 연산은 규약상
금지다(CLAUDE.md 실행 티어). 137 MB 이진 파싱과 재조립은 운전석이
grammar-debug 노드로 투척한다. 스크립트가 **인자만 받아 그대로 돌아가도록**
작성하라. 소규모 자기검사(수십 KB 이하 합성 데이터)는 허용한다.

산출물 2개:
- `scripts/uv_t2n9_offline.py`
- `validation/uv_t2n9/PREREG.md`

## 1. 계기 (방금 착지)

디렉터리 `/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/`

| 파일 | 내용 |
|---|---|
| `linepop_iter10` (137 MB) | LCMFLP01-v1. (line, shell)별 n_l, n_u, tau_used, tau_from_pops, S_l^pop, S_l^used, eps_l, w + flags + EPAY disposition |
| `linepop_iter10.manifest.json` | 아래 전재 |
| `chieta_iter10` | LCMFCE01. 조립된 chi, eta (내림차순 nu) |
| `emiss_ab_iter10.{A,B,B2}` | E-계열 A/B 판본 |

매니페스트 요지: `iteration=10`, `field_generation=10`, `n_shells=50`,
`n_bins=1000`, `selected_shells=5`, `selected_lines=601371`, `rows=1169145`,
`row_bytes=76`, `lambda_window_A=[600,3000]`, `tau_min=1e-12`,
`chi_line_roundtrip_bitwise=true`, `eta_line_epoch="pre-EPAY, pre-split"`,
`epay_disposition_counts={legacy_source:5000, thick_exempt:10696,
rate_shape_replaced:34304, scalar_rescaled:0}`,
`epay_scale_not_reproducible=true`, `clamp=0`, `fallback=0`,
게이트 `eps_phys=1, src_nlte=0, epay=2, epay_smin=5, epay_taubin=10, epay_hotf=0`.

**레코드 배치는 추측하지 마라.** R-T2 계기를 심은 writer 소스(`src/` 안,
`LCMFLP01` 문자열로 grep)를 읽어 필드 순서·타입·엔디언·헤더 길이·부속
테이블(선 인덱스 테이블로 보이는 약 48 MB가 별도로 존재한다)을 소스에서
확정하고, 그 근거를 파일·줄 번호로 사전등록에 적어라. 매니페스트의
`rows*row_bytes`와 실제 파일 크기가 맞지 않는 이유를 설명하지 못하면
fail-closed로 멈춰라.

## 2. 과업 T2 — 인구-native chi+eta 단일인자 시험

기존 T2가 UNRESOLVED로 남은 이유는 하위준위 population과 선별 line-chi
분해가 없어서였다(`docs/CODEX_UV_T1T2.md`). 그 계기가 이제 있다.

세 판본을 **오프라인에서 동일한 수송 연산자로** 재조립하고 UV 밴드
B0~B4에서 CMFGEN 대비 비를 낸다.

- **A**: 현행 조립 그대로 (기준선). `chieta_iter10`과 bitwise 일치해야 한다.
- **B2**: eta만 인구형으로 교체 (기존 E-계열 판본, chi는 A와 동일).
- **C**: **chi와 eta를 둘 다 인구에서** — chi는 `tau_from_pops` 기반,
  eta는 `chi * S_l^pop`. 이것이 그동안 구성 불가였던 단일인자 시험이다.

### 사전등록 판독 (측정 전에 확정하고 PREREG.md에 적을 것)

세 갈래 중 어디에 떨어지는지가 그대로 결론이다. 사후에 기준을 고르지 마라.

1. **C가 A를 몇 퍼센트 이내로 재현** → 조립형과 인구형이 이미 등가이고,
   UV 초과는 전적으로 **연산자**(결맞음 산란 채널) 몫이다. Stage 3.2(ALI)가
   유일 전선임이 확증된다.
2. **C가 A와 유의하게 다르나 여전히 CMFGEN보다 훨씬 큼** → 조립과 연산자가
   함께 기여한다. 각각의 몫을 정량하라.
3. **C가 CMFGEN 수준으로 내려옴** → 조립이 전부였다. (E6·T1과 상충하므로
   이 경우 두 결과의 모순을 반드시 해명하라.)

"몇 퍼센트 이내"의 경계값을 **측정 전에** 숫자로 못박아라. 근거도 함께.

## 3. 과업 N9 — EPAY 폐기의 국지화와 기전 확인

매니페스트는 전역 셀 수만 준다. 다음을 추가로 낸다.

1. **셸별·밴드별 처분 분율.** s>=5의 45,000셀 중 76.2%가
   `rate_shape_replaced`라는 전역 수치를, UV 밴드(B0~B4)와 셸 축으로 분해하라.
2. **에너지 가중 분율.** 셀 수 가중은 물리적 무게를 왜곡한다. **UV 방출
   에너지의 몇 퍼센트가 `rate_shape_replaced` 셀에서 나오는가**가 본 판정의
   핵심 수치다.
3. **기전 확인.** `rate_shape_replaced` 셀에서 재구성된 선 선원함수가
   실제로 `B(T_e)`인지 덤프에서 직접 확인하라. 참이면 그 영역에서는 인구가
   어떤 값이어도 형광이 나타날 수 없다는 것이 구성상 따라온다.
4. **`epay_scale_not_reproducible=true`의 정체.** 이 플래그를 세운 조건을
   소스에서 찾아 무엇이 재현 불가인지 서술하라. 재현 불가라면 그 자체가
   결함 후보이므로 대장 등재 문안까지 작성하라.

## 4. 규율 (전건 필수)

- **세대 fail-closed**: `iteration=10`, `field_generation=10`,
  linepop sha256 `84d1849dafd1c796dac77c4037b19683e3ef1d5ddb72dd0e6bf701490b05a1cc`
  를 검사하고 불일치 시 즉시 중단하라.
- **음성 대조 의무**: A 판본 재조립이 `chieta_iter10`과 bitwise 일치하는
  것을 확인한 뒤, **의도적 결함을 주입해 그 검사가 FAIL하는 것을 시연**하라.
  시연 없는 PASS는 자격이 없다.
- **clamp/floor 금지**: 스크립트 어디에도 clamp·floor·fallback을 두지 마라.
  비유한값이 나오면 세지 말고 중단하라. clamp/fallback/nonfinite 카운터는 0이어야 한다.
- **결정론**: 동일 입력 2회 실행이 byte-identical이어야 한다(스크립트에
  자기검사로 넣어라).
- `src/` 아래 생산 코드를 수정하지 마라. GPU·모델 런을 돌리지 마라.
  git commit 하지 마라.
- 스크립트는 `--linepop`, `--chieta`, `--outdir`를 인자로 받고, 무거운 경로에
  진입하기 전에 입력 검증을 마쳐라. 실행 시간·메모리 상한을 문서화하라
  (grammar-debug는 32코어 공용 노드다).

## 5. 보고

전체 보고서는 `docs/CODEX_UV_T2N9_STEP1.md`에 쓰고, `-o` 요약에는 다음만
담아라: 레코드 배치 확정 근거, 사전등록 경계값과 그 근거, 스크립트 실행
명령 한 줄, 예상 자원 사용량, 미해결 항목.
