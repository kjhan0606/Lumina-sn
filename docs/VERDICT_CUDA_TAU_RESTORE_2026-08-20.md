# 판정 — `src/lumina_cuda.cu:10160`·`:10196` CUDA 벌크 tau 복원의 등록부 지위 (GR-2b, 2026-08-20, 판정자 Fable)

발주: GR-2 판정문(`docs/VERDICT_TAU_BULK_TRANSPLANT_2026-08-20.md`) §6 의 GR-2b 발주 +
운전석 발주문(분담 개정14). 아래 [실측]은 전부 판정자가 HEAD(`513ee92`) 작업트리에서
직접 읽어 확인한 것이다. 실행 0회(로그인 노드 규약 — 빌드·selftest 미실행). 판정 대상
파일(`src/lumina_cuda.cu` · `scripts/check_tau_writer_generation.py` ·
`docs/VERDICT_TAU_BULK_TRANSPLANT_2026-08-20.md`)은 작업트리=HEAD 일치
[실측 `git status --porcelain` 무출력 · `git diff HEAD --stat` 무출력].

```
판정: (iv) 「진단 save/restore(관측자 상태 재생)」 — writer 도 transplant 도 아닌
      제3의 류. 계약 = 격리(env 가드 블록 밖 도달 불가) + 복원 항등(저장한 공적
      바이트의 재생) + 표면 폐포(블록이 변이하는 공적 표면 전부의 복원).
      현 실물은 이 계약을 부분 충족한다 — 배열 4종 바이트는 복원되나 tau 세대
      스칼라·tau_validity 는 비복원(armed 런 한정 잔존 결손, 대장 기재).

Q1 지위             = (iv). (i) 기각 — 생산이 아니라 등록 생산물의 재생; 괄호를 씌우면
                      census 의 "production write" 집합에 진단 관측자가 편입된다.
                      (ii) 기각 — preflight·세대 이식·소유권 이전 전부 부재.
                      (iii) 기각 — 클램프 판별식 통과(값 비변조)·기본 OFF·물리 무접촉.
                      단 armed 런에서 세대 무신호 복원 = 계약 letter 결손 → 붉은 대장행.
Q2 cuda_writers=0   = 조작은 아니나(ASSIGN census 실패 분기가 앞을 막는다) 측정면
                      과장이다. GR-4: 리터럴 폐지→실측 카운트 출력 + 벌크 census 신설
                      (전 src, 4건 전수 등록) + 비인증 잔여 명시. 명세 §3 (U1~U5+NC).
Q3 env 게이트       = 실재한다 [실측]. 8개 공적 벌크 memcpy 전부가
                      LUMINA_NLTE_FINAL_RESOLVE 단일 파생 가드 블록 안 — 조건부
                      도달성("도달 ⇒ env truthy")까지는 정적 인증 가능, 런타임 무장
                      여부·armed 런의 결손은 정적 인증 밖(§4).
Q4 동류 미인증 경로 = 있다. 같은 블록의 pops 복원 2·S_l 복원 2·jbar 주입 1+복원 1 —
                      전부 각자의 측정면에 census 부재 또는 사각(§5). 블록 밖 친척
                      3건(:10845·:1853·plasma:20136)은 별류로 처분.
태생 커밋           = 커밋 계보상 a97d0e1 (2026-08-05) [실측 pickaxe 2종·경로 한정
                      유일]. 저작 시점은 모른다(3주 덩어리 커밋 — git 복원 불가).
                      운전석의 "특정 실패"는 경로 무제한 pickaxe 재현으로 해명(§6).
```

---

## 1. 판정의 근거 실측

1. **[실측] 블록의 정체.** `src/lumina_cuda.cu:10094-10210`, `main()`(`:6826` 시작) 안
   pure-CMFGEN 수렴 후·스펙트럼 산출 전의 복합문. 구조:
   - `:10095-10097` — `static int fr_on` 을 `getenv("LUMINA_NLTE_FINAL_RESOLVE")` 에서
     1회 파생(기본 OFF). `fr_on` 의 출현은 이 4행이 전부다 [실측 grep — `:10095-10098`
     외 0건]. 블록 안 `goto`/라벨 0건 [실측].
   - `:10098` — `if (fr_on && nlte.enabled && enable_nlte)` 가드.
   - `:10134-10138` — 공적 pops/tau/S_l/jbar 를 사적 버퍼로 **저장**(공적 상태 무변조).
   - `:10143` — resolve_ema: `nlte_solve_all_gpu` 재솔브 → `:10159-10161` pops/tau/S_l
     **복원**. `:10164-10166` — (무장 시) raw jbar 스냅샷을 공적 `jbar_line` 에 **주입**
     → `:10168` resolve_raw 재솔브 → `:10195-10198` pops/tau/S_l/jbar **최종 복원**,
     `:10199-10201` 디바이스 tau 재업로드.
2. **[실측] 재솔브는 공적 tau/S_l 을 등록 writer 로 실제 재작성한다.**
   `nlte_solve_all_gpu`(`:1049-2064`)는 `:1938` 에서 `nlte_update_tau_sobolev` 를
   호출하고("Route both CPU and GPU solve lanes through one tau/source writer" 주석),
   그 impl(`nlte_update_tau_sobolev_with_authority`, `src/lumina_plasma.c:19457`)은
   require(`상대:7`)→쓰기(tau + `line_source_S` `:19591` + `tau_validity` 스탬프
   `상대:66-67,118-119`)→mark(`상대:143`) 괄호를 완주한다. 즉 armed 런에서 GATE1 은
   공적 tau 세대를 재솔브당 +1 전진시키고, **복원 memcpy 는 그 뒤에 세대 무신호로
   바이트만 되돌린다.** (참고: `:10316` 의 "GPU solve 는 tau writer 를 안 부른다"
   주석(2026-06-20)은 화석이다 — `:1938` 실물과 모순 [실측].)
3. **[실측] 복원 목록의 폐포는 불완전하다.** 저장·복원 목록 = 배열 4종 바이트뿐.
   `opacity.tau_required/computed/first_consumer_generation`(`src/lumina.h:219-221`)과
   `tau_validity` 슬랩은 저장도 복원도 안 된다. 따라서 armed 런의 블록-후 상태 =
   「바이트=수렴 세대 G 의 것, 세대 스칼라=G+1~G+2, validity=마지막 재솔브의 스탬프」.
   블록의 printf "converged state restored ... downstream spectra unaffected"(`:10202`)
   가 실제 인증하는 것은 배열 4종+디바이스 tau 바이트까지다.
4. **[실측] authority 슬랩은 무접촉.** `g_ew_tau_authority` 스왑 3곳(GR-2 §4-1)은 전부
   plasma 측이고 writer impl 은 그것을 입력으로 소비만 한다(`:19607` 위임) — GATE1 이
   authority 를 교란하지는 않는다.
5. **[실측] 하류 소비는 바이트 직접 소비다.** pure-exit 경로의 하류는 formal
   적분(`compute_formal_integral_spectrum`, `src/lumina_plasma.c:22002`)과 CMF 경로로,
   판독한 소비 지점(`:22113`·`:22251`·`:22625`)은 tau/S_l 바이트를 세대·validity 대조
   없이 읽는다(전수 판독은 아님 — §8). THEN-MC 분기는 스펙트럼 뒤 `:10324` 에서 등록
   writer 의 신규 괄호가 tau/S_l/validity/세대를 전면 재생산하므로 결손이 그 지점에서
   자연 폐합된다 [실측 코드 순서].
6. **[실측] 게이트의 사각과 문안.** `CUDA_ASSIGN`(`scripts/check_tau_writer_generation.py:29-32`)
   은 원소 대입 정규식뿐 — memcpy 무감. PASS 문안 `:114` 의 `cuda_writers=0` 은
   하드코딩 리터럴이다. 단 `:77-81` 이 CUDA 원소 대입 발견 시 PASS 도달 전에 FAIL
   반환하므로 **리터럴이 거짓을 찍는 경로는 없다** — 문제는 조작이 아니라 「writers」
   라는 낱말이 측정면(원소 대입, 2파일)을 넘는 전칭으로 읽힌다는 것이다. 음성 대조
   4번(`:94` rogue CUDA 원소 대입)은 실제 검출된다 — 원소 대입면은 진짜로 census 된다.

## 2. 질문 1 — 지위: **(iv) 진단 save/restore(관측자 상태 재생)**

**(i) "괄호 의무를 지는 writer" 기각.** census 의 계약 문서는 "every **production**
write" 다 [실측 docstring `:2`]. 이 경로는 생산이 아니다 — 새 물리값을 만들지 않고,
등록 writer 의 괄호 안에서 태어난(세대 G) 공적 바이트를 같은 블록 안에서 저장했다가
재생한다. 괄호를 씌우면 (a) 진단 관측자가 생산 writer 집합에 편입되어 closed-set 의
의미가 흐려지고, (b) "세대 G+3 이 이 바이트를 생산했다"는 기재가 되는데 그 생산의
실체는 재생이라 producer 기재가 실물과 어긋난다. 재괄호가 원리적으로 불가능한 것은
아니나(§H-2 원칙: 앵커를 고쳐 통과시키는 것은 인증 확장이 아니다), 옳은 기술(記述)은
"writer 등록" 이 아니라 "별류 등록 + 결손의 붉은 기재"다.

**(ii) transplant 기각.** GR-2 transplant 계약의 3요소 — 사적 생산의 괄호 폐합 증명,
fail-closed preflight, 값+세대 계보의 원자적 이식 — 가 전부 부재다 [실측: preflight
호출 0건·세대 필드 접촉 0건·사적 생산 없음]. 목적도 다르다: transplant 는 소유권
이전, 이것은 상태 되감기다.

**(iii) 위반 기각 — 단 무조건이 아니다.** 세 갈래:
- **클램프 판별식**("정확해가 위반 가능한 가드인가"): 통과 — 어떤 값도 선호 방향으로
  변조하지 않고, 정확해가 위반할 수 있는 가드가 아니다. 복원되는 바이트는 등록 생산물
  그 자체다.
- **도달성**: 기본 OFF env 게이트 뒤에 있고(§4), 게이트 OFF 런의 물리·장부에 무접촉.
- **armed 런의 계약 letter**: 여기가 유일한 결손이다. 복원 memcpy 는 census docstring
  이 금지하는 바로 그 형상 — "세대 카운터가 신호하지 않은 tau 셀 변경" — 이고(§1-2),
  복원 폐포도 불완전하다(§1-3: 세대 스칼라·validity 비복원 → armed 런에 바이트/메타
  불일치 상태 잔존; pure-exit 시 프로세스 종료까지 지속). 이것을 (iii) 전면 위반으로
  올리지 않는 근거: 피해면이 armed **진단** 런의 장부 의미론에 국한되고(하류 소비는
  바이트 직접 소비 [실측 §1-5]), 물리 수치 비변조이며, §H·§I·GR-2 의 일관 논리 —
  "코드는 정당, 잣대가 이 류를 상정 못 했다" — 의 적용 범위 안이다. **처분 = 두 결손
  (α 무신호 복원 · β 메타 비복원)을 붉은 대장행으로 기재**, 수리는 캠페인 밖(잣대 수리
  계약 — src 0줄), 재개 트리거: ①GATE1 블록 형상 변경 ②armed 런 산출물(levelpop_resolve
  덤프 이외의 것)이 판정 잣대로 쓰이게 될 때 ③src 동결 해제 후 첫 src-접촉 캠페인의
  rider. 수리 시 본보기 = 복원을 등록 writer 의 신규 괄호로 감싸거나(THEN-MC `:10324`
  가 이미 하는 일), 세대 스칼라·validity 를 저장·복원 목록에 넣는 것 — 어느 쪽인지는
  그때의 판정 사안이다.

**따라서 (iv).** 이 류의 계약을 문장으로 고정한다(GR-4 등록부 문안의 정본):

> **진단 save/restore 계약**: (1) **격리** — 저장·재솔브·복원 전체가 기본-OFF env
> 게이트의 단일 가드 블록 안에 있고, 게이트 OFF 런에서 블록은 도달 불가다.
> (2) **복원 항등** — 복원되는 각 공적 배열의 바이트는 같은 블록 안에서 저장된 공적
> 생산물과 동일하다(저장·복원 짝의 문면 대응). (3) **표면 폐포** — 블록이 변이하는
> 공적 표면 전부가 복원 목록에 있어야 한다. (4) **무신호의 명시** — 복원은 세대 계약
> letter 상 무신호 변이이므로 writer 로 세지 않되, 별도 등록부에 열거되고 초록불
> 문안이 이 류의 존재·게이트·비인증 잔여를 명시한다.
>
> 현 실물(`:10160`·`:10196`)은 (1)(2) 충족 [실측], (3) 부분 충족(배열 4종만 — 세대
> 스칼라·validity 누락, 그리고 재솔브의 접촉 표면 전수는 미검증 §8), (4) 는 이 판정과
> GR-4 가 신설한다.

## 3. 질문 2 — `cuda_writers=0` 문안의 처분과 GR-4 명세

**진단**: 조작 아님(실패 분기가 리터럴 앞을 막는다 [실측 §1-6]) — 그러나 측정면
과장이다. "writers" 의 실측면은 「원소 대입 정규식 × 2파일」인데 문안은 전칭으로
읽히고, 실제로 벌크 writer 2건이 그 사각에 있었다. §H-2·§I·GR-2 §6 에 이어 네 번째
실증이다. **처분 = 문안 정직화 + 정규식(벌크 census) 확장 + 류 등록부 신설 — 셋 다,
별도 스크립트 신설은 불요**(GR-4 기대 변경집합 (2)가 이미 memcpy 계열 신설을 소유한다
[실측 사다리 §6] — 그 안에 이 판정의 등록부를 싣는다).

**GR-4 가 구현할 명세 (U1~U5 — 전부 정적, 코드는 GR-4 의 몫):**

- **U1 (벌크 census, 전 src)**: 목적지 표현식에 `tau_sobolev` 를 포함하는 모든
  `memcpy`/`memmove`/`memset` 을 `src/*.c`·`src/*.cu` 전체에서 수집(GR-2 T5b 와 동일
  기구 — 이중 구현 금지, 같은 census 를 두 등록부가 나눠 소비한다). 오늘의 전수 실측
  기대치 = 정확히 4건: transplant 2(`lumina_plasma.c:21271`·`:21379`, GR-2 T1~T7) +
  save-restore 2(`lumina_cuda.cu:10160`·`:10196`, 본 판정). 등록 밖 매치 =
  `unregistered bulk tau writer (memcpy)` FAIL.
- **U2 (가드 앵커+블록 스팬)**: `fr_on` 파생 2행(`:10095-10097` 문면)과 가드행
  `if (fr_on && nlte.enabled && enable_nlte) {`(`:10098` 문면)을 **정확일치 pin** 하고,
  가드의 여는 중괄호부터 brace-매칭으로 스팬을 뽑아 save-restore 등록 2건이 그 스팬
  **안**임을 검사한다. 추가로 `fr_on` 토큰의 파일 내 출현이 이 4행뿐임을 검사(제2
  파생·재대입 금지). 스팬 추출은 GR-3 `gate_source_lib` 채택 — 단 이 스팬은 함수가
  아니라 가드 블록이므로 lib 에 "앵커-시작 brace-스팬" 이 없다면 그 확장도 GR-3/GR-4
  의 몫이다(현행 `function_span` 유용 금지 — §I-5 오매치 결함).
- **U3 (문면 pin)**: 두 복원 memcpy 의 현재 문면(`:10160`·`:10196`)을 정확일치로 각
  1건 pin. 스팬 안 그 외의 tau-목적지 벌크 호출 = FAIL(저장 memcpy `:10136` 은
  목적지가 사적 버퍼라 U1 census 에 원리적으로 안 걸린다 — 걸리면 그것대로 FAIL 이
  옳다).
- **U4 (문안 정직화)**: PASS 줄에서 하드코딩 리터럴 폐지. 명세(문구는 GR-4 재량,
  내용은 고정): ① CUDA 원소-대입 카운트는 **실측값**으로 출력(`cuda_assign_writers=<len>`
  류) ② 벌크 census 결과를 류별 카운트로 출력(`bulk_tau=4/4 registered: transplant=2,
  diag_saverestore=2(gate=LUMINA_NLTE_FINAL_RESOLVE)`) ③ 비인증 잔여의 존재를 두 판정문
  참조로 한 줄 명시(GR-2 §3 의 6류 + 본 판정 §3 말미 목록). 과장 금지가 이 캠페인의
  병명이다 — 초록불은 측정면을 말해야 한다.
- **U5 (음성 대조 — §E 의무, 이름 있는 사유 고정)**: (a) 복원 memcpy 1건을 가드 블록
  **밖**으로 복제 → `unregistered bulk tau writer (memcpy)` · (b) 가드행에서 `fr_on &&`
  제거 → `saverestore guard anchor missing` 계열 · (c) `fr_on` 파생 2행 삭제 → 동일
  계열 · (d) 스팬 **안** rogue 원소 대입 `opacity.tau_sobolev[0]=0.0;` → 기존
  `duplicate/unregistered CUDA raw tau writer`(save-restore 등록이 ASSIGN census 의
  면제가 아님의 생존 증명) · (e) 두 복원 중 1건의 문면 변조(예: 저장·복원 배열 전치)
  → U3 pin FAIL.

**이 표면이 인증하지 못하는 것 (U4-③의 목록에 이대로 — §H-2 교훈)**:
1. **런타임 env 상태** — 정적 인증은 "복원 실행 ⇒ `LUMINA_NLTE_FINAL_RESOLVE` truthy"
   조건부 도달성까지다. 어떤 런처가 무장하는지는 게이트 밖(§4 실측 참조).
2. **armed 런의 계약 결손 2건**(§2 α·β) — 등록은 용인 목록이지 무해 인증이 아니다.
3. **재솔브의 접촉 표면 폐포** — `nlte_solve_all_gpu` 가 배열 4종 밖의 공적 상태
   (파티션/within-SL 스탬프·솔버 내부 상태·`nlte_writeback_ion_stage` env 분기 등)를
   변이하는지의 전수는 미검증이다(§8) — "downstream spectra unaffected" 주장 전체는
   이 게이트의 측정면 밖.
4. **디바이스 표면**(`d_tau_sobolev` 정합) — GR-2 §3-5 와 동일.
5. **별칭·매크로·함수 포인터 우회** — 문면 census 의 원리적 한계(GR-2 §3-4·6 동일).

## 4. 질문 3 — env 게이트: 실재하고, 조건부 도달성까지는 정적 인증 가능

- **[실측] 게이트 실재**: §1-1. 8개 공적-배열 memcpy(`:10159-10161`·`:10166`·
  `:10195-10198`) 전부가 `:10098` 가드 블록 안이고, `fr_on` 파생은 단일·유일하며
  기본 OFF, 블록 안 라벨/goto 0건. 스냅샷 포획측(`:9485` `fr_snap`)도 같은 env 의
  독립 파생이나 목적지가 사적 버퍼라 공적 writer 가 아니다 [실측].
- **정적 인증 가능성**: 가능 — 단 **조건부 도달성**("이 memcpy 가 실행되면 env 가
  truthy 였다")의 형태로만. U2 의 앵커+스팬 검사가 그 인증 기구이고, U5-(a,b,c)가
  그 생존 증명이다. C 언어 수준의 잔여(매크로 재정의·longjmp 등)는 문면 census 가
  원리적으로 못 본다 — U4-③ 목록으로 정직하게 남긴다.
- **"게이트 꺼진 생산 런" 실태 [실측]**: 현행 정본 러너 3종(`run_coevolve_s01.sh` ·
  `run_det_convergence_2026-08-08.slurm` · `run_manual_det_with_tripwire.sh`)은 이 env
  를 설정하지 않는다. 무장 런처는 parity26~42 계열 + 구 결정론 프로브
  (`sbatch_gpu_determinism_probe_{h100,a100}.sh`) [실측 grep]. **K36 R6 실런(08-09~18,
  Codex 진행)이 어느 러너로 돌았는지는 이 판정에서 특정하지 못했다 — 모른다**(§8).
  만약 R6 가 구 프로브 스크립트 계열로 돌았다면 그 bit-identity 증명은 armed 경로를
  포함한 채의 identity 다 — identity 주장 자체는 그로써 훼손되지 않으나(결정론은 경로
  포함 여부와 무관한 byte 비교), "생산 경로와 같은 코드가 돌았다" 는 별개 주장이 된다.
  확인은 운전석 몫으로 남긴다.
- **도달 가능 시 지위 변동 여부**: 발주문의 조건 질문에 답한다 — armed 런에서 도달
  가능하고 실제로 공적 상태를 만진다. 그래도 (iv) 가 유지되는 근거는 §2-(iii) 기각
  문단의 세 갈래다. 단 그 유지는 **결손 2건의 붉은 기재와 한 몸**이다 — 기재 없는
  (iv) 는 §H-2 가 금지하는 인증 과장이 된다.

## 5. 질문 4 — 같은 계급의 미인증 경로: **있다 — 같은 블록 안 3표면 + 블록 밖 친척 3건**

같은 가드 블록 안, tau 와 동일한 save/restore 형상 [실측]:

| 표면 | 지점 | 측정면 소속 | 오늘의 census 실태 [실측] | 처분 |
|---|---|---|---|---|
| `nlte_level_populations` | `:10159`·`:10195` 복원 | A2-07 population 계약(세대 `lumina.h:704-705`) | `a2_07_population_census.py` 는 `src/lumina_plasma.c` **만** 읽는다(`:12`) — CUDA 사각 | 본 판정의 save-restore 등록부에 tau 와 같은 행으로 열거(U1 census 의 대상 배열 확장은 GR-4 재량 밖 — **A2-07 census 의 CUDA-사각 자체를 대장 기재**, 확장은 별도 단 발의 |
| `line_source_S` | `:10161`·`:10197` 복원 | tau 와 **같은 writer**(`:19591`)가 생산하나 S_l writer census 는 **어디에도 없다**(`a2_09` census 는 fallback census [실측 §1 판독], `a2_08` 은 allowlist 토큰뿐) | census 부재 자체가 발견 | 대장 기재: 「S_l writer 표면 무잣대」 — 소유 후보 = tau census 의 자매면(같은 writer 이므로 GR-4 류 확장이 자연스러우나 **계약 확장은 사다리 밖** — 발의만) |
| `jbar_line` | `:10166` **주입**(raw 스냅샷 게시) + `:10198` 복원 | 소속 잣대 없음(A2-06 은 line_jbar 추정자 모듈 계약 — 공적 `opacity.jbar_line` writer census 아님) | census 부재 | 대장 기재. 주의: `:10166` 은 복원이 아니라 **대체 필드의 일시 게시**다 — save/restore 계약 (2) 복원 항등의 적용을 받지 않는 별개 문장이며, 등록부에 실을 때 「inject」로 구분 표기할 것 |

블록 **밖** 친척(같은 sweep 이 적발 [실측 grep]) — 본 판정의 등록 대상이 아니고 각자
별류다:

- `src/lumina_cuda.cu:10845` — THEN-MC `LUMINA_NLTE_JBAR_POPS` 블록의 트립와이어
  revert(pops memcpy 되돌림 + **등록 writer 재괄호로 tau 재생산** `:10846` — tau 측은
  깨끗하나 pops 측은 무신호 벌크). env+트립와이어 이중 가드. **처분**: A2-07 CUDA-사각
  대장행에 같이 기재.
- `src/lumina_cuda.cu:1853` — `nlte_solve_all_gpu` **내부**의 C1 restore(중첩쌍 공유
  블록 되돌림) — 생산 스팬 내부의 알고리즘 단계다. writer census 의 "스팬 안" 개념이
  CUDA lane 에는 아직 없다는 사실만 기재.
- `src/lumina_plasma.c:20136` — 위와 동형의 CPU lane C1 restore. A2-07 census 의 기존
  측정면 안이니 그 census 가 어떻게 다루는지는 GR-4 범위 밖 — 기재만.

## 6. 태생 커밋 — 커밋 계보상 `a97d0e1`, 저작 시점은 모른다

- [실측] 경로 한정 pickaxe 2종이 **유일 커밋**을 낸다:
  `git log -S 'LUMINA_NLTE_FINAL_RESOLVE' -- src/lumina_cuda.cu` = `a97d0e1`(2026-08-05)
  단독. `git log -S 'restore the converged state so downstream formal' -- src/lumina_cuda.cu`
  = 동일 단독. `-G 'memcpy\(opacity\.tau_sobolev'` 도 동일 단독.
- [실측] 운전석의 "다른 커밋들" 은 **경로 무제한** pickaxe 로 재현된다:
  `git log -S 'LUMINA_NLTE_FINAL_RESOLVE'`(무제한) = 5커밋 — `513ee92`(GR-2 판정문이
  이 env 를 인용)·`79e11e6`(검증 원장)·`8d30991`·`9efb95c`(노브 대장)·`a97d0e1`.
  src 태생은 여전히 `a97d0e1` 하나다. GR-2 §6 의 "a97d0e1 태생 [실측]" 은 **확인**된다.
- **모른다**: `a97d0e1` 은 1,369파일·363,861삽입의 3주 동결-해소 덩어리라 [실측
  `git show --stat`] **저작 시점**은 git 에서 복원 불가다. 정황 [추정]: 이 게이트를
  무장·소비한 parity26-diag 캠페인이 07-25 이므로 저작은 그 무렵으로 추정되나, 무장
  런처들 자체도 `a97d0e1` 로 함께 커밋되어 [실측] git 증거로는 구별되지 않는다.

## 7. 분기 지시 — GR-4 가 할 것

판정이 (iv) 이므로 GR-2 §5-4 의 조건부("GR-2b 판정이 착지해 있으면 그 명세대로")가
발동한다 — known-violation 붉은 등재·부분폐합 분기는 **불발**. 단:

1. **사전등록 기대치 정정 확정**: 사다리 §6 P1 의 `cuda_writers=0` 문안 기대는 무효
   (GR-2 §5-4 의 선언을 본 판정이 확정). GR-4 집행 기록에 정정을 남기고, P1 기대를
   §3 U4 의 3요소 문안으로 교체한다.
2. **등록부 2부 구성**: transplant(GR-2 T1~T7) + 진단 save-restore(본 판정 U1~U5).
   벌크 census 기구는 하나(U1=T5b), 등록부가 둘이다. 등록부는 판정을 요구하는
   화이트리스트다 — 이 판정문 밖의 재량 등록 금지(사다리 §G-3-1 유지).
3. **NC 표 증보**: 부록 C-2 에 U5 의 5건을 GR-2 T7 의 5건과 별도 행으로 추가(사유
   문자열은 U5 의 것으로 고정).
4. **대장 기재 발주(운전석)**: ① §2 결손 α(무신호 복원)·β(메타 비복원) — 붉은 행,
   재개 트리거 3종 명문 ② §5 표의 3행(A2-07 CUDA-사각 · S_l writer 무잣대 · jbar
   무잣대+inject) ③ §5 블록 밖 3건 — 전부 「조용한 대장 기재」 규율로, 수리·확장 단은
   이 캠페인에서 개설하지 않는다(src 0줄 + 사다리 범위 계약).
5. **운전석 확인 1건(게이트 밖)**: K36 R6 실런의 러너가 `LUMINA_NLTE_FINAL_RESOLVE`
   를 무장했는지(§4) — 무장이었다면 R6 정본 기재에 "armed 경로 포함 identity" 를
   주석하라. 게이트·등록부와 무관한 기록 정확성 사안이다.

## 8. 이 판정이 모른다고 적는 것

1. **저작 시점**(§6) — 커밋 계보 밖이라 git 로 복원 불가.
2. **K36 R6 실런의 러너와 env 무장 여부**(§4) — 러너 스크립트 3종의 비무장만 실측했다.
3. **재솔브의 공적 접촉 표면 전수** — `nlte_solve_all_gpu`(1,016행)와 그 피호출부의
   전수 판독은 하지 않았다. 복원 목록(4배열+디바이스 tau) 밖의 부수효과 존재 여부는
   미검증이고, 그래서 §3 비인증 3항에 남겼다.
4. **하류 스펙트럼 경로의 전수** — §1-5 는 표본 판독(3지점)이다. 세대·validity 를
   대조하는 하류 소비자가 어딘가 있다면 armed 런 결손 β 의 피해면이 넓어진다 — 그
   발견은 지위를 바꾸지 않고 대장행의 무게만 바꾼다.
5. **armed 런 결손의 런타임 발화 실적** — 코드 문면 판정이다. 과거 parity/프로브
   산출물에서 바이트/메타 불일치가 실제 관측치를 오염시켰는지는 조사하지 않았다
   (그 산출물들은 levelpop 덤프 소비가 주였고, 덤프는 복원 **전** 기록이라 [실측
   코드 순서] 오염 경로가 좁다 — 단 이것도 [추정]이다).

— 판정자 Fable, 2026-08-20. 근거 파일: `src/lumina_cuda.cu` · `src/lumina_plasma.c` ·
`src/lumina.h` · `scripts/check_tau_writer_generation.py` · `scripts/a2_07_population_census.py` ·
`scripts/a2_09_emissivity_census.py` · `scripts/run_coevolve_s01.sh` ·
`scripts/run_det_convergence_2026-08-08.slurm` · `scripts/run_manual_det_with_tripwire.sh` ·
`scripts/launch_parity*.sh` · `scripts/sbatch_gpu_determinism_probe_*.sh` ·
`docs/VERDICT_TAU_BULK_TRANSPLANT_2026-08-20.md` · `docs/RUNG_GATE_REPAIR_LADDER_2026-08-20.md`
§4·§6 · `git`(pickaxe·show·status 실측).
