# 판정 — `:21271`·`:21379` 벌크 tau 이식의 등록부 지위 (GR-2, 2026-08-20, 판정자 Fable)

발주: `docs/RUNG_GATE_REPAIR_LADDER_2026-08-20.md` §4 (GR-2, 트리 무변조).
증거 패킷: 운전석(`/tmp/claude-10396/gr2/EVIDENCE.md`). 판정은 fresh Fable — 아래 [실측]은
전부 판정자가 HEAD(`6d16a9a`) 작업트리에서 직접 읽어 확인한 것이다. 실행은 0회(로그인 노드
규약 — 빌드·selftest 미실행). 판정 대상 파일(`src/lumina_plasma.c`·
`scripts/check_tau_writer_generation.py`)은 작업트리=HEAD 일치 [실측 `git status`/`git diff`].

```
판정: (ii) transplant — 괄호 의무 대신 「preflight 증명 + 세대 계보 이식」이라는
      별개 계약을 지는 별개 류

Q1 지위               = (ii). (i)은 범주 오류 — 재괄호는 장부를 거짓으로 만든다.
                        (iii) 기각 — 클램프 판별식 통과 + 설계 실물 + fail-closed 원자성.
Q2 정적 인증 표면      = 사전등록 최소선(스팬+호출 실존+필드 접촉)은 불충분.
                        검사 7종(T1~T7)으로 확장하고, 비인증 잔여 6류를 초록불 문안에 명기.
Q3 §H-5-3 ②·③ 처분   = 대장 기재로 종결 + 재개 트리거 명문. 이 캠페인 내 후속 단 미개설
                        (src 0줄 계약·A2-10 동결에 이중 저촉 — 집행 불능 단은 개설하지 않는다).
                        단 같은 계급의 부속 발견 1건(커밋 시점 공적 세대 결속 부재)을 추가 기재.
★신규 실측            = src/lumina_cuda.cu:10160·:10196 벌크 tau 복원 2건(a97d0e1 태생,
                        게이트 사각) — 등록 전 GR-2b 판정 발주 필요 (§6).
열린 질문 해소         = preflight 음성 대조는 소스에 실존한다. 단 전부 unwired 타깃 안이고
                        커버리지가 부분적이다 (§7).
```

---

## 1. 판정의 근거 실측 — 운전석 패킷에 없던 것 위주

패킷의 좌표·수치는 전부 재확인했고 오류 0건이다(§H-2 계열 좌표 포함; `f6c2eb6^` 0건 ·
`f6c2eb6` 2건은 `git show | grep -c` 로 재실측). 아래는 판정자가 **추가로** 읽은 사실이다.

1. **[실측] 괄호 의무는 후보 안에서 이미 이행된다.** 이식되는 tau 슬랩은 등록된 writer
   들이 후보의 사적 `OpacityState` 위에서 생산한 것이다:
   `nlte_population_candidate_produce_tau_source`(`src/lumina_plasma.c:20734`)가
   `compute_tau_sobolev`(`:20767`) → (env 시) `apply_overlap_corrections` →
   `nlte_update_tau_sobolev_with_authority`(`:20779`)를 **`&candidate->opacity`** 에 대해
   호출한다 — 세 이름 모두 게이트의 등록 writer(또는 그 impl)이고, 각자 자기 스팬 안에서
   require→write→mark 괄호를 정상 수행한다. 즉 **공적 슬랩에 도착하는 모든 tau 바이트는
   등록 writer 의 괄호 안에서 태어났다.** 이식은 미괄호 생산이 아니라 완성품의 이사다.
2. **[실측] 괄호 폐합은 두 번 검증된다.** 생산 직후
   (`:20784-20791` — `tau_required==0 || tau_computed!=tau_required` 이면 후보 abort) 와
   커밋 preflight(`candidate_material_commit_preflight` `:21066`, 조건 `:21168-21170`)에서.
   preflight 는 추가로 tau 슬랩 별칭 금지(`candidate->tau_sobolev==opacity->tau_sobolev`
   `:21102`)를 검사한다.
3. **[실측] 세대 계보는 공적 계보의 연속이다.** 후보의 opacity 뷰는 공적 객체의 얕은
   구조체 복사로 시작하므로(`src/nlte_population_candidate.c` `prepare_opacity_view`:
   `candidate->opacity = *public_opacity`) tau 세대 스칼라는 공적 값에서 출발하고, 사적
   괄호 실행마다 +1 된다. 이식되는 세대값(`:21296-21301`·`:21403-21408`)은 따라서
   순차 흐름에서 공적 커밋-전 값보다 **엄밀히 크다**. `tau_first_consumer_generation=
   computed` 이식은 은폐가 아니라 정직한 기재다 — 후보 내부의 A2-08/A2-09 생산이 그
   세대의 슬랩을 이미 소비했으므로, 0 으로 리셋하면 공적 첫-소비 장부가 거짓이 된다.
4. **[실측] fail-closed 원자성은 헤더 계약 그대로다.** 두 커밋 함수 모두 guard
   (`:21244`·`:21351`)가 첫 공적 바이트 변경 **앞**에 있고, 첫 memcpy 이후 실패 반환
   경로가 없다(본문 전수 판독 — memcpy·구조체 대입·포화 카운터 병합·free 뿐).
   실패 시 공적 상태 바이트 보존은 테스트가 memcmp 전수로 주장한다(§7).
5. **[실측] 생산 호출부는 2곳뿐이고 둘 다 build→commit→free 단일 소유 흐름이다**
   (`:16003` bundle — A2-10 production solve · `:16194` seed — A2-INIT predictor).
   후보 인터리브·이중 커밋은 오늘 호출 규율상 도달 불능이다. 단 이것은 규율이지 계약이
   아니다 — §4-3 에 부채로 잇는다.

## 2. 질문 1 — 지위: **(ii) transplant**

**(i) "괄호 의무를 지는 writer" 는 범주 오류다.** 괄호(require→write→mark)의 목적은
"세대 카운터가 신호하지 않은 tau 셀 변경이 없다"는 소비자 보증이다(게이트 docstring
[실측]). 이식 지점에 공적 require/mark 를 강제하면: (a) 공적 세대가 후보 계보와 무관한
번호로 다시 매겨져, 후보 안에서 그 슬랩을 소비하며 스탬프된 A2-08/A2-09 발행물의 세대
참조와 어긋나고, (b) `tau_first_consumer_generation` 장부가 "아직 미소비" 로 거짓말하게
된다. 즉 **재괄호는 실제로 일어난 생산을 기술하지 못하고, 일어나지 않은 생산을 인증하는
연극이 된다.** 괄호는 이미 옳은 자리(사적 생산)에서 이행됐다(§1-1).

**(iii) "위반" 기각.** 세 갈래로 검사했다:
- **클램프 판별식**: preflight 는 거부-가드다. 정확해(올바르게 지어진 번들)는 구성상
  이를 위반할 수 없고, 위반 = 부서진 후보이며 처분은 값 수리가 아니라 **거부 + 공적
  바이트 보존**이다 [실측 코드 + 테스트 주장]. 물리 수치를 만지는 지점이 없다.
- **계약 실질**: 이 경로는 §H-2 가 이미 "SH-RADEQ-5 의 설계된 트랜잭션 커밋" 으로 읽은
  실물이고, 그 설계가 지키려는 것 — 공적 상태의 원자적 세대 전이, 부분 발행 금지 — 을
  괄호 계약보다 **더 강하게** 지킨다(원소 writer 는 쓰기 도중 실패하면 반쯤 쓴 슬랩을
  남길 수 있으나, 이식은 preflight 뒤 무실패 구간만 갖는다 [실측 §1-4]).
- **일관성**: §H(원소 writer 2건)·§I(방출률 19건)의 판정 논리 — "코드는 정당, 잣대가
  이 류를 상정 못 했다" — 가 이 경로에 그대로 적용된다. 위반 선언은 잣대의 상정 실패를
  코드에 전가하는 것이다.

**따라서 (ii).** 이 류의 계약을 문장으로 고정한다(GR-4 등록부 문안의 정본):

> **transplant 계약**: 등록된 커밋 함수 안에서, (1) 등록 writer 들의 괄호 안에서 생산이
> 완결되고 괄호 폐합이 증명된 사적 슬랩을, (2) fail-closed preflight 가 공적 바이트
> 변경 전에 전건 검증한 뒤, (3) 값과 **세대 계보를 함께**, 원자적으로(첫 공적 바이트
> 변경 후 무실패) 공적 소유자에 이식한다. 괄호 의무는 지지 않는다 — 괄호는 생산 시점에
> 사적 객체에서 이행되며, transplant 는 그 결과의 소유권 이전이다.

## 3. 질문 2 — 정적 인증 표면: 최소선은 **불충분**, 검사 7종으로 확장

사전등록 최소선(등록 함수 스팬 내 + 스팬 내 preflight 호출 실존 + 세대 이식 필드 접촉
실존)의 결함 3개:
- "호출 실존" 은 **가드됨**을 인증하지 않는다 — memcpy 뒤의 호출, 반환값을 버리는 호출도
  실존이다.
- "필드 접촉" 은 `=0` 사보타주·전치 대입도 통과시킨다.
- 류의 **배타성**이 없다 — transplant 스팬이 원소 대입의 은신처가 되거나, 등록 밖 벌크
  쓰기가 남는 것을 막는 조항이 없다.

**GR-4 가 구현할 명세 (T1~T7 — 전부 정적, 전부 정확일치 pin, 코드는 GR-4 의 몫):**

- **T1 (스팬)**: transplant 등록부 = {`nlte_population_candidate_commit_bundle`(`:21238`),
  `nlte_population_candidate_commit_seed_material`(`:21346`)} — WRITERS 와 **서로소**
  (한 함수가 두 류에 속하면 FAIL). 스팬 추출은 GR-3 의 `gate_source_lib`(부록 B, 정의-앵커)
  채택 — 현행 `function_span` 은 호출부 오매치 결함(§I-5)이 있다.
- **T2 (가드 존재+위치+관용구)**: 각 스팬 안에 해당 preflight 의 **guard 블록 정확일치
  앵커**(`if(!candidate_bundle_commit_preflight(` … `return NLTE_CANDIDATE_COMMIT_FAILED;`
  / seed 변형)가 실존하고, 그 매치 오프셋이 스팬 내 **첫 벌크 tau 쓰기보다 앞**이어야
  한다. 호출 실존 검사로는 안 된다(위 결함 1).
- **T3 (벌크 쓰기 pin)**: 스팬당 tau memcpy 문장 정확일치 앵커 **정확히 1건**
  (`:21271`·`:21379` 의 현재 문면). 스팬 내 그 외의 tau-목적지 벌크 호출 = FAIL.
- **T4 (세대 이식 삼중 pin)**: `tau_required_generation`·`tau_computed_generation`·
  `tau_first_consumer_generation` 세 대입(`:21296-21301`·`:21403-21408`)을 **문장
  정확일치**로 pin — 접촉이 아니라 문면이다(위 결함 2). 위치는 벌크 쓰기 **뒤**(HEAD
  실물 순서) — 어긋나면 FAIL 로 리팩터 시 재판정을 강제한다.
- **T5 (류 배타성 census)**: (a) 원소 대입 census(기존 ASSIGN + GR-4 별칭 확장)가
  transplant 스팬 안에서 **0건**이어야 한다 — transplant 등록이 ASSIGN census 의 면제가
  되어선 안 된다. (b) 벌크 census(목적지 표현식에 `tau_sobolev` 를 포함하는 모든
  `memcpy`/`memmove`/`memset`)를 **`src/*.c`·`src/*.cu` 전체**로 넓혀 수집하고, 등록
  스팬 밖의 매치 = `unregistered bulk tau writer` FAIL. (현행 게이트의 2파일 한정은
  가정이었다 — 오늘은 전 src 에서 4건뿐임을 실측했으나(§6), 그 사실을 가정이 아니라
  검사로 만들라.)
- **T6 (preflight 본문 pin)**: `candidate_material_commit_preflight` 스팬 안에 tau 계약을
  지키는 두 조건의 정확일치 앵커 — (α) 괄호 폐합 조건(`:21168-21170` 문면), (β) tau
  별칭 금지 조건(`:21102` 문면). "preflight 호출 실존" 이 속을 비운 preflight 를
  가리키는 것을 막는다. **preflight 의 나머지 조건들(closure 1e-10·CDF 단조·status 전수
  등)은 이 게이트의 측정면 밖이다 — pin 하지 않고, 인증 범위에 넣지도 않는다**(그것은
  A2-08/09/10 계약의 몫).
- **T7 (음성 대조 — §E 의무)**: 신설 주입, 각각 이름 있는 사유 고정(부록 C-2 에 추가 —
  기존 "수리 후 6건" 은 이 판정으로 늘어난다): (a) guard 블록 제거 →
  `transplant not preflight-guarded:<함수>` · (b) memcpy 를 guard 앞으로 이동 → 동일
  사유(위치 검사의 생존 증명) · (c) 세대 대입 1건 제거 →
  `transplant does not carry the tau generation:<함수>` · (d) 스팬 밖 rogue 벌크 memcpy
  (= 부록 C-2 주입 5) · (e) transplant 스팬 **안** rogue 원소 대입 →
  `unregistered raw tau writer`(T5a 의 생존 증명).

**이 표면이 인증하는 것**: 가드 관용구의 실존·위치, pin 된 단일 벌크 쓰기, pin 된 세대
이식 문면, 두 census 패턴에 대한 류 폐포(src 전체), preflight 의 tau-관련 2조건 문면.

**인증하지 못하는 것 (초록불 문안에 이대로 명기하라 — §H-2 교훈)**:
1. **preflight 의 런타임 실행·통과** — 정적 게이트는 원리적으로 못 본다. 실행 증거는
   unwired selftest 2건에 있고 그 처분은 GR-7 이다(§7).
2. **preflight 본문의 의미론적 완전성** — pin 한 2조건 밖의 조건이 약화되어도 초록이다.
3. **커밋 시점 공적 세대 단조성** — 코드에 그 검사가 없으므로(§4-3) 게이트가 인증할 수
   없다. 없는 것을 있는 것처럼 초록이 말하면 안 된다.
4. **임의 지역 별칭 경유 쓰기**(`double*t=...->tau_sobolev; t[i]=...`) — 정규식 census 의
   원리적 한계. GR-4 별칭 확장은 명명된 별칭 집합까지다.
5. **디바이스 측**(`d_tau_sobolev`) 정합 — 이 게이트의 측정면 밖.
6. **함수 포인터·매크로 우회** — 오늘 0건 [실측]이나 census 는 문면 검사다.

## 4. 질문 3 — §H-5-3 ②·③의 처분: **대장 기재로 종결 + 재개 트리거**

### 4-1. ② 슬랩↔authority 런타임 결속 부재 — 확인, 대장 기재

[실측] `g_ew_tau_authority` 는 맨 `int*`+count 전역(`:701-702`)이다. 스왑 지점 3곳
(`:20316` 레거시 solve 경로 · `:21310` bundle 이식 · `:21417` seed 이식) 전부 슬랩
설치와의 **인접성**만으로 결속하고, 배열에 자기 슬랩·세대와 묶는 스탬프가 없다. 소비
술어(`nlte_tau_line_shell_authorized_by` `:8933`)는 `nshells` 일치만 검사한다 — 세대 G 의
슬랩에 세대 G′ 의 authority 가 붙어도 nshells 만 맞으면 무감이다.

**처분 = 대장 기재로 종결(이 캠페인).** 근거: (a) 수리는 src 편집이고, 이 캠페인은
src 0줄 계약 + A2-10 귀속 동결에 이중으로 저촉된다 — **집행할 수 없는 단의 개설은 계획
연극이다**; (b) 오늘 도달 가능한 발화 경로가 없다 — 스왑 3곳 전부 슬랩 갱신과 같은 문장
블록 안이다 [실측]; (c) known-red 교훈(처분 없는 등재 금지)은 **재개 트리거의 명문**으로
충족한다. **재개 트리거**: ①스왑 3곳 중 어느 하나의 형상 변경 ②후보 동시성·제2 커밋
경로 신설 ③src 동결 해제 후 첫 src-접촉 캠페인(그 사전등록에 이 행을 rider 로 실을 것).

### 4-2. ③ authority 거부 카운터 부재 — 확인, 대장 기재

[실측] 술어는 거부 시 0 반환뿐, 카운터 증가 없음. 소비자는 조용히 LTE 폴백하되 값은
정의되고 validity 로 표기된다(`a209_upper_population_for_tau` `:8953-8999`) — §H-4-3 의
"조용한 성능저하는 아니나 조용한 분기" 판정을 재확인한다. **처분 = 대장 기재로 종결**,
재개 트리거는 4-1-③ 과 동일(동결 해제 후 첫 계측 확장의 rider). 발행 값의 은폐가
아니므로 ② 보다도 완만한 부채다.

### 4-3. ★부속 발견(같은 계급, 추가 기재): bundle 커밋의 공적 세대 결속 부재

[실측] seed preflight 는 공적 상태에 **직접 결속**한다(`:21211` 이하 —
`plasma->T_e_generation`·`atom/nlte population_committed_generation`·공적 te_publication
manifest 대조). 그러나 **bundle 경로의 preflight 는 후보 내부 정합만 검사하고, 커밋
시점의 공적 세대와는 아무 비교도 하지 않는다** — required 세대는 begin 시점에 공적+1 로
파생될 뿐이다(`:15914-15915`). 원리적 함의: 같은 공적 상태에서 지어진 두 후보를 차례로
커밋하면 같은 세대 번호가 다른 바이트로 두 번 설치된다 — 세대 카운터의 변조-검출 목적이
정확히 그 시나리오에서 무력화된다. 오늘의 도달성: **0** — 호출부 2곳 모두 단일
build→commit→free 흐름 [실측 §1-5]. 이것은 규율이지 계약이 아니므로, ② 와 같은 행으로
**대장 기재 + 동일 트리거**. 수리 시 seed preflight 의 공적 결속이 본보기다. 정적
게이트는 이를 보상할 수 없다(§3 비인증 3항) — 초록불 문안에 남긴다.

## 5. 분기 지시 — GR-4 가 할 것

판정이 (ii) 이므로 **부분폐합 분기는 발동하지 않는다** — 단, §6 의 CUDA 발견이 새 조건을
단다.

1. 등록부·검사를 §3 T1~T7 명세대로 구현한다(§H-5-1 의 원소-writer 수리 — `_with_authority`
   교체·별칭·memcpy 확장 — 와 같은 단, 같은 커밋).
2. 부록 C-2 의 NC 표를 T7 의 5건으로 증보한다(고정 사유 문자열은 T7 의 것).
3. **초록불 문안**: PASS 줄이 transplant 류 카운트와 함께 §3 "인증하지 못하는 것" 6류의
   존재를 한 줄로 가리키게 한다(문서 참조로 충분 — 과장 금지가 이 캠페인의 병명이다).
4. **CUDA 2건(§6)**: T5b 의 넓힌 census 가 반드시 이들을 적중한다. GR-4 철회·분기 조항
   ("예상 밖 쓰기 지점 → GR-2 계급 판정 후 등록") 대로 **등록하지 말고**: GR-2b 판정이
   착지해 있으면 그 명세대로, 아니면 두 행을 `known-violation`(판정 대기) 로 붉게 등재
   하고 **host-transplant lane 완전 폐합 + CUDA lane 판정 대기**로 보고한다. GR-4 P1 의
   사전등록 기대 `cuda_writers=0` 은 이 발견으로 **무효** — 집행 기록에 정정을 남겨라.
5. 사전등록 §11 계열 규율 유지: 이 판정문 밖의 재량 등록 금지 — 등록부는 판정을 요구하는
   화이트리스트다.

## 6. ★판정 밖 신규 실측 — CUDA 진단 복원 벌크 writer 2건 (GR-2b 발주 필요)

[실측] `src/*.c`·`src/*.cu` 전수 grep: 목적지가 tau 슬랩인 벌크 호출은 정확히 4건 —
판정 대상 2건(`:21271`·`:21379`) + **`src/lumina_cuda.cu:10160`·`:10196`**:

- 소속: `LUMINA_NLTE_FINAL_RESOLVE` env 로 무장되는 withParityP GATE1 진단 블록
  (`:10096` 부근 gating [실측]). 수렴 상태의 pops/tau/S_l/jbar **바이트를 저장** → 진단
  재솔브 2회(공유 host writer 경유 — 세대는 정상 괄호로 전진) → **원본 바이트를 memcpy 로
  복원**. 세대 스칼라는 저장·복원 목록에 **없다** — 복원 후 슬랩은 "바이트=원본, 세대=
  전진" 상태다(소비자에겐 신선한 세대에 원본 값 — 스테일 위장은 아니다) [실측 코드 문면;
  런타임 함의는 [추정]].
- 계보: `a97d0e1`(2026-08-05) 태생 [실측 `git log -S`] — **f6c2eb6 이전부터** 게이트
  사각(정규식이 memcpy 무감)에 있었다. §H-2·§I-5 가 이것을 못 본 것은 두 판정 모두
  lumina_plasma.c 창에 집중했기 때문이다 — 사각의 존재 자체가 이 판정으로 세 번째 실증됐다.
- **예비 판독 (판정 아님 — 사전등록 질문 밖)**: transplant 류가 아니다(preflight 없음·
  세대 이식 없음·목적은 진단 A/B 의 상태 복원). "진단 save/restore" 라는 제3류로 읽히나,
  env-무장 경로라는 점·세대 스칼라 비복원의 정당성 여부는 별도 판정 사안이다.
  **운전석은 GR-2b(같은 계급, fresh 판정)를 발주하라** — 증거의 대부분은 이 절이다.

## 7. 열린 질문의 해소 — preflight 음성 대조: **실존, 그러나 unwired + 부분 커버리지**

[실측 — 소스 판독; 실행은 못 했으므로 빌드·PASS 여부는 모른다]:

- `tests/a2_10_seed_commit_selftest.c:214-289`: seed preflight 에 음성 대조 **3건**(공적
  Te 세대 오류·te_manifest provenance 훼손·공적/시행 Te 바이트 불일치) — 각각
  COMMIT_FAILED + 공적 소유자 전수 memcmp 바이트 보존 주장 + 양성 대조(이식 후
  `public_tau[0]` 착지 확인).
- `tests/nlte_candidate_tau_selftest.c:404-533`: bundle 경로 음성 대조 — build 실패의
  fail-closed(공적 무변조 memcmp)·`residual_status` 주입 → COMMIT_FAILED + 전수 바이트
  보존·양성 커밋의 세대 착지(T_e_generation=5·population=8)와 소유권 박탈 확인.
- **커버리지 구멍** [실측 grep]: preflight 의 tau-괄호-폐합 조건(`:21168-1170`)·tau 별칭
  조건(`:21102`)을 표적하는 주입은 **없다** — §3 T6 이 정확히 그 두 조건을 정적으로 pin
  하는 이유다.
- **둘 다 unwired 다** [실측]: 두 make 타깃(`Makefile:473`·`:488`)의 참조는 `clean` 목록
  (`:311-312`)뿐 — §0-A-1 unwired 27 명단과 정합. 즉 **음성 대조는 존재하되 아무도
  관측하지 않는다.** 처분은 GR-7 이 이미 소유한다(스텝 1 실측 대상). GR-7 판정자에의
  전달 사항: 이 두 타깃은 transplant 계약의 유일한 실행 증거이므로 처분 후보 중 「은퇴」는
  이 판정과 충돌한다 — 배선(또는 known-red 전환)이 정합한다.
- a2_07 population selftest 는 커밋 경로를 **건드리지 않는다** [실측 grep 0건] — 배선된
  게이트 중 이 계약을 실행으로 보는 것은 현재 없다.

## 8. 이 판정이 모른다고 적는 것

1. **두 selftest 의 현재 빌드·PASS 여부** — 실행 금지 티어에서 판정했다. GR-7 스텝 1 의 몫.
2. **CUDA 복원 경로의 런타임 무해성** — 코드 문면만 읽었다(§6 은 예비 판독이다).
3. **08-08~08-18 작업트리들에서 이 경로의 일시 변형 여부** — §H-4-2·§I-6 과 같은 계급으로
   복원 불가.
4. **T2·T4 의 위치 검사가 모든 리팩터 형상에서 안정적인지** — 정확일치 pin 은 의도적으로
   깨지기 쉽게(fail-closed) 설계했다. 깨짐 = 재판정 강제이지 오탐이 아니다 — 이 성질이
   싫다면 그것은 게이트 완화 요구이고, §I-2 의 금지(알터네이션으로 초록 표면 키우기)가
   선례다.

— 판정자 Fable, 2026-08-20. 근거 파일: `src/lumina_plasma.c` · `src/lumina_cuda.cu` ·
`src/nlte_population_candidate.{h,c}` · `scripts/check_tau_writer_generation.py` ·
`tests/nlte_candidate_tau_selftest.c` · `tests/a2_10_seed_commit_selftest.c` · `Makefile` ·
`docs/GATE_RECOVERY_INVENTORY_2026-08-18.md` §H·§I · 사다리 §4·§6·부록 C-2.
