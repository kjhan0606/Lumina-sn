# astra 사전감사 2회차(코드 검수) 1회차 — D-C-1 구현 (gpt-6-astra, read-only, 2026-09-06)

발주 = `/gpfs/kjhan/lumina/dproxy_audits/dc1/astra_code/ORDER_ASTRA_CODE_REVIEW_DC1.md`.

폐합 차단 3건으로, 현재 산출물은 스테이징 불가입니다.

P=[사전등록](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PREREG_DC1_NE_CLOSURE_2026-09-06.md), O=[발주 6](/gpfs/kjhan/lumina/dproxy_audits/dc1/ORDER_DC1_CHANGESET.md).
S=[lumina_plasma.c](/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth/src/lumina_plasma.c), T=[selftest](/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth/tests/dc1_ne_closure_selftest.c), G=[stager](/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth/scripts/stage_superlevel_b.sh), C=[CMF](/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth/src/lumina_cmfgen.c). 아래 번호는 현재 파일의 행이다.

| Q | 답 | 근거 |
|---|---|---|
| Q0 | NO | 봉인 내부 목적지로 우회하는 경로가 허용된다(G:184,337–348,3481–3484). OFF는 폐합 본체·부동소수 연산·배열 변경·진단이 0(S:20147–20148); 실물 byte 동일성 G-A는 미측정이다. |
| Q1 | NO | 발주는 사전등록 표 전부를 참조하며 범위를 좁히거나 넓히지 않았다(O:5,8–11). 추가 헤더 선언의 명세 누락은 아래 기재한다. |
| Q2 | YES | 고정-T·자유-T 합류 뒤 폐합→`root_ne`·`root.n_e` 복사→bundle commit(S:16331–16372). seed도 build 성공 뒤 commit 직전(S:16551–16568). |
| Q3 | YES | 대상은 6원소의 전하 1·2(S:20029–20032). Saha 전하합·NLTE 준위합 치환(S:20106–20131), 세 자연로그(S:20193–20196)가 P:9–11 및 기존 전하 규약(S:3090–3093)과 일치한다. |
| Q4 | PARTIAL | 0·비유한값은 floor 없이 거부(S:20178–20206). 공개 물리 commit 이전에 양쪽 실패 경로가 자원을 정리한다(S:16349–16381,16562–16565). 다만 진단 행은 먼저 발행된다. |
| Q5 | PARTIAL | 폐합값은 즉시 plasma·opacity에 복사된다(S:21825–21829,21939–21942). 지정 DET의 같은 반복 소비는 physics snapshot(C:7549); CMF 재조립은 통상 다음 pass(C:7420). 소비·파생량의 차이를 아래 기재한다. |
| Q6 | NO | residual 검사가 다른 필드를 읽는다(T:83–84,113). seed 대조도 E-1b를 시험하지 않는다(T:112,116–120). 비-NLTE 단계 누락 사례는 거부가 아니라 성공을 요구한다(T:129–131). |
| Q7 | PARTIAL | base exports·launcher 복제, ON의 env 1개 추가, 새 binary SHA 기록, 자원·자기 기술 7곳은 구현됐다(G:193,515–516,930–976,3500–3518,3664–3665). 추가 헤더는 source identity에서 빠진다(G:17–22). |

- **«폐합 차단» Q0 — 봉인 경로 우회:** 기존 `slb_20260829T101641Z_census_ion1/../../…`를 경유하여 정규화 결과가 `RII_ROOT/slb_20990906T000000Z_dc1_ne_closure_off`인 신규 목적지를 두 선행 검사 모두 수락했다(rc=0). 이후 `mkdir`·`cp`가 봉인 내부에 쓴다(G:3481–3482). 검사만 실행했으며 생성·복사는 하지 않았다.
- **«폐합 차단» Q6 — residual 검사 무력화:** `strstr(line,"residual=")`는 먼저 나온 `dlog_saha_residual=`를 선택한다. 동일 libc 파서로 실제 `residual=9`, `residual=nan`, residual 필드 삭제 모두 해당 검사가 통과함을 재현했다(T:83–84,113).
- **«폐합 차단» Q6/E-1b — seed 시연 부재:** `SEED` fixture는 `q_closed=240`, `q_saha=250`, `dlog_resplit≈−0.040822`인데 이를 검사하지 않는다(T:116–120; P:28). “seed-like” shell도 Saha 단계 `(40,10)`과 NLTE `(30,15)`의 전하합만 우연히 같다(T:47–49,112). 실제 seed 역시 NLTE core를 거치므로 LTE 동일성은 보장되지 않는다(S:16551,21516–21518).
- **«기재로 족함» Q4 — 진단 선발행:** 뒤 shell에서 거부되면 앞 shell의 사설 배열 변경·정상 형식 진단 행이 남는다(S:20167–20218; T:140–142). 후보 폐기로 공개 물리의 부분 발행·자원 누수는 없지만, 이 행들은 성공한 commit의 증거가 아니다.
- **«기재로 족함» Q5 — 소비 시점:** CMF는 plasma n_e를 우선하여 χ_es·χ_ff를 계산한다(C:2077–2104). 따라서 다음 Saha 계산보다 먼저 방사장에도 영향이 간다. 같은 반복 재조립은 `JINC_CONT` 조건부(C:7571–7582)이며 base에는 그 키가 없다. 상태 SHA 함수는 두 n_e 배열을 포함하지만(C:1699–1700), 그 호출은 지정 DET가 우회하는 CUDA-host 경로에 있다([CUDA:8072](/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth/src/lumina_cuda.cu:8072), 8306).
- **«기재로 족함» Q5 — 폐합 전 파생량 유지:** CPU χ/η·BF·열원장은 폐합 전에 생성되며(S:21522–21529), 폐합 후 n_e와 함께 발행된다(S:21869–21885). 계약의 “다음 반복 입력”만으로 동시 폐합된 파생량이나 순수 Saha 되먹임을 주장할 수 없다.
- **«기재로 족함» Q6 — 음성 명칭·빌드 줄:** 비-NLTE 단계 누락은 `rc=0`, n_e=120을 요구하므로 누락 거부의 시연이 아니다(T:129–131). 머리 빌드 줄(T:2–4)도 후보·열원장 구현을 빠뜨렸고, 생성 어셈블리에 해당 외부 호출이 남아 링크 입력이 불완전하다.
- **«기재로 족함» Q1/Q7 — 변경 명세:** [후보 헤더:172](/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth/src/nlte_population_candidate.h:172)의 추가 선언은 계약 변경표(P:56–60)와 stager의 source identity·보존 patch 목록(G:17–22,233)에 없다.

검증: base 복원 identity 4/4, 제공 patch와 현재 diff 일치, env 생성물 2종의 메모리 재생성 byte 동일. CPU 구문 검사·`bash -n`·`git diff --check`·strict-env 통과. stager 추출 로직은 양성 2·env 음성 4 및 launcher·helper 복제를 확인했다.
세션의 파일 쓰기 금지로 C 실행 selftest·stager 전체 selftest·보고 파일 작성은 수행하지 않았다. 스테이징·제출·GPU 실행 없음.

CODE_REVIEW: 차단 3  
SAFE_TO_STAGE: NO