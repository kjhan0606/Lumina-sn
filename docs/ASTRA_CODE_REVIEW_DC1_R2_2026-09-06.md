# astra 사전감사 2회차(코드 검수) 재검수 — D-C-1 구현 (gpt-6-astra, read-only, 2026-09-06)

발주 = `/gpfs/kjhan/lumina/dproxy_audits/dc1/astra_code2/ORDER_ASTRA_CODE_REVIEW_DC1_R2.md`.

1회차 «폐합 차단» 3건은 해소됐습니다. 신규 폐합 차단은 발견하지 못했습니다.

P=[계약 개정 2](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PREREG_DC1_NE_CLOSURE_2026-09-06.md), O=[발주 6](/gpfs/kjhan/lumina/dproxy_audits/dc1/ORDER_DC1_CHANGESET.md).
S=[소스](/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth/src/lumina_plasma.c), T=[selftest](/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth/tests/dc1_ne_closure_selftest.c), G=[stager](/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth/scripts/stage_superlevel_b.sh).
C=[CMF](/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth/src/lumina_cmfgen.c), U=[CUDA](/gpfs/kjhan/lumina/dproxy_audits/worktree_364ee72_dualauth/src/lumina_cuda.cu). 번호는 현재 파일의 행입니다.

| Q | 답 | 근거 |
|---|---|---|
| Q0 | PARTIAL | 지정 봉인 경로 3종 모두 rc=70. 상대 성분·봉인 내부·symlink 거부와 독립 복사·inode 검사가 연결된다(G:338–378,3480–3505). OFF는 폐합 본체·부동소수 연산·배열 변경·진단 0(S:20147–20148); 실물 G-A는 미측정. |
| Q1 | NO | 발주는 계약 변경표 전체를 참조한다(O:5–11). 헤더·경로 검사·identity 확대도 개정 2에 등록됐다(P:57–62). |
| Q2 | YES | 고정-T·자유-T 합류 후 폐합→두 `root_ne` 복사→bundle commit(S:16331–16372). seed는 build 성공 후 commit 직전(S:16551–16568). |
| Q3 | YES | 전하=`ion_pop_stage`, 대상 6원소 II/III만 준위합으로 치환한다(S:3090–3093,20029–20032,20106–20131). 세 자연로그와 분해식도 계약과 일치(S:20193–20196; P:9–11). |
| Q4 | YES | 0·비유한값을 floor 없이 거부하며 공개 물리 발행 전에 후보·자원을 정리한다(S:20178–20206,16377–16381,16562–16565). 진단 선발행은 아래 기재. |
| Q5 | YES | plasma·opacity에 동시 복사(S:21825–21829), 같은 반복 snapshot(C:7549), 다음 pass CMF 재조립(C:7420)에 들어간다. 개정 2의 소비 시점 기재와 일치(P:12). |
| Q6 | PARTIAL | residual 필드 선택과 SEED 값 검사는 수정됐다(T:88–90,119,125–133). 양성·음성의 나머지 검사 범위와 실행 검증 제한은 아래 기재. |
| Q7 | YES | 실제 base와 경로 정규화한 OFF exports가 동일하고 ON만 env 1개 추가(G:526–527). 새 binary SHA·source identity 등록(G:3518–3537), 자기 기술 7곳(G:941–986), launcher·helper 복제를 확인했다. |
| R1 | YES | 경로 3종: 상대 성분/봉인 내부/상대 성분으로 각각 거부(G:345–353). glibc 2.34에서 residual=9·nan·필드 삭제 모두 실패(T:89–90,119). E-1b는 측정으로 강등됐고(P:29), SEED의 ln(240/250)=−0.040821994520255166 검사에 0 대입은 실패(T:133). |
| R2 | NO | 정상 신규 경로 15모드 모두 통과, 기존 331050 계약 재검사 rc=0. 1·2회차 소스 patch도 동일하다. identity 목록 확대에 따른 메타데이터 차이는 아래 기재(G:17–23,309). |

- **«기재로 족함» Q0:** 실물 OFF byte 동일성은 아직 입증되지 않았다. 스테이징 적합 판정과 별개로 ON 발주는 기존 G-A 통과 뒤다(P:21–23,37).
- **«기재로 족함» R2:** identity 목록은 기존 DPROXY 모드에도 4→5로 확대된다. 따라서 새 base 클론의 identity JSON은 이전 것과 byte 동일하지 않으며, 이전 4항목 identity는 새 목록 검사와 불일치한다(G:17–23,262–264,309). 이는 개정 2에 등록된 변경이다(P:62).
- **«기재로 족함» Q4:** 뒤 shell에서 거부되면 앞 shell의 정상 형식 진단 행이 남는다(S:20167–20218). 후보 폐기로 공개 물리의 부분 발행은 없으며, 계약도 거부 commit 행을 판독에서 제외한다(P:41).
- **«기재로 족함» Q5:** χ_es·χ_ff는 폐합 n_e를 소비하지만(C:2077–2104), 함께 발행되는 CPU χ/η·BF·열원장은 폐합 전 값이다(S:21522–21529,21869–21885). 상태 SHA는 두 n_e 배열을 포함하나(C:1699–1700), 해당 호출은 지정 DET가 우회하는 경로에 있다(U:8072,8306). 개정 2가 이 소비 차이를 수용했다.
- **«기재로 족함» Q6:** 비-NLTE 단계 누락 사례는 n_e=120의 성공을 확인하는 합 민감도 검사다(T:143–145). residual 파서는 `bogus`를 0으로 읽고 음수도 허용한다(T:90,119); 출력 손상 검증에는 한계가 남는다.

검증: CPU 구문·`bash -n`·`git diff --check`·strict-env 통과. base 복원 identity 4/4, env 생성물 2종 메모리 재생성 byte 동일, stager exports 양성 2·음성 4 및 launcher/helper 복제 확인.
세션의 파일 쓰기 금지로 C 실행 selftest·stager 전체 selftest·보고 파일 작성은 수행하지 않았다. 스테이징·제출·GPU 실행 없음.

CODE_REVIEW: 통과  
SAFE_TO_STAGE: YES