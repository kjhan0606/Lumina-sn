# LUMINA-SN 프로젝트 상설 규약 (세션 자동 로드)

## 실행 환경 티어 (user 2026-08-01 확정)
- **경량 오프라인**: **로그인 노드(syntax·grammar) 실행 전면 금지(user 08-01)** — 파일 조회·grep·판독 등 비연산만 로그인 허용. 단일 CPU 소형 작업은 **grammar-debug 노드**(nested ssh: `ssh grammar "ssh grammar-debug '<cmd>'"`; 32코어·/gpfs·/home 공유·슬럼 불요) 또는 slurm job-per-run. 대형은 아래 티어.
- **대형 오프라인**(분 단위 초과 — KA 전체 사다리·고정밀 oracle·대형 리플레이): **lageunha 직접 투척**(ssh+백그라운드, OMP=60). 투척 전 `ssh lageunha uptime`으로 부하 확인 — 선주 작업(예: ramses_zoom3d) 포화 시 grammar CPU slurm(양호 노드 지정)으로 폴백. 타 사용자 등장 시 즉시 양보.
- **CMFGEN**: grammar CPU slurm 또는 lageunha 수동만(syntax 금지). **OMP=16 절대.** slurm은 `--time` 명시(백필 자격).
- **GPU 생산 런**: slurm job-per-run, 파티션 h200→h100(full-NLTE 80GB — a40 제외). 상주 러너·interactive 할당 금지.
- **분업**: codex=스크립트 작성까지, 제출·투척·대외 발송=운전석(Claude). codex 샌드박스는 네트워크 차단됨.
- **불량 노드 상시 제외(user 08-01 지시, 관리자 해소 확인 시까지)**: grammar 제출 시 **모든 sbatch에 `--exclude=grammar072,grammar078,grammar080` 기본 적용**(072=/gpfs 기록 불가 의심·078/080=/home 마운트 불량). slurm 산출 경로는 /gpfs(GPFS scratch 규약).

## 검증 규약 (요지 — 상세는 memory/MEMORY.md 자동 로드)
- 물리 복원 사다리: 수리 단위=물리 계약 1개, rung당 패치+기대 변경집합 사전등록, B=증분 byte-배터리.
- 음성 대조 의무: 게이트는 주입 결함으로 FAIL을 시연해야 PASS 자격.
- clamp/floor 금지(판별식: 정확해가 위반 가능한 가드·표현은 잘못된 것). 발견의 처분=조용한 대장 기재.
- src-편집/트리-변조 태스크는 한 번에 1개(발주 전 가동 중 변조 태스크 0 확인). C 리뷰는 안정 산출물(패치 파일)만.
- 런 발주 3요건(offline-first): 기전 오프라인 특정·수리안 오프라인 검증·기대치 사전등록 후 판정런 1회.

## 논문 (Overleaf)
- 저장소: `~/BACKUP/Eunha.A1/Claude/overleaf_lumina_paper1` (git.overleaf.com/6a6d705cc431f91124274896). 격식 영문은 **~/WRITING.md 전 규칙** 준수 + 제출 전 자가점검 grep. 그림은 python(matplotlib)+내장 레이아웃 검사기 0위반. 푸시 전 `git pull --rebase`(웹 수정 보존).
