# astra 사전감사 1회차 — D-C-1 계약 (gpt-6-astra, read-only, 2026-09-06)

발주 = `/gpfs/kjhan/lumina/dproxy_audits/dc1/astra_pre/ORDER_ASTRA_PREAUDIT_DC1.md`. 헤더 `model: gpt-6-astra`.

P=[사전등록](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PREREG_DC1_NE_CLOSURE_2026-09-06.md), S=`src/lumina_plasma.c`의 `364ee72`+보존 패치 복원본이다. **소스 identity 4/4 일치**를 재현했으며, S의 인용행은 복원본 기준이다.
R=[331050 런루트](/gpfs/kjhan/lumina/superlevel_b_h200x2_k36/slb_20260903T080407Z_dproxy_provenance_k42_ion1_h200), B=[base 판독](/gpfs/kjhan/lumina/dproxy_audits/readout_331050_fable/out_331050/readout.json).

| Q | 답 | 근거 |
|---|---|---|
| Q1 | YES | 하나의 혼합 전하합을 발행하는 계약이다(P:8–12). [상위 설계](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/DESIGN_DC_DD_CMFGEN_CONTRAST_2026-09-05.md:48)의 증분 사다리와 일치한다. CMFGEN의 동시 Newton 계 자체라는 뜻은 아니다(D:48–52). |
| Q2 | PARTIAL | 전하 규약과 12이온 집합은 맞다(S:3090–3093; R/stdout:207–218). GPU wrapper도 CPU 트랜잭션으로 위임한다(`lumina_cuda.cu`:1071–1083). 그러나 DET 후보 경로는 지정 말단 커밋 전에 반환한다(S:20627–20654). |
| Q3 | PARTIAL | E-1의 자기검사, E-3~5의 측정 구분과 **쌍별 ΔQ=−Δn_II는 맞다**. 이것을 저장된 Saha `n_e` 대비 전체 변화로 적용한 E-2는 성립하지 않는다(P:25–29; S:3099–3105,20850–20905). |
| Q4 | PARTIAL | **G-A base 자기대조는 참**: 메모리 재판독 네 열 모두 338/338 문자열 동일, epoch 3행 확인. G-D도 rc=0·provenance 685/685·registry 포함·공통338, G-F 최대 산포 1.21e−9로 통과한다. G-C·G-E는 아래 차단 사항이 있다(P:34–39; B:84–119,139–209). |
| Q5 | NO | 정상 유한값의 수치 구간 공백은 없다. 기대 크기 1e−5와 판정 문턱 1e−4 사이도 NC-A/B가 포함한다. 미수렴 분류에는 논리적 충돌이 있다(P:26,45–48,99–102). |
| Q6 | PARTIAL | 새 바이너리의 OFF 동일성까지 요구하므로 **1런 감축은 불가**다. OFF+ON은 제한된 두 반복 효과 측정에는 충분한 구성이나, 아래 계약 결함을 해소하지는 않는다(P:17–22,62). |
| Q7 | PARTIAL | 비음수 population과 양의 쌍 총량이 보존되면 폐합합은 양수다. 기존 commit은 음수·비유한값을 거부하지만 0은 허용한다(`population_contract.c`:846–848). P:9–11에는 0일 때 발행 전 처분이 없다. |

- **«폐합 차단» Q2 — 삽입 경로:** P:54의 말단 commit(S:20758)은 DET가 사용하는 `private_population_core=1` 경로에서 미도달이다. 이 경로는 S:20633에서 별도 commit 후 반환하며, 공개 발행은 후보 bundle에서 수행한다(S:20998–21002,21623–21633).
- **«폐합 차단» Q3 — E-2 전제:** 정확한 관계는 `n_e_closed−n_e_saha=(Q_Saha−n_e_saha)−Σ₆종 Δn_II`다. E-2는 기존 Saha 잔차와 Si/S/Ca 기여를 빠뜨렸다. 따라서 1e−4 초과만으로 쌍 총량 보존 또는 II/III 집합 전제의 붕괴를 판정할 수 없다(P:12,26,47).
- **«폐합 차단» Q4/G-C — 피연산자:** `resolved_lumina.sha256`는 exports 해시 파일이다([생성 코드](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/stage_a210_z2a_probe.sh:252)). base부터 기록값 `29a91bf4…`와 실제 exports 해시 `cc3155e6…`가 다르다. 이 파일의 단순 `cmp` 동일성은 현재 봉인 입력의 동일성을 검증하지 못한다(P:36).
- **«폐합 차단» Q4/G-E — 세대 수:** publication 2회와 전체 NLTE commit 수를 동일시했다. 실제 공개 NLTE commit에는 seed `1→2`와 반복의 세대 3·4가 있어 총 3회다. “커밋마다 기록”과 `50×2`가 양립하지 않는다(P:11,38; R/stderr:419; B:143,191).
- **«기재로 족함» Q5:** 미수렴은 G-D의 rc=0 조건을 깨므로, 문면의 NC-N 조건과 NC-X가 충돌한다. JSON 분기는 NC-X를 우선한다(P:37,47–48,99–102).
- **«기재로 족함» Q7:** 전하 기여의 전부 소실·underflow는 0, 비정상 입력·합산 overflow는 비유한값의 경로다. 이 봉인 레인의 정상 보존 상태에서 그런 발생은 확인하지 못했다. 0에서는 residual/dlog가 정의되지 않아 G-E를 만족하지 못하지만, 발행 전 처분은 미기재다. floor 추가를 허용하는 계약도 없다(P:9–11; S:20049–20069).
- **«기재로 족함» Q3:** NC-A의 작은 `dlog(n_e)`만으로 “스펙트럼 귀결 없음”은 도출되지 않는다. E-5는 별도의 측정이다(P:29,45).
- **«기재로 족함» Q4:** G-A에 적힌 판독 명령은 필수 `--out-dir`가 생략됐다(P:34; `readout_dproxy_sl_331050.py`:2691). 비교 술어 자체의 base 자기대조는 재현됐다.

PRE_AUDIT: 차단 4
**운전석 확인(저자 실행)**: 차단 4 전건 소스로 재현 — DET 공개 n_e 는 후보 bundle 의 `memcpy(public_plasma->n_electron, candidate->electron_density)`(worktree `:21794`·seed `:21908`)이고 말단 commit `:20928` 은 `private_population_core` 경로에서 미도달; Saha 잔차는 `emit_diagnostics` 시 stdout 에만(기본 미출력); `resolved_lumina.sha256` 은 스테이징 전 exports 해시(재작성 후 값과 불일치 `29a91bf4` 대 `cc3155e6`); 공개 commit = seed 1→2 + R7 iter 0·1 = 3.
