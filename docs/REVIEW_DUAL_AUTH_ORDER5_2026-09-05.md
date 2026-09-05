# 코드 검수 2회차 — DET-DUAL-AUTH-2 변경집합 개정 1 (Fable fresh, read-only, 2026-09-05)

검수자 = Fable fresh(1회차와 다른 컨텍스트). 대상 = 발주 5 산출(stager RESCALE 훅, 1회차 대비 diff 447행; `spectrum_contrast.py` 무변경). 탐침 `/gpfs/kjhan/lumina/dproxy_audits/readout_331050_fable/order5/fable_review_probe/`.

| 질문 | 판정 | 핵심 증거 |
|---|---|---|
| R1 F-1 해소(치환·중복 0·2행 diff) | **YES** | `rewrite_exports` 444–447 값 치환, 475–477 append 삭제; 실 RII base→arm 91행·RESCALE 키 1개·G-0 형식 diff 정확히 `<"0"`/`>"1"` 2행; base 모드 재작성 = 봉인 331050 exports byte-동일 |
| R2 F-2 해소 | **YES** | 740–742 base 분기 `="0"` 1개 요구; 봉인 331050 재적용 4검사 rc=0 |
| R3 F-3 해소 | **YES** | suffix·job·mode.txt·EXPECTED_DELTA·OWNER·STAGE_SEAL·post-model sealer 7곳 RESCALE 모드명; dry-stage 대 331050 diff 가 그 7곳 + RESCALE 행뿐, 자원·진단·landing byte-동일 |
| R4 타 모드 회귀 | 없음 | 13모드 rc=0; 기존 루트 재검증 실패 2건은 1회차 함수로도 동일(선재) |
| R5 픽스처·음성 실효 | **YES** | base 키 `"0"` 실존 픽스처; 음성 14; 검사 11곳 각각 무력화 mutant 전건 selftest FAIL |
| R6 봉인 무접촉 | **YES** | 봉인·SLB 부모 `-newer` 0, dry-stage inode 공유 0, repo 추적 파일 변경 0 |

**«폐합 차단» 0.** **«기재로 족함»**: ① python 거부 rc=1 무 FATAL 행(선재) ② **base 331050 은 post-model seal 오판으로 landing 없음 — arm 도 같은 결말 예상, 사전등록 판정 경로(`work/`·readout)와 무관, 결함으로 오독 금지** ③ identity 의 git_head 차이(G-1 은 binary sha 만 봄; 운전석은 worktree `364ee72`+패치로 base 와 동일하게 맞춤) ④ dry-stage 산출 5.1 GB 가 probe dir 에 잔존.

**SAFE_TO_STAGE: YES.**
