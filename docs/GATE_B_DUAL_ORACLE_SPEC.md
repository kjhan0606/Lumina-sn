# Gate B: dual frozen-cell oracle — 작업 명세

작성: 운전석(Claude), 2026-07-31. 발주 대상: Codex A(구현)/B(테스트)/C(리뷰).
배경 합의: docs/RE_RE_ARTIS_CMFGEN_PARITY_DIAGNOSIS_2026-07-31.md §P2 + RE³ §5 (P2 합의),
수리 시간표 2층 본체 (1층 계기 배치 = 완료, withParityAB sha 122a7eee).

## 0. 목적

수송·솔버 전체를 돌리지 않고, **한 셀의 동결 입력**(T_e, n_e, per-bin J, 준위 인구)에
대해 **생산 코드 경로가 실제로 내는** 율/불투명도/방출률/열 항을 두 레인과 비교하는
결정론적 오프라인 하니스.

- **Lane C (CMFGEN acceptance — Phase 1, 이번 발주)**: CMFGEN toy06 19.48d 자체런의
  같은 셸 값과 비교. CMFGEN이 최종 acceptance reference.
- **Lane A (ARTIS method — Phase 2, 후속)**: 같은 입력에 대한 ARTIS 산식
  (../artis-ref 소스) 대비 rate arithmetic + matrix topology 비교. ARTIS는 방법 reference.
- **KA 사다리 중간 단 (Phase 1b, oracle 착지 후 별도 발주)**: formal 적분기
  KA-2(순수 e-scattering 보존 대기, L_out/L_in=1) · KA-3(LTE S=B 슬랩, 해석해 대조).
  기존 KA-1(영점 1.000000028 PASS, FORMAL_FIX 수용시험)의 윗 단.

## 1. 처분 규약 (필수 — 위반 시 리젝)

1. **튜닝·수리 금지**: oracle이 찾은 어긋남의 처분 = 보고서 기재뿐. oracle 내부에
   클램프/플로어/보정 신설 금지. 생산 코드의 물리 변경 금지 (이번 발주는 계측만).
2. **게이트 default-OFF**: 신규 게이트 미설정 시 생산 경로 **byte-identical**
   (빌드 후 OFF-인증 배터리 포함 — §6).
3. **결정론**: oracle 실행 경로는 CPU 단일스레드 우선. GPU 커널 호출이 불가피하면
   사유와 결정론 근거를 보고서에 명시.
4. **생산 함수 그 자체를 호출**: 비교 대상 수량은 생산 경로(lumina_plasma.c /
   lumina_cuda.cu의 실제 함수)가 산출해야 함. 별도 재구현으로 만든 값의 비교는
   무효 (생산 경로가 피검체다).
5. commit/push/reset/파일 삭제 금지 (유저 지시 없이).

## 2. 동결 입력 (실물 경로)

- **Lumina 측 (기준선 = parity50, 현행 채택 EMA 수축계)**:
  `logs/coevolve_consume_parity50/lumina_plasma_state.csv` (셸별 T_e·n_e·T_rad·W),
  `lumina_levelpop.csv`, `lumina_c1_bins.csv` (per-bin J), `lumina_jbar_dump.csv`.
  형식·열 정의는 writer 코드에서 실측 확인 (추측 금지 — feedback_data_import_rigor).
  ※ D-4 처분 확정 시 parity57 기준으로 재실행 가능해야 함 → 입력 디렉토리는
  환경변수/인자로 주입 (하드코딩 금지).
- **CMFGEN 측**: `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/{RVTJ, <ion>PRRR, GENCOOL}`,
  J 진리 = jnu4 런 `/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/` EDDFACTOR.
  기존 리더 재사용: `validation/cmfgen_toy06_19p48d/analysis/extract_jnu.py`,
  `cmp_rvtj_T_ne_vs_published.py`, scripts/의 PRRR 파서 (있는 것 먼저 찾아 쓰고,
  새로 짜면 왕복항등 검증 포함).
- **셸 선택 (3셀 파일럿)**: s0(심부, parity57 ΔT_e 최대 지점) · s8(광구) · s45(외곽).
  Lumina 셸↔CMFGEN depth 대응은 속도 매칭으로 확립 (기존 분석물에 선례 있음 —
  validation/.../analysis/ 참조; 대응표를 보고서에 명시).

## 3. 산출 수량표 (셀당, Lane C 비교 대상)

| 분류 | 수량 | CMFGEN 앵커 |
|---|---|---|
| bf | per-ion Γ(광이온율, 바닥+전준위 합), α(재결합, spont+stim 분리), chi_bf, eta_bf | PRRR (율), RVTJ(상태) |
| ff | chi_ff, eta_ff, ff 냉각률 | GENCOOL |
| bb | 대표 이온(Si II/III·S II/III·Fe II/III/IV·Co III) 선별 jbar·펌핑율 상/하 | EDDFACTOR(J), GENCOOL(냉각) |
| collisional | 위 이온 대표 전이 C_ul/C_lu | GENCOOL(coll 항) |
| thermal | 가열/냉각 대장 항목별 분해 (GENCOOL 대응 형식으로 정렬) | GENCOOL |
| 상태 | n_e, 이온분율, 대표 준위 b_k | RVTJ, PRRR |

산출 불가 항목은 삭제하지 말고 사유 명기 (검증불가 고아 금지 —
feedback_no_unverifiable_orphans).

## 4. 구현 형태 (구현자 재량 항목 포함)

- 실행형: 신규 `bench_frozen_oracle.c(u)` (repo 루트 bench_nlte_rates.c 패턴) 또는
  `lumina_main` 게이트 모드 `LUMINA_ORACLE_CELL=<shell>` — **구현자 판단**. 단 §1-4
  (생산 함수 직접 호출) 충족이 판단 기준.
- 덤프: `lumina_oracle_cell_s<N>.csv` (수량표 전 항목, 항목명·값·단위·산출 함수명).
- 비교자: `scripts/oracle_compare_cmfgen.py` — 항목×셸×(Lumina, CMFGEN, 비율) 표
  CSV+MD. 판정 없음(REPORT-ONLY). 단위 환산은 스크립트 안에서 명시적으로.

## 5. 사전등록 기대치 (참고용 — 판정 아님, 어긋나도 수리 금지)

- n_e: 기존 실측 ~1.92× (광구) 수준의 어긋남이 재현될 것.
- b_k: 2-20× 대역 (기존 원장).
- thermal: s0 근방 "가스가 욕보다 2000-2600K 참" 서사에 대응하는 가열 결손이
  항목별로 어디에 앉는지가 이번 파일럿의 핵심 관찰.
- D-4 관련: MA_LINE_DESTRUCT 채널의 열화 항이 thermal 표 어디에 나타나는지 표시.

## 6. 검증 배터리 (Codex B)

1. OFF-인증: 게이트 미설정 빌드가 기존 산출물과 byte-identical (표준 4종 CSV,
   짧은 구성으로 가능하면 CPU 경로; GPU 필요 시 러너 큐잉은 운전석에 요청만).
2. oracle 실행 (3셀) → 덤프 완결성 (수량표 전 항목).
3. 자기일관 스모크: 같은 입력 2회 실행 byte-identical (결정론).
4. 비교자 왕복: CMFGEN 파서의 값 3개를 원시 파일에서 손으로 대조 (행:열 명기).

## 7. 산출물

- `docs/CODEX_GATEB_A_IMPL.md` (A: 구현 보고 — diff 요약·판단 근거·미해결)
- `docs/CODEX_GATEB_B_TEST.md` (B: 배터리 결과)
- `docs/CODEX_GATEB_C_REVIEW.md` (C: 독립 리뷰 — A/B 보고서 미열람 조항)
- 운전석 통합 후 V-원장·대장 기재는 운전석 몫.
