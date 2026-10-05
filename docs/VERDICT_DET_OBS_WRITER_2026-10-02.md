# 판정 — DET-OBS-WRITER 단계 2 R′ (run2)

- 계약: `docs/PREREG_DET_OBS_WRITER_2026-10-02.md` (개정 1–5, HEAD e7ad3ca; 기안 astra)
- 판독기: `scripts/det_obs_readout.py` (커밋 cfd45d8, sha256 `a530e0460d7a…` = `result.json` 의 `readout_sha256`)
- 판정자: 운전자(Opus 5.5) — **구현자·판정자 겸임**(배분 개정25-w). 이 판정문에는 운전자의 구현·처분 이탈이 기재된다.
- 장부 정본: 사전등록 집행 기록 표. 이 문서는 지목만 한다.

## 1. 결과

판독 `N/Rprime/readout2/result.json` (N = `/gpfs/kjhan/lumina/cmfgen_map/obs_writer_20261002`): **validity VALID, 사유 0**, mode `rprime`, 짝 판독은 `PENDING_EPRIME`.

같은 view(같은 J·물질·세대)에서 원천 보존 작성기 N 과 기존 작성기 L 의 색비 차이 W_R′ = log10(C_N / C_L), C_j = L_j / L_OPT:

| 대역 | W_R′ (dex) | 범주(±0.30) |
|---|---|---|
| FUV0 912–1200 Å | **−1.073** | down |
| FUVa 1200–1500 Å | **−0.614** | down |
| FUVb 1500–2000 Å | −0.090 | flat |
| NUV 2500–4000 Å | −0.077 | flat |

대역 광도(erg s⁻¹, 같은 런):

| | FUV0 | FUVa | FUVb | NUV | OPT | 전체 |
|---|---|---|---|---|---|---|
| N | 6.07e38 | 9.31e39 | 1.48e41 | 5.31e41 | 1.28e42 | 3.96e42 |
| L | 4.38e39 | 2.33e40 | 1.11e41 | 3.86e41 | 7.79e41 | 2.90e42 |
| log N/L | −0.86 | −0.40 | +0.13 | +0.14 | +0.22 | +0.14 |

NUV/OPT 색: N 0.415, L 0.496. (참고로 NRB-E 판정의 공개 잣대 값은 0.775 였다. 그 비교는 이 단의 판정 대상이 아니다.)

**판정 문장**: 같은 상태에서 매핑 S_l·Milne η·η_dep·미매핑 산란을 실제로 소비하는 작성기로 바꾸면, FUV0·FUVa 의 OPT 대비 색이 각각 −1.07·−0.61 dex 낮아진다. FUVb·NUV 색은 ±0.1 dex 안이다. 절대 광도는 OPT·전체가 +0.22·+0.14 dex 높아진다. 이것은 **작성기 효과(같은 view)** 이며 상태 변화 Δ_N(두 상태 FUV 분기)는 E′ 없이 판정하지 않는다(계약 §5).

## 2. 실행·게이트

- job 412765 `LUMINA_detobs_Rp2`, h200 syn104, **H200 NVL ×2**(143,771 MiB, UUID 상이), CPU 16, 64G, COMPLETED 0:0, 10:29:27, RESTARTS 0, MaxRSS 32.5 GB.
- footer: `partition=h200 gpu_kind=H200 gpu_profile=h200x2 allocated_gpus=2 allocated_cpus=16 model_rc=0 terminal_state=SUCCESS`.
- 작성기: `det_obs_state=SUCCESS`, mode 1, src_nlte 1, 세대 rad 7 / material 7, view digest 전후 3개 동일(`a339b3c9e536bfcc`).
  - 소비 선 19,712,245(매핑 16,721,517 · 미매핑 2,990,728), 점프 1.48e9, `mapped_B_fallback=0`, `unmapped_line_source_positive=0`, 원천 대조 최대 상대차 0·0, nsub 상한 도달 0.
  - 벽시계 N 417 s, L 250 s.
  - **χ_e 대조(개정 4 상한)**: `status=PASS max_ratio=0.745`(shell 47·bin 246, n=4·m=4). 옛 (A+S)×1e-12 척도의 진단값은 `old_dev=1.929e-11` 로 **run1 실패값과 같은 수치가 재현**되었고, 그 셀을 포함한 전 셀이 새 상한을 지킨다.
- G-ID(제출 전): `N/evidence/rev5/gid_pre_rp2` PASS(사유 0). G-RUN·G-READOUT: 판독 결과 사유 0.
- signed·영 τ 추가 성분: `UNSUPPORTED_SOURCE_DOMAIN`(단독으로 G-RUN 사유 아님, 계약대로).

## 3. 경로 — run1 X 와 개정 4·5

1. **run1(411170, A100×2)**: 작성기 `FAILED(CHI_E_RECONCILE dev=1.929e-11)`, 판독 X. 운전자가 처분 Q2-2에서 제안하고 astra 개정 3이 계약화한 (A+S)×1e-12 척도에는 누적합 반올림(L=Σw, H=Σwε)을 제한하지 못하는 계약 결함이 확인됐다. run1의 관측 편차는 이 결함과 양립하지만, 실패 셀의 L·H·S·E가 보존되지 않아 실제 원인을 확정하거나 물리 불일치를 배제하지 못한다. run1 의 X 는 소급 변경하지 않는다.
2. **개정 4**(d766259): 조립기 연산에서 유도한 바깥쪽 반올림 상한 B_up. 반례 fixture(옛 FAIL·새 PASS)와 결함 17 유지.
3. **개정 5**(e7ad3ca): user 10-05 결정으로 run2 를 h200x2 로. 근거는 syn104 예약 중 형식 없는 GPU 요청이 거부된다는 운전자 실측.
4. 코드 감사(gpt-6-sol) 3회 — 개정 4 1회차 4건·2회차 4건, 개정 5 7건, 합계 «폐합 차단» 15건. 운전자 폐합 기록은 15건의 수리·처분을 보고한다(`N/evidence/rev5/code_audit_closure.txt`). 종료감사(B2)가 지적한 대로 항목별 처분문 9개가 인용한 판독기 해시(c7b8…)는 최종 판독기(a530…)와 달랐다. 운전자는 c7b8 판본을 재구성해 해시 일치를 확인했다(`N/evidence/rev5/det_obs_readout.closure_basis_c7b8.py`). 두 판본의 차이는 감사 항목 키 수정 1건(`N/evidence/rev5/readout_c7b8_to_a530.diff`)이다. 이 연결을 담은 `_v2` 처분문으로 해당 9개 항목을 갱신했고, 원본 처분문과 `code_audit_closure.attempt1.txt`는 보존했다. **개정 5 감사 7건은 재감사 없이 운전자 처분으로 폐합**했다.

## 4. 한계 (과대 주장 금지)

- **겸직**: 운전자가 C·판독기·러너를 구현하고 이 판정을 쓴다. 독립 검사는 gpt-6-sol 감사(코드)·astra 종료감사(이 문서)·user 다.
- **cmfgen_run 실패 주입 종단 미검증**: 작성기 실패가 R7·footer·model_rc 를 지나가는 경로는 CPU fixture 로 재현하지 못했다(코드 구조·흐름 지속 시험·run1 실측으로 갈음: run1 은 작성기 실패에도 `model_rc=0`·footer SUCCESS 였다).
- **하드웨어**: run2 는 H200, R·run1 은 H100/A100 이다. 기존 ④(lumina_spectrum.csv)의 run 간 비교에는 하드웨어 차이가 섞인다. W_R′ 는 같은 런 안의 N 대 L 이라 이 영향이 없다.
- W_R′ 에는 계약 §6.2 의 N·L 선 선택 문턱 차이(1e-12 대 1e-6)도 포함된다. 개별 원천의 독점 기여를 측정한 값이 아니다. W_R′ 자체의 하드웨어 불변성은 검증하지 않았다(서로 다른 하드웨어의 두 런을 직접 비교하지 않는다는 범위에서만 영향이 없다).
- **W_R′ 는 작성기 차이**다. 물리(상태) 변화도 아니고 CMFGEN 대비 개선 주장도 아니다. N 이 L 보다 "옳다"는 것은 계약의 원천 보존 정의에 따른 것이며, 공개 스펙트럼 대비 판단은 별도 단이다.
- run1 실패 셀은 보존되지 않았다. run2 에서 같은 `old_dev` 가 재현되고 전 셀이 상한을 지킨 것은 반올림 귀속과 **양립**하는 증거이지 run1 셀의 직접 대조는 아니다.
- 비용: run2 footer 기준 경과 10.489시간, H200×2 약 20.98 GPU·시간(Slurm 경과 10:29:27). 정밀 해 1회 소요는 run1(A100×2) 같은 회차의 약 0.79배(2,596 s 대 3,305 s, 2,717 s 대 3,456 s; `EXACT-MULTIGPU-TIMING reported_total_s`), footer 시간비는 약 0.813. 계약은 사전 예측에 가속률을 쓰지 않았다.

## 5. 다음 (user 결정)

- E′(같은 h200x2·B1·모듈, 계약 §2.4) 실행 여부 — 실행하면 Δ_N·Δ_L 과 두 상태 FUV 분기를 판정한다. 경과 약 10.5시간, H200×2 약 21 GPU·시간 예상이며 완주시간 보장은 아니다. **user 승인 전 실행하지 않는다.**

## 6. 분장 장부 (이 단 칸만; 정본은 사전등록 집행 기록)

| 단계 | 규약상 담당 | 실제 | 이탈 |
|---|---|---|---|
| 개정 4·5 기안 | astra | astra(개정 4 1회차 잘림 → 재발주) | 없음 |
| 구현(C·판독기·러너) | 운전자(25-w) | 운전자 | 없음 |
| 코드 감사 | gpt-6-sol | gpt-6-sol 3회 | 개정 5 차단 7건 재감사 없이 운전자 폐합 |
| 판정 | 운전자 | 운전자 | 겸직(기재) |
| 종료감사 | astra | astra 1회(`N/astra_closeout.md`): 폐합 불가·차단 2(B1 run1 귀인 단정, B2 폐합 근거 판본 연결) → OLD/NEW 반영·판본 연결 보완 | 기안자가 자기 계약 판정을 감사(이해관계 기재) |

## 7. 이후

user 가 E′ 를 승인했다(2026-10-05 「돌려요.」, `N/evidence/rev5/USER_APPROVAL_EPRIME.txt`). E′ = job 413614 `LUMINA_detobs_Ep1`(h200x2, 짝 G-ID `N/evidence/rev5/gid_pre_ep1` PASS). 두 상태 판정은 E′ 완료 뒤 별도 판정문 갱신으로 한다.
