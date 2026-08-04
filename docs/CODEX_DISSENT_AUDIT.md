금지된 `docs/FABLE_*` 계열은 근거로 사용하지 않았다. 파일 수정·commit·모델 런도 하지 않았다.

| 쟁점 | 판정 | 핵심 |
|---|---|---|
| 1 | **운전석 주장 부분 지지** | 커버리지/선원함수 설명은 배제되지 않았다. 다만 `B(T_e)` 폴백은 Co IV 미매핑 때문이 아니라 이번 캡처에서 전 선에 적용된 전역 기본값이고, “불투명도당 4배 과방출”은 성립하지 않는다. |
| 2 | **운전석 주장 지지** | `ε_eff`는 post-clamp payload의 집계량일 뿐 unclamped 물리 ε 측정이 아니다. 5247은 독립 측정이 아니라 정의상 역수다. C08 발화 횟수는 해당 산출물에서 복원할 수 없다. |
| 3 | **운전석 주장 부분 지지** | 고정장 시험은 결합 되먹임 오차를 최종 면책하지 못한다. 그러나 실제 핀은 전체 J가 아니라 Gph 소비 J이며, 원 판정문은 이미 조건부다. 더구나 “Fe 완전재현”은 최종 대장 자체와 모순된다. |

## 쟁점 1 — Co IV 형광 깔때기

**판정: 운전석 주장 부분 지지**

### 1. 캠페인 사슬은 현상·중간 인과까지는 닫혔지만, 유일한 상류 원인까지 닫히지 않았다

산출물이 직접 지지하는 범위는 다음이다.

- trapping 부족은 반증됐고, FUV 결손은 색 재분배 문제다: [trapping_audit/VERDICT.md:13](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/trapping_audit/VERDICT.md:13).
- CMFGEN J를 주면 radeq 식이 18,277 K를 내므로 13,120 K 냉근은 소비장의 기근과 연결된다: [radeq_ledger_audit/VERDICT.md:85](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/radeq_ledger_audit/VERDICT.md:85).
- MC event ledger에서 Co IV가 1290–2000 Å 방출의 84%, 전체 deep emission의 80.9%를 차지한다: [reddening_localization/VERDICT.md:13](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/reddening_localization/VERDICT.md:13), [같은 파일:87](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/reddening_localization/VERDICT.md:87).

그러나 마지막 문서는 원인을 확정한 것이 아니라 Co 이온균형과 MC thermalization A/B를 “design only — no runs”로 남겼다: [reddening_localization/VERDICT.md:134](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/reddening_localization/VERDICT.md:134). radeq 판정도 실제 소비장과 pumping 부호에 관해 명시적 caveat를 둔다: [radeq_ledger_audit/VERDICT.md:106](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/radeq_ledger_audit/VERDICT.md:106).

따라서 “Co IV pile → field starvation → cold root”는 지지되지만, 이를 **유일하게 MC 선수송 결함으로 확정**한 것은 증거보다 강하다.

반대로 “커버리지 가능성이 전혀 검토되지 않았다”도 틀렸다. 같은 범죄대장은 비-NLTE 선의 `B(T_e)` thermal source를 별도 **POSSIBLE coverage gap**으로 이미 기록한다: [CRIMINAL_RECORD.md:92](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/criminal_record/CRIMINAL_RECORD.md:92). 즉 공존은 모순이라기보다 아직 분리되지 않은 MC-side와 deterministic-side 후보에 가깝다.

### 2. Co IV는 실제로 기본 NLTE 창 밖이고, 이번 캡처의 사용 선원함수는 `B(T_e)`다

기본 NLTE 대상은 Co II/III(`ion=1,2`)뿐이다: [lumina_plasma.c:7677](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:7677). 매핑은 Z와 ion stage가 정확히 일치해야 한다: [lumina_plasma.c:14220](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14220). Stage-IV 배열은 별도 gate용이다: [lumina_plasma.c:7686](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:7686).

다만 운전석의 인과 서술은 틀렸다.

```c
Sl = src_nlte ? Sl_pop : 0;
if (Sl <= 0) Sl = B(Te);
```

실제 분기는 [lumina_cmfgen.c:785](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:785)와 production [같은 파일:1789](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1789)에 있다. 캡처 manifest의 `gates.src_nlte=0`이므로 **매핑된 선까지 모두** `B(T_e)`를 사용한다: [linepop_iter10.manifest.json](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/linepop_iter10.manifest.json).

따라서 “Co IV가 창 밖이어서 B 폴백”은 반증된다. 정확한 표현은 “Co IV는 창 밖이며, 동시에 이번 런은 전역 `SRC_NLTE=0`이라 모든 선이 B를 쓴다”이다. 커버리지 결함은 여전히 population/source 완전성 문제지만, 이번 mapsplit만으로 Co IV에 특유한 B 부여 원인이라고 할 수 없다.

### 3. 캠페인은 커버리지 설명을 배제하지 않는다

오히려 trapping audit은 전체 선을 보존하더라도 98%의 FUV 선이 super-level 내부 Boltzmann population을 쓴다는 population-accuracy 문제를 남긴다: [trapping_audit/VERDICT.md:108](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/trapping_audit/VERDICT.md:108). “CMFGEN도 같은 Co IV complex를 가진다”는 [reddening_localization/VERDICT.md:127](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/reddening_localization/VERDICT.md:127) 수준은 복합체 존재를 확인할 뿐, line별 population·`ETAL/CHIL`·원자 데이터의 동등성을 증명하지 않는다.

**UNRESOLVED:** 커버리지와 MC 재순환을 결판내려면 같은 epoch/depth에서 Co IV line별 `S_l=ETAL/CHIL` 또는 `n_u/n_l`, 그리고 Stage-IV/source gate A/B가 필요하다.

### 4. “단위 불투명도당 4배 과방출”은 반증

수치 자체는 맞다.

- 미매핑 energy fraction: `0.0468308053`
- 미매핑 `chi_line_sum` fraction: `0.0114057671`
- 비율: 약 4.106

근거는 [uv_mapsplit.json](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/uv_mapsplit/uv_mapsplit.json)의 `headline_selected_five_shells`다. Co IV가 미매핑 방출의 45.176%로 1위인 것도 CSV `SELECTED5/BALL/Co IV` 행에서 확인된다: [uv_mapsplit_unmapped_ion_rank.csv](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/uv_mapsplit/uv_mapsplit_unmapped_ion_rank.csv).

하지만 두 분율의 가중이 다르다.

- opacity 지표: `Σw`
- energy: `Σ(w·ε·S_l·Δν)`

정의는 [report.md:5](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/uv_mapsplit/report.md:5)에 명시돼 있다. 따라서 비율 4.106은 `ε`, `S_l`, `Δν`, 셸/주파수 분포를 함께 포함한 **share ratio**이지 물리적인 “방출/불투명도”가 아니다. 더구나 비교한 opacity 쪽에는 `Δν` 적분도 없다.

`B>S_true`의 가능성은 있다. CMFGEN에서 BALL `J/B`가 s0=0.8000, s8=0.7815이고, 두 준위식 `S=(1-ε)J+εB`를 가정하면 `J<B`, `ε<1`에서 `B>S`다: [CODEX_EMISS_E8.md:215](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E8.md:215). 하지만 같은 문서가 line별 `ETAL/CHIL` 부재 때문에 실제 equivalent source를 역산할 수 없다고 명시한다: [같은 파일:221](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_EMISS_E8.md:221).

따라서 **“B가 참 source보다 커서 과방출”은 가능성이지 측정된 결론이 아니다.**

## 쟁점 2 — ε_eff와 5247

**판정: 운전석 주장 지지**

`chi_line_th`와 `eta_line`은 실제로 clamp된 선별 ε를 쓴다: [lumina_cmfgen.c:1795](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1795). `S_fixed`는 그 `eta_line`을 포함하고, scattering remainder도 `chi_es`로 들어간다: [같은 파일:2038](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:2038), [같은 파일:2075](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:2075).

E8의 정의는:

- `eps_eff_source = ∫(eta_fixed/chi_total)dν / ∫(eta_total/chi_total)dν`
- `gain = 1/eps_eff_source`

이다: [summary.json:374](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e8/summary.json:374). 값도 `0.00019056728565`, `5247.4903894`로 확인된다: [summary.json:472](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e8/summary.json:472).

따라서:

- `ε_eff`는 **post-clamp payload에서 측정한 집계 source fraction**으로는 유효하다.
- microscopic/unclamped line ε의 측정값이나 평균으로 해석하면 안 된다.
- 5247은 독립적인 재순환 이득 측정이 아니라 같은 집계량의 산술적 역수다.
- `S_fixed`에는 continuum 등도 포함되므로 “모든 값이 오직 line ε의 측정”이라는 표현도 부정확하다.

C08은 공식적으로 counter `N`이다: [CODEX_CLAMP_CENSUS_2026-07-31.md:38](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_CLAMP_CENSUS_2026-07-31.md:38). E8 산출물과 이번 linepop payload에는 raw ε와 clamp reason이 없고 post-clamp `el`만 있다. 경계값과 정확히 같은 행을 세더라도 “원래부터 1e-5/1”인지 floor/cap 발화인지 구별할 수 없다.

현재 소스에는 향후 진단용 `eps_applied_diff_rows` manifest 필드가 마련돼 있다: [lumina_cmfgen.c:1342](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1342). 그러나 지정된 validation 및 runner 산출물에서 이 manifest가 실현된 결과는 찾지 못했다. 따라서 E8 epoch의 발화 횟수는 **UNRESOLVED**다.

`denom<=0 → 1.0`은 [lumina_plasma.c:8471](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:8471)에 있다. 이는 “미등록 line → -1 → caller가 1”과 다른 분기다. C08 문구는 미등록 ε만 적고 있으므로, 현재 census상으로는 **별도 미등록 대체 항목**이다. 같은 ε 위험군으로 C08을 확장하려면 사유별 카운터가 필요하다.

마지막으로 대역 평균 `1.90567e-4 > 1e-5`는 개별 선 미발화를 전혀 함의하지 않는다. 일부 선이 floor에 붙어 있어도 다른 선과 continuum의 가중합이 floor보다 커질 수 있다.

## 쟁점 3 — “율 무죄 최종”

**판정: 운전석 주장 부분 지지**

### 실제 핀 범위

쌍둥이 런은:

- `T_e`를 전 셸 CMFGEN 표로 whole-state pin하고,
- CMFGEN J-table을 **Gph photoionization 적분에서만** 대체한다.

런 설계 자체가 이를 명시한다: [sbatch_te_jtable.sh:11](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/sbatch_te_jtable.sh:11). 코드도 J-table이 “thermal balance, line transfer, MC/deterministic estimator를 건드리지 않고 photoionization integral만” 바꾼다고 명시한다: [lumina_plasma.c:9760](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:9760). 런 로그에서 50개 셸 Te pin과 214,208,908회 Gph table 소비를 확인한다: [stdout.log:217](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_a10_kx_tetab_jtab/stdout.log:217).

따라서 운전석의 “J 전체를 핀했다”는 요약은 틀렸지만, **시험 대상인 ionization-rate 소비 J는 핀됐다**는 핵심은 맞다.

### 원문은 이미 조건부다

rate certification의 첫 질문부터 “CMFGEN field와 populations를 pin했을 때 σ-table과 1000-bin integration이 Γ를 재현하는가”로 한정한다: [rates_certification/VERDICT.md:6](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/rates_certification/VERDICT.md:6).

또한 다음을 명시적으로 인증 범위 밖에 둔다.

- Lumina J estimator
- Lumina 자체 level populations
- recombination/Milne side
- fallback/sub-bin 경로

근거: [rates_certification/VERDICT.md:246](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/rates_certification/VERDICT.md:246).

따라서 원 판정문의 정확한 결론은 “σ-bake+ν 적분 rate machine이 고정된 진리장·진리 population에서 factor급 오차를 만들지 않는다”이다. 별도의 정확한 문구 “율 무죄 최종”은 검색된 원문에서 확인되지 않아 그 문구의 출처는 **UNRESOLVED**다.

### “Fe 완전재현, Co만 10× 실패”는 반증

같은 twin의 최종 대장은 다음을 기록한다: [final_ledger.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/co3_closure_trace/final_ledger.csv).

- s2 Fe: CMFGEN/twin = 1.377
- s6 Fe: 0.715
- s8 Fe: **4.249**
- s8 Co: **17.255**

즉 Co 잔차가 Fe보다 크기는 하지만 Fe도 s8에서 4.25배 틀린다. 범죄대장의 “twin reproduces Fe fully” [CRIMINAL_RECORD.md:63](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/criminal_record/CRIMINAL_RECORD.md:63)는 자체 최종 대장과 양립하지 않는다.

Co 결과는 시험이 고정점에서의 **직접적인 종별 rate 오차**를 잡을 판별력이 있음을 보인다. 그러나 깨끗한 Fe 음성대조나 “Co만의 10×”를 입증하지는 않는다.

### 되먹임 민감도

고정 시험은 `R_Lumina(J*,T*)`와 진리 상태를 한 점에서 비교한다. 그러므로:

- 그 점에서도 남는 직접 rate 오류는 검출한다 — Co가 그 사례다.
- `dR/dJ`가 잘못됐거나, 잘못된 rate가 J를 바꾸고 그 J가 다시 rate를 증폭시키는 loop 오류, 또는 `J=J*`에서 우연히 사라지는 오류는 검출하지 못한다.

따라서 운전석의 “J를 매개하는 모든 율 오류를 볼 수 없다”는 표현은 너무 넓지만, **결합된 J–rate 되먹임까지 최종 면책할 수 없다는 결론은 맞다.**

결판에는 고정점 한 점이 아니라 J perturbation에 대한 rate 응답/Jacobian 또는 pin-release된 self-consistent 대조가 필요하다. 현재 산출물로는 그 부분을 **UNRESOLVED**로 남겨야 한다.