# LUMINA-SN: 스펙트럼/준위점유 진단 브리프 (2026-06-14)

## 0. 한 줄 요약
DDC15 0.976d CMFGEN self-test에서 **T_e/n_e는 gold와 0.5% 일치**하는데
**최종 emergent 스펙트럼은 망가짐**(너무 파랑). 중간검증(T_e,n_e)은 통과,
최종검증(스펙트럼) 실패. 증거는 NLTE 준위점유가 틀렸음을 가리킴.
이것이 (A) 결정론적 2준위 선소스의 구조적 결손(다준위 형광 누락)인지,
(B) rate-matrix/준위해 코드의 특정 버그인지 판정 요청.

## 1. 아키텍처
- 두 수송 경로가 동일 plasma 솔버(compute_plasma_state, nlte_solve_all)를 공유.
  - **MC 경로**: 실/가상 패킷 + macroatom 형광(다준위 재분배 있음).
  - **pure-CMFGEN 경로**: 결정론적 1000-bin tangent-ray formal solve (2준위 선소스만).
- 차이는 "누가 J_ν를 만드나"뿐. plasma/NLTE 솔버는 공유.
- 현재 검증은 pure-CMFGEN으로 plasma 수렴 → 스펙트럼 합성.

## 2. 작동하는 것
- pure-CMFGEN deterministic plasma로 T_e 프로파일이 gold(CMFGEN) 대비 ~0.5% RMS.
- n_e도 캠페인 누적으로 양호(frozen-in, core-ray pathlength fix 등 적용 후).

## 3. 실패하는 것 (최종 산출물 = 스펙트럼)
- 스펙트럼 합성 경로 5종:
  - P1 MC 실패킷, P2 MC 가상패킷, **P3 형식적분 Lucy1999 관측자계(P-Cygni 가능)**,
    **P4 CMF 유한프로파일+선겹침**, P5 comoving(셸간 Doppler 없음 → P-Cygni 구조적 불가).
- P3 선소스 버그(J→line_source_S) 수정(commit b00be0d) 후: 선은 형성됨
  (deep minima 0→28(P3)/72(P4))**그러나 너무 파랑**:
  피크 ~3000-4000Å, flux의 49%가 <4000Å. gold는 ~6600Å 빨강 피크.
- P3/P4 일치(둘 다 line_source_S 사용) → 합성 경로 문제 아님. 입력 S_l이 문제.

## 4. 준위점유가 틀렸다는 직접 증거
### (a) Super-thermal S_l (job 165958, S_l dump)
- gold-matched T_e=4315K(0.5% 정확)에서 선 소스 함수
  S_l = (2hν³/c²)/(g_u n_l/(g_l n_u) − 1) 를 B(T_e)와 비교:
  - **광학대역 S_l/B 중앙값 = 91.5** (90배 super-thermal)
  - UV 선의 64%가 super-thermal.
- 즉 상위준위 n_u가 LTE@T_e 대비 과점유 → 선소스가 B(T_e)보다 훨씬 큼.
- 핵심 질문: 이 super-thermal이 **물리적**(CMFGEN도 비-LTE 선소스를 가짐;
  radiative pumping/형광 캐스케이드의 정상 결과)인가, 아니면 **버그**인가?

### (b) 특이행렬 (pure-CMFGEN run, GPU getrf)
- 49 셸 중 **4개 셸의 NLTE rate matrix가 특이**(cublasDgetrfBatched info>0).
- 16개 stderr 경고: `[NLTE-FALLBACK] GPU pair ... shell=N ret=.. info=.. -> Boltzmann@T_rad`.
- 특이 슬롯은 Boltzmann@T_rad로 폴백 → 그 셸/이온의 준위점유가 LTE@T_rad로 대체됨.

### (c) Hang (frozen-plasma → MC macroatom 배선 시)
- 특이행렬에서 나온 NaN transition probability →
  macroatom escape 루프 `probability += NaN; if (probability > event)`가 항상 false →
  패킷이 interaction cap(5000 internal / 100000 total)까지 갈림 → 수 시간 hang.
- 사용자 지시: **증상(NaN guard/sanitize) 패치 금지. 행렬이 왜 특이한가 = 준위점유가
  왜 틀렸나의 근본을 파라.**

## 5. 판정 요청
1. **Super-thermal S_l(91×)이 물리인가 버그인가?**
   - 4315K에서 광학선의 상위준위가 LTE 대비 90배 과점유가 정상 범위인가?
   - rate-matrix(아래 코드) 구성에서 상위준위를 과점유시키는 항이 있는가?
   - 특히 van Regemorter 충돌(×0.2 스케일, ×0.2의 근거?), 자발방출/유도방출,
     binned J_line(nlte_get_J_at_nu)이 상위준위로 펌핑하는 균형이 맞는가?
2. **4/49 특이행렬의 원인**은? (보존행 붕괴? bb-고립 상위준위? 밀도≈0 이온?
   T_e 고/저에서 충돌항 underflow?) 어느 Z/이온/셸인지 진단 방법 제안.
3. **(a)와 (b)가 같은 뿌리인가?** (둘 다 틀린 준위점유)
4. **(A) 구조적 결손(다준위 형광 누락)인지 (B) 특정 버그인지** 판정과 근거.
   - (A)면: 2준위 결정론 선소스가 UV서 과방출 → MC macroatom 경로(형광)로
     스펙트럼을 합성해야 하는가, 형광을 결정론 경로에 이식해야 하는가?
   - (B)면: 어느 코드 줄/항이 문제인지.

## 6. 핵심 코드

### 6.1 rate matrix 조립 — bound-bound 라디에이티브+충돌 (lumina_plasma.c:6645-6717)
```c
/* ---- Radiative bound-bound rates from line data ---- */
for (int line = 0; line < n_lines; line++) {
    ... (이온/준위 매핑) ...
    double nu_line = atom->line_nu[line];
    double J_line = nlte_get_J_at_nu(nlte, shell, nu_line);   // binned mean intensity

    double R_absorb = atom->line_B_lu[line] * J_line;
    double R_stim   = atom->line_B_ul[line] * J_line;
    double R_spont  = atom->line_A_ul[line];

    double dE = fabs(E_upper - E_lower) * EV_TO_ERG;
    int g_lo = atom->level_g[lower_global];
    int g_up = atom->level_g[upper_global];
    double f_lu = atom->line_f_lu[line];

    double C_up = 0.0;
    if (T_e > 0.0 && dE > 0.0) {
        double exp_factor = exp(-dE / (K_BOLTZMANN * T_e));
        if (f_lu > 1e-10) {
            C_up = VAN_REGEMORTER_COEFF * n_e * f_lu *
                   exp_factor / (g_lo * sqrt(T_e)) * 0.2;     // <-- ×0.2 스케일
        } else {
            C_up = 8.63e-6 * n_e * AXELROD_OMEGA *
                   exp_factor / (g_lo * sqrt(T_e));
        }
    }
    double C_down = (g_lo>0 && g_up>0 && T_e>0.0) ?
        C_up * ((double)g_lo/(double)g_up) * exp(dE/(K_BOLTZMANN*T_e)) : 0.0;

    double total_up   = R_absorb + C_up;
    double total_down = R_stim + R_spont + C_down;

    double f_lo = FRAC_OF(fl_lo_g);   // within-superlevel Boltzmann fraction
    double f_up = FRAC_OF(fl_up_g);
    ACM(i_up, i_lo) += total_up   * f_lo;
    ACM(i_lo, i_up) += total_down * f_up;
    ACM(i_lo, i_lo) -= total_up   * f_lo;
    ACM(i_up, i_up) -= total_down * f_up;
}
```
(이후 photoionization/recombination 블록, 충전보존행으로 닫음 — 필요시 추가 제공)

### 6.2 line_source_S 계산 (lumina_cuda.cu:841-866, plasma.c:7407-7435 동일)
```c
double nu_l = C / lam_cm;
double src_prefac = 2*H*nu_l*nu_l*nu_l/(C*C);
for (int s=0; s<n_shells; s++) {
    double n_lower = nlte_level_populations[nlte_lo*n_shells + s];
    double n_upper = nlte_level_populations[nlte_up*n_shells + s];
    double stim_corr = 1.0 - (g_lo*n_upper)/(g_up*n_lower);  // clamp >=0
    double tau = SOBOLEV_COEFF*f_lu*lam_cm*t_exp*n_lower*stim_corr;
    double S_l = 0.0;
    if (n_lower>0 && n_upper>0) {
        double ratio = (g_up*n_lower)/(g_lo*n_upper);
        double denom = ratio - 1.0;
        if (denom > 1e-30) S_l = src_prefac/denom;   // super-thermal when denom small
    }
    line_source_S[line*n_shells + s] = S_l;
}
```

### 6.3 특이행렬 처리 (lumina_cuda.cu:519-592, GPU 경로)
- getrf info[i]>0 → 특이 → 그 셸은 Boltzmann@T_rad 폴백(준위점유 LTE@T_rad로 대체).
- 추가로 inv-ceiling sanity gate: 상위준위가 ground의 (g_i/g_0)*margin 초과 시도 폴백.
- 비-finite(NaN/Inf) 출력도 폴백.

### 6.4 nlte_get_J_at_nu (binned 평균강도)
- pure-CMFGEN가 채운 1000-bin J_nu에서 nu_line이 속한 bin의 J를 보간/반환.
- 라디에이티브 펌핑 R_absorb = B_lu * J_line의 J가 이 값.

## 7. 참고
- 메모리: project_spectrum_pipeline_fluorescence.md(통합진단),
  project_inner_ne_fuv_thermalization.md(super-thermal S_l ↔ T_e 폭발 동일뿌리 주장).
- 사용자 원칙: 근본원인만, band-aid 금지, 결론 전 검증.
</content>
</invoke>
