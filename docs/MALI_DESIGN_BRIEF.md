# LUMINA-SN: MALI (Sobolev-escape NLTE rate matrix) 설계 검증 브리프 (2026-06-15)

## 목표
두꺼운 선(τ≫1)이 자기일관적으로 thermalize(S_l→B(T_e))되도록 **NLTE 준위 rate matrix**를 고친다 — operator-split에서 CMFGEN의 동시선형화 등가물. 이게 되면 super-thermal S_l이 치유되고 MC macroatom이 올바른 선 features를 만든다.

## 확정된 문제 (이 세션 + A4 메모리 수렴)
DDC15 0.976d: T_e/n_e는 gold 0.5% 일치하나 준위점유가 super-thermal(S_l/B 광학 median ~1947, τ>30 두꺼운 선의 99% super-thermal). 이걸 MC에 넣으면 패킷이 47× 과다 상호작용(선이 흡수 대신 재방출) → featureless 스펙트럼(6/15가 6/10보다 나쁨). 6/10 표준MC는 thermal 준위 → 깨끗한 features.

## 뿌리 (코드 실측)
`lumina_plasma.c:6700` bound-bound rate에 **Sobolev escape β_esc 없음**:
```c
double J_line   = nlte_get_J_at_nu(nlte, shell, nu_line);   // binned ambient J
double R_absorb = atom->line_B_lu[line] * J_line;           // bare, no beta
double R_stim   = atom->line_B_ul[line] * J_line;
double R_spont  = atom->line_A_ul[line];                    // full A_ul, no beta
```
bare J를 쓰니 두꺼운 선의 복사율이 안 사라짐 → 약한 충돌을 압도 → 준위가 J/cascade를 따라 super-thermal.

## 제안 MALI 형태 (codex가 "host ~8줄"이라 한 것)
Sobolev escape probability를 bb 복사율에 곱한다:
```c
double tau_l = opacity->tau_sobolev[(size_t)line*n_shells + shell];
double beta  = radeq_beta_esc(tau_l);   // (1-e^-tau)/tau, 이미 존재(plasma.c:3472)
double R_absorb = atom->line_B_lu[line] * J_line * beta;
double R_stim   = atom->line_B_ul[line] * J_line * beta;
double R_spont  = atom->line_A_ul[line] * beta;
```
**물리**: 두꺼운 선 β→0 → 복사율→0 → 충돌(C_up/C_down, detailed-balance쌍)이 비율을 지배 → n_u/n_l=(g_u/g_l)e^{−hν/kTe}=Boltzmann → **S_l→B(T_e) thermalized**. 얇은 선 β→1 → 복사율 그대로 → 물리적으로 비-thermal 허용. = CMFGEN-like 균형. (Sobolev cancellation: 갇힌 (1−β) 광자는 흡수·자발방출서 상쇄 → 자기일관성이 해석적으로 처리됨.)

## 과거 실패와의 구분 (반복 방지)
**7 strikes는 MALI가 아니라 ε-소비(consumption)였음**: SRC_BLEND/eps_phys/line-RE — 이미 틀린(super-thermal) S_l을 J-solve·에너지수지서 downstream으로 누그러뜨림(증상). 불안정 이력은 frozen-S_l을 J-solve에 먹인 데서 옴. **MALI는 정반대 — rate matrix를 고쳐 준위가 애초에 thermalize돼 나오게(근본).** rate matrix엔 β_esc/Λ* 전무(과거 한 적 없음, 메모리 확인).

## 검증 질문
1. **형태 정확성**: `β·J_line` 곱셈이 옳은 Sobolev rate인가, 아니면 J_ext(연속체)를 binned J에서 분리해야 하나? binned J_line이 선 자체 방출을 포함하면 β·J_line이 self-coupling을 이중계상 안 하나(Sobolev cancellation이 binned framework서도 성립?)? 검증: 두꺼운 선 β→0서 정말 Boltzmann으로 가는지, stim emission/detailed balance 보존되는지.
2. **Λ* 전처리 필요?**: Sobolev escape가 local approximate-Λ 역할을 해서 별도 Λ* 전처리 불요인가(self-coupling 상쇄로)? 아니면 cross-line/binned 안정성 위해 diagonal-Λ* 전처리가 추가로 필요한가(Rybicki-Hummer 1991/92)?
3. **안정성**: 이게 operator-split-STABLE인가(과거 ε-소비 7실패와 구조적으로 왜 다른가)? β_esc만 넣으면 불안정해지는 경로 있나?
4. **super-thermal 주범 치유 여부**: 이 세션 진단 — super-thermal엔 두 뿌리: (a) bb rate β_esc 누락(MALI가 고침), (b) top-stage III 연속체-앵커 부재(IV 없어 들뜬준위 bf 0). MALI(a)만으로 O III/S III 들뜬 super-thermal이 치유되나, 아니면 (b) 천장도 별도로 필요한가? (충돌이 약한 sub-critical 밀도서 β→0이 정말 Boltzmann을 강제하나 — C_up/C_down이 약해도 비율은 Boltzmann이라 OK일 듯, 확인.)
5. **구현 범위**: bb rate는 CPU 어셈블리(GPU GEMM은 bf-only) → MALI는 CPU만, CUDA 무접촉 맞나? tau_sobolev가 이 시점에 채워져 있나(어셈블리 순서)? S_l writer(7407)도 β 영향받나?
6. **falsifier**: 두꺼운 선 S_l/B → O(1) 붕괴 + T_e/n_e 0.5% 유지 + MC features 형성. 단일 env 게이트 A/B.

원칙: 패치 금지, 근본·detailed-balance, 구현 전 검증. 과거 7-strikes 반복 금지. 구체 verdict + 구현 스케치 + 가장 날카로운 falsifier.
</content>
