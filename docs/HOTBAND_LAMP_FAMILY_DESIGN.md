# Hot band s36-40: 램프 가족 진단 확정 + 물리 fix 설계

날짜: 2026-07-04 (전용 세션). 베이스라인 커밋 64de233 (fix22/repro22, bit-identical 검증).

## 1. 확정된 진단 (falsifier 체인)

진단 게이트 `LUMINA_DIAG_BF_DARK=smin:smax:emin_eV` (cmfgen_assemble에서 S_fixed의
방출항만 소등, 흡수(chi_t)는 유지 — 물리 아님, falsifier 전용) + `LUMINA_DIAG_LINE_DARK=1`
(η_line도 소등) + `LUMINA_DIAG_DARK_ITERS=N` (iter<N만 소등).

| Arm | 게이트 | 결과 (s36-40 mean T_e) | 판정 |
|-----|--------|------------------------|------|
| — (repro22) | 없음 | 75,308 K (40k-106k) | 기준선 hot band |
| A | s36-40 bf만 | 71,750 K — 불변 | 자기 bf 램프는 유지자 아님; J(35eV)=5e-9~3e-8이 s43 피크로 유입 = **이웃(turn-up) Wien꼬리 조명 확증** |
| B | s36-49 bf | s36-37: 11.4k/12.1k (CMFGEN급!) / s38-49로 hot zone 이동 (35-76k) | bf를 죽이면 **line 채널이 이어받음**: JDUMP에 30.7 eV (J~1.3-3e-6)·41.1 eV (6e-7) 좁은 스파이크 = binned FUV forest η_l=χ_l·ε·B(T_e); ε~1e-5여도 B(60-100kK) Wien + 산란증폭(chi_line≈chi_es, Λ*~0.5, J/S_fixed~10×). 41.1 eV가 S III→IV(34.8)/Si III→IV(33.5) strip 지속 |
| D | s36-49 bf+line | **12,551 K — 밴드 소멸** | 매끈한 단조 U자 11.1k(s36)→37.2k(s49); s36-43 = 1.07-1.15×CMFGEN; s48-49 PIN(20940) 이탈, 진짜 root; s30-35 회귀 없음(0.95-0.99로 개선) |

오프라인 계산기 교차검증 (scripts/offline_cell_balance.py --jdump, Arm B 실측 J):
- Arm B J 그대로: hot + Si V/S IV 100% (in-code 재현 ✓)
- ≥30 eV 완전소등 + η_nt=0: **전 외곽 매끈 U자 17k→30k, Si III/S III 100%, hot band 소멸**, CMFGEN 1.2-1.6× (정직한 closure gap)
- 소등 + flat η_nt=0.05: far-edge(s43+)만 재과이온 (s49 Si V 92%) → far-edge 잔차는 flat-eta NT 과강 = **per-ion Spencer-Fano 명명 gap의 몫** (hot band와 별개)

핵심 대조 (s38, v=31,928 km/s): CMFGEN Si III 74%/S III 70%/T=11,123 K.
s37↔s38 분기 = 황: S III(냉각제) 91% vs S IV 87%. H_dep(s38)=5.7e-10 (deposition-closure).

**결론: hot band = 채널-불가지론적 열적 램프 가족 (η=χ·B(T_e), bf든 line이든).**
한 채널만 고치면 (A-full의 bf-Milne, Arm A/B의 bf-dark) 다음 Wien-꼬리가 이어받아
attractor를 재점화한다 — A-full 실패의 진짜 이유. G1(라인 S_l=B) + bf 램프 = 한 뿌리.

## 2. 코드상 비일관 (직독)

| 채널 | opacity | 방출 | 비일관 |
|------|---------|------|--------|
| bf | χ_bf: dilute-Boltzmann pops @**T_rad** (plasma.c:3048) | χ_bf·B(**T_e**) (cmfgen.c S_fixed) | pops와 Planck 온도 불일치 → T_e≫T_rad시 LTE 극한 없음 |
| line | τ_sob: n_ion(rate-solved ✓) × 준위 dilute-Boltzmann @**T_rad** (plasma.c:966-990) | η_l=χ_l·ε·B(**T_e**) (eps_phys 열적 몫) | 동일 구조의 짝 불일치 |

## 3. Fix 설계 — Arm E가 분기 결정

**Arm E (cold-branch continuation): iter 0-7 소등 → iter 8-11 정직한 램프 복원 — 판정: 재점화.**
s37-40이 4 iter 만에 26k/44k/66k/69k로 복귀. **단 점화 파면이 방향성**: s36은 10.9k로
생존, 가열이 s40(69k)→s37(26k)로 바깥→안 감쇠 = **점화원은 far edge(s43-49)의
정직한 warm root(30-37kK) Wien 꼬리**가 밖에서 공격, S III 차폐가 안쪽을 방어.
⟹ cold-branch 선택 단독 출하 불가; 기록의 "all-cold 자기일관"은 far edge가 냉각됐을
때만 성립 (far edge 30-37k는 D에서도 남는 정직한 root — flat-eta NT 과강 포함).

**확정 fix 스택 (orthodox, 인과 순서):**
1. **bf 방출 일관화** (최대 레버, 전 셸의 인공 χ_bf·B(T_e) 램프 제거): η_bf를
   재결합 물리로 — S_bf 소스함수형(A-full 기계 재기용, 롤백 코드 기록 있음) +
   χ_bf 일관 pops. n_e~5e4-8e5에서 재결합 연속체 ≪ LTE-bf → far-edge 35eV 출력
   수 orders 감소. A-full 단독 실패는 이제 설명됨(line 채널 + far-edge 점화원) —
   단독 판정이 아니라 스택의 일부로 재평가.
2. **per-ion Γ_nt** (Spencer-Fano/Lotz per-ion; ARTIS 레시피 기록) — far edge
   37k→~24-30k, Wien 꼬리 추가 ~2 orders 감광. 계산기 근거: eta=0 → s49 30k,
   eta=0.05 → 43k, CMFGEN 24.4k.
3. line 채널: ε·B(T_e)는 정직한 냉각복사이므로 유지 — 1+2 후 점화원이 충분히
   어두운지 falsifier로 확인. 재점화 잔존 시에만 τ_sob pops의 T_e-일관화 추가.

**스택 falsifier**: 게이트 전무 상태 full run → hot band 미형성 + s36-43 ≈ 1.05-1.15×
+ far edge ≤1.3× + s0-35 무회귀 + compare_toy06_full.py (formal) 무회귀.
주의: 에너지 관점 — 정직한 램프 총출력은 냉각 Λ(=H_dep~5e-10)로 유계여야 함.
현재 스파이크장 J~1e-6은 그 수지를 초과 (장부에 없는 에너지).

## 4. 잔여 명명 gap (hot band와 분리됨)

1. **per-ion Γ_nt** (far-edge s43+ 과이온, flat eta=0.05 과강) — ARTIS Spencer-Fano 레시피.
2. **정직한 closure gap** (전소등시에도 CMFGEN 대비 1.1-1.5×) — 냉각제 atomic data 부족분.
3. 스펙트럼 전선 (formal too-UV/MC too-red) — 본 세션 무관, 미해결 유지.

## 5. 도구

- scripts/run_bfdark.sh (Arms A/B/D/E), scripts/analyze_bfdark.py (판정+J35 프로파일)
- 덤프: logs/stage1_toy06_bfdark{A,B,D,E}/ (plasma_state, jnu, spectra, stdout)
- figures/bfdark_j35_profile.png
- offline_cell_balance.py --jdump <경로> (이번 세션 추가)

## 6. 구현 시도 결과 (2026-07-05 자정) — per-channel 소스패치 한계 확정

- **milne2 (BF_MILNE=1, meta-only)**: 실패 — 밴드 잔존(66k). 잔여 램프 = 비-meta 준위 열적 χ·B(91kK) + line 채널.
- **에너지 장부 감사 (결정타)**: P_emit(>30eV) vs H_dep — s39(91.8kK) **2.1×10⁷배**, s38 4.8×10⁶배,
  cold s43 1.2배(자연 유계). Arm B line 채널도 10³-10⁴배. 기전 = hot 셸끼리 방출≈흡수하는
  unpaid 복사욕 (per-shell r1은 net만 봐서 못 잡음; ARTIS=e-packet 보존, CMFGEN=전역 Newton이 봉쇄).
- **milne3 (BF_MILNE=2, 전준위)**: 실패+회귀 — 밴드 72.5k(s40 128k) + **valley 오염(s30-35→20-34k)**.
  원인: valley의 n_+(Si IV/V) 상류 과이온화 인플레 → 재결합 글로우 과대. 상류 pops가 틀린 곳에선
  어떤 소스함수 형태도 정직 불가. (지난 세션 meta-only 채택 근거가 옳았음.)

**최종 경계**: 출하 fix = ① 에너지-보존 방출 부기 (ARTIS kpkt 미러: 셸은 흡수+deposition만큼만
방출) 또는 ② transport-측 RE 적분 강제 ∫χ(S−J)dν = H_dep per shell (CMFGEN 방식) — 기존
option-2 integral-RE / A4 방향과 합류. per-channel 소스함수 교체로는 닫히지 않음 (양 방향 실증).

Arm D(1.07-1.15×CMFGEN)는 목표 상태의 존재증명으로 유지. 게이트/기계 전부 env-off 기본(레거시
byte-identical), repro23 bit-identity 검증으로 상태 신뢰 재확립.
