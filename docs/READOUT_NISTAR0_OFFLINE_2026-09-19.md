# 판독 — DET-CONT-NISTAR-0 오프라인 귀속: CONT 초열비 1e79 = Si V 깊은 엣지 Saha 인자 (2026-09-19, 런 0회)

운전자(Claude) 판독. 상위 = `docs/PREREG_DET_CONT_NISTAR0_2026-09-19.md`(census 계약). **경량 사전감사(Opus 4.8)가 Q-LEAN L4 «폐합 차단»으로 「census 런이 오프라인 정적 조회 대비 과provision」이라 판정 → 오프라인 탐침으로 귀속(런 0회).** 도구 = repro (해시 `READOUT_TOOLS.sha256`), 입력 = run5 `nlte_levels_final.csv`·`ionization_energies.csv`(덱)·twin T_e.

## 1. §0 항등 확인 (사전감사 재현)
Lumina η_bf = `eta_opacity·S_l` 에서 n_level 소거(`lumina_plasma.c` 8548·8611·8615·8416·8336) ⟹ η_bf 이미 CMFGEN n_i* 형(Opus 4.8 사전감사 독립 재현). 브리프의 「재정의」는 no-op. 1e79 는 η 가 아니라 census 비 η/(χ·B) = 1/b_l = n_l*/n_l 의 발산.

## 2. 오프라인 인자 배제·귀속 (런 0회)
- **(i) n_l 붕괴 배제**: 8373 floor `n_level<1e-30 → skip` ⟹ 비 ≤ n_l*·1e30; 1e79 엔 n_l* ≥ 1e49 필요(물리 n_l* ≤ ~1e10) ⟹ n_l 붕괴 단독 불가.
- **(ii) U_next 이상 배제**: run5 `nlte_levels_final.csv` 로 텍스트북 U(T_e)=Σg·e^{-(E-E_g)/kT} 재계산 — S/Si/Ca I·II·III 전건 U ≥ g_ground(정상, 최소 1.00). 분배함수 소멸·tiny-positive 없음.
- **(iii) 깊은 엣지 = 귀속** [실측]: n_l*/n_next = n_e·(g/2U_next)·(h²/2πm_e kT)^{1.5}·e^{χ/kT}. 덱 이온화에너지 × twin T_e(셸20 8849K, kT 0.763 eV):

| 이온 | χ(eV) | e^{χ/kT} | n_l*/n_next |
|---|---|---|---|
| **Si V** | **166.77** | **9.5e94** | **2.42e79** |
| Ca V | 84.41 | 1.2e48 | 7.8e32 |
| S V | 72.59 | 2.2e41 | 5.6e26 |
| Si IV | 45.14 | 5.1e25 | 2.7e10 |

**Si V(Ne-like 닫힌 껍질, χ 166.8 eV)의 n_l*/n_next = 2.42e79 가 CONT run1 census 셸20 max 1.60e79 와 일치**(지상준위 근사). S V·Ca V 는 1e33 이하로 far below. ⟹ **1e79 = Si V 의 깊은 재결합 엣지 Saha 인자를 낮은 twin T_e 에서 계산한 값**이다. U_next 버그·데이터 손상 아님 — 잘못 채워진 고이온의 정확한 Saha 물리.

## 3. 읽기 — 이건 근인이 아니라 하류 증폭기다
- Si V 는 9000K 에서 **존재하면 안 되는 이온**(V 이온화 ~수십만 K). CONT 의 과이온화가 Si 를 V 로 밀어올렸고, 그 잘못된 인구의 Milne 소스가 e^{167/kT}~1e95 로 초열이 된다. ⟹ **1e79 는 과이온화의 증상이자 되먹임 증폭기**이지 독립 버그가 아니다.
- **mean 10–30 은 정직한 과이온화 비**(CONT ion.csv 셸20 q2/q1=231, base r=8.79 ⟹ ~26)로, 이것이 실제 병리다.
- ⟹ **근인 = decoupled 조립기 solve(완전 선형화 부재)** — CMFGEN 은 인구·장 동시 해라 Si 가 V 로 과이온화하지 않아 깊은 엣지 소스가 켜지지 않는다. Lumina 는 lagged solve 라 과이온화 → 깊은 엣지 → 되먹임.

## 4. 처분·다음
- **1e79 에 값싼 배선 수리는 없다.** U_next·데이터 정상, 엣지는 정확한 Saha. 유일한 수리 = 고이온 과이온화를 막는 것 = **완전 선형화(인구·장 동시 해)** — 큰 아키텍처 변경. 또는 국소 완화(9000K 에서 비물리적으로 켜지는 깊은-엣지 고이온 소스를 이온화 연속체에서 배제 — 물리적 처방 설계 필요, clamp 아님).
- census GPU 런 **불필요**(사전감사 L4 실현 — 오프라인이 귀속). DET-CONT-NISTAR-0 은 **구현·런 없이 오프라인 귀속으로 종결**(NIS0 분기 라벨 대신 「오프라인 귀속」).

## 5. 주장 상한
Si V 귀속은 지상준위·twin T_e·이온쌍 n_l~n_next 가정 위의 자릿수 일치(2.42e79 대 1.60e79). 정확한 셸별 지배 준위·n_next 실측값은 census 가 줄 수 있으나(불요 판단), mean=과이온화·근인=커플링 결론은 바뀌지 않는다. 완전 선형화가 실제 수리인지는 미검증(CMFGEN 방어층 후보).
