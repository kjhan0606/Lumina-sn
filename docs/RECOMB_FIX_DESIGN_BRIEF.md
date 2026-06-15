# LUMINA-SN: 재결합 항 fix 설계 검증 브리프 (2026-06-14)

## 확정된 버그 (삼중검증 + falsifier 테스트 통과)
`src/lumina_plasma.c:6812` (CPU assembly) 및 동일 GPU 경로:
```c
double R_rec = R_bf * n_star_ratio;
```
여기서
- `R_bf = ∫ 4π J_ν σ_ν/(hν) dν`  (광이온화율, binned J_ν에 대한 적분; GPU는 static-K GEMM R_bf=K·J)
- `n_star_ratio = n_e·λ³·(g_lev/2g_ion)·exp(χ_lev/kT_e)` = Saha-Boltzmann 준위/이온 LTE 비 (plasma.c:6793-6809)

**문제**: R_rec을 R_bf(=J 적분)에 비례시켜, 차가운 외곽서 이온화 J_ν→0이면 R_rec→0 → 자발(라디에이티브) 재결합이 사라짐. 물리적으론 자발재결합은 J와 무관하게 n_e·n_ion·α_rr(T_e)로 계속돼야 함. 결과: 하위이온 준위가 연속체와 decouple → drain/반전 → 선소스 S_l 광학 폭발(median 3260×).

**falsifier 확정**: O+Si를 nebular Saha로 고정(rate-solve 우회)하니 광학 S_l/B **2960→77 (38×) 붕괴**. τ>30 갇힌 라인 99.3% super-thermal(물리 불가). 주범 O II/III, 차가운 외곽(J_ν Wien붕괴)서 최악.

## 검증된 재사용 빌딩블록: `frozenin_alpha_rr` (plasma.c:1917-1996)
B(T_e)에 대한 Milne 재결합계수[cm³/s], **자발항 2hν³/c² 포함**:
```c
double B = (2.0*H*nu_c³/c²) / expm1(hν/kT_e);
Rbf += 4π·B·σ/(h·nu_c)·dnu;                       // ∫4πσ B/hν dν
...
a_tot += Rbf · λ³ · g_lev/(2·U_ion) · exp(χ_l/kT_e);  // = n*_lev/(n_e n_ion) factor
// + 선택적 DR (LUMINA_FROZENIN_DR), U_II 분배함수 보정(Si II/C II ^2P)
```
문헌검증: 외곽 ⟨Z⟩~0.53 plateau를 ZERO fitting 재현(scripts/frozen_in_milne_prototype.py). cmfgen σ_bf 필요.

## 제안 fix (검증 요청)
일반(비-LTE) 재결합률:
```
R_rec = n_star_ratio · ∫ (4πσ_ν/hν)(2hν³/c² + J_ν)·e^(−hν/kT_e) dν
      = R_rec_spont(T_e만, J무관)  +  R_rec_stim(J_ν 가중적분)
```
- J=B(T_e) 극한서 (2hν³/c²+B)e^(−x)=B 이므로 ∫(4πσ/hν)B dν로 환원 → LTE Saha 고정점 보존(=frozenin_alpha_rr의 Rbf_B 구조와 일치).
- 현재 코드는 (a) 자발항 2hν³/c², (b) e^(−hν/kT_e) 가중 둘 다 누락.

## 검증 질문
1. **물리 정확성**: 위 일반형이 표준 Milne 재결합률로 옳은가? n_star_ratio가 이미 exp(+χ/kT_e)·λ³를 품고 있는데, 적분 안의 e^(−hν/kT_e)와 어떻게 정합하나(이중계상 없는가)? J=B서 LTE Saha로 환원됨을 확인.
2. **frozenin_alpha_rr 재사용 vs 인라인**: 자발/LTE 부분 = frozenin_alpha_rr의 per-level Rbf_B 구조와 동일. 이걸 per-(level,shell) 자발-재결합으로 재사용하고 stimulated 보정만 더하는 게 옳은가, 아니면 assembly 빈루프(plasma.c:6776)서 전체 가중적분을 인라인 계산하는 게 옳은가?
3. **GPU 구현**: GPU R_bf_table은 static-K GEMM(R_bf=K·J, K=σ·4π/hν·dν, T_e 무관). 자발항은 J무관이라 per-(level,shell) precompute 가능(T_e 의존); stimulated는 e^(−hν/kT_e) 가중이 T_e의존이라 static-K로 못 넣음. 최선 구현은? (자발-재결합 테이블 K_rec_spont[level,shell] + stimulated 별도?)
4. **회귀 안전성**: inner(뜨겁고 J≈B) 영역은 현재 코드가 이미 작동. 새 항이 LTE 고정점을 보존해 inner를 안 깨는가? PER_ION_RESCALE/frozen-in 경로와 이중계상 없는가(frozen-in은 별도 게이트)?
5. **DR/U_II**: frozenin_alpha_rr의 U_II 분배함수 보정·DR을 NLTE assembly 재결합에도 적용해야 하나, 아니면 per-level이라 무관한가?

원칙: 패치 금지, detailed-balance 복원, 구현 전 물리 재검증. 구체적 verdict + 구현 스케치 요청.
</content>
