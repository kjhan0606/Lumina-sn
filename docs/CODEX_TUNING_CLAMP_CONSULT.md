## 결론

1. **현재 물리 파라미터 튜닝은 의미가 없다.** 스펙트럼을 맞출 수는 있어도, 얻은 값은 물리 파라미터가 아니라 구조 결함을 상쇄하는 보상계수가 된다. 지금 허용되는 것은 진단용 감도시험과 고정점을 바꾸지 않는 수치 알고리즘 조정뿐이다.

2. **cap/floor 자체는 유효할 수 있지만 물리 결함의 수리 수단은 아니다.** 정확한 물리 제약을 집행하거나, 고정점을 보존하거나, 오차를 엄밀히 제한하는 경우만 정당하다. 계산된 물리량을 임의 기준값으로 대체하면 거의 항상 위험하다.

## 질문 1 — 파라미터 튜닝

### 판정: 현재는 과학적 의미가 없다

현재 모델 오차가 튜닝하려는 파라미터와 같은 출력 방향을 지배한다.

- bf 흡수는 모든 continuum 기여를 확률적으로 추첨하지 않고, 빈마다 최대 기여 `best_ip` 하나만 고른 뒤 단일 activation level로 바꾼다: [lumina_plasma.c:6591](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6591), [lumina_plasma.c:6847](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6847), [lumina_plasma.c:6857](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6857). 수송에서는 그 정수 하나를 그대로 읽는다: [lumina_cuda.cu:3372](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:3372), [lumina_cuda.cu:5530](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5530). 이것은 계수 오차가 아니라 사건 공간 자체의 축소다.

- `chi_bf`는 실제로 `n_level*sigma`만 더한다: [lumina_plasma.c:6798](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6798), [lumina_plasma.c:6801](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6801). 같은 구현이 stimulated recombination을 버렸다고 명시한다: [lumina_plasma.c:6811](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:6811). 따라서 bf opacity 계수나 이온화 문턱을 조절하면 D-3을 보정하는 값으로 흡수된다.

- 공유 이온은 겹치는 pair를 순차적으로 푼 뒤 낮은 이온 블록을 저장·복원한다: [lumina_cuda.cu:1030](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:1030), [lumina_cuda.cu:1055](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:1055), [lumina_cuda.cu:1633](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:1633). 행렬의 기본 continuum block도 낮은 이온의 ionization edge 하나로 구성된다: [lumina_plasma.c:14893](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:14893). 이 토폴로지에서는 재결합률·damping을 바꿔도 올바른 다단계 이온 평형을 식별할 수 없다.

- formal 적분 실측은 `L_out/L_total_in=17.738`이다: [stdout.log:38070](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity56/stdout.log:38070). 즉 상대 에너지 초과가 약 **+1,674%**다. 이 크기의 비보존에서는 선 세기·색·이온화의 목적함수가 모두 잘못 정규화된다.

- D-4 하나만 OFF로 바꾼 단일변수 A/B에서 formal 적분이 **−21.8968%**, shell 0의 \(T_e\)가 **−2896.8 K** 변했다: [CODEX_PARITY57_V3_ADVERSARIAL.md:5](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_PARITY57_V3_ADVERSARIAL.md:5). 같은 입력을 적용하면 OFF 뒤에도 \(L_{\rm out}/L_{\rm total}\approx13.85\)로 추정되어 여전히 약 **+1,285%** 비보존이다. 사건 수로는 terminal의 0.530%였던 채널이 스펙트럼을 22% 움직였다는 점도 단순 사후 계수보정이 불가능함을 보여준다.

따라서 지금 기준 스펙트럼에 맞춘 계수는 대략

\[
\theta_{\rm fit}=\theta_{\rm physics}
+\text{D-1/D-3/D-5 보상}
+\text{formal 비보존 보상}
+\text{clamp 보상}
\]

이 되어 원래 물리량과 분리되지 않는다. 좋은 스펙트럼 일치는 가능하지만, 다른 epoch·조성·밀도에서 재현될 근거가 없다.

### 언제부터 튜닝이 의미가 생기는가

다음 조건이 순서대로 충족돼야 한다.

1. **사건·방정식 폐합**

   - bf continuum/level 추첨과 `ν_edge/ν` 에너지 분기가 구현·검증됨.
   - `chi_bf`와 재결합 방출이 같은 detailed-balance 정의를 사용함.
   - macro-atom 열화가 단일 fair draw임.
   - 겹치는 ion pair가 아니라 element-wide 또는 동등한 동시 다단계 solve가 최종 인구를 소유함.

2. **보존 폐합**

   - \( |L_{\rm out}-L_{\rm in}|/L_{\rm in} \)이 목적 스펙트럼 정확도보다 충분히 작아야 한다.
   - 예를 들어 광대역 5% 정확도가 목표라면 전역 에너지 오차는 **≤1%**, 또는 적어도 목표 허용오차의 1/5 이하가 합리적이다.
   - particle/charge 및 packet energy ledger도 같은 수준으로 닫혀야 한다. 상호작용 cap에 의한 미계상 에너지는 0이거나 명시적 오차예산에 들어가야 한다.

3. **수치 수렴과 해상도 독립성**

   - 반복수, 패킷수, 주파수 격자, line cull, damping을 변화시켰을 때 스펙트럼 변화가 목표 허용오차보다 작아야 한다.
   - 서로 다른 초기조건이 같은 고정점으로 와야 한다.
   - clamp 발화 건수만이 아니라 에너지·냉각·복사율 가중 효과가 측정돼야 한다.

4. **식별가능성**

   - 튜닝할 계수의 변화가 MC 잡음과 discretization 오차보다 최소 약 3σ 커야 한다.
   - 하나의 스펙트럼만 맞추지 말고 ion fraction, \(T_e\), \(J_\nu\), 주요 rate ledger 및 여러 epoch/조성을 함께 맞춰야 한다.
   - 원자자료 불확실성 범위 안의 계수만 조정하고 hold-out 사례에서 검증해야 한다.

### 지금도 정당한 좁은 튜닝

- 고정점을 바꾸지 않는 damping·trust-region·선형솔버 tolerance 조정.
- 패킷수·격자수·반복수 같은 정확도/비용 조정.
- 결함 위치를 판별하기 위한 ablation sweep.
- 외부 원자자료에 실제 불확실성이 있는 근사 계수의 감도분석.

다만 이 결과를 관측 적합이나 CMFGEN-동등 물리 파라미터로 해석해서는 안 된다.

## 질문 2 — cap/floor clamp

### 판정: 수치 보호에는 조건부 유효, 물리 수리에는 무효

다음 질문에 모두 “예”여야 정당하다.

1. **정확해가 경계를 위반할 수 없는가?**  
   예: 확률 \(0\le p\le1\), 로그 입력 \(u>0\). 단, 인구가 음수라는 이유로 사후에 양수 floor를 넣는 것은 해당하지 않는다. 음수는 솔버 실패 신호다.

2. **최종 고정점을 보존하는가?**  
   반복 스텝만 제한하고 수렴 후 clamp가 비활성이라면 가능하다. 최종 \(J_\nu,n_i,S_\nu,\chi_\nu\)를 직접 자르면 고정점을 바꾼다.

3. **쌍대항과 보존법칙을 함께 유지하는가?**  
   흡수와 방출, 상향·하향률, 이온화·재결합, 에너지·입자수를 한쪽만 clamp하면 무효다.

4. **오차상계가 있는가?**  
   제거된 냉각·에너지·복사율 합이 허용오차 이하임을 증명해야 한다. 발화율이 낮다는 사실만으로는 부족하다.

5. **정제 한계에서 사라지는가?**  
   패킷수·격자·정밀도를 높이면 clamp의 가중 효과가 0으로 가야 한다.

6. **문턱 민감도가 작은가?**  
   문턱을 10배 올리거나 내리거나 제거했을 때 핵심 결과가 허용오차 안에 있어야 한다. 최적 fit이 cap 경계에 붙으면 cap이 물리를 대신하고 있다는 신호다.

### 정당하거나 조건부로 정당한 실제 사례

- **RNG의 `log(0)` 방지:** [lumina_cuda.cu:3444](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:3444)에서 난수를 `1e-300` 이상으로 둔 뒤 로그를 취한다. \(u<10^{-300}\)인 측도만 바꾸며 물리 상태나 rate를 floor하지 않는다. 정당하다.

- **\(T_e\) 반복 스텝을 0.5–2배로 제한:** [lumina_plasma.c:8223](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:8223). 이는 업데이트만 제한하므로 충분히 반복해 같은 root에 도달한다면 정당한 damping이다. 단, 최대 반복수에 걸려 clamp된 상태를 “수렴해”로 채택하면 무효다. 현재는 직접 발화 카운터가 없다는 한계가 있다.

- **기여 상계에 의한 line cull:** [lumina_plasma.c:9786](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:9786), [lumina_plasma.c:9808](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:9808). 라인당 최대기여 상계가 `0.01*H_dep/N_line`보다 작을 때 제거하므로, 상계 가정이 맞다면 전체 누락을 약 1%로 제한한다. 이는 목표 오차가 1%보다 느슨하고 실제 누적 누락도 계측될 때만 정당하다.

- **NLTE matched pair의 `stim_corr→0`, `τ→1e-100`:** [lumina_cuda.cu:1791](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:1791), [lumina_cuda.cu:1798](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:1798). 같은 인구로 \(S_l\)도 0이 되어 흡수·방출이 함께 사라지는 비-maser 모델 범위라면 `1e-100`은 사실상 zero sentinel이다. 실제로 matched NLTE 이온에서는 `S_l==0`과 `τ==1e-100`가 일치했다: [ADVERSARIAL.md:98](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/clamp_census/ADVERSARIAL.md:98). 다만 maser가 물리적으로 중요할 수 있는 문제에는 정당하지 않다.

### 위험하거나 무효인 실제 사례

- **음수 NLTE 인구를 `1e-30`으로 교체:** [lumina_cuda.cu:1550](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:1550). 이후 재분배·재정규화·인구비에 들어가므로 작은 절대값이라고 무해하지 않다. 실제 스펙트럼 상태에서 exact `1e-30` 흔적이 **2,727개**, 최대 \(b_k=7652\)였다: [ADVERSARIAL.md:38](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/clamp_census/ADVERSARIAL.md:38). 현 런의 폭발 주범으로 입증된 것은 아니지만, 솔버 실패를 물리적 미량인구로 바꾸는 방식이므로 위험하다. 올바른 수단은 양수 제약/log-space solve와 잔차 검사다.

- **Ω floor:** 실측 collision strength를 넣은 뒤에도 `Υ<floor`면 올린다: [lumina_plasma.c:8016](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:8016), [lumina_plasma.c:8041](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:8041). census 구성에서는 2,584,132 전이 중 **2,278,264개, 88.16%**가 잘렸고 증폭배율은 중앙값 **253×**, p90 **9×10⁴**, p99 **2.5×10⁷**, 최대 **4.2×10⁹**였다: [ADVERSARIAL.md:80](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/cmfgen_toy06_19p48d/analysis/clamp_census/ADVERSARIAL.md:80). 이는 안전망이 아니라 충돌냉각 모델의 교체다. 다만 최신 `OMEGA_CMFGEN=1` 구성에서는 소스가 이 floor를 강제로 무효화한다: [lumina_plasma.c:7881](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:7881).

- **`eps_floor=1e-5`와 미등록 라인 `eps=1`:** [lumina_cmfgen.c:231](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:231). 상한 `eps≤1` 자체는 확률 제약이라 정당하지만, 아래쪽 floor는 약한 열화선을 강제로 흡수선으로 만들고, `el<0→1`은 테이블 부재를 완전 열화로 해석한다. 현재 `LINE_EPS_PHYS=1`인데 발화율과 에너지 가중 효과가 계측되지 않아 유효성을 입증할 수 없다.

- **\(J_\nu\)를 `factor×W B_\nu`로 자르는 cap/floor:** [lumina_plasma.c:4301](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:4301). 현재는 OFF지만, 켜면 계산된 복사장을 기준 Planck장으로 치환해 형광·photoexcitation을 직접 튜닝한다. shot-noise 처리라면 estimator variance/packet refinement로 정당화해야 하며, 기준 스펙트럼 적합 노브로 쓰는 것은 무효다.

- **상호작용 cap 뒤 packet energy 삭제:** 루프는 [lumina_cuda.cu:5341](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5341)에서 잘리고, `force_escape=0`이면 에너지를 `d_E_truncated`에 넣고 스펙트럼에서는 삭제한다: [lumina_cuda.cu:5867](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5867), [lumina_cuda.cu:5890](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:5890). 실측 삭제는 **343/100,000=0.343% packet**이지만 에너지 비율은 미측정이고, 깊이 갇힌 packet만 선택적으로 삭제하므로 편향이 무작위 0.343%가 아니다. 에너지 보존 계산에서는 무효다.

- **소스 미해결을 \(B(T_e)\)로 대체:** binned 경로는 NLTE 소스를 기본적으로 끄고 모든 참여 라인에 Planck source를 쓴다: [lumina_cmfgen.c:225](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:225). 이는 clamp라기보다 모델 대체다. 특히 네트워크 밖 이온에서 `S_l=0`은 “물리적으로 0”이 아니라 “미계산”인데, 소비자가 이를 열적 소스로 해석한다. 문턱 튜닝으로 해결할 문제가 아니다.

요약하면, **clamp는 표현영역·수렴경로·엄밀한 오차예산을 보호할 때만 도구이고, \(n_i,J_\nu,S_\nu,\chi_\nu,\Omega\) 같은 물리 해를 관측에 맞게 직접 자르는 순간 새로운 비문서화 물리모델이 된다.**