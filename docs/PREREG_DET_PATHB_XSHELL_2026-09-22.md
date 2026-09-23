# 사전등록 — DET-PATHB-XSHELL: 실측 셸외 계수 변화에 대한 sh24 광이온화율 방향응답

**표적:** epoch 6 기준상태에서 sh24를 고정하고, 다른 셸의 조립 흡수·산란·방출 계수만 epoch 6→7 실측 방향으로 바꿨을 때 기존 형식해·대각 ALI가 만드는 sh24 `Γ_phot` 변화의 부호와 크기를 측정한다.

상태 = **기안**. 실행 승인·구현 완료·판정이 아니다.  
기안자 = **sol** (`gpt-5.6-sol`, Codex 계열; 개정25-h, user 2026-09-22).  
표적 선정 = **astra** (`gpt-6-astra`, user 2026-09-22 권위 명세).  
최종목적 = **MC 수송으로 CMFGEN 수준 SN Ia 스펙트럼 재현**.

## §0. 정정·선행 사실

1. **이 단은 솔버 수리가 아니라 측정이다.** 생산 인구 궤적에 새 이득을 넣지 않는다. `cmfgen_solve_J`의 기존 형식해·산란 소스·대각 ALI를 그대로 재호출하여 제한된 반사실 장만 계산한다.

2. **391837 산출물만으로 런 0회 평가는 불가능하다.** 확인 결과 `/gpfs/kjhan/lumina/cmfgen_map/pathb_local_20260921/run2/work`에는 `lumina_cmfgen_jnu.csv`, frozen χ/η dump, 재시작 상태가 없다. `resolved_lumina.exports`에도 `LUMINA_CMFGEN_JDUMP`와 `LUMINA_CMF_FROZEN_CHIETA_*`가 없었다. LOCAL은 J·조립 계수·ALI 초기장을 복원할 자료를 남기지 않았다.

3. **실측 기준점은 391837의 sh24 epoch-last 행이다.**

| 양 | epoch 6, gen 146 | epoch 7, gen 167 | 실측 변화 |
|---|---:|---:|---:|
| `x_lag` | 133.6492638 | 109.3956961 | `Δx_24 = −24.2535677` |
| `x_map` | 297.3683328 | 407.9299558 | +110.5616230 |
| `gamma_phot` | 0.02824526205 | 0.03874831249 | `ΔΓ_obs = +0.01050305044 s⁻¹` |
| `n_e` | 1.491381653×10⁷ | 1.491437645×10⁷ | 약 +0.0038% |

   온도는 두 행 모두 9231.3 K이고 비열 이온화 항은 0이다. 이 표는 새 런의 재현 referent이며, 셸외 기여의 부호를 미리 정하는 예측값이 아니다.

4. **방향응답은 유한 실측방향 응답이다.** 전체 셸외 계수 변화 `C_out^7−C_out^6`를 한 번에 적용한다. 무한소 편미분, 전체 야코비안, 보편적인 음의 대각 도함수로 부르지 않는다.

5. **연산자 형태를 발명하지 않는다.** tri 절단, N×N 결합, `coupled_newton_solve_all`, 새 Λ 연산자, 준위 권위 통합을 넣지 않는다. 결과 부호만 미지이며, 사용 연산자는 기존 `cmfgen_solve_J`로 고정한다.

6. **astra 코드감사 #6 수리는 selftest의 생산 함수 직접 호출을 요구했다.** 이에 `cmfgen_xshell_production_selftest` 테스트 진입점 선언, `selftest_a2_10_radeq` selftest 빌드 타깃, `scripts/run_a2_10_selftest.py` 러너를 변경집합에 포함하여 5파일에서 8파일로 확장한다. 이 셋은 selftest 빌드 전용이다. 판정런 생산 GPU 바이너리 `lumina_cuda`는 `cuda_build7`의 직접 `nvcc` 명령으로 빌드되어 `Makefile`을 소비하지 않으므로 생산 nvcc 바이너리에 접촉하지 않는다. 이 확장은 user가 2026-09-22 승인했다.

## §1. 물리 계약 1개

**계약 질문:**  
**“391837 계보를 재현한 동일 생산 궤적에서 epoch 6의 조립 상태와 초기 장을 기준으로 삼고, sh24의 조립 계수·T·`n_e`·LTE 준위 가중치를 epoch 6 값에 고정한 채 모든 `s≠24` 셸의 조립 수송 계수만 epoch 7 값으로 교체하면, 기존 binned 형식해와 대각 ALI가 산출하는 sh24 S II 광이온화율은 `Δx_lag,24<0` 방향에 대해 음의 투영응답을 보이는가?”**

판정런은 하나다. 정상 생산 6 iterations를 먼저 그대로 수행하고, 그 안에서 보존한 epoch 6·7 상태로 다음 두 shadow 평가만 추가한다.

- `NULL6`: epoch 6 계수 + epoch 6 ALI 초기 장. epoch 6 생산 장의 재현 대조.
- `XSHELL`: `s=24`는 epoch 6 계수, `s≠24`는 epoch 7 계수 + `NULL6`과 동일한 epoch 6 ALI 초기 장·동일 경계·동일 ALI 반복수.

shadow 장은 생산 `NLTEConfig`나 커밋 뷰에 쓰지 않는다. `XSHELL`은 기존 커밋 직전 `cs.J`와 같은 배열·격자의 반사실 장이며, 정확히 같은 canonical 변환과 LTE 가중치로 `Γ_phot`만 읽는다.

### 말하지 않는 것

- “최근접 tri면 충분하다.”
- “sh24 자신의 이온화 증가에 대한 보편적인 음의 편미분이다.”
- “셸간 결합이 필요하다” 또는 “셸간 결합이 원인이다.”
- “전체 야코비안의 한 열 또는 대각을 측정했다.”
- “CMFGEN의 중간 셸 `f_II`에 도달한다.”
- “안정된 고정점 또는 최종 MC 스펙트럼이 보장된다.”
- 셸별 기여의 가법 분해, 특정 인접 셸의 원인 귀속.
- 준위 solve·`n_e` solve·인구 권위의 정합 또는 수정.

### Q-GOAL — 기안자 자기답

**결론 = PASS.**

이 단은 최종목적의 중간 병목인 중간 셸 S II/III 오차를 직접 고치지는 않는다. 대신 LOCAL에서 확인된 sh24의 잘못된 총감도 부호가 실제 셸외 수송 변화에서 공급되는지 측정한다. 이를 생략하면 근거 없이 tri/N×N 또는 이중 인구 권위를 구현하거나, 실제 원인이 아닌 셸외 채널에 계산비용을 쓸 수 있다.

산출은 후속 선형화 투자 여부를 결정한다.

- 음의 셸외 기여가 분해능보다 크면, 실측 수송방향을 보존하는 후속 선형화 후보를 설계할 근거가 생긴다.
- 미소하거나 반대 부호이면, 이 표적을 우선 원인에서 내리고 준위·`n_e`·산란/ALI 초기화 등 다른 장 생성 원인을 검토한다.

### Q-LEAN — 기안자 자기답

**결론 = APPROPRIATE.**

필수 측정은 다섯 개뿐이다.

1. `Γ_6`: epoch 6 생산 장의 sh24 S II `Γ_phot`.
2. `Γ_null`: epoch 6 상태 shadow 재현값.
3. `Γ_out`: 셸외 혼합 상태의 sh24 S II `Γ_phot`.
4. `ΔΓ_out = Γ_out−Γ_null`.
5. `R_out = ΔΓ_out/Δx_24` 및 크기비 `q_out=ΔΓ_out/ΔΓ_obs`.

소비자는 각각 다음과 같다.

- `Γ_null−Γ_6`: 같은 초기 장·같은 ALI 반복수를 실제로 재현했는지 판정한다.
- `ΔΓ_out`: 셸외 계수 변화의 방향응답을 판정한다.
- `R_out`: 관측된 `Δx_lag,24` 방향에 투영한 부호를 판정한다.
- `q_out`: 관측된 전체 `Γ_phot` 변화와 비교한 유한 기여 크기를 기재한다.

삭제한 계측:

- 전체 N×N 응답행렬.
- 셸별 one-at-a-time 49회 평가.
- 최근접 tri 스캔.
- 준위·`n_e` 재풀이.
- 스펙트럼 재작성.
- 새 J 진단 CSV 전건 출력.
- `κ`, 새 뉴턴 이득, 새 착지 게이트.

추가 수송 평가는 `NULL6`과 `XSHELL` 두 번뿐이다. `NULL6`은 유한 ALI 초기장 혼입을 검출하는 필수 음성대조라 삭제하지 않는다.

## §2. 기전·측정 정의

### 2.1 런 0회로 확정된 것

- `cmfgen_solve_J`는 조립된 `chi_tot`, `chi_es`, `S_fixed`, 초기 `J`, 내부 경계 `T_inner`, ALI 반복수를 소비한다.
- 대각 ALI에서 각 반복의 소스는  
  `S = S_fixed + (chi_es/chi_tot) J_old`이다.
- `formal_solve_bin`의 형식해와 대각 `lambda_star`가 그대로 사용된다.
- LOCAL 생산 경로에서 `LUMINA_CMFGEN_LAMBDA_TRI`는 비활성이고 `LUMINA_CMFGEN_ALI_ITER=8`이다.
- `Γ_phot` 생산 변환은 canonical 방사장, 생산 주파수 격자, 레벨별 BF 단면적과 LTE 준위 가중치를 사용한다.
- epoch는 `radeq_set_line_re_source` 호출 세대와 대응하며 epoch 6·7은 두 연속 조립/장 세대다.

### 2.2 조립 계수 상태

수송이 실제 소비하는 셸별 상태를 다음 튜플로 정의한다.

\[
C_s^{(k)}(b)=
\left[
  \chi_{\rm tot,sb}^{(k)},
  \chi_{\rm es,sb}^{(k)},
  S_{\rm fixed,sb}^{(k)}
\right].
\]

감사 표기용으로만 다음을 함께 기록한다.

\[
\chi_{\rm abs,eff}=\chi_{\rm tot}-\chi_{\rm es}, \qquad
\eta_{\rm fixed}=\chi_{\rm tot}S_{\rm fixed}.
\]

혼합 상태는 보간이나 재조립 없이 원시 배열을 셸 단위로 선택한다.

\[
C_s^{X} =
\begin{cases}
C_s^{(6)}, & s=24,\\
C_s^{(7)}, & s\ne24.
\end{cases}
\]

새 floor·clamp·정규화·선형 보간을 넣지 않는다. 기존 형식해 내부의 기존 수치 분기만 유지한다.

### 2.3 고정 상태와 두 평가

epoch 6 조립 직후, 생산 `cmfgen_solve_J` 호출 전에 다음을 복제한다.

- `C^(6)`.
- 생산 ALI 초기 장 `J_init^(6)`.
- 주파수 격자와 기하.
- `T_inner`와 경계 설정.
- sh24 `T_e`, `n_e`.
- sh24 S II의 생산 LTE 준위 가중치 또는 그것을 재현하는 고정 입력·해시.

epoch 6 생산 solve가 끝나면 `J_prod^(6)`을 보존한다. epoch 7 조립 후 `C^(7)`을 보존한다.

그 뒤 생산 배열과 분리된 shadow `CMFGENState`에서 실행한다.

\[
J_{\rm null} =
 {\cal T}_{8\,{\rm ALI}}\!\left(C^{(6)},J_{\rm init}^{(6)},BC^{(6)}\right),
\]

\[
J_{\rm out} =
 {\cal T}_{8\,{\rm ALI}}\!\left(C^{X},J_{\rm init}^{(6)},BC^{(6)}\right).
\]

두 평가의 초기 장·기하·내부 경계·ALI 반복수는 bit-동일해야 한다. 조기 종료 조건을 새로 넣지 않는다.

### 2.4 `Γ_phot` 변환

sh24 S II의 `Γ`는 새 근사 적분으로 계산하지 않는다. `parity_gamma_phot_checked`와 `nlte_bf_gamma_canonical`이 쓰는 생산 계산을 공용 helper로 인수분해하거나, 그 계산을 호출하는 명시적-field adapter를 둔다.

요구사항:

- 생산과 동일한 `nu`, bin contract, canonical BF 적분.
- 생산과 동일한 CMFGEN BF 단면적 선택 및 fallback 정책.
- epoch 6의 고정 `T_e`와 LTE 준위 가중치.
- 두 shadow 장에 동일한 가중치 해시.
- blocked/unsampled/out-of-grid 레벨이 하나라도 생기면 fail-closed.
- `Γ(J_prod^6)`가 생산 census의 epoch 6 `gamma_phot`과 맞아야 한다.
- 같은 검산을 epoch 7 생산 장에도 적용한다.

### 2.5 투영과 크기

\[
\Delta\Gamma_{\rm out}=\Gamma_{\rm out}-\Gamma_{\rm null},
\]

\[
R_{\rm out} =
\frac{\Delta\Gamma_{\rm out}}
     {x_{\rm lag,24}^{(7)}-x_{\rm lag,24}^{(6)}}.
\]

`Δx_24<0`이므로 `ΔΓ_out>0`이면 `R_out<0`이다. 이때만 “실측 셸외 변화가 관측된 음의 투영부호를 제공한다”고 말한다.

크기 비교는

\[
q_{\rm out} =
\frac{\Delta\Gamma_{\rm out}}
     {\Gamma_{\rm prod}^{(7)}-\Gamma_{\rm prod}^{(6)}}
\]

로 기록한다. 이는 비선형 유한 교란의 비교비이며 가법적인 원인 분율이 아니다. `q_out`은 0–1 범위로 clamp하지 않는다.

## §3. 변경집합과 산출물

### 변경집합 — DET-PATHB-XSHELL (표)

| 경로 | 분류 | 허용 변경 |
|---|---|---|
| `src/env_universe.h` | 계측 게이트 | `LUMINA_XSHELL_RESPONSE`를 strict env universe에 등록 |
| `src/lumina.h` | 측정 API | 명시적 canonical J에서 기존 생산 방식의 S II `Γ_phot`을 계산하는 좁은 read-only helper 선언 |
| `src/lumina_plasma.c` | Γ 변환 계측 | 기존 `parity_gamma_phot_checked`/`nlte_bf_gamma_canonical` 계산을 재사용하는 explicit-field adapter; 생산 호출의 산술·정책 불변 |
| `src/lumina_cmfgen.c` | 라이브 오케스트레이션 | `cmfgen_run`에서 epoch 6·7 상태 캡처, `NULL6`·`XSHELL` shadow solve, 요약·증거 writer; 생산 `cs`·commit 경로 무접촉 |
| `tests/a2_10_radeq_selftest.c` | 직접시험·음성대조 | 상태 선택, 동일 초기장, sh24 freeze, canonical Γ 변환, fault-injection teeth |
| `Makefile` | selftest 빌드 | `selftest_a2_10_radeq` 새 타깃만 추가; 생산 nvcc 빌드·기존 타깃 무변경 |
| `src/lumina_cmfgen.h` | 테스트 API | `cmfgen_xshell_production_selftest` 테스트 진입점 선언만; **CMFGENState 필드 추가 금지 유지** |
| `scripts/run_a2_10_selftest.py` | selftest 러너 | 신규, selftest 바이너리 실행·baseline+negative 검증 |

### 변경집합 끝

`expected_extra`는 위 8파일과 정확히 같다.

변경 금지:

- `src/lumina_cuda.cu`.
- 새 `.cu`.
- `Makefile`: 생산 nvcc 빌드 명령·기존 타깃 무변경(`selftest_a2_10_radeq` selftest 타깃만 추가 허용).
- `src/lumina_cmfgen.h`: `CMFGENState` 필드·생산 시그니처 무변경(`cmfgen_xshell_production_selftest` 테스트 진입점 선언만 허용).
- `formal_solve_bin` 전달식.
- 대각 ALI 갱신식.
- `cmfgen_commit_jnu`.
- 인구 solve·`n_e` solve·준위 solve.
- `pop_djdn_log_newton`.
- `coupled_newton_solve_all`.
- tri 저장·활성화.
- 기존 χ/η 조립식.

### 신규 증거

- `lumina_xshell_response.bin`: `C6`, `C7`, `J_init6`, `J_prod6`, `J_null`, `J_out`의 재현용 binary64 자료.
- `lumina_xshell_response.json`: 차원·세대·배열 해시·고정값·`Γ`·`ΔΓ`·`R_out`·`q_out`·게이트 결과.
- 기존 `lumina_pop_djdn.csv`와 `lumina_ion_channels.csv`: 생산 궤적 referent.

full J CSV, 셸별 49개 파일, 스펙트럼 사본은 만들지 않는다.

## §4. 게이트와 음성대조

수치 항등 허용오차는 **상대오차 `1e-6`**로 고정한다. 이는 직전 DRIVE/LOCAL census의 실제 십진 출력 재계산 그레인보다 여유가 있는 기존 판정 허용오차다. `1e-9`는 사용하지 않는다. 배열 동등성·해시는 산술 허용오차가 아니라 bit-동일로 판정한다.

### G-ID — 계보·실행 동일성

전건 필요:

- base 계보 = 391837의 봉인 입력·물리 exports.
- `ITER=6`, `LUMINA_CMFGEN_ALI_ITER=8`.
- `LUMINA_POP_DJDN_CHI=1` 유지.
- `LUMINA_CMFGEN_LAMBDA_TRI`, JDUMP, FROZEN_CHIETA, NT, FDHOLD 비활성.
- `LUMINA_COUPLED_NEWTON=0`.
- exports 델타 = `+LUMINA_XSHELL_RESPONSE=1` 하나.
- binary·source manifest·runner의 세 신원 일치; `source.sha256`의 `non-OK=changeset` 집합은 §3 변경집합 8파일과 정확히 일치.
- runner는 live `cmfgen_run`을 실행하며 `lumina_cuda.cu` 레거시 루프를 계측점으로 사용하지 않음.

### G-RUN — 정상 생산 완주

전건 필요:

- `terminal_state=SUCCESS`, `model_rc=0`.
- 정상 생산 outer iterations=6.
- 생산 `cmfgen_solve_J` 세대 7개와 shadow 평가 2개가 구분돼 기록됨.
- 기존 census·ion channels·최종 publication 실재.
- NaN/Inf, allocation fallback, shadow→production write 검출 0.

### G-CAP — 실측방향 캡처

전건 필요:

- epoch 6·7 계수 세대가 각각 정확히 한 번 캡처됨.
- `C6`, `C7` 차원·격자·기하 동일.
- `s≠24`에서 `C7−C6`의 비영 원소가 적어도 하나 존재.
- `XSHELL`의 sh24 튜플은 `C6` sh24와 bit-동일.
- `XSHELL`의 모든 `s≠24` 튜플은 `C7` 해당 셸과 bit-동일.
- `NULL6`과 `XSHELL`의 `J_init`, 경계, `T_inner`, ALI 반복수 해시 동일.
- 새 clamp·floor·보간 적용 건수 0.
- epoch 6·7 물질 상태(sh24 `T_e`, `n_e`, S II partition function)가 R7 물질 커밋 직후 각각 정확히 한 번 캡처됨(`xshell_capture_material`, epoch 당 `material_count==1`; 0회 또는 2회 이상이면 FAIL).
- 캡처된 epoch 6 `T_e`, `n_e`, partition 이 모두 유한이고 partition `>0`.
- 관측 `parity_gamma_phot_observed`(epoch 6·7, sh24, S II)가 둘 다 성공.
- 관측 epoch 6 `T_e`·`n_e` 와 캡처 epoch 6 `T_e`·`n_e` 가 각각 §4 표준 상대오차 `≤1e-6`(`XSHELL_REL_TOL`) 이내. 이 두 항은 산술 허용오차 비교이며 bit-동일을 요구하지 않는다(§4 규정: 배열·해시만 bit-동일).
- 위 항 중 하나라도 실패하면 stderr 에 `[XSHELL][G-CAP-DETAIL]` 한 줄(각 conjunct 값·관측/캡처 원시값·허용오차)을 남긴다.

### G-NULL — 유한 ALI 초기장 음성대조

전건 필요:

- `J_null`과 저장된 `J_prod6`이 bit-동일.
- `Γ_null`과 `Γ_prod6`이 상대오차 `≤1e-6`.
- `NULL6` 계수에 한 비트를 의도적으로 바꾸는 selftest가 hash/장 재현 게이트를 FAIL.
- `XSHELL`의 초기 장을 epoch 7 장으로 바꾸는 selftest가 초기장 동일성 게이트를 FAIL.

이 게이트가 실패하면 셸외 효과와 유한 ALI 초기장 효과를 분리하지 못했으므로 결과 부호를 읽지 않는다.

### G-RATE — 생산 `Γ_phot` 변환 동일성

391837 referent:

- epoch 6: `Γ=2.824526205e-2 s⁻¹`.
- epoch 7: `Γ=3.874831249e-2 s⁻¹`.
- sh24 `T_e=9231.3 K`.

새 런의 생산 census가 위 값과 상대오차 `≤1e-6`로 재현되고, 명시적-field helper가 새 런 자체의 epoch 6·7 생산 census와 각각 상대오차 `≤1e-6`로 일치해야 한다. grid·sigma 선택·LTE weight 해시도 기록한다.

### G-DIR — 투영 가능성

전건 필요:

- epoch-last `Δx_24`가 유한하고 음수.
- `Γ_out`, `Γ_null`, `ΔΓ_out`, `R_out`, `q_out`가 모두 유한.
- 분해능  
  `εΓ = max(|Γ_null−Γ_prod6|, 1e-6·max(|Γ_null|,|Γ_out|))`
  를 원시 값으로 기록.
- `q_out`을 0–1로 제한하거나 부호를 교정하지 않음.

### 게이트 음성대조

| 음성대조 | 주입 | 반드시 실패할 게이트 |
|---|---|---|
| NC-NULL | `NULL6`의 계수 한 값을 변조 | G-NULL |
| NC-INIT | `XSHELL`에 epoch 7 초기 J 사용 | G-CAP 또는 G-NULL |
| NC-SH24 | 혼합 상태의 sh24 계수를 epoch 7로 교체 | G-CAP |
| NC-RATE | LTE 가중치 하나를 변조 | G-RATE |
| NC-DEAD | 게이트 ON이나 shadow solve 호출 생략 | G-RUN/G-DIR |
| NC-OFF | `LUMINA_XSHELL_RESPONSE=0` | 신규 산출 0, 생산 경로·기존 산출 byte-동일 |

## §5. 기대치와 분기

### 런 없이 얻을 수 없는 수

다음은 모두 **미지이며 부호를 예측하지 않는다.**

- `Γ_null`.
- `Γ_out`.
- `ΔΓ_out`.
- `R_out`.
- `q_out`.
- `εΓ`.
- 어떤 주파수 bin이 응답 크기를 지배하는지.
- 셸외 실측 방향이 음의 투영부호를 제공하는지.

알려진 값은 391837의 재현 referent뿐이며, 예상 결과로 사용하지 않는다.

### 분기 표

게이트 실패가 하나라도 있으면 먼저 `XSHELL-X`다.

| 분기 | 배타적 판정식 | 뜻 | 다음 결정 |
|---|---|---|---|
| **XSHELL-A** | 전 게이트 PASS ∧ `ΔΓ_out > εΓ` ∧ `Δx_24<0` | 분해 가능한 `R_out<0`; 실측 셸외 변화가 관측된 음의 투영부호를 제공 | 실측 방향을 보존하는 셸간 선형화의 최소 표현을 별도 계약으로 설계. 최근접 tri 충분성은 미판정 |
| **XSHELL-B** | 전 게이트 PASS ∧ `ΔΓ_out ≤ εΓ` | 셸외 기여가 분해능 이하거나 `R_out≥0`인 반대 방향 | 이 표적을 우선 원인에서 내림. 준위·`n_e`·산란/ALI 등 다른 장 생성 원인을 다음 표적으로 비교 |
| **XSHELL-X** | 하나 이상의 게이트 FAIL 또는 필수 값 부재 | 미시연 | 물리 결론 없음. 실패한 계측·오케스트레이션만 수리하고 같은 계약을 재현 |

`XSHELL-B` 안에서는 판정문에 `미소`, `반대 부호`를 수치로 구분해 기재하되 새 분기를 만들지 않는다.

## §6. 결정·자원·보존

- 판정런 = **1회**.
- base = 391837 입력·물리 설정·6 iterations.
- 추가 수송 = 동일 대각 ALI의 shadow 평가 2회(`NULL6`, `XSHELL`).
- 자원 상한 = 1 node, GPU 2개, CPU 16개, memory 64 GiB, walltime 20시간.
- 허용 파티션 = `h100,a100`; H200 제외 상태를 유지한다.
- 정상 생산 궤적이 먼저 완료되지 않으면 shadow 결과를 판정에 쓰지 않는다.
- `NULL6`/`XSHELL`은 생산 `cs`, canonical production view, 인구, `n_e`, 준위 배열을 변경하지 않는다.
- 기존 391837과 새 런은 보존한다. 기존 산출물을 덮어쓰지 않고 새 run root를 사용한다.
- 실패 시 신규 run root를 재사용하지 않는다.
- 산출 크기 상한과 `ulimit -f`를 runner에 명시한다.
- XSHELL-A도 구현 승인이 아니다. 후속 선형화의 형태·비용은 별도 계약과 사용자 결정 대상이다.

## §7. 분장 장부

| 단계 | 규약상 담당 | 실제·상태 |
|---|---|---|
| 표적 선정 | astra | **완료** — user가 제공한 `gpt-6-astra` 권위 명세 |
| 단계 계획·계약 기안 | sol, 개정25-h | **완료** — `gpt-5.6-sol`, 이 본문 |
| 기계 프리플라이트 | 운전자 | 미집행 |
| 경량 사전감사 Q-GOAL·Q-LEAN | Opus 4.8 fresh | 미집행 |
| 구현·직접 selftest | Opus 4.8 | 미집행 |
| 코드 간단 감사 | Fable fresh | 미집행 |
| 봉인·빌드·스테이징·제출·계측·판독 | 운전자 Claude | 미집행 |
| 판정 | 운전자 Claude | 미집행 |
| 종료감사 | Opus 4.8 fresh, 운전자 필요 판단 시 | 미집행 |

## §8. 주장 상한

이 단이 말할 수 있는 최대치는 다음뿐이다.

> “391837 계보의 epoch 6 기준에서 sh24의 조립 계수·물질 상태·LTE 가중치·초기 장·경계를 고정하고, 다른 모든 셸의 조립 수송 계수만 epoch 6→7 실측 방향으로 교체했을 때, 기존 형식해와 대각 ALI가 산출한 sh24 S II `Γ_phot`의 유한 방향응답은 `ΔΓ_out`, 투영값은 `R_out`, 관측 변화 대비 크기는 `q_out`이었다.”

XSHELL-A라면 “그 실측 셸외 방향이 음의 투영부호를 제공했다”까지 말한다.

다음으로 확장하지 않는다.

- 최근접 tri 충분성.
- 전체 또는 희소 야코비안의 형태.
- 보편적인 음의 대각 도함수.
- 셸외 결합의 필요성·유일 원인성.
- 특정 셸의 원인 귀속.
- CMFGEN `f_II` 도달.
- 고정점 안정성.
- MC 스펙트럼 재현.

## §9. 집행 기록 뼈대

1. **기안**
   - 모델: `gpt-5.6-sol`
   - 날짜: 2026-09-22
   - 방식: read-only stdout
   - 파일화: 미실행

2. **저자 확인**
   - 공유 컨텍스트·`CLAUDE.md`·AGENTS 지시 확인: 완료
   - 요청된 배경문서 4건 확인: 완료
   - 391837 J/frozen/restart 산출 부재 확인: 완료
   - repo 실존 경로 확인: 완료

3. **기계 프리플라이트**
   - `check_prereg_preflight`: 미집행
   - `check_gate_referents`: 미집행
   - `check_doc_canonical`: 미집행
   - `check_dispatch_preflight`: 미집행

4. **경량 사전감사**
   - 모델 ID:
   - Q-GOAL:
   - Q-LEAN:
   - 폐합 차단:
   - 처분:

5. **구현·직접시험**
   - 구현 모델 ID:
   - 변경집합 실제:
   - 변경집합 외 접촉:
   - selftest:
   - NC 시연:
   - 생산 OFF byte 동일성:

6. **코드 간단 감사**
   - 감사 모델 ID:
   - 첫 판정 — 봉인 접촉 가부:
   - 기존 형식해·대각 ALI 무변경:
   - shadow→production write 0:
   - canonical Γ 변환 재사용:
   - 차단 및 수리:

7. **봉인·빌드·스테이징**
   - source manifest:
   - binary SHA-256:
   - runner SHA-256:
   - run-build identity:
   - exports 델타:
   - ITER/ALI 상수:
   - 새 run root:

8. **판정런**
   - Slurm job:
   - node/partition/GPU:
   - 시작·종료:
   - terminal state:
   - elapsed/MaxRSS:
   - 생산·shadow solve 수:

9. **판독**
   - G-ID:
   - G-RUN:
   - G-CAP:
   - G-NULL:
   - G-RATE:
   - G-DIR:
   - `Γ_prod6`, `Γ_null`, `Γ_out`:
   - `ΔΓ_out`, `εΓ`, `R_out`, `q_out`:
   - 분기:

10. **판정·종료감사**
    - 운전자 판정:
    - 주장 상한 준수:
    - 종료감사 모델 ID:
    - 재현 상태:
    - 최종 처분:

11. **개정 1 — §G-CAP 캡처↔관측 열역학 conjunct 사후 등록**
    - 기안: Fable fresh(`claude-fable-5-1`, 개정25-k), 2026-09-23, read-only 본문 반환.
    - 성격: **이미 가동 중인 코드의 문서화**다. 추가된 conjunct 는 첫 빌드부터 `g_cap` 논리곱에 있었고 계약 본문에만 없었다. 게이트 판정식·분기 표·측정식(`ΔΓ_out`, `R_out`, `q_out`, `εΓ`)·허용오차(§4 `1e-6`)는 바꾸지 않는다.
    - run1(job 397672) 실패 경위: 바로 이 conjunct(`cap_obsne`, epoch 6 `n_e`)에서 G-CAP FAIL → XSHELL-X. 원인 = 물질 캡처가 관측(`parity_gamma_phot_observed`)보다 material iteration 한 단계 앞선 상태를 읽음(관측 대 캡처 상대차 4.46e-5).
    - 수리 요지: ① 캡처 호출을 R7 물질 커밋 직후(stamp 검사 뒤)로 정렬해 관측과 같은 공개 상태를 읽게 함 ② 비교를 원래의 byte-동일 memcmp 에서 1e-9 를 거쳐 §4 표준 상대오차 1e-6(`XSHELL_REL_TOL`)으로 정착. 수리 전 4.46e-5 는 1e-6 에서도 FAIL 이므로 허용오차 변경이 run1 실패를 통과로 바꾸는 것은 아니다.
    - 감사 이력: Fable 코드감사 → Fable 재감사 → astra 재감사 통과.
    - **적용 시점: 재런 job 398939 가동 후**의 사후 개정이다. 재런 결과를 보고 쓴 것이 아니라 가동 중 문서화이나, 판정문은 이 순서를 그대로 기재한다.
    - 프리플라이트(`check_prereg_preflight`·`check_gate_referents`·`check_doc_canonical`): 운전자 집행, 미집행.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-PATHB-XSHELL (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:(?:src|tests)/[A-Za-z0-9_/.]+\\.(?:c|h)|Makefile|scripts/[A-Za-z0-9_/.]+\\.py)",
    "symbol": "LUMINA_XSHELL_RESPONSE",
    "roots": ["src", "tests"],
    "expected_extra": [
      "src/env_universe.h",
      "src/lumina.h",
      "src/lumina_cmfgen.c",
      "src/lumina_plasma.c",
      "tests/a2_10_radeq_selftest.c",
      "Makefile",
      "src/lumina_cmfgen.h",
      "scripts/run_a2_10_selftest.py"
    ]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
    "metrics": {
      "count": "len(v)",
      "gate_fail": "sum(1 for x in v if x < 0.5)",
      "negative": "sum(1 for x in v if 0.5 <= x < 1.5)",
      "small_or_nonnegative": "sum(1 for x in v if x >= 1.5)"
    },
    "rules": [
      {
        "name": "XSHELL-X",
        "predicate": "count == 0 or gate_fail > 0"
      },
      {
        "name": "XSHELL-A",
        "predicate": "count > 0 and gate_fail == 0 and negative == count"
      },
      {
        "name": "XSHELL-B",
        "predicate": "count > 0 and gate_fail == 0 and small_or_nonnegative == count"
      }
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []},
      {"name": "gate-fail", "mix": [[0.0, 1], [1.0, 1]]},
      {"name": "negative-response", "mix": [[1.0, 1]]},
      {"name": "small-or-nonnegative", "mix": [[2.0, 1]]}
    ],
    "residual": "XSHELL-X"
  },
  "references": [
    {
      "path": "docs/VERDICT_DET_PATHB_LOCAL_2026-09-22.md"
    },
    {
      "path": "docs/VERDICT_DET_DJDN_DRIVE_2026-09-21.md"
    },
    {
      "path": "docs/DESIGN_PATHB_LINEARIZATION_2026-09-19.md"
    },
    {
      "path": "docs/PREREG_DET_PATHB_LOCAL_2026-09-21.md"
    },
    {
      "path": "CLAUDE.md"
    },
    {
      "path": "Makefile"
    },
    {
      "path": "src/lumina_cmfgen.h"
    },
    {
      "path": "scripts/run_a2_10_selftest.py"
    },
    {
      "path": "src/lumina_cmfgen.c",
      "flags_existing": [
        "cmfgen_solve_J",
        "formal_solve_bin",
        "cmfgen_commit_jnu",
        "cmfgen_run"
      ],
      "flags_planned": [
        "LUMINA_XSHELL_RESPONSE",
        "xshell_capture_epoch",
        "xshell_shadow_solve",
        "xshell_write_evidence"
      ]
    },
    {
      "path": "src/lumina_plasma.c",
      "flags_existing": [
        "nlte_bf_gamma_canonical",
        "parity_gamma_phot_checked",
        "radeq_set_line_re_source"
      ],
      "flags_planned": [
        "parity_gamma_phot_explicit_field"
      ]
    },
    {
      "path": "src/lumina.h",
      "flags_existing": [
        "radeq_set_line_re_source"
      ],
      "flags_planned": [
        "parity_gamma_phot_explicit_field"
      ]
    },
    {
      "path": "src/env_universe.h",
      "flags_existing": [
        "LUMINA_CMFGEN_JDUMP",
        "LUMINA_CMFGEN_LAMBDA_TRI",
        "LUMINA_COUPLED_NEWTON"
      ],
      "flags_planned": [
        "LUMINA_XSHELL_RESPONSE"
      ]
    },
    {
      "path": "tests/a2_10_radeq_selftest.c",
      "flags_existing": [
        "int main"
      ],
      "flags_planned": [
        "LUMINA_XSHELL_RESPONSE",
        "XSHELL_NEG_NULL",
        "XSHELL_NEG_INIT",
        "XSHELL_NEG_SH24",
        "XSHELL_NEG_RATE"
      ]
    }
  ]
}
```
