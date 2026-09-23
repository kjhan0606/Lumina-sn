# 판정 — DET-PATHB-XSHELL: 분기 XSHELL-A (2026-09-24)

판정 = **운전자**(Opus 5.5, 개정25). 계약 = `docs/PREREG_DET_PATHB_XSHELL_2026-09-22.md`(개정 1 = 66ea920 포함). 판정런 = **398939**(`/gpfs/kjhan/lumina/cmfgen_map/pathb_xshell_20260922/run2`, syn101 **a100 — A100-SXM4-80GB ×2**(footer·gpu_identity; astra 종료감사 정정), ITER=6, 47,833 s = 13h17m13s, COMPLETED·SUCCESS·model_rc=0, binary cuda_build9 `35c6dbc2…`). 계보 = 391837. 판독기 = `pathb_xshell_20260922/judge_xshell.py` **rev3, sha `8d5b7e6c…` — run2 증거 생성 전 고정**(`JUDGE_FROZEN.txt` 22:38, job RUNNING·evidence 부재 기록), 음성대조 9/9(`judge_nc/summary.json`), 산출 `judge_run2.json`(절대경로 호출 — §6). ⚠판독기는 run1 자료로 개발·시험됐으므로 이 고정은 **결과 비열람을 뜻하지 않는다**(run1 = run2 bit-동일); 「골대 이동 없음」의 근거는 **계약 분기식·기준이 불변**이라는 것이다(astra 종료감사). 상태 = **판정 / 종료감사 Fable·astra 통과(§8)**.

## 1. 분기 = **XSHELL-A**

§5 판정식: 전 게이트 PASS ∧ ΔΓ_out > εΓ ∧ Δx_24 < 0 — 충족(ΔΓ_out/εΓ = 3.46e5). εΓ 를 census 기준(|Γ_null−Γ_census6|)으로 읽어도 A(astra 사전 지적 반영, 판독기 rev3). 바이너리 자체 분기(XSHELL-A)와 판독기 독립 재판정 일치.

**「sh24 계수를 epoch 6(C6)에 고정하고 s≠24 계수만 epoch 7(C7)로 바꾼 shadow solve(생산 형식해 + 대각 ALI 8회, J_init6 warm-start)에서 sh24 S II Γ_phot 이 분해능보다 크게 증가했다 — 실측 셸외 계수 변화가 관측된 음의 투영부호(R_out<0)를 제공한다.」**

## 2. 게이트 (json 6/6 PASS · 판독기 재현은 G-CAP 의 payload 밖 조건을 제외한 부분 독립 — problems 0)

| 게이트 | 독립 재현 근거 | 결과 |
|---|---|---|
| G-ID | 바이너리 3-way sha · exports 델타 = `+LUMINA_XSHELL_RESPONSE=1` 하나 · 소스 매니페스트 대 391837 차이 = §3 변경집합 8파일 정확 · ITER 6·ALI 8·CHI 1·COUPLED_NEWTON 0·TRI off | PASS |
| G-RUN | footer SUCCESS·rc 0·outer 6 · 생산 solve 7·shadow 2 · 산출 4종(spectrum 포함) · production_state_untouched | PASS |
| G-CAP | payload sha·크기·헤더 · 셸외 변경 셀 재계수 60466 = json · (C^X 혼합 bit·J_init/edges 해시·캡처 계수·obsT/obsne 는 payload 밖 → json 의존, 개정 1 로 계약 등록) | PASS |
| G-NULL | J_null ≡ J_prod6 **bit-동일** · Γ_null = Γ_prod6 = Γ_census6 (rel 0.0) | PASS |
| G-RATE | census 6·7 대 391837 referent rel 5.2e-11·5.0e-11 · T_e 9231.3 · explicit=census(rel 0) · census CSV gamma_lag 교차 | PASS |
| G-DIR | Δx_24 = −24.2536 (census x_lag 차와 rel 6e-12) · 전 값 유한 | PASS |

## 3. 측정 (런에서만 얻은 수)

| 양 | 값 | 비고 |
|---|---|---|
| Γ_null (= Γ_prod6) | 2.82453e-2 s⁻¹ | referent 재현 |
| Γ_out | 4.31944e-2 s⁻¹ | |
| **ΔΓ_out** | **+1.49491e-2 s⁻¹** | Γ_prod6 의 **+52.9 %** |
| εΓ | 4.32e-8 s⁻¹ | = 1e-6·Γ_out (G-NULL bit-동일로 첫 항 0 — §5 기재) |
| R_out | −6.164e-4 s⁻¹ per unit x | 부호 정규화값 — s·g 와 단위 다름 |
| **q_out** | **1.4233** | 분모 Γ_prod7−Γ_prod6 = 1.0503e-2 |
| **s_out** = (ΔΓ_out/den_24)/Δx_lag | **−6.489** | den_24 = 9.498e-5 (ion_channels gen146, r 재구성 rel 4e-12) |
| s_7 / g_7 / s_7−g_7 (census) | −4.559 / 19.210 / −23.768 | LOCAL 판정(9e59ddf)과 동일 값 — 추출 교차검증 |
| sh24 J 변화 bin | 930/1234 | 상대 최대 bin 780·829·843·851·859 = 821·633·588·564·540 Å (최대 2.5×; bin→Å 은 저장 격자 규약 환산, payload 에 주파수 좌표 없음) |

**재현성:** run1(397672, G-CAP 실패로 미인증)과 **rates·projection·payload sha 전부 bit-동일**. 차이는 캡처 n_e 하나(14914481 → 14913816, 관측과 정렬). ⟹ G-CAP 수리는 게이트 캡처만 바꾸고 측정 경로를 건드리지 않았다(Fable·astra 재감사 주장의 실측 확인).

## 4. 해석 (두 알고리듬-물리 감사가 요구한 한정 포함)

1. **A 는 부호 문장이다.** G-NULL 이 bit-동일을 요구하므로 εΓ 는 1e-6·Γ 로 붕괴한다(Fable 사전감사 «수리 요»). 크기는 q_out·s_out 로만 읽는다.
2. **크기:** 셸외 계수 변화 단독으로 sh24 S II Γ 를 epoch 6 값의 +53 % 올린다. q_out = 1.42 > 1 — 이 응답이 같은 걸음의 생산 전체 변화(Γ_prod7−Γ_prod6)보다 크다. **전체 변화에서 이 응답을 뺀 잔차 = −42 %** 이며, 이는 뺄셈으로 정의한 잔차이지 초기장·자기계수·교차항을 각각 측정해 합한 값이 아니다. sh24 에서 J_init6 대 J_prod6 가 일부 bin 에서 최대 5.08× 다르므로(8회 절단이 수렴과 멂) 잔차에는 절단·warm-start 산물이 섞인다 — **물리적 상쇄로 읽지 않는다**(Fable·astra 종료감사 정정). 원인 분율이 아니다(§2.5).
3. **s 단위:** s_out = −6.49 로 **s_7(−4.56)과 같은 부호·1.42 배 크기**. s_out/s_7 = 1.42351 은 q_out(1.42331)과 rel 1.4e-4 차이(den 의 epoch 6→7 변화분)로 거의 같다 — **같은 ΔΓ_out 의 다른 정규화이며 독립 교차검증이 아니다**. s_7−g_7 = −23.77 대비 0.27, **가법 분율 아님**(유한 교란, 교차항 존재). 뜻하는 바: DRIVE·LOCAL 의 대각/국소 이득 g 는 명시적 셸외 계수 변화 응답을 포함하지 않으며(g 정의상 — 구성상 참), 그 경로는 이 걸음에서 |s_7| 보다 큰 크기를 낸다. s−g 의 원인 귀속도, 「생산 수송에 셸간 결합이 없다」도 뜻하지 않는다.
4. **어디서:** sh24 J 의 **상대 변화** top bin 은 821·633·588·564·540 Å. S II 문턱(NIST) = 바닥 4S° 531.26 Å · 2D° ≈576.8 Å · 2P° ≈610.9 Å ⟹ **821·633 Å 는 세 준위 어느 것도 광이온화하지 못한다**(상대 최대 bin 821 Å 은 Γ 무관); 문턱 안쪽은 588 Å(2P° 만)·564·540 Å(2D°·2P°) 뿐. |ΔJ| 절대 최대는 1500–2100 Å 로 역시 Γ 무관 — 「상대 변화 top bin」은 Γ 위치 지표로 부적절하다(초판 「540–633 Å 가 엣지 사이」는 오류, Fable·astra 정정). **Γ-가중 지배 대역은 측정하지 않았다.** 참고(측정 아님): Fable proxy(σ∝ν⁻³·LTE 3준위, T=9231 K)는 ΔΓ/Γ≈0.50(실측 0.53)이며 기여가 2D°·2P° 문턱 직단파장에 집중.
5. **연산자 한정:** 측정 연산자 = 생산 형식해 + **대각 ALI 8회 절단**, J_init6 warm-start. 고정 계수 아래 산란원천의 J 응답(`S=S_fixed+(χ_es/χ_tot)J_old`)과 형식해의 셸간 전달은 **포함**된다. 빠지는 것 = 물질 상태 변화(준위·n_e·T_e)에 따른 계수·S_fixed 재조립과 그 되먹임, 비대각 ALI(초판 「산란원천 재조립이 빠진다」 포괄 표현은 오류 — astra 정정). 방향 = epoch 6→7 실측 한 걸음(임의 방향 아님·대표 방향 검증 아님).

## 5. 주장 상한

말할 수 있는 것: 「391837 계보·ITER=6 생산 궤적의 epoch 6→7 걸음에서, sh24 계수를 고정하고 셸외 계수만 바꾼 8회 절단 대각 ALI 응답이 sh24 S II Γ_phot 을 +1.495e-2 s⁻¹(+53 %) 올렸고, 이는 음의 투영부호(R_out<0)이며 s 단위로 −6.49(s_7=−4.56, s_7−g_7=−23.77)다.」

말하지 않는 것: **−42 % 잔차의 물리적 구성**(절단·warm-start 산물과 분리 불가) · Γ 를 지배하는 파장 대역(미측정) · **비국소 결합이 필요/충분하다**(8회 절단·warm-start 연산자·한 걸음 실현치; 수렴 연산자 기준 판단 아님) · 셸간 야코비안의 형태·최근접 tri 충분성 · s−g 불일치의 원인 분율 · 이 방향이 대표 방향 · 준위·n_e 되먹임 포함 응답 · CMFGEN f_II 도달·고정점·스펙트럼. ⚠§1 Q-GOAL 첫 bullet(「분해능보다 크면 후속 선형화 근거」)을 이 A 로 충족했다고 옮기지 않는다 — εΓ 가 수치 잡음 수준이라(§4-1) A 단독은 투자 근거가 아니다; 근거가 되는 것은 크기(q_out 1.42·s_out −6.49)다.

## 6. 기재 (조용한 대장)

- **계약-코드 간극**(Fable 개정 1 (a) + astra 사전감사): G-RUN 코드는 관측 성공·spectrum 을 추가 요구 · `complete` 는 view 캡처 포함 8계수 · epoch 7 캡처값은 관측 일치 미검사 · §2.3 「solve 전 물질 캡처」 대 코드 R7 커밋 후 캡처(T_e 고정 프로파일이라 Γ 변환 입력 불변) · §5 「모두 미지」 과대(G-NULL·G-RATE PASS 시 Γ_null·εΓ 는 사실상 정해진 수) · json `new_clamp_floor_interp_count` 는 writer 리터럴 0(측정값 아님) · q_out 분모 census/explicit(이 런 bit-동일).
- **재런과 run1 의 bit-동일 (사후 확인):** run2 측정 산출은 run1 과 bit-동일했다. 그러나 run1 은 당시 계약으로 인증 불가(MODEL_FAILED·rc=1·G-CAP FAIL)였고, run1 수치를 본 뒤 계약을 고쳐 그 수치를 인증하는 것은 골대 이동이다 ⟹ **재런은 사전등록의 비용이지 낭비가 아니다**(Fable 종료감사; 초판 교훈 문장은 역전이라 철회. astra 는 「재런 불필요는 사전 증명 아님」으로 같은 방향). **공정한 교훈 = 계측 음성대조의 구멍:** selftest 가 관측값을 캡처+1 ULP(`nextafter`)로 모형화해, 캡처가 material iteration 한 단계 늦은 결함을 원리적으로 잡을 수 없었다. 이후 캡처 정합 NC 는 **다른 iteration 의 실제 값**을 주입해야 이빨이 있다.
- 판독기 rev1→rev3 은 전부 run2 증거 생성 전 수정·재고정(`JUDGE_FROZEN.txt`); 근거는 머리글 ⚠ 참조.
- 판독기 사용 조건: run root 를 **절대경로**로 줄 것 — 상대경로면 exports `<ROOT>` 정규화가 깨져 G-ID 거짓 FAIL(fail-closed, Fable 종료감사).
- 판독기는 일부 불일치를 `problems` 에만 남기므로 일반 검증기로 인증되지 않는다(이 런은 problems 0, astra).
- 중첩 CLI(Opus 5.5)·각 감사의 모델 자기보고는 제공 자료로 독립 확인되지 않는다(astra) — 호출 로그의 stderr 헤더·자기보고만 근거.

## 7. 분장 장부 (실제)

| 단계 | 실제 담당 |
|---|---|
| 기안 | sol (개정25-h) · 개정 1 = Fable (25-k) |
| 물리·대수 사전감사 | astra (간단 기안 검사, 25-g) |
| 구현 | sol (25-i) · G-CAP 캡처 정렬 = sol · §4 tol 최종화 = Opus 5.5 (`claude-opus-5-5`, 중첩 CLI) |
| 코드감사 | astra 3라운드(25-i) · G-CAP 수리 = Fable(«수리 요» 1) |
| 재감사 | Fable → astra (25-l) |
| 알고리듬-물리 정합 사전감사(종료감사 ② 선행) | Fable(잰다·수리 요 1) · astra(부분적으로 잰다·수리 요 3) |
| 판독기 작성 / 감사 | 운전자 / Fable(조건부 가·수리 요 4 반영) |
| 빌드·스테이징·제출·판정 | 운전자 (Opus 4.8 → Opus 5.5, 09-23 전환) |
| 종료감사 | Fable + astra (§8) |

## 8. 종료감사 (개정25-m, 두 축: ① 물리적 확고성 ② 알고리듬-물리 정합, 각 독립 fresh)

| 감사 | 모델 | «폐합 차단» | 재현 | 정정(1회차 전건 반영, 개정18③) |
|---|---|---|---|---|
| Fable | `claude-fable-5-1` | **0** · 폐합 가 | §3 전 수치·판독기 재실행 byte-동일·run1/run2 payload sha 동일 | §4-4 bin 목록 오류 · §4-2 −42 % 표현 · §4-3 항등 재표현 · §6 교훈 역전 · 판독기 절대경로 조건 |
| astra | `gpt-6-astra` | **0** · 폐합 가 | §3 전 수치·NC 9 재판독·소스 sha = run2 매니페스트 | 노드 h100→a100 · §2 제목 과장 · §4-5 산란원천 표현 · §4-4 NIST 문턱 · 골대 근거 · 모델 자기보고 미검증 |

두 감사가 독립으로 같은 곳(§4-4 bin 목록, §4-2 잔차, q_out 재표현)을 잡았고, 서로 다른 것(Fable: 교훈 역전·절대경로 / astra: 노드·산란원천·골대 근거)도 잡았다. 분기·게이트 판정치 변경 없음.
