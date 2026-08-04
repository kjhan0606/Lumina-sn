# Wave 3 갈림길 해석 — Fable 자문 (2026-07-31)

읽기 전용 오프라인 분석. 소스 수정·런·git 없음. 모든 수치는 기존 산출물에서 독립 재계산했다.

## 0. 한 줄 판정

**Wave 3의 "무개선/악화"는 element-wide 구조의 실패가 아니라, 구조 수리가 성공해서 가림막이 벗겨진 결과다.** 행렬은 자기 입력에 정확히 평형한다(플럭스 폐합 1.0000 실측). 문제는 입력이다: 동결된 이온화/재결합 유효 균형이 **모든 링크·모든 셀에서 +0.6~+3.2 dex 과이온화**돼 있고, pair-wise 레인은 Saha 폐쇄·pin이 이를 가려 왔다. s0의 D −58%는 D-5 구조 수리의 실물 증거이고, s0 Fe IV "악화"는 stage-V 창 절단의 산술 아티팩트로 **정량 해명된다**(과충전 0.0111 vs CMFGEN V+ 몫 0.0107).

## 1. 독립 재계산 — B2 수치 검증과 신규 실측

### 1.1 B2 d_k 전량 재현 (로그숫자 자체 검증)

`/tmp/w31_on_a.JuCpDY/`·`/tmp/w31_s0_fe.C6wf6v/`의 solution CSV `ion_total`/manifest `n_element`로부터 stage 분율과 `d_k=|log10(p/p_ref)|`를 재계산, B2 표(`docs/CODEX_WAVE3_B2_TEST.md:29-32`, `:47-52`)와 **전 항목 일치**:

| 셀·원소 | EW 분율 (II,III,IV) | 앵커비 (II,III,IV) | d_k 재현 |
|---|---|---|---|
| s8 S | (1.145e-4, 0.9091, 0.0908) | (**0.0051**, 0.930, **1340**) | 2.2934/0.0313/3.1270 ✓ |
| s8 Fe | (1.442e-6, 0.6076, 0.3924) | (**0.0458**, 0.609, **143.7**) | 1.3395/0.2151/2.1575 ✓ |
| s0 Fe | (2.781e-13, 3.291e-5, 0.99997) | (**0.0280**, **0.108**, 1.0111) | 1.5528/0.9670/0.00479 ✓ |

p_ref는 조건부 CMFGEN 지도값(`docs/CODEX_ABS_STATE_5154.md:32,36-37`).

핵심 방향 신호(B2 표에는 없는 부호): **s8은 II 붕괴(196~22× 결핍)+IV 폭증(1340/144×) — 양쪽 다 이온화 방향. s0은 II·III가 앵커를 관통해 반대편 결핍으로 착지**(pair II +4.89 dex 과잉 → EW −1.55 dex 결핍, 6.4 dex 스윙; III +1.06 → −0.97 dex). EW 해는 세 셀·원소 모두에서 pair 대비 일관되게 **이온화 쪽으로** 이동했다.

### 1.2 링크별 유효 균형 — 전 링크 과이온화 실측

provenance의 bf 채널(rad/coll/nt)과 solution `restored_population`으로 stage 경계 플럭스를 재계산. 정상상태 폐합 up/dn=1.0000 확인(해가 조립 rate의 충실한 SE 해라는 독립 증명):

| 링크 | Γ_eff [s⁻¹] | α_eff [cm³/s] | EW n_hi/n_lo | CMFGEN비 | **과이온화** |
|---|---:|---:|---:|---:|---:|
| s8 S II↔III | 13.54 | 2.275e-12 | 7.94e3 | 43.4 | **+2.26 dex** |
| s8 S III↔IV | 1.062e-3 | 1.418e-11 | 9.99e-2 | 6.94e-5 | **+3.16 dex** |
| s8 Fe II↔III | 370.5 | 1.173e-12 | 4.22e5 | 3.17e4 | **+1.12 dex** |
| s8 Fe III↔IV | 1.457e-2 | 3.011e-11 | 6.46e-1 | 2.74e-3 | **+2.37 dex** |
| s0 Fe II↔III | 5.295e5 | 9.686e-13 | 1.18e8 | 3.07e7 | **+0.59 dex** |
| s0 Fe III↔IV | 1.620e3 | 1.154e-11 | 3.04e4 | 3.24e3 | **+0.97 dex** |

부호가 6/6 동일(과이온), 크기는 광구(s8)로 갈수록·상위 링크로 갈수록 커진다. 행렬 구조 결함의 패턴이 아니라 **이온화율 입력(장·단면적·추정기) 결함의 패턴**이다.

### 1.3 플럭스 운반자 분해 — 지상채널이 아니다

상향 이온화 플럭스의 지상 SL 몫: s8 S II→III **0.29%**, S III→IV 1.80%, Fe II→III 3.99%, Fe III→IV **0.74%**. 과이온화의 96~99.7%는 **여기준위 광이온화 채널**이 운반한다. 그중:

- **Fe III의 C48 lumped super-level(idx 201, E=13.13 eV, sl_id=100, 멤버 full level 다수)이 단독으로 III→IV 플럭스의 64.2%** (within-ion 인구분율 2.1e-5로).
- **Fe II lump(idx 100, E=6.52 eV)가 II→III의 21.0%** — Fe 두 링크 모두 최대 운반자가 C48 lump다.
- S는 lump 몫이 작고(II lump 3.42%), 대신 metastable/여기 SL들(E=1.9~3.1 eV, 그리고 E=17~19.6 eV 고준위)이 분산 운반.
- 이 lump들은 stdout의 `[ARTIS super-cutoff] K=100: 21581 levels lumped` 투영 산물이며, C2 리뷰가 §7 계약 위반으로 FAIL시킨 바로 그 C48이다(`docs/CODEX_WAVE3_C2_REVIEW.md:33-34`).

### 1.4 s0 Fe IV "악화"의 정량 해명 — 창 절단 부기

CMFGEN s0 Fe에서 II+III+IV 합 = 0.9893; **나머지 0.0107은 창 밖(V+)**. EW 창(II–IV)은 보존행으로 원소 전량을 II–IV에 강제하므로, D-5 제거로 II/III 과잉(절대질량 ~3.2%)이 해소되면 그 질량은 갈 곳이 IV뿐이다. 실측: EW IV/앵커 = 1.0111, 즉 **과충전 +0.0111 ≈ CMFGEN V+ 몫 0.0107**(4% 이내 일치; 앵커 3유효숫자 정밀도 내). diagnostics의 boundary fraction 1.375e-2(`docs/CODEX_WAVE3_B2_TEST.md:53`)가 가리키는 것도 동일 창 결손이다.

**따라서 "aggregate −58%인데 Fe IV 악화"의 정체: II·III의 3.3/0.17 dex 구조 회복은 실물이고, IV의 Δd=+0.0017은 spec §1.3.2의 boundary 규칙(`docs/WAVE3_D5_ELEMENT_WIDE_NLTE_SPEC_2026-07-31.md:42`)이 정확히 잡으라고 설계된 창 절단 아티팩트다.** 스펙이 s0 acceptance 전에 V stage 포함을 이미 요구하고 있다.

### 1.5 pair 레인이 s8에서 "좋았던" 이유 — 가림막

- pair 레인의 이온분할 폐쇄는 배너 자기선언대로 `LTE Saha @T_e,W=1 + per-ion pin(residual)`(`docs/V3_FUNCTION_AUDIT_VERIFICATION_2026-07-31.md:205-206`). s8 pair S II가 앵커 1.27× 이내였던 것은 rate가 맞아서가 아니라 **Saha@T_e가 우연히 CMFGEN NLTE와 근접**해서다.
- pair 오라클 자체가 D-5 캡처 서명을 갖는다: stage≥2의 Γ/α는 전부 `ion is not a lower member of an assembled NLTE pair`(`src/lumina_plasma.c:166-179`) — III→IV 균형은 pair 레인에서 애초에 같은 방식으로 측정된 적이 없다.
- 참고 대조(주의: 가중 인구가 달라 순수 산술 비교 아님): pair 오라클 S II Γ_total=1.907 s⁻¹, α_total=4.406e-12 → 함의 III/II=577. EW 유효값은 13.54/2.275e-12 → 7940. **같은 동결장에서 두 레인의 유효 균형이 13.8× 차이** — 구조(여기준위 재이온화 경로) + fixup의 신규 bf 산술(`docs/CODEX_WAVE3_1_FIXUP.md:8`)이 혼재하며 아직 분리되지 않았다(§4.2 ARTIS 행렬 대조 미실시, `docs/CODEX_WAVE3_B2_TEST.md:68`).

### 1.6 입력장 단서

- bf 추정기 소비의 **34.9%가 fallback**(positive 5.159e6 vs fallback 2.763e6, oracle s8). fallback의 실체는 MC 미표본 bin에서 `pref*J_bin`(binned C1 J_nu) 적분(`src/lumina_plasma.c:15627-15641`).
- 동결장의 알려진 왜곡: 2000–2500Å 점유율 66×, 2500–3000Å 108–111×, EUV 505–912Å **0.009×**(`docs/CODEX_ABS_STATE_5154.md:110-119`) — 여기준위 문턱(UV/근UV)이 보는 장은 과잉, 지상 문턱(EUV)은 기근. 여기준위-지배 과이온화와 방향이 정합.
- manifest의 `T_rad=10470.09324`가 s0과 s8에서 자릿수까지 동일 — 과거 "T_rad 전셸 10470핀=잣대 결함" 사건의 반향일 수 있으므로 writer 검증 전 증거로 쓰지 않는다(기록만).

## 2. Q1 — aggregate-vs-component 신호의 이야기 서열

| 순위 | 이야기 | 지위 | 증거 |
|---|---|---|---|
| 1 | **s0: D-5 구조 수리 실물 + stage-V 창 절단 아티팩트.** II/III 과잉(pair 저이온 과잉의 실체) 해소는 element-wide가 상위 drain을 보게 된 직접 결과; freed mass가 창 안 IV로 강제 착지해 IV만 +0.0017 악화 | **확정**(정량 일치 §1.4) | 과충전 0.0111 = V+ 몫 0.0107; boundary 1.37e-2 |
| 2 | **s8: 정직한 행렬 + 과이온화 입력.** 전 링크 +1.1~+3.2 dex 과이온, 96~99.7%가 여기준위 채널; pair는 Saha 폐쇄가 가림. 구조 수리가 가림막을 제거하자 입력 오차가 그대로 노출 → "무개선/악화" | **확정**(부호 6/6, 폐합 1.0000) | §1.2–1.3, §1.5 |
| 3 | **C48 lump가 Fe 과이온화의 주 운반자.** lump의 bf rate 구성(σ identity·문턱·within-SL 가중·n*)이 물리인지 아티팩트인지 미분리 | **미결**(운반자 확정, 원인 미정) | Fe III lump 64.2%, Fe II lump 21.0% |
| 4 | **fixup 신규 bf 산술 혼입.** EW-vs-pair는 구조+산술 이중 차이; §4.2 ARTIS 대조 미실시라 산술 무죄가 미증명 | **미결** | §1.5 13.8× 차이 미분해 |
| 5 | **앵커 fixed-T 편향.** 미세단계 앵커값은 지수적 T 민감(dex급) — 절대 d_k 크기는 조건부. 단 pair/elem **상대** 판정과 s0 부호 관통은 앵커로 설명 불가 | **부분 기여**(단독 설명 불가) | §4 참조 |
| 6 | 보존행 자체의 결함, 행렬 조립 오류 | **기각** | rank/residual/폐합/재현성 전부 클린(`docs/CODEX_WAVE3_B2_TEST.md:5-14`) |

## 3. Q2 — element-wide는 여전히 옳은 표적인가

**구조로서는 옳고, s8 acceptance 배치로서는 틀렸다.**

- s0이 스펙의 D-5 기전 예측(`docs/V3_FUNCTION_AUDIT_VERIFICATION_2026-07-31.md:185-194`: pair 부동점 ≠ element-wide 부동점)을 그대로 실증했다: 상위 drain을 보게 하자 저이온 과잉 4.9 dex가 즉시 붕괴. 이것은 지도의 "저항 축"(s0 IGE 전 성분 R, `docs/CODEX_ABS_STATE_60_OVERLAY.md:13-15`)이 **처음 움직인 사건**이다. N3/Wave1이 못 움직인 축을 구조 수리가 움직였다 — 지도 구조가 말해온 "s0=구조 지배" 진단과 정합.
- 반면 s8은 지도상 N3(장 측 변경)에 반응해 앵커 방향으로 움직여 온 셀(`docs/CODEX_ABS_STATE_60_OVERLAY.md:17-20`)이고, pair의 근접성은 Saha 폐쇄 보정의 산물이었다. **s8의 지배 결함은 이온화율 입력 내용(장 C1/C2 + C48 lump + 추정기 fallback)이며 Stage 2A가 명시적으로 동결하는 바로 그것이다**(`docs/WAVE3_D5_ELEMENT_WIDE_NLTE_SPEC_2026-07-31.md:30-35`). 25% 개선 사전등록은 "s8 국소 어긋남의 지배 원인=pair 구조"를 암묵 가정했는데, 실측이 그 가정을 반증했다. 이는 Wave-1 교훈(제거된 버그가 다른 결함을 보상 중이었다)의 재연이다.
- 결론: EW assembler는 유지·전진. 폐기·pivot 사유 없음. 단 acceptance의 무게중심을 "s8 25% 개선"에서 "구조 지배 셀(s0, 그리고 s20 frozen 입력 확보 시 s20)의 성분별 회복 + 입력 결함의 별도 원장 등재"로 재배치해야 한다. 등재는 조용히, 수리는 별도 결정(원장 규약).

## 4. Q3 — 앵커 한계 감사

- 앵커는 fixed-T·MAXCH 3.46e3% 미수렴 조건부(`docs/CODEX_ABS_STATE_5154.md:12-14`). 미세단계 분율의 T 민감도는 `d(log10 ratio) ≈ (χ/kT+1.5)/ln10 · (ΔT/T)`. CMFGEN s8 T_e≈10.4 kK 기준: S III→IV(χ=34.8 eV) **~1.75 dex/10%T**, S II(23.3 eV) ~1.2, Fe III(30.7 eV) ~1.5, Fe II(16.2 eV) ~0.85. 즉 s8 IV 성분의 절대 d_k(2.2~3.2 dex)는 앵커 T 오차만으로 dex급 이동 가능 — **절대 크기는 조건부 수치로만 취급.**
- 그러나 **판정 구제는 어렵다**: (i) improvement는 같은 앵커에 대한 pair/elem 거리 비율이고, elem은 s8에서 pair보다 일관되게 더 이온화 쪽이다. 판정이 뒤집히려면 앵커가 ≳1 dex 이온화 방향(더 뜨겁게)으로 움직여야 한다. (ii) 가용 예측은 반대 방향이다: dig_B6는 released-T에서 S II 에지 T_b 9550–9650 K vs fixed-T 10200 K, 즉 **~6% 냉각**을 예측(`/gpfs/kjhan/cmfgen_runs/R8_RESUME_NOTE.txt:2`). 냉각 앵커는 미세 고이온 분율을 낮춰 s8 과이온화 어긋남을 오히려 키운다. (iii) s0 II의 6.4 dex 부호 관통과 링크 과이온화 부호 6/6은 어떤 그럴듯한 앵커 이동으로도 지워지지 않는다.
- **relT2 착지 시 재판정 목록**: §4.3/§4.4 d_k 전표와 improvement 백분율, 지배 분율 절대차, §1.2 표의 CMFGEN비 열(과이온화 dex 재산정), T_e 사다리(s8 1.15×), 그리고 "pair Saha 폐쇄의 우연 근접"이 released-T에서도 유지되는지. **사전등록 기대(드라마화 금지, 방향만)**: 앵커 냉각 시 s8 양 레인 d_IV 동반 증가·elem이 더 증가, s0 결론 불변.
- **relT2에 홀드를 걸지 말 것**: 직전 released-T 시도(job 385770)는 it49 NaN으로 발산했고(`R8_RESUME_NOTE.txt` 2026-07-24 항), relT2는 현재 great-iteration 41로 진행 중일 뿐 착지 보장이 없다. 재판정은 착지 시 공짜로 얹는다.

## 5. Q4 — 권고: **(b) 행렬 채널 물리내용 감사** (단일 primary)

이유: 기계 게이트는 전부 클린하므로 남은 미결은 **rate 내용**뿐이고(§2 이야기 3·4), 필요한 원자료가 전부 디스크에 있어 오프라인으로 완결된다(offline-first 3요건의 "기전 오프라인 특정" 단계가 정확히 이것이다). 구체 표적 3개:

1. **C48 lump bf 구성 감사** — Fe III lump(64%)·Fe II lump(21%)의 σ identity, 유효 문턱, within-SL Boltzmann 가중, n* 구성을 provenance·identity CSV와 `src/lumina_element_wide.c` bf assembler(:245-430 영역)에서 추적. lump rate가 물리인지(멤버 준위 정당 합산) 투영 아티팩트인지 판정. C2 FAIL#4(C48 계약)를 여기에 흡수.
2. **여기준위 Γ 독립 재계산** — 최대 운반자 SL들의 Γ를 동결 입력(`/gpfs/kjhan/lumina_runner2/logs/coevolve_consume_parity59/lumina_c2_bfr_dump.csv`의 per-bin `bfr`/`j_nu_count`, 동 디렉터리 `lumina_c1_bins.csv`, CMFGEN σ_bf 격자)에서 손으로 적분해 provenance `aggregated_rate`와 대조. fallback bin(34.9%) 기여를 분리 집계. C2 FAIL#1(외부 기준 checksum)의 실질을 이것으로 닫는다.
3. **EW-vs-pair 동일링크 rate 원장** — 같은 (l→t) 전이의 pair 레인 산술(`src/lumina_plasma.c:15600-15707`)과 EW 산술을 수치 대조해 §1.5의 13.8×를 "구조 몫 vs 산술 몫"으로 분해. 이것이 §4.2 ARTIS 대조 공백의 최소 대체물이다.

감사 종결 후의 후속(별도 발주, 이번 결정 아님): s0 **Fe II–V 창 확장** 1건이 유일한 판정런 후보다. 사전등록 기대: IV 과충전 0.0111 소멸 → s0 Fe 성분별 전항 회복(§4.4 s0 축 PASS). 스펙 §1.3.2가 이미 요구하는 확장이므로 스펙 개정도 불필요.

기각 사유 — (a) C2 계약 4건 선수리: B2의 경계질량 상한이 보여주듯 판정을 못 뒤집고(S −137%, Fe −23%), 항목 1·4는 위 감사에 흡수, 항목 3(31→33 layout 누출)은 COMMIT 레인 전 필수지만 shadow 판정과 무관. (c) relT2 홀드: 선행 시도 발산 전력 + 예측 방향이 구제 반대(§4). (d) 셀 pivot: 지도가 지목하는 구조 지배 셀(s0)은 이미 이번에 측정됐고 −58%가 그 결과다 — pivot의 실질은 V-창 확장이며 위 후속에 포함. s20은 frozen 입력 아카이브 부재로 지금 불가(`docs/CODEX_WAVE3_A_IMPL.md:34`).

## 6. Q5 — 최저비용 판별 측정 (오프라인, 신규 런 0)

**Fe III lump(idx 201) 단일 준위의 Γ 삼중 대조.** 스크립트 1개, 입력 4파일(전부 존재):

```
A = EW provenance의 aggregated_rate (lump→IV 전 route 합)     [/tmp/w31_on_a.JuCpDY/..z26_s008_provenance.csv]
B = 손 적분: Σ_bin σ_bf(bin)·[bfr>0 ? σ·bfr : 4π/hν·J_bin·Δν]  [parity59 lumina_c2_bfr_dump.csv + lumina_c1_bins.csv + σ 격자]
C = 손 적분 B에서 J를 CMFGEN jnu4 벤치 J_nu(s8)로 치환          [validation/cmfgen_toy06_19p48d/ 계열]
```

판독표:

| 결과 | 귀속 | 다음 수순 |
|---|---|---|
| A ≠ B (>0.1 dex) | EW bf 산술 버그 (이야기 4 확정) | assembler 수리 후 재측정 |
| A ≈ B, C가 과이온화 붕괴 | **장 내용이 진범** (이야기 2 확정·구조 무죄 종결) | Stage-3/C1/C2 원장 등재, Wave 3 구조 PASS 재표기 |
| A ≈ B ≈ C | lump/원자데이터 내용 (이야기 3 확정) | C48 투영·σ 데이터 감사 확대 |

이 한 측정이 이야기 2·3·4를 동시에 갈라 Wave 3 갈림길의 잔여 불확실성 대부분을 제거한다. 같은 스크립트를 S II SL4(E=3.05 eV, EUV 문턱 611Å에서 Γ=31.8 s⁻¹ — EUV 0.009× 기근과 표면상 모순이라 그 자체로 의심 수치)에 재사용하면 fallback 내용 검증까지 겸한다.

## 7. 검증 상태 표

| 주장 | 상태 | 경로 |
|---|---|---|
| B2 d_k 9개 값 | 독립 재현 ✓ | solution/manifest CSV 재계산 |
| EW 해의 SE 충실성 | 실측 ✓ (up/dn=1.0000) | provenance×solution 플럭스 폐합 |
| 전 링크 과이온화 +0.6~+3.2 dex | 실측 ✓ (조건부 앵커 기준) | §1.2 표 |
| 여기준위/lump 플럭스 지배 | 실측 ✓ | §1.3 |
| s0 IV 과충전=V+ 몫 | 정량 일치 ✓ (4% 이내) | §1.4 |
| 과이온화의 장 vs 산술 vs lump 귀속 | **미결** | §6 판별 측정 대기 |
| relT2 재판정 영향 | 방향 예측만 사전등록 | §4 |
