# 얇은 선의 큰 `eps_l`, C11 80.9%, 무활성화 방출 감사

조사일: 2026-08-03. 기존 캡처와 원자자료만 사용했다. 모델·수송·GPU 런은 하지 않았고, 8 GB 이벤트 원장은 열지 않았다. `src/`, 원본 `VERDICT.md`, `scripts/uv_mapsplit_offline.py`도 수정하지 않았다.

## 결론

1. **조사 1 — `van_regemorter_trap`은 이 네 선에는 해당하지 않는다.** 네 선 모두 CMFGEN에서 가져온 전이별 충돌강도 tier 1을 사용하며, 작은 원자자료 `A_ul=0.368–3.859 s^-1`와 `C_ul=26.7–31.1 s^-1`가 결합해 `C/A=8.07–77.45`가 된다. Fe IV·Ni IV는 주어진 자료 안에서 정상 물리로 판정한다. Co IV는 충돌자료 자체가 Fe III 대용이라고 명시돼 있어 절대 물리 판정은 **`UNRESOLVED`**다.
2. **조사 2 — 80.9%는 `mc_J` 장이 아니라 과거 128M 이벤트 원장의 에너지 원장이다.** 분자는 s0–2, 1290–2000 Å의 방출선 이온축 Co IV 에너지 13.6314688이고, 분모는 같은 s0–2의 전 파장 방출 이벤트 에너지 16.8389519다. 몫은 0.8095200이다.
3. **조사 3 — 현재는 `UNRESOLVED`.** 496,950건/3.69273247의 존재와 비-LINE 기원 후보 경로는 확인했지만, 이들이 어느 채널·이온·대역·셸에 몰렸는지와 전체 유효 선방출 에너지 몫은 8 GB 원장을 한 번 통과하기 전에는 알 수 없다. 이를 계산하는 스트리밍 스크립트와 자기시험만 준비했다.

## 조사 1 — 얇은 선인데 `eps_l`이 큰 이유

### 정의와 세대

생산식은 `src/lumina_plasma.c:8471-8473`의

```text
C_ul = n_e * coeff / (g_up * sqrt(T_e))
eps' = C_ul / (C_ul + A_ul * beta_esc)
beta_esc = (1-exp(-tau))/tau
```

이다. 아래 계산에는 clamp/floor/cap/fallback/대체값을 적용하지 않았다. 입력이 정의역 밖이면 스크립트가 중단한다. 계산 산출물은 `validation/codex_eps_thin/investigation1_eps_thin.json`이며 정의는 그 파일의 `definitions`와 실제 계산이 일치한다.

주의할 세대 차이가 하나 있다.

- 사용자가 지정한 상태는 `/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/lumina_plasma_state.csv`의 `shell_id=0`, 필드 `n_e=4.627433e9 cm^-3`, `T_e=21227.639444 K`인 **최종 상태**다.
- `scheme_fracture_s0_line_rank.csv`의 `eps_l`은 `/gpfs/.../linepop_iter10`의 iter-10 상태에서 저장됐다. 그 덤프의 `selected_shell_state`는 s0에서 `T_e=21081.34859856876 K`, `n_e=4481706543.77714 cm^-3`다.

따라서 요청한 최종 상태로 `C/A`를 계산하고, 캡처 `eps_l` 재생은 같은 세대인 iter-10 상태로 따로 검사했다. 최종 상태를 캡처 `eps_l`의 생성 상태인 것처럼 섞지 않았다.

### 실제 `C_ul`, `A_ul`, 비율

요청한 최종 상태의 결과다. `tau_used`와 `eps_l` 출처는 `/gpfs/.../scheme_fracture_s0/scheme_fracture_s0_line_rank.csv`의 동명 필드, `g_up`은 `data/tardis_reference_toy06_19p48d_sivcaiv/levels.csv:g`, `A_ul`은 같은 모델의 `line_list.csv:A_ul`, `coeff`는 해당 `ige_col_Z_ion_cmfgen.bin`의 전이별 `Upsilon(10400 K)`에서 계산했다.

| line_id | ion, λ (Å) | `tau_used` | `Upsilon(10400)` | `coeff` | `g_up` | `A_ul` (s⁻¹) | `C_ul` (s⁻¹) | `C/A` | `eps'(tau)` | `eps(tau→0)` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 776418 | Fe IV 2835.740 | 0.1269702 | 0.5852 | 5.0496908e-6 | 6 | 1.400 | 26.73024 | 19.0930 | 0.953119 | 0.950231 |
| 748621 | Co IV 2734.832 | 0.0786278 | 0.7280 | 6.2819120e-6 | 7 | 0.368 | 28.50251 | 77.4525 | 0.987736 | 0.987253 |
| 774507 | Fe IV 2829.359 | 0.0528495 | 0.3906 | 3.3704874e-6 | 4 | 0.880 | 26.76221 | 30.4116 | 0.968966 | 0.968165 |
| 635410 | Ni IV 2280.055 | 0.0701410 | 0.9085 | 7.8394465e-6 | 8 | 3.859 | 31.12323 | 8.06510 | 0.893062 | 0.889687 |

얇은 극한과 현재 `tau`의 차이가 작다는 것도 보인다. 큰 `eps_l`의 직접 원인은 탈출확률 억제가 아니라 `C/A`가 크다는 것이다.

같은 세대 재생에서는 다음과 같이 캡처 필드와 정확히 일치했다.

| line_id | iter-10 `C_ul` (s⁻¹) | iter-10 `C/A` | 재생 `eps'` | CSV `eps_l` |
|---:|---:|---:|---:|---:|
| 776418 | 25.9781223 | 18.5558017 | 0.9518275211900334 | 0.9518275211900334 |
| 748621 | 27.7005269 | 75.2731708 | 0.9873850827624182 | 0.9873850827624182 |
| 774507 | 26.0091966 | 29.5559053 | 0.9680957558994530 | 0.9680957558994530 |
| 635410 | 30.2475104 | 7.83817321 | 0.8903060728205255 | 0.8903060728205255 |

### `A_ul`의 원자자료 계보와 전이 성격

`line_list.csv:A_ul`은 임의의 작은 값이나 로더 오류가 아니다. CMFGEN 원본 `osc_data`의 같은 전이 행과 정확히 일치한다.

| line_id | 준위 전이 | `line_list.A_ul` | CMFGEN 원본 파일·행의 `A` |
|---:|---|---:|---|
| 776418 | `3d5_6Se[5/2] → 3d5_4Pe[5/2]` | 1.400 | `/gpfs/kjhan/cmfgen_21jun23/atomic/FE/IV/19apr23/osc_data:1033`, 1.4000 |
| 774507 | `3d5_6Se[5/2] → 3d5_4Pe[3/2]` | 0.880 | 같은 파일 `:1034`, 0.8800 |
| 748621 | `3d6_5De[4] → 3d6_3De[3]` | 0.368 | `/gpfs/kjhan/cmfgen_21jun23/atomic/COB/IV/19apr23/osc_data:1045`, 0.3680 |
| 635410 | `3d7_4Fe[9/2] → 3d7_2Fe[7/2]` | 3.859 | `/gpfs/kjhan/cmfgen_21jun23/atomic/NICK/IV/19apr23/osc_data:1041`, 3.859 |

네 전이는 모두 같은 짝수 패리티의 `3d^n` 배치 사이이며 스핀 다중도도 6→4, 5→3, 4→2로 바뀐다. 따라서 보통의 허용 E1선이 아니라 **E1 금지/인터콤비네이션 성격의 약한 전이**라는 해석이 작은 `f_lu≈3.2e-10–2.4e-9`와 작은 `A`에 맞는다. 파일은 정확한 M1/E2/혼합 다극자 라벨을 제공하지 않으므로 그보다 세밀한 전이 종류는 단정하지 않았다. 여기서 “원자자료 참값”은 적어도 생산 입력 파일의 참값임을 뜻한다. 실험실 정확도까지 독립 인증한 것은 아니다.

### `coeff`의 출처와 처방

캡처 환경은 `PARITY59_INSTR.env`의 `LUMINA_ARTIS_PARITY=1`, `LUMINA_OMEGA_CMFGEN=1`, `LUMINA_CMFGEN_LINE_EPS_PHYS=1`이다. `stdout.log:259,353`도 다음을 기록한다.

```text
29840 tabulated (no floor), 1742025 van-Regemorter, 812267 OMEGA_SET=0.1
radeq coeff = 8.629e-6*Omega_CMFGEN(T=10400K), NO floor
```

네 준위쌍은 모두 29,840개 **tabulated tier 1**에 포함된다. `src/lumina_plasma.c:8683-8693`의 실제 분기는 `coeff=8.629e-6*Upsilon_CMFGEN(10400 K)`이며, 이 네 선에는 van Regemorter도 `OMEGA_SET=0.1`도 적용되지 않았다. 원본 `col_data` 전이행과 가져온 바이너리의 준위쌍을 모두 대조했다.

- Fe IV: `/gpfs/.../FE/IV/19apr23/col_data:39,44`; 파일 헤더 `:15-17`은 Zhang & Pradhan, `ZP97_FeIV_col`을 출처로 든다.
- Ni IV: `/gpfs/.../NICK/IV/19apr23/col_data:42`; 헤더 `:15-16`은 Fernández-Menchero et al. (2019)을 든다.
- Co IV: `/gpfs/.../COB/IV/19apr23/col_data:195`; 그러나 헤더 `:12-13`은 `Zha96_FeIII_col`과 “Using FeIII values?”라고 명시한다.
- 가져오기 계보는 `data/tardis_reference_toy06_19p48d_sivcaiv/coldata_cmfgen_manifest.csv`의 Fe IV/Co IV/Ni IV 행이며 세 파일 상태는 `OK`다.

충돌은 복사 허용 E1 선택규칙에 묶이지 않으므로, 금지 복사선의 작은 `A`와 `Upsilon~0.4–0.9`가 동시에 존재하는 것 자체는 모순이 아니다. 이 밀도에서는 그 조합이 `C~30 s^-1`을 만든다.

### 판정과 폴백 `B(T_e)`와의 곱

- **Fe IV·Ni IV: 정상 물리(주어진 원자자료 모델 안에서).** 약한 복사 붕괴율과 전이별 close-coupling 충돌강도가 실제 `C≫A`를 만든다.
- **Co IV: `UNRESOLVED`.** 계산·로딩·분기에는 결함 증거가 없고 van Regemorter 함정도 아니다. 다만 Co IV 전이별 값이 Fe III 자료의 대용이므로, 진짜 Co IV `Upsilon(0↔20,T)` 또는 독립적인 전이별 계산/실험이 있어야 `eps≈0.99`의 절대 물리성을 결판낼 수 있다.
- **`van_regemorter_trap`: 이 네 선에 대해 기각.** 그 폴백 분기에 들어가지 않았다.

`scheme_fracture_s0.json:definitions`는 `eta_l=w*eps_l*S_l_used`라고 정의하고, 행별 재계산도 네 선 모두 비트 수준에서 일치했다. 또한 payload 헤더 `src_nlte=0`이므로 `src/lumina_cmfgen.c:787-799`에서 전 선이 `S_l=B(T_e)`를 쓴다. 실제 곱은 다음과 같다.

| line_id | `eps_l*S_l_used` | `1-eps_l` |
|---:|---:|---:|
| 776418 | 1.64226662e-3 | 4.81725% |
| 748621 | 1.72335059e-3 | 1.26149% |
| 774507 | 1.67166288e-3 | 3.19042% |
| 635410 | 1.57459951e-3 | 10.9694% |

따라서 높은 `eps`가 과대한 Planck 폴백을 거의 감쇠하지 않은 채 `eta`에 싣는다는 의미에서 **두 항은 생산식에서 실제로 곱해진다**. `docs/VERIFICATION_REGISTERS.md:55`의 기존 비교는 `B(21228)/B(18760)=2.66`(912 Å), 2.11(1200 Å), 1.79(1553 Å), 1.26(5000 Å)다.

다만 `eps≤1`은 그 자체가 1보다 큰 증폭률은 아니다. 동일한 진짜 `eps`를 가정하면 `eta_prod/eta_truth=B(T_e)/S_truth`로 `eps`는 상쇄된다. 두 개의 독립적인 과대계수가 곱해진다고 말하려면 `eps_prod/eps_truth`도 알아야 한다. Fe IV·Ni IV에는 현재 그 두 번째 편향의 증거가 없고, Co IV만 위 이유로 `UNRESOLVED`다.

### 자기검사와 음성 대조

`scripts/codex_eps_thin_offline.py`의 가벼운 실행 결과:

- iter-10 `eps_l` 4개 재생 최대 절대오차: **0.0**.
- 원본 `osc_data.A` ↔ `line_list.A_ul`, 충돌 바이너리 준위쌍, `eta=w*eps*S` 왕복: **PASS**.
- 결함 주입 `g_up→g_up+1`: 최소 `eps` 절대오차 0.0017762로 **기각 PASS**.
- 잘못된 최종 상태로 iter-10 `eps`를 재생하는 세대 혼합: 최대 절대오차 0.0027564로 **기각 PASS**.

## 조사 2 — C11 80.9%의 정확한 출처

### 원본과 재계산

최초 문장은 `validation/cmfgen_toy06_19p48d/analysis/reddening_localization/VERDICT.md:18-20`에 있고, C11은 이를 `criminal_record/CRIMINAL_RECORD.md:64`로 옮겼다. 원 산출물은 다음 두 CSV다.

- `taskB_band_ledger.csv`: `group=s0-2`의 11개 전 파장 band 행 `emitE` 합 = **16.838951877972704**. `NUV_1290_2000.emitE` = **16.21967887878418**, 따라서 pile/all = **0.9632237799789306**.
- `taskB_top_ions.csv`: `group=s0-2`, `role=EMIT_NUVpile_1290_2000`, `Z=27`, `ion_idx=3`의 `E` = **13.631468772888184**. 따라서 Co IV/pile = **0.8404277837287241**.

그러므로

```text
80.95200266424967%
= 13.631468772888184 / 16.838951877972704
= 96.32237799789306% * 84.04277837287241%
```

이다. 결과 파일은 `validation/codex_eps_thin/investigation2_c11_origin.json`이다.

### 분자·분모 정의

생성 스크립트 `taskB_event_forensics.py:35-39,59-75,80-118`에 따른 정확한 정의는 다음과 같다.

- 원장: `logs/coevolve_consume_a10_kx_gphall/lumina_events.bin`, iter 11의 **CAP128M 저장 prefix**.
- 방출: `etype∈{2(line emit),4(kpkt ff),5(kpkt fb)}`.
- 파장: `2.99792458e18/EventRec.nu_comov`, 즉 이벤트 자체의 파장.
- deep: 이벤트 `shell∈{0,1,2}`.
- 분자: deep, 1290≤λ<2000 Å 방출 중 `line_id>=0`이고 **방출선** 테이블의 `(Z,ion_number)=(27,3)`인 `EventRec.energy` 합.
- 분모: 11개 ledger band가 덮는 전 파장의 deep 방출 `EventRec.energy` 합. 여기에는 `etype=4,5` 연속체 방출도 들어가므로 “deep line forest가 방출한 모든 erg”라는 문장은 엄밀히는 넓게 쓴 표현이다.

즉 **이벤트 원장의 packet comoving energy weight**이며 `mc_J` 장 덤프도 제3의 산출물도 아니다. 이온축은 활성화선이 아니라 방출선이다.

### 현재 400M 캡처와의 관계

현재 `/gpfs/.../pile_ion_attribution/pile_ion_attribution.json:legacy_co_iv_84_percent_check`는 별도 캡처의 400M 비무작위 prefix에서, `etype==2`, 유효 방출선, **방출선 표 파장** 1290–2000 Å만 분모로 쓴다. 같은 필드의 수치는:

- 방출선 이온축 Co IV: `164.82180114541927 / 784.6372296781043 = 0.2100611529904553`.
- 활성화선 이온축 Co IV: `2.8909815431121615 / 784.6372296781043 = 0.003684481737245861`.

이는 80.9%와 생성 세대, 저장 prefix, 파장축, 방출 종류, 분모, 귀속축이 모두 같지 않다. 따라서 현재 캡처에서 80.9%를 맞추는 재계산은 하지 않았고, 두 현재 수치를 과거 주장과 직접 비교하는 것도 금지한다. **80.9%의 출처 정의는 RESOLVED**, 현재 캡처에서의 동일 정의 재현은 입력 원장이 다르므로 적용 대상이 아니다.

## 조사 3 — 활성화 짝이 없는 방출 496,950건

### 현재 확정된 사실

출처는 `/gpfs/.../pile_ion_attribution/pile_ion_attribution.json:pairing_diagnostics`다.

- `line_emissions_without_activation = 496950`
- `line_emission_energy_without_activation = 3.692732472009652`
- `activation_records = 199491999`
- `recognized_terminal_records = 200010004`
- `activations_unpaired_at_stored_prefix_tail = 7858`

기존 정의는 같은 packet의 가장 최근 `etype==1`을 활성화로 잡고, 지정 terminal channel에서 지운 뒤 `etype==2 && line_id>=0`이 활성화 없이 나오면 누락으로 센다. 7,858은 저장 prefix 끝에서 **활성화 뒤 터미널이 아직 저장되지 않은 반대 방향의 미완성 상태**이므로 496,950개의 무활성화 방출을 설명하지 않는다.

캡처 `stdout.log:339-340,37980`은 `lambda_max=0.0`, 970,557,175 시도 중 첫 400,000,000건 저장, 뒤 570,557,175건 drop을 기록한다. `src/lumina_cuda.cu:4686-4689,9021-9025`도 필터가 꺼졌고 cap이 **꼬리를 버리는 prefix 절단**임을 보인다. 따라서 파장 필터나 저장 파일 앞머리 절단으로 선활성화가 사라졌다는 설명은 이 캡처에는 맞지 않는다.

반면 코드에는 활성화선 없이도 유효한 선방출을 만들 수 있는 경로가 실제 존재한다.

- `src/lumina.h:104`: channel `0x12 KPKT_COLLEXC`는 thermal pool→macro-atom→line이다.
- `src/lumina.h:108-118`: `0x16 KPKT_COLLEXC_BB`는 이와 달리 LINE(bb) 활성화에서 출발하는 전용 태그다.
- `src/lumina.h:126-137`: `0x31 MA_ACT_BF`, `0x38 MA_RAD_DEEXC`, `0x40 RPKT_BF_ABS`가 있어 bf 활성화 뒤 선으로 끝나는 경로를 구분할 수 있다.

따라서 496,950건 전부를 곧바로 “원장 기록 누락”으로 부를 수 없다. 최소한 `0x12`, 그리고 앞선 `etype=3`/`0x40` 뒤 `0x15` 또는 `0x38`인 부분은 비-LINE 기원의 합법적 후보다. 반대로 `0x16`인데 선활성화가 없거나, 선기원만 가능한 조합이 무활성화로 나오면 기록/짝짓기 결함의 강한 증거가 된다.

### 남은 계산과 정의

`scripts/codex_unpaired_emissions_offline.py`를 준비했다. 실제 8 GB 입력은 실행하지 않았고 `--self-test`만 실행했다. 실데이터 통과는 다음을 한 번에 계산한다.

- 무활성화 방출의 `EventRec.chan`별 count/energy.
- 방출선 표의 `(Z, zero-based ion_number)`별 count/energy.
- 방출선 표 파장 대역 `LT600`, B0–B4, `GT3000`별 count/energy.
- 이벤트 `shell`별 count/energy.
- `(ion, band, shell)` 교차표와 각 bucket의 전체 유효 선방출 대비 몫.
- `etype=3` 등 직전 terminal 이후 기원과 방출 channel 교차표.
- 두 분모를 별도로 계산한다: **모든** `etype==2 && valid line_id` 선방출 에너지, 그리고 스키마상 내부 방출 `etype∈{2,4,5,8}` 에너지. 3.69273247의 두 전역 몫을 모두 출력한다.

정의되지 않은 0 분모는 JSON `null`, CSV `UNDEFINED`로만 쓴다. 대체값은 넣지 않는다. 실제 패스는 기존 `pairing_diagnostics`의 네 수치(count, energy, activation count, terminal count)와 tail 7,858을 그대로 재생하지 못하면 중단한다.

합성 자기시험 결과는 다음과 같다.

- bf→k-packet `0x12` 무활성화, 정상 LINE 활성화→`0x38`, 무활성화 `0x16`, terminal 뒤 상태 삭제, 청크 경계 상태 보존: **PASS**.
- `0x50 ESCAPE`를 terminal에서 고의로 빼는 결함 주입: 정답 missing energy 25가 잘못된 14로 변해 **결함 기각 PASS**.

### 현재 판정

**`UNRESOLVED`**. 비-LINE 경로의 존재는 소스로 확인했지만 496,950건 중 그 몫, 지배 이온·대역·셸, 전체 유효 선방출 및 전체 내부 방출 에너지 대비 3.69273247의 몫은 아직 정의되지 않았다. 기존 작은 JSON에서 이를 역산하거나 B0–B4만을 전체 분모로 대체하지 않았다.

## 운전석 잔여 작업

계산 노드에서 CPU 오프라인 패스 한 번만 실행한다. GPU는 쓰지 않는다.

```bash
python3 scripts/codex_unpaired_emissions_offline.py \
  --run-heavy \
  --outdir /gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/codex_unpaired_emissions
```

예상 입력은 기존 8 GB `lumina_events.bin`과 20.7 MB `lumina_events_lines.bin`이다. 출력은 `unpaired_summary.json` 및 channel/ion/band/shell/교차표 CSV다. 실행 후 이 문서의 조사 3 `UNRESOLVED`를 그 결과로만 갱신해야 한다. 49 GB `lumina_jbar_dump.csv`를 읽거나 새 모델을 돌릴 잔여 작업은 없다.
