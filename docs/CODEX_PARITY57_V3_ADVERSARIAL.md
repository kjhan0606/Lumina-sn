## 판정표

| 주장 | 판정 | 독립 검증 결과 |
|---|---|---|
| C1. 단일변수 V1 | **CONFIRMED** | 두 `RESOLVED CONFIG`와 두 `RUN FOOTER`의 전체 블록 diff는 `LUMINA_MA_LINE_DESTRUCT=1→0` 한 줄뿐이다. 바이너리·인수·EMA 계수는 동일하다. |
| C2. FORMAL-CONS 62.37→48.72, −22% | **CONFIRMED** | CSV 독립 사다리꼴 적분은 `1.930315250633e44→1.507638800621e44 erg/s`, 즉 **−21.8968%**다. stdout 반올림값 및 CONSWIN과 일치한다. |
| C3. “b4/1113 Å 지표” 3.908e4→3.742e4 | **REFUTED** | 수치는 존재하지만 `b_k`가 아니라 **level 4의 n_k**다. level 4의 기준 전이는 **1206.5 Å**이고, 1113.2 Å는 level 3→9/10/11이다. 실제 b4 변화는 1.4290→1.4172, −0.83%다. |
| C4. max\|ΔT_e\|=2897 K@s0 | **CONFIRMED** | `ΔT_e=T57−T50=−2896.830664 K`이며 전 셸 최대 절댓값은 shell 0이다. parity57이 더 차갑다. |

## C1 — V1과 산출물 무결성

두 설정 블록은 [parity50 stdout:4](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity50/stdout.log:4)–125와 [parity57 stdout:4](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity57/stdout.log:4)–125이다. 전체 diff의 유일한 차이는 다음 한 줄이다.

- parity50: [LUMINA_MA_LINE_DESTRUCT=1](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity50/stdout.log:70)
- parity57: [LUMINA_MA_LINE_DESTRUCT=0](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity57/stdout.log:70)

RUN FOOTER 전체 diff도 동일하게 한 줄뿐이다: [parity50:38144](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity50/stdout.log:38144), [parity57:38915](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity57/stdout.log:38915).

동일성 세부사항:

- 바이너리: 양쪽 모두 `lumina_cuda.withParityAA` — [p50:49](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity50/stdout.log:49), [p57:49](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity57/stdout.log:49)
- 현재 바이너리 SHA-256: `2d0cfec5504344753dd77893f0692bac5838970760f314b521a42a009b6f5b22`; mtime은 두 런보다 앞선 `2026-07-30 01:11:12 +0900`
- EMA: 양쪽 `LUMINA_COEVOLVE_JBAR_DAMP=0.5` — [p50:34](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity50/stdout.log:34), [p57:34](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity57/stdout.log:34)
- 인수도 동일: `100000 12 spectrum nlte` — [p50:124](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity50/stdout.log:124), [p57:124](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity57/stdout.log:124)

신선도:

- parity50 `.run_start`: `2026-07-30 15:14:13.968 +0900`; 핵심 산출물은 +9102∼+9108초
- parity57 `.run_start`: `2026-07-31 02:43:37.982 +0900`; 핵심 산출물은 +6268∼+6271초
- 모두 owner `kjhan:kjhan`
- 런처는 `.run_start`보다 새 파일만 복사한다: [run_coevolve_s01.sh:146](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_coevolve_s01.sh:146)–151
- writer 흔적: levelpop [p50:37975](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity50/stdout.log:37975), [p57:38746](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity57/stdout.log:38746); resolve_raw [p50:38065](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity50/stdout.log:38065), [p57:38836](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity57/stdout.log:38836); formal/plasma [p50:38072](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity50/stdout.log:38072), [p57:38843](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity57/stdout.log:38843)

## C2 — FORMAL-CONS

stdout 실측:

- parity50: `L=1.930315e44`, `62.37×L_inj`, CONSWIN `63.2835` — [stdout:38070](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity50/stdout.log:38070)
- parity57: `L=1.507639e44`, `48.72×L_inj`, CONSWIN `49.4265` — [stdout:38841](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity57/stdout.log:38841)

CSV의 2,000개 점을 코드와 같은 사다리꼴 규칙으로 독립 적분했다. 원 코드 정의는 [lumina_plasma.c:16851](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:16851)–16858이다.

| 값 | parity50 | parity57 | 변화 |
|---|---:|---:|---:|
| 독립 적분 L | 1.930315250633e44 | 1.507638800621e44 | −21.8968% |
| L/L_inj | 62.3736454 | 48.7158395 | −21.8968% |
| CONSWIN | 63.2835055 | 49.4264697 | −21.8968% |

CSV 반올림으로 인한 stdout 적분과의 차이는 약 `1.3×10⁻⁷` 상대오차다.

## C3 — 지표 정의 오류

인용 수치는 [parity50 resolve_raw:9032](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity50/lumina_levelpop_resolve_raw.csv:9032)와 [parity57 resolve_raw:9032](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity57/lumina_levelpop_resolve_raw.csv:9032)에 실제로 있다.

| 열 | parity50 | parity57 | 변화 |
|---|---:|---:|---:|
| `n_k`(7열) | 3.908138e4 | 3.742235e4 | **−4.2451%** |
| `b_k`(9열) | 1.4290 | 1.4172 | **−0.8258%** |

기존 판정 스크립트도 b 계수는 9열을 읽도록 정의한다: [judge_parity35.py:61](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/judge_parity35.py:61)–67.

파장 연결도 다르다.

- Si III level 4의 ground 전이: **0→4, 1206.5 Å** — [line_list.csv:282646](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/line_list.csv:282646)
- 1113.174/1113.204/1113.230 Å: **3→11/10/9** — [line_list.csv:249727](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/line_list.csv:249727), [249739](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/line_list.csv:249739), [249752](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/line_list.csv:249752)

1113.23 Å에 대응하는 b9은 `3.0779→3.0175`, 즉 약 −1.96%다: [p50:9037](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity50/lumina_levelpop_resolve_raw.csv:9037), [p57:9037](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity57/lumina_levelpop_resolve_raw.csv:9037).

따라서 “3.908e4→3.742e4”라는 숫자 변화만 맞고, `b4`, `b_k`, `1113 Å`라는 식별은 모두 맞지 않는다.

## C4 — 전 셸 ΔT_e

`ΔT_e ≡ T_e(parity57)−T_e(parity50)`로 계산했다.

| 순위 | shell | T50 (K) | T57 (K) | ΔT_e (K) |
|---:|---:|---:|---:|---:|
| 최대 | 0 | 21202.683 | 18305.853 | **−2896.831** |
| s0 제외 1 | 1 | 18424.011 | 16814.013 | −1609.998 |
| 2 | 2 | 16475.786 | 15856.436 | −619.350 |
| 3 | 11 | 11503.170 | 11199.982 | −303.188 |
| 4 | 3 | 15661.949 | 15422.775 | −239.173 |
| 5 | 17 | 9026.900 | 9223.617 | +196.717 |

근거: [parity50 plasma_state](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity50/lumina_plasma_state.csv:2), [parity57 plasma_state](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity57/lumina_plasma_state.csv:2).

## 적대 질문

### A1. 게이트의 정확한 의미론

`LUMINA_MA_LINE_DESTRUCT`의 물리적 소비처는 터미널 `ma_line_eps` 추가 추첨이다.

- 게이트 ON일 때만 device eps 테이블 할당: [lumina_cuda.cu:240](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:240)–243
- host eps 테이블도 게이트 ON일 때만 할당: [lumina_plasma.c:3693](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:3693)–3710
- `eps=C_down/(C_down+Aβ)` 저장: [lumina_plasma.c:4214](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:4214)–4218
- 선택된 터미널 전이에서 추가 RNG 추첨: [lumina_cuda.cu:4376](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4376)–4387

반면 `kp_deact`는 그대로 유지된다.

- 같은 `C_down`을 `kp_deact`에 더함: [lumina_plasma.c:4182](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:4182)–4240
- `p_kpacket=kp_deact/(sum_rates+kp_deact)`: [lumina_plasma.c:4526](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:4526)–4532
- GPU 선행 추첨: [lumina_cuda.cu:4220](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4220)–4239

parity57 stdout에도 `p_kpacket`이 계속 출력된다: [stdout:442](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity57/stdout.log:442). 반대로 `[MA-LINE-DESTRUCT]` 출력은 없다.

따라서 정확한 의미는 **“ma_line_eps 터미널 추가 추첨 OFF, kp_deact 선행 추첨 ON”**이다. 그 외 소비처는 테이블 업로드·해제·counter/reset·진단 출력뿐이며 별도 물리 경로는 발견되지 않았다.

### A2. 전체 채널 OFF 효과인가?

**아니다. 정당한 선행 `p_kpacket` 몫은 남아 있다.**

단순 2준위에서는 먼저 `p=C/(C+Aβ)`로 k-packet 추첨을 하고, 실패 후 동일 `p`로 터미널 추첨을 다시 하므로 합성 확률은 `p+(1−p)p=2p−p²`다. parity57은 두 번째 항 `(1−p)p`만 제거한다.

따라서 −21.9%, C3 행 변화, −2897 K는 이 구성에서 **중복 터미널 추첨을 제거했을 때의 결합계 응답**이다. “정당한 1p까지 포함한 채널 전체 OFF 효과”는 아니다. 다만 NLTE/열/수송이 다시 수렴한 결과이므로 이를 미시 확률 차이와 수치적으로 동일시할 수는 없다.

### A3. ΔT_e 방향

shell 0에서 parity57은 parity50보다 **2896.8 K 낮다**. 기존 서사가 “Lumina 가스가 CMFGEN보다 2000–2600 K 차갑다”는 방향이라면, 이번 변화는 그 차가운 방향과 **같은 부호**다. 별도의 CMFGEN 대비 크기 판정은 하지 않았다.

### A4. −22%를 운반하는 파장 구간

각 점의 비율 `Rλ=F57/F50<1`인 연속 구간을 만들고, 각 구간을 `∫(F50−F57)dλ`로 순위화했다.

| 순위 | 연속 구간 (Å) | 구간 L57/L50 | ΔL (erg/s) | 전체 순감소 기여 |
|---:|---:|---:|---:|---:|
| 1 | **2191.625–3234.875** | 0.7319 | 2.6694e43 | **63.15%** |
| 2 | **3566.375–4229.375** | 0.8381 | 5.3350e42 | **12.62%** |
| 3 | **1431.125–1762.625** | 0.5519 | 3.0486e42 | **7.21%** |

상위 3구간 합계가 순감소량의 **82.99%**다. 구간 경계 근거는 두 formal CSV의 동일 파장행이다: [parity50:97](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity50/lumina_spectrum_formal.csv:97), [175](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity50/lumina_spectrum_formal.csv:175), [316](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity50/lumina_spectrum_formal.csv:316) 및 대응하는 [parity57 CSV](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/logs/coevolve_consume_parity57/lumina_spectrum_formal.csv:97).

추가 무결성 검사에서 두 런의 formal/plasma/세 levelpop 파일에 NaN·Inf는 없었고, `n_k<0`도 0건이었다.

**종합: V3 스탬프 불가 — 런 무결성 자체의 BLOCKING 결함은 없지만, C3의 `n_k↔b_k` 및 `1206.5↔1113 Å` 오인이 보고서 BLOCKING 결함이다.**