# 단 — 연속체 사건 측도의 단일 정의 (사전등록, 2026-08-08)

발견 경위: R7 판정 런에서 MC 팔이 `RADEQ_TERM_MISSING` 으로 막혔다.
따라가 보니 **패킷 800개가 전부 첫 스텝에서 죽어** 복사장이 비어 있었다.
그 원인을 세 소비 지점에서 대조한 결과가 이 단이다.

---

## 계약 (하나)

> **연속체 사건 측도(bound-free event measure)는 단일 정의를 가지며,
> 그것을 소비하는 모든 지점이 같은 정의와 **같은 부재 정책**을 쓴다.
> 부재 시 다른 양을 조용히 대입하지 않는다.**

---

## 1. 실측 — 같은 부재에 대해 세 지점이 **서로 다르게** 행동한다

| 지점 | 사건 측도가 없을 때 | 결과 |
|---|---|---|
| **CPU 전송** `lumina_transport.c:571-579` | `blocked_negative_transport++` 후 **패킷을 재흡수하고 끝낸다** | 전송 전멸 |
| **CPU a208** `lumina_plasma.c:8183-8188` | `bfnet = legacy - ff` 로 **대체**하고, `bfnet < 0` 일 때만 차단 | 조건부 통과 |
| **GPU 전송** `lumina_cuda.cu:6124` | `d_chi_bf` 로 **조용히 대체**, **부호 검사 없음** | 무조건 통과 |

```c
/* CPU 전송 — 죽인다 */
if (bf && bf->enabled) { if (!bf->event_enabled) {
    a208_counters()->blocked_negative_transport++;
    fprintf(stderr,"[A2-08][BLOCKED] consumer=T03 reason=EVENT_MEASURE_UNAVAILABLE rc=3\n");
    pkt->status = PACKET_REABSORBED; return; } }

/* CPU a208 — 대체하되 부호는 본다 */
double event_bf = (bf && bf->event_enabled && bf->event_chi_bf) ? bf->event_chi_bf[k] : bfnet;
if (!isfinite(event_bf) || event_bf < 0.0) { ctr->event_measure_unavailable++; return 5; }

/* GPU 전송 — 조용히 대체, 검사 없음 */
const double *bf_event_grid = (bf_event_enabled && d_bf_event_chi) ? d_bf_event_chi : d_chi_bf;
```

**두 양은 같은 양이 아니다.**
- `event_chi_bf` += `chi_spont_base` (`lumina_plasma.c:7929`) — **자발 흡수만**, 정의상 ≥ 0
- `chi_bf` += `chi_contrib` (`:7927`), 그리고 a208 이 쓰는 `bfnet = chi_bf - ff` — **순(net)**,
  유도 재결합 때문에 **음수가 될 수 있다**

사건 측도는 확률 측도다.  음수일 수 있는 순 불투명도를 사건 확률로 쓰는 것은 정의 위반이며,
A2-08 의 상태 열거에 `BLOCKED_NEGATIVE_OPACITY_SEMANTICS` 가 있는 이유가 그것이다.
⟹ **GPU 전송은 그 검사 없이 대입한다.**

## 2. 노출 규모

```
LUMINA_BF_OPACITY=1 을 켜는 런처      : 299개
LUMINA_FIX_BF_CONTINUUM_EVENT=1 런처 :   0개
```
⟹ 저장소의 **모든 생산 런**이 사건 측도 없이 돌았다.
GPU(생산 경로)는 조용히 대체했고, CPU MC 는 죽는다 — 그래서 아무도 몰랐다.
CPU MC 팔은 `8a9f861`(A2-08 폐합, "정직한 BLOCKED 표면화") 이후 사실상 **작동 불가**다.

★역설: **정직한 쪽(CPU)이 멈추고, 조용한 쪽(GPU)이 생산을 계속했다.**
차단은 옳았지만 다른 팔에 같은 잣대가 적용되지 않아, 결과적으로 결함이 은폐됐다.

## 3. 이 노브의 정체

`docs/CODEX_WAVE2_A_REPAIR_REPORT_2026-07-31.md:97` 이 OFF 를 *"argmax/RNG 순서 유지"*
라는 **정당한 대안 경로**로 적고 있고, `docs/ARTIS_PARITY_GAP_AUDIT.md:45` 는 ON 을
*"D6 [GATED REPAIR]"* 로 적는다.  즉 설계 의도는 "두 경로가 있다" 였다.

그런데 지금 OFF 경로는 **대안이 아니라 비작동**이다(CPU) 또는 **다른 양의 조용한 대입**
이다(GPU).  이름은 `FIX_`(선택적 수리)인데 실체는 **필수 부품**이다.

---

## 4. 기대 변경집합 (사전등록)

1. 단일 접근자 `bf_event_measure_get(bf, shell, nu, double *out)` 가 **값과 상태**를 함께 낸다
   (`OK` / `UNAVAILABLE` / `NEGATIVE` / `OUT_OF_GRID`).
2. 세 소비 지점이 전부 그 접근자를 쓴다 — CPU 전송·CPU a208·GPU 전송.
3. 부재 정책이 **하나**다.  이름 있는 사유로 차단하며, 조용한 대체는 없다.
4. OFF 경로를 유지한다면 그것은 **암묵적 폴백이 아니라 이름과 provenance 를 가진 별도 생산자**
   여야 한다(`EVENT_MEASURE_SPONTANEOUS` / `EVENT_MEASURE_LEGACY_ARGMAX`).
5. GPU 커널이 부호 검사를 갖는다.

**변경하지 않는 것**: `chi_spont_base`·`chi_contrib` 의 계산식, argmax 경로의 물리.

---

## 5. 음성 대조

| # | 주입 | 기대 |
|---|---|---|
| **NE1** | ★주입 없음 — **현행 코드가 그대로 음성 대조다.** event OFF 로 CPU 전송·CPU a208·GPU 전송을 각각 관측 | 오늘: 죽음/조건부통과/무조건통과 **3종 불일치**. 수리 후: **세 지점 동일 판정** |
| **NE2** | `bfnet < 0` 인 (셸,빈)을 만들어 GPU 전송에 태운다 | 수리 후 GPU 도 차단. 오늘은 **음수를 확률로 쓴다** |
| **NE3** | event ON | 세 지점 모두 통과하고 CPU MC 가 **실제로 수송한다**(T03 블록 0) |
| **NE4** | 접근자를 우회해 `chi_bf` 를 직접 읽는 코드 추가 | 컴파일 또는 게이트에서 걸려야 한다(우회 불가 구조) |

## 6. 측정 (이 단의 산출)

| | 무엇 |
|---|---|
| **ME1** | `bfnet < 0` 인 (셸,빈) 비율 — GPU 가 **음수를 사건 확률로 쓴 빈도** |
| **ME2** | ★**GPU 생산 구성에서 event ON vs OFF 스펙트럼 차이.** 299개 런처가 전부 OFF 로 돌았으므로, 이 차이가 곧 **기존 결과에 실린 오차의 크기**다 |
| **ME3** | CPU MC 가 event ON 에서 실제로 수송하는가(T03 블록 0, 복사장 빈 채움률) |

⚠ME2 는 **과거 결론의 재평가 범위**를 정하는 수치다.  나오기 전에는
"기존 GPU 결과가 틀렸다"고 말하지 않는다 — 대체된 양의 차이가 작을 수도 있다.

## 7. 게이트

| | 기준 |
|---|---|
| **E1** | NE1~NE4 전항 기대대로.  특히 NE1 이 **세 지점 동일 판정**을 보여야 한다 |
| **E2** | event ON 일 때 CPU MC 와 GPU 가 **같은 사건 측도 배열**을 쓴다(해시 대조) |
| **E3** | ME1~ME3 수치가 대장에 기재됨 |
| **E4** | 회귀: event ON 구성에서 기존 GPU 런과 **바이트-parity**(계약만 추가했으므로) |

---

## 8. ★Fable 판단이 필요한 것 (물리 결정)

**OFF 경로(legacy argmax)는 유지할 물리인가, 죽은 코드인가?**

- 유지한다면: 별도 provenance 를 가진 정식 생산자로 승격하고, 두 팔이 **같은 것**을 쓰게 한다.
- 죽었다면: 노브를 제거하고 사건 측도를 **필수**로 만든다(기본 ON 이 아니라 **선택지 없음**).

근거 자료: 위 §3 의 두 문서, ME2 의 스펙트럼 차이.
⚠운전석은 이 판단을 하지 않는다.  ME2 를 재서 올린다.

---

## 9. 순서

Γ단(감마 소유권) 게이트 폐합 후 착수.  R6 와는 독립이다.
⚠이 단이 열려야 **Γ4(M1/M2 측정)** 를 닫을 수 있다 — M1/M2 는 A2-10 성공을 요구하는데
지금 MC 팔은 전송이 죽어 A2-10 에 도달하지 못한다.
