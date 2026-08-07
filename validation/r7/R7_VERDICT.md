# R7 판정 — 발행 위상 (2026-08-08 02:10)

사전등록 `validation/r7/CODEX_R7.md` §3 · 검사 `validation/r7/DRIVER_INSPECTION.md`.

## 판정: **PASS** (자기 계약에 한해)

| 계약 | 관측 | 판정 |
|---|---|---|
| commit/view 뒤 a208·a209 가 **물질 갱신 앞**에 선다 | `lane=MC iter=0: 위상 view -> a208 -> a209` | **PASS** |
| 동세대 삼중항 `o=e=r` | `[A2-10][PRE] lane=MC te=1 r=1 line=1 o=1 e=1 m=1` — 넷이 모두 1 | **PASS** |
| a208 이 **현 복사장에 결박** | 수리 전 `com=1 rad=0` → 수리 후 `r=1 o=1` | **PASS** |
| pure lane 에 a209 신설 | DET 에서 a209 가 호출되고 **원인을 지목하며** 차단(`blocked_stale_line=1`) | **PASS**(R6 경계, 사전등록됨) |
| R8: 실패 시 `(T_e,t)` 보존 + 표면 종료 | `te_manifest_preserved=1 generation_preserved=1 material_update=BLOCKED action=TERMINATE` | **PASS** |

검사기 독립 판정: `PUBLICATION_PHASE records=1 violations=0 verdict=PASS`.

★**R8 의 음성 대조가 실물로 시연됐다.** 주입할 필요가 없었다 — A2-10 이 실제로 실패했고
(`RADEQ_TERM_MISSING`), 그 실패에서 T_e 매니페스트와 세대가 **둘 다 보존**된 채
표면화 종료했다.  수리 전 코드는 이 자리에서 `T_e_generation = 0` 으로 지웠다.

## 차단 지점의 이동 — 이것이 R7 성립의 실질

```
수리 전 DET : [A2-10][PRE] ... opacity: com=1 rad=0 | emissivity: com=0
              [CMFGEN][FATAL] radiative-equilibrium T_e not qualified    ← 원인 이름 없음

수리 후 DET : [R7][PHASE] view -> a208 (r=1 o=1)
              [A2-09][BLOCKED] R7_PUBLICATION_BLOCKED blocked_stale_line=1  ← 원인 지목

수리 후 MC  : [R7][PHASE] view -> a208 -> a209  (o=e=r=1)
              [A2-10][BLOCKED] RADEQ_TERM_MISSING, (T_e,t) 보존, TERMINATE
```

## 이 판정이 **말하지 않는 것**

MC 런의 복사장은 **비어 있었다**(아래 §신규 결함).  따라서 이 PASS 는
**세대 장부와 위상**에 대한 것이지, 물리적으로 유의미한 MC 런에 대한 것이 아니다.
R7 의 계약이 위상·세대이므로 판정 자격은 성립하나, 내용 검증은 MC 가 실제로
수송할 수 있게 된 뒤에 다시 한다.

---

# ★신규 결함 — MC 전송이 기본 구성에서 죽어 있다

## 사실

```
lumina_transport.c:571-579
    if (bf && bf->enabled) {
        if (!bf->event_enabled) {
            ... fprintf("[A2-08][BLOCKED] consumer=T03 reason=EVENT_MEASURE_UNAVAILABLE rc=3");
            pkt->status = PACKET_REABSORBED;   ← **패킷을 죽인다**
            return;
        }
```
`bf->event_enabled = lumina_fix_bf_continuum_event_enabled()` (`lumina_plasma.c:7430`)
= `getenv("LUMINA_FIX_BF_CONTINUUM_EVENT")`, **기본 0**.

**저장소의 어떤 런처도 이 변수를 켜지 않는다**(`knob_scrap_ledger.py` 의 목록 항목이 유일한 언급).

⟹ `LUMINA_BF_OPACITY=1` 이고 이 노브가 꺼져 있으면 **모든 패킷이 첫 스텝에서 재흡수**된다.
실측: 800 패킷 → T03 블록 **정확히 800건** → 복사장 전 빈 UNSAMPLED
→ `a210_rebin_checked_J` 가 옳게 거부 → `RADEQ_TERM_MISSING`.

## 왜 이것이 노브 표면 감사의 표적인가

이름이 `FIX_` 로 시작하는 **선택적 수리 노브**처럼 보이지만, 실제로는 **필수 부품**이다.
OFF 경로가 ON 경로와 *다른* 것이 아니라 **치명적**이다 — 폴백이 없고 패킷을 죽인다.
판별식("OFF 경로에 ON 경로가 가진 것이 없다 ⟹ 스위치가 아니라 추가")의 극단 사례다.

기본값이 곧 **작동하지 않는 구성**이라는 뜻이므로, 노브 표면 동결 대장에서
S-CONTRACT(계약의 일부)로 재분류하고 노브를 없애거나 기본 ON 으로 만들어야 한다.
⚠수리는 별도 단이다.  여기서는 **기재만** 한다.

---

# ★내 잣대 실패 — 오늘 밤 세 번째, 같은 계열

| # | 무엇 | 어떻게 드러났나 |
|---|---|---|
| 1 | `LUMINA_PURE_CMFGEN=0` 을 넘겼으나 하니스의 `eval` 이 덮어써 **DET 를 MC 라 불렀다** | R7 이 새로 찍는 `lane=` 표지 |
| 2 | 그것을 고친 뒤에도 **DET 런처의 env 로 MC 를 돌렸다** — MC 에 필요한 노브가 없는 구성 | T03 블록 수 = 패킷 수 |
| 3 | 야간 판정 기준을 **단 경계보다 넓게** 잡아(`MATERIAL_PHASE_COMMITTED` 요구) R7 을 FAIL 로 적었다 | 검사기가 `violations=0 PASS` 로 반대 판정 |

★규약: **"그 env 를 넘겼다" ≠ "그 env 로 돌았다" ≠ "그 팔의 온전한 구성으로 돌았다".**
셋은 각각 별도 관측으로 확인해야 한다.  그리고 **판정 기준은 단 경계와 정확히 같아야 한다** —
넓히면 남의 단의 실패를 내 단의 실패로 적게 된다.
