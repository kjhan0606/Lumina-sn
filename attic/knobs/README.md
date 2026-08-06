# 창고 (attic/knobs)

user 지시(2026-08-07): **"비물리와 계측용 노브들을 소스코드들로부터 스크랩해서 창고에
쳐박아놓고, 판정으로 인정된 물리 배선도와 0층에서 확증된 자산을 바탕으로 1층 검사를 시작."**

여기는 **삭제가 아니라 보관**이다. 스크랩된 노브마다 원래 위치·감싸던 분기·
남긴 무조건 경로를 대장에 남긴다. 되살릴 근거가 필요하면 여기서 찾는다.

- `KNOB_SCRAP_LEDGER.json` / `.md` — 분류 대장(생성: `scripts/knob_scrap_ledger.py`)
- `<file>/` — 파일별 제거 기록(제거된 분기 원문 + 남긴 경로)

## 원칙

**생존이 예외, 스크랩이 기본.** 살아남으려면 셋 중 하나여야 한다:
입력(경로·자원)이거나 · 0층 계약이 요구하거나 · 판정된 물리 배선도에 있거나.

제거는 **분기를 없애고 현 값을 무조건 경로로 못박는 것**이다. 그래야 현 설정에 대해
바이트 동일이 유지되고, "그 값이 옳은 물리인가" 라는 질문이 **하나로 닫힌다**
(노브가 있는 한 "다른 값을 넣으면 다른 물리" 가 따라붙어 판정이 안 닫힌다).

## 파일별 정독 결과 (기계 분류가 틀린 사례)

| 파일 | 기계 분류 스크랩 | 정독 후 실제 스크랩 | 비고 |
|---|---|---|---|
| `lumina_main.c` | 11 | **11** | 전부 스위치였다 |
| `lumina_atomic.c` | 9 | **0** | 전부 덧붙임·가드·입력이었다 |

`lumina_atomic.c` 9건의 정체:
- `TOPSTAGE_ANCHOR` — 최상단 이온 바닥준위 주입(ARTIS SINGLE_LEVEL_TOP_ION 부분 구현)
- `ALPHA_SPINGATE` + `SPINGATE_MULT` — 스핀-금지 재결합 판별(Fe α 5× 근원 수리)
- `FIX_BF_STIM_RECOMB` + `BF_CLUMP_FACTOR` — 유도재결합 + clumping.
  주석이 명시한다: *"Keep the field absent on the gate-OFF path"* — OFF 는 그 물리가 없다
- `FIX_BF_CONTINUUM_EVENT` + `MA_RADRECOMB_TARGET` — D-1 연속체 event selector 수리
- `CMF_EPAY` — A2-17 은퇴 **강제 가드**. 스크랩하면 폐기된 분류기를 요구하는 설정이 조용히 통과
- `T_INNER_FIX` — CONFIG-PREC 우선순위 사슬(argv > env > config.json > default)의 env 단

⟹ **일괄 sweep 이었다면 물리 수리 4건·강제 가드 1건·계약 1건을 지웠다.**
파일별 정독 없이는 스크랩하지 않는다.

## 미해결 (기재)

`LUMINA_T_INNER_FIX` 는 한 이름으로 **두 가지**를 했다:
(a) CONFIG-PREC 의 초기 T_inner 지정 [lumina_atomic.c, 존치]
(b) 반복 중 T_inner 핀(update_t_inner 무력화) [lumina_main.c, 스크랩됨]
현 설정이 이 env 를 쓰지 않으므로 동작은 불변이나, **의미가 갈라진 채 남았다**.
L1-5(물질 입력의 물리화)에서 판정할 것.

## 기계 분류의 한계 (음성대조 실패 기록)

`scripts/knob_block_classify.py` 를 만들어 "블록이 무엇을 하는가" 로 스크랩 안전성을
가르려 했다. **음성대조를 통과하지 못했다.**

정답이 이미 있는 `lumina_atomic.c` 9건(정독 결과 전부 스크랩 금지)에 돌린 결과:

| 도구 판정 | 수 | 실제 |
|---|---|---|
| ADDS-PHYS | 3 | 맞음 |
| GUARD | 2 | 맞음 |
| UNKNOWN | 2 | 보류 — 그 중 `TOPSTAGE_ANCHOR`(최상단 이온 바닥준위)가 여기 빠졌다 |
| **DIAG-ONLY** | **2** | **틀림** — `FIX_BF_CONTINUUM_EVENT`(물리 수리) · `SPINGATE_MULT`(입력) |

`TOPSTAGE_ANCHOR` 가 새는 이유: `if (!(e && atoi(e) != 0)) return;` 라는
**early-return 가드** 관용구는 중괄호 블록이 아니라 잡히지 않는다.

⟹ **어떤 기계 기준도 스크랩을 승인하지 못한다.** 도구의 용도는 *읽는 순서* 뿐이고,
승인은 정독만 할 수 있다. 남은 ~300건은 본질적으로 수작업이며, 그렇기 때문에
**승인된 배선도(F→G) 이후로 미루는 것이 옳다** — 살아남지 못하는 함수에 붙은 노브는
그 함수와 함께 사라지므로 읽을 대상 자체가 줄어든다.
