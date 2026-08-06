# 자율진행 보고 — 2026-08-07 (user 취침 중)

지시: *"4 거부까지 도달하시오"* + *"앞으로 6시간동안 자율진행"*.
**4단계 도달 완료.** 이후 census 잔여 부류를 계속 닫는 중.

---

## 1. 노브 표면 동결 1–4단계 — 완료

| 단계 | 상태 | 증거 |
|---|---|---|
| 1 보고만 | ✅ | 음성대조: 가짜 노브 2개 주입 → 정확히 2개 탐지 |
| 2 집계 | ✅ | 런처 404 · env 설정 11,101 · **죽은 노브 1,384(12.5%)** |
| 3 이관 | ✅ 부분 | T3 는 must-unset 22종을 소스에서 유도. 함대 이관은 처분 대기 |
| 4 거부 | ✅ 게이트 뒤 | `LUMINA_ENV_STRICT=1` 인 런만 거부. 양성·음성 대조 통과 |

**배터리 새 덱·구 덱(통제) 양쪽 PASS** — 동작을 바꾸지 않았음을 확인.

거부를 전역으로 켜지 않은 이유: 그러면 C2 사고(런처 83–174 사망)를 자발적으로
반복한다. 기본값은 report-only 이고, 신규 런처만 STRICT 를 켠다.

## 2. ★이 작업이 드러낸 것 (물리 판정에 영향)

### 2.1 죽은 노브 1,384건 — 복사평형·coupled 솔버

런처가 설정하지만 **src 가 읽지 않는** 노브:

```
RADEQ_COOL_NLTE_ONLY 102 · RADEQ_COOL_ESCAPE 99 · RADEQ_COOL_NONNEG 97
COUPLED_TDEP 88 · COUPLED_JNU_PHOTOION 84 · COUPLED_JNU_LSTAR 81
COUPLED_LAMBDA_STAR 81 · RADEQ_LINE_RE 78 · RADEQ_LINE_RESPOND 77
```

검증: `COUPLED_JNU_PHOTOION`·`RADEQ_LINE_RESPOND`·`CN_DAMP` 는 **src 언급 0회**.
`NLTE_FALLBACK_TE` 는 `lumina_cuda.cu:1482` **주석에만** 있다.

⟹ **회귀 대장의 `gate_set`(111 항목)에 아무 일도 하지 않은 설정이 들어 있다.**
이 노브를 변화시킨 과거 A/B 는 동일 설정끼리 비교했을 수 있다. **판정 필요.**

### 2.2 챔피언 설정이 재현 불가

참조 런처가 `LUMINA_CMF_EPAY=2`·`LUMINA_CMF_EPAY_HOTF=0` 을 켜는데 A2-17 이
*"retired scalar hot/cold classifier"* 를 제거해 **로드 단계에서 거부**된다.
전수 **재현 불가 런처 36개**. 정본 `validation/instrumentation_debt/CHAMPION_UNRUNNABLE.json`.

⟹ 그 런처들의 과거 결과는 현 코드로 재현할 수 없다. **판정 필요.**

## 3. census 잔여 진행 (전부 완료)

- **C7 봉인 검증기 신설** — `scripts/verify_seals.py`. 불변 봉인 12건 OK.
  ★첫 실행에서 **A2-13 무봉인 편집**을 적발했다:
  `implementation_start_manifest` V1 이 `seal_status=BLOCKED_GIT_READ_ONLY` +
  `source_edit_started=true` 인데 A2-14/15 의 V2 로 교체되며 그 기록이 트리에서 사라졌다.
  A2-13 은 A2-18 재판정에서 **폐합으로 집계**돼 있다. 판정 필요.
  증거 `validation/a2_13_15/A2_13_UNSEALED_IMPLEMENTATION.json`.
- **C1 판정** — sha256 리터럴 4개는 부채 아님(결박 대상 인접·자기검증).
  대신 **A2 게이트 기본 덱이 `_ftos` 3 · bare 3 으로 갈림** — 게이트가 풀릴 때
  어느 덱에서 판정하는지 미정.

### 기존 항목

- **C7 분류**: 재현 고아 29건을 4부류로 — `seal_with_verifier` 2 ·
  **`seal_no_verifier` 11(최대 미결)** · `measurement` 12 · `status_record` 4.
  검증기 없는 봉인은 봉인이 아니다.
  자기부채 상환: 내가 어제 인라인으로 만든 산출물에 생성 스크립트를 붙였다
  (`scripts/a2_16_launcher_debt_report.py`).
- **C3 수리**: K 게이트 음성1 을 **구성형 변조**(`npy-30col`)로 바꿨다.
  살아있는 결함 덱에 의존하던 것을 없앴다(Fable Q3-2 지적 해소). 배터리 검증 중.

## 4. T3 — ★미해결. 가설 다섯 개가 다 틀렸다

잡 225869(`_vac`)·225870(`_jnu4`)·226374(단독) 제출. **여섯 번째 실패까지 진행 중.**

세운 가설과 결과:

| # | 가설 | 결과 |
|---|---|---|
| 1 | 두 잡이 같은 저장소에서 `make` 동시 실행 | **틀림** — 사전 빌드로 고쳤으나 동일 증상 |
| 2 | footer 직후 사망 | **틀림** — footer 는 `atexit` 핸들러, 진행 표지가 아니다 |
| 3 | env 를 기억으로 다시 써서 34종 누락 | **부분 맞음** — 고치자 오류 메시지가 나오기 시작했다 |
| 4 | 폐기 노브(`LUMINA_CMF_EPAY_HOTF`) | **맞음** — 소스 유도 unset 으로 해소, 로더 통과 |
| 5 | 덱(`_vac`) 문제 | **틀림** — 두 덱이 구조적으로 동일(파일집합·형상·계약) |
| 6 | 동시 실행 GPU 경합 | **미검증** — 단독 잡 226374 로 시험 중 |

225870 은 **로더를 통과해** `Line order: DESCENDING (correct, 2220952 pairs)` 까지 갔다가
36초에 죽었다. 225869 는 52초. 둘 다 exit 1, **오류 메시지 없음**.

⚠ 로그의 `universe=482` 는 잡이 **구 `lumina_cuda`** 를 썼음을 뜻한다 —
헤더를 483 으로 재생성한 뒤 `lumina`(CPU)만 리빌드했고 CUDA 는 안 했다.
지금 리빌드하면 실행 중인 단독 잡과 경합하므로 **하지 않는다**(가설 1 의 실수 반복 금지).
단독 잡이 끝나면 리빌드 후 재시험한다.

**교훈(자기 기재)**: T3 에서 가설을 다섯 번 세우고 다섯 번 틀렸다. 매번 "고쳤다"고
보고했고 매번 증상이 남았다. 로그를 시간순으로 읽었다고 가정한 것(가설 2)과
내가 한 일을 잘못 보고한 것(가설 3, "금지 2종만 제거"라 했으나 실제로는 34종 누락)이
근원이다. 추측을 쌓지 말고 **깨끗한 단일 실험**으로 갔어야 했다.

## 5. ★이 세션의 반복 실패 (자기 기재)

**첫 목록이 짧았던 것이 열 번.** 생산 덱 오인 · 6파일만 비교 · `--deck` 배선 ·
C2 98→157 · 157→83–174 · 레지스트리 3종 누락 · 하드거부 사이트 3→4 ·
env 12→22→23 · env 418→482→483 · T3 env 50→16.

⟹ 오늘 만든 도구는 전부 **자기 수치를 못 믿게 만드는 쪽**이다:
`spread_ratio` · drift 대조기 · UNKNOWN stamp · 완전성 게이트 · ENV-SURFACE 스캔.

## 6. user 판정 대기 (우선순위)

1. **죽은 노브 1,384건** — 과거 A/B 판정에 걸린다. 대장 재해석 필요?
2. **챔피언 재현 불가 36런처** — EPAY 기전 복원 / 재현 불가 표시 / 새 기준 수립
3. **★A2-13 무봉인 편집** — A2-18 재판정에서 폐합으로 집계돼 있다.
   사후 검증 / 재봉인 / 기재만 중 판정
4. C2 함대 이관 처분(C10 때문에 재설계 필요 — 비-NLTE 런처는 사망이 아니라 no-op)
5. **게이트 4종이 어느 덱에서 판정하는가** — A2 게이트 기본 덱이 `_ftos` 3 · bare 3 으로 갈림
6. T3 착지 후 대장 v3 첫 점 기록

## 7. 다음 사람이 바로 할 수 있는 것

- `python3 scripts/verify_seals.py` — 봉인 무결성 (현재 PASS, SUPERSEDED 1건 표시)
- `python3 scripts/check_legacy_knob_registry.py` — 노브 레지스트리 drift (현재 PASS)
- `python3 scripts/instrumentation_debt_census.py` — 부채 수치 + **패턴 민감도**
- `python3 scripts/derive_env_universe.py` — env 전집 재도출(헤더는 별도 생성 필요)
- `python3 scripts/deck_provenance_audit.py` — 덱 계보 stamp 갱신
- `bash` 로 T3: `sbatch -p h200,h100 --export=NONE,T3_DECK=<덱>,PKTS=400000,NITER=12 scripts/run_t3_deck_ab.slurm`
  ⚠ 먼저 `make lumina_cuda` 를 **한 번만** 돌릴 것(잡 안에서 빌드하지 않는다)
