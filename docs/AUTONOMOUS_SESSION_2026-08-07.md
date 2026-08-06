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

## 3. census 잔여 진행

- **C7 분류**: 재현 고아 29건을 4부류로 — `seal_with_verifier` 2 ·
  **`seal_no_verifier` 11(최대 미결)** · `measurement` 12 · `status_record` 4.
  검증기 없는 봉인은 봉인이 아니다.
  자기부채 상환: 내가 어제 인라인으로 만든 산출물에 생성 스크립트를 붙였다
  (`scripts/a2_16_launcher_debt_report.py`).
- **C3 수리**: K 게이트 음성1 을 **구성형 변조**(`npy-30col`)로 바꿨다.
  살아있는 결함 덱에 의존하던 것을 없앴다(Fable Q3-2 지적 해소). 배터리 검증 중.

## 4. T3 — GPU 큐 대기

잡 225869(`_vac`) · 225870(`_jnu4`) PENDING. 타 사용자 18잡 대기 중.
네 번의 실패를 거쳐 원인 3개를 확정했다:
1. 두 잡이 같은 저장소에서 `make` 동시 실행 → 사전 빌드로 수리
2. env 를 참조 런처에서 **기억으로 다시 씀**(50종 중 16종만) → 소스 상속으로 수리
3. `LUMINA_CMF_EPAY_HOTF` 등 폐기 노브 → 소스 유도 unset 으로 수리

## 5. ★이 세션의 반복 실패 (자기 기재)

**첫 목록이 짧았던 것이 열 번.** 생산 덱 오인 · 6파일만 비교 · `--deck` 배선 ·
C2 98→157 · 157→83–174 · 레지스트리 3종 누락 · 하드거부 사이트 3→4 ·
env 12→22→23 · env 418→482→483 · T3 env 50→16.

⟹ 오늘 만든 도구는 전부 **자기 수치를 못 믿게 만드는 쪽**이다:
`spread_ratio` · drift 대조기 · UNKNOWN stamp · 완전성 게이트 · ENV-SURFACE 스캔.

## 6. user 판정 대기 (우선순위)

1. **죽은 노브 1,384건** — 과거 A/B 판정에 걸린다. 대장 재해석 필요?
2. **챔피언 재현 불가 36런처** — EPAY 기전 복원 / 재현 불가 표시 / 새 기준 수립
3. **`seal_no_verifier` 11건** — 봉인에 검증기 부여
4. C2 함대 이관 처분(C10 때문에 재설계 필요)
5. T3 결과 착지 후 대장 v3 첫 점 기록
