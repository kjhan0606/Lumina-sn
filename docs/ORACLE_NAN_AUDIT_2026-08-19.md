# NaN 함정 소급 감사 — 그리고 ★내 경보의 정정 (2026-08-19)

`ORACLE_COMPROMISED_2026-08-19.md` 의 후속. 런 0회, 오프라인 판독.

## 1. NaN 종료 런 전수 — 8개 (평가자가 센 5개보다 많다)

| 디렉터리 | NaN | `MAXCH=0.0` | 마지막 iter |
|---|---|---|---|
| `toy06_19.48d` | 3 | 2 | 65 |
| `toy06_19.48d_conv` | 12 | 3 | 49 |
| `toy06_19.48d_dc` | 5 | 3 | 5 |
| `toy06_19.48d_deepdamp` | 3 | 3 | 67 |
| `toy06_19.48d_freeze` | 3 | 3 | 69 |
| `toy06_19.48d_tame` | 8 | 3 | 67 |
| `toy06_2d` | 10 | 2 | 2 |
| `wrpop_coiii_20260728` | 12 | 3 | 49 |

`MAXCH = 0.0000000000000000` 은 **NaN 의 서명**이다(`solveba_v13.f` 의 비교가 NaN 에 전부 거짓 →
INCREASE/DECREASE 가 초기값 0 유지). ⟹ **"MAXCH ≤ 1%" 형 게이트는 이들을 완벽 수렴으로 인증한다.**
이름이 `_conv` 인 것까지 있다.

## 2. ★그러나 — 프로젝트는 이미 알고 있었고, 공식 판정을 내려 두었다

`validation/a2_00_oracle/` 의 매니페스트들이 **적격성을 명시적으로 등급화**해 두었다.

`toy06_19p48d_modern.manifest.json`:
| 항목 | 판정 | 사유(원문 요지) |
|---|---|---|
| `CMFGEN_FILE_INTEGRITY` | **PASS** | 해시·직접접근 스키마 검사 완료 |
| `CMFGEN_NONLINEAR_CONVERGENCE` | **FAIL** | 마지막 3회 최대 population 증가 **[13800, 8980, 3460]%**, 1% 초과 |
| `CMFGEN_PHYSICAL_ORACLE` | **INELIGIBLE** | nonlinear=FAIL; FIX_T evidence 동반 |
| `CMFGEN_SNAPSHOT_REPLAY` | **ELIGIBLE** | FINISH_REC=1 로 파일 재생은 허용, **cross-file 물리 게이팅은 불허** |

`toy06_19.48d_jnu4.manifest.json` 도 동일 구조이며
`finish_rec_contract.is_physical_convergence = **False**`,
statement = *"FINISH_REC proves completed EDDFACTOR writes, **not nonlinear convergence**"*.

⟹ **오라클은 "파일 무결성·재생" 으로만 인증됐고, "물리" 로는 명시적으로 부적격 판정을 받았다.**
그 판정에 쓰인 수치(13800/8980/3460%)는 오늘 내가 독립으로 추출한 것과 **동일**하다.

## 3. 내 경보의 정정

`ORACLE_COMPROMISED_2026-08-19.md` 는 "미수렴 오라클로 11일을 대조했다" 를 **새 발견**처럼 적었다.
**새 발견이 아니다.** 2026-08-04 에 이미 등급화돼 있었다.

**진짜 문제는 그것이 아니라 이것이다** — 울타리는 있었는데 **넘어갔다**:
- `CMFGEN_PHYSICAL_ORACLE = INELIGIBLE` 이고 `SNAPSHOT_REPLAY` 는 *"cross-file 물리 게이팅 불허"* 인데,
- 08-17 성분 대조는 `LINEHEAT`·`NETRATE` 를 **cross-file 로 물리 판정에 썼다**
  (흡수 1/172 · 상쇄조건 27–105 · net 비 4.37e5).
- 그 대조 산출물 어디에도 **적격성 검사 인용이 없다**.

⟹ 처분은 "오라클을 고쳐라" 가 아니라 **"부적격 판정을 무시한 인용을 회수하라"** 다.
`ORACLE_COMPROMISED` 의 이분 분리(오라클 무의존 결론은 유지 / 의존 결론은 보류)는 **그대로 유효**하며,
이제 그 근거가 "내 오늘 판독" 이 아니라 **프로젝트 자신의 08-04 판정**이다.

## 4. 부수 — `ophys` 는 계약 검사조차 통과하지 못했다

`validation/a2_00_oracle/check_ophys.stdout.txt`:
```
ERROR OPHYS_GAP MISSING_REQUIRED_FILE:NETRATE / TOTRATE / CHI_DATA / ETA_DATA / LINEHEAT
ERROR OPHYS_GAP MISSING_ATTESTATION:CMFGEN_ORACLE_ATTESTATION.json
FAIL CMFGEN_ORACLE_CONTRACT exit_code=16
```
⚠**주의**: 이 검사 시점 이후 그 파일들이 쓰였을 수 있다(현재 존재한다).
검사 시각과 파일 mtime 의 선후는 **미확인** — 그러나 **적격 인증이 없다**는 사실은 변하지 않는다.

## 5. 처분

- 8개 NaN 런: **대장 기재**. 이들을 수치 출처로 인용한 결론은 발견되지 않았다
  (인용은 build guide·recipe·postmortem 류이며 수치 공급이 아니다) — 다만 `toy06_19.48d` 계열
  146건 인용의 전수 성격 분류는 **미완**으로 남긴다.
- **신설 게이트 권고**: 수렴 판정에서 `MAXCH == 0.0` 을 **PASS 로 읽지 않는다**(NaN 서명).
  기존 게이트가 이 값을 통과시키는지 별도 확인 필요.
- **회수 대상**: 적격성 인용 없이 CMFGEN 내부장부를 물리 판정에 쓴 결론 — 08-17 성분 대조가 그것이다.
