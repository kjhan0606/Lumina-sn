# 발견 — A2-09 방출률 발행체의 **신원 해시 3/4 가 한 번도 채워지지 않는다**

날짜: 2026-08-20 · 발견 경위: DET-PHYSCMP 단의 발화점 오프라인 특정 중 부수 발견
성격: **계약 미이행**(구현이 SPEC 을 따르지 않음). 수치 오류가 아니라 **신원 위조 가능 상태**.
처분: [[feedback_wrong_values_register_quietly]] — 조용히 기재. 수리는 별도 계약.

---

## 1. 사실

`docs/SPEC_A2_09_10_V1.md:90-103` 은 `CpuEmissivityPublication` 의 신원 필드를 이렇게 규정한다:

```
atomic_model_sha256
grid_manifest_sha256
source_manifest_sha256
```

[실측] 전 저장소에서 이 셋에 **값을 계산해 넣는 코드가 없다**:

| 필드 | 선언 | 생산 코드에서 값을 쓰는 곳 | 읽는 곳 |
|---|---|---|---|
| `atomic_model_sha256` (em) | `emissivity_publication.h:26` | **없음** | — |
| `grid_manifest_sha256` | `emissivity_publication.h:26` | **없음** | `physics_comparison.c:132`(게이트) · `:419`(JSON) |
| `source_manifest_sha256` | `emissivity_publication.h:27` | **없음** — 전 저장소 출현이 **선언 한 줄뿐** | 없음 |
| `cdf_manifest_sha256` | `emissivity_publication.h:27` | ✅ `emissivity_publication.c:120` | — |

[실측] `a209_publication_init` 은 `memset(p,0,sizeof(*p))` 로 시작한다
(`emissivity_publication.c:71`), `lumina_main.c:100` 도 `OpacityState` 전체를 0 으로 민다.
⟹ 세 필드는 **결정론적으로 NUL 64개**다. 미정의 쓰레기값이 아니라 확실히 빈 문자열이다.

## 2. 왜 20일 넘게 안 드러났나

이 셋 중 **읽는 코드가 있는 것은 `grid_manifest_sha256` 하나**이고, 그 독자는
`physics_comparison.c` 뿐이다. 그리고 [실측] `/gpfs/kjhan/lumina` 전 코퍼스에서
`PHYSICS-COMPARISON` 문자열을 가진 `stderr.log` 은 **2026-08-19 L4 런 하나**다.

⟹ **읽는 자가 한 번도 실행되지 않았으므로 빈 필드가 한 번도 문제를 일으키지 않았다.**
결정론 팔이 처음으로 T_e 세대를 커밋하자마자 이 게이트에 닿았고, 즉시 죽었다.

## 3. ★시험이 결함을 우회하고 있었다

[실측] 이 필드를 검사하는 게이트에는 음성 대조가 붙어 있다. 그런데 픽스처가
**생산 경로에 없는 값을 손으로 합성해 넣는다**:

- `tests/physics_comparison_selftest.c:75` — `fill_hash(em.grid_manifest_sha256,'b');`
- `tests/physics_comparison_regrid_selftest.py:78` — `"grid_manifest_sha256":"d"*64`
- `tests/det_convergence_selftest.py:104` — `"grid_manifest_sha256": "c" * 64`

⟹ **시험은 언제나 통과하고 생산은 통과할 수 없다.** 게이트의 문언은 옳았고 주입 지점이
생산 경로 밖이었다. [[feedback_audit_the_yardstick_first]] 계열이며, 이 프로젝트가 기록해 온
잣대 사고에 하나를 더한다: **픽스처가 생산자를 대신 채우면 그 필드의 생산자 부재는
영원히 보이지 않는다.**

## 4. 무엇이 위태로운가 (수치가 아니라 **귀속**이다)

이 셋은 "이 방출률이 어떤 원자모형·어떤 주파수 격자·어떤 소스항에서 나왔는가" 를 봉인하는
필드다. 비어 있으면:

- 두 스냅샷이 **다른 격자**에서 나왔는데 같다고 비교될 수 있다(격자 신원 봉인 부재)
- `physics_comparison` 이 산출할 JSON 의 `grid_manifest_sha256` 은 **빈 문자열**이 된다 —
  즉 게이트가 통과하도록 필드만 채우면 **비교 산출물이 신원 없는 채로 나간다**
- MC 팔과 DET 팔의 대조에서 "같은 격자였다" 를 **증명할 수단이 없다** — 이 캠페인의
  차동 잣대가 기대는 바로 그 전제다

⚠**따라서 수리는 "hex64 를 통과시키는 아무 값"이 아니다.** 실제 `nu_edge` 배열과
원자모형·소스항에서 유도한 해시여야 한다. 통과만 시키는 수리는 클램프와 같은 성질의
거짓 통과다.

## 5. 인접 의심 (미확인 — 이 문서의 주장 아님)

같은 크루드 스캔에서 writer 0 으로 나온 필드가 더 있다. **확인하지 않았다.**

- `src/jnu_seed.h`: `source_geometry_sha256` · `source_payload_sha256` — `.c/.cu` 출현 0줄
  (단 `shape_sha256`/`edge_sha256` 은 `jnu_seed.c:31` 에서 **읽힌다** ⟹ 같은 구조의
  '읽기만 하고 쓰지 않는' 짝일 가능성)

별도 감사 항목으로 남긴다. **여기서 단정하지 않는다.**

## 6. 처분

- 이 발견 자체는 **기재로 끝난다**. 수리는 별도 계약 = 별도 커밋
  ([[feedback_one_contract_one_commit]]).
- DET-PHYSCMP 단은 **측정 단**이며 이 수리를 포함하지 않는다.
- 수리 단을 세울 때의 계약: "A2-09 발행체는 자기 격자·원자모형·소스항의 해시를 스스로
  계산해 봉인한다", 음성 대조는 **생산 경로에서** 격자를 바꿔 해시가 바뀌는 것을 보여야 한다
  (픽스처 주입 금지 — 그것이 이 결함을 숨긴 방법이다).
- `docs/CLASSIC_DEBT_CENSUS.md` 와 `docs/VERIFICATION_REGISTERS.md` 의 B(검증대상)에 올린다.
