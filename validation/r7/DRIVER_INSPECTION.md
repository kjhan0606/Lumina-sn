# R7 코드 검사 — 운전석 (2026-08-08)

분담 개정10: 코딩=Codex, **검사=운전석**.  이 문서가 그 검사다.
대상 `validation/r7/CODEX_R7.md`(839행, gpt-5.6-sol effort high, 00:15 착지).

기계 검사(`scripts/r7_apply_and_check.sh`)는 통과했다 — 새 `getenv` 0, clamp/floor 0.
기계가 "세대 술어 변경 15건" 으로 센 것은 **전부 `+` 줄**, 즉 검사를 지운 것이 아니라
새로 **추가**한 fail-closed 단언이다.  계약 완화의 반대 방향이다.

아래는 기계가 볼 수 없는 것 — 의미론, 범위, 소비-생산 관계 — 을 읽은 결과다.

---

## 1. ★범위 위반 1건 — 분리했다

### 결정론 팔의 감마 침착이 **한 번도 계산된 적이 없다**

패치가 `lumina_cmfgen.c` 반복 앞에 다음 두 줄을 **신설**한다.

```c
if (gamma)
    compute_gamma_deposition(gamma, atom, plasma, geo);
```

이것은 발행 위상 수리가 아니라 **누락된 물리 항의 신설**이다.  분리해서 적용하지 않았다
(`/tmp/claude-10396/r7_apply/part_03_scoped.patch`).  근거와 사실관계는 아래.

**사실 (전부 실측)**

| 확인 | 결과 |
|---|---|
| `compute_gamma_deposition` 호출부 전수 | `lumina_main.c:603` **단 하나**. `lumina_cmfgen.c` 에는 없다 |
| 그 603 행의 도달성 | MC 반복 루프 안. DET 분기는 `lumina_main.c:338` 에서 `cmfgen_run` 을 부르고 **370 행에서 `return 0`** — 603 에 도달하지 않는다 |
| `gamma_deposition_init` (`lumina_plasma.c:15775`) | `heating_rate = calloc(...)` ⟹ **전 셸 0** |
| A2-10 이 그 항을 쓰는 곳 (`lumina_plasma.c:12060`) | `double qg=(c->gamma&&c->gamma->heating_rate)?c->gamma->heating_rate[s]:0;` |
| A2-10 이 결측을 잡는가 | 아니다. `blocked_missing_term` 은 **J 재빈닝 실패만** 센다 |

⟹ **결정론 팔의 복사평형은 방사성 가열 항 `q_γ ≡ 0` 으로 풀려 왔다.**
19.48일 Ia 초신성에서 국소 에너지원은 ⁵⁶Ni→⁵⁶Co→⁵⁶Fe 붕괴가 사실상 전부다.

**왜 아무도 몰랐나** — 이것이 「고리 밖 감사」의 표적 그 자체다.
감마 침착은 A2-10 이 **소비하되**, 결정론 팔에는 **생산자가 없다**.
그리고 감마 침착에는 **세대 도장이 없다** — 그래서 GEN-GUARD 도, A2-10 의
동세대 삼중항 검사도, 이것을 보지 못한다.  삼중항은 opacity·emissivity·radiation 만 본다.

**아직 주장하지 않는 것**(잣대 감사):
- 이것이 심부 T_e 결손(캠페인 확정 사슬의 "가스가 자기 욕보다 2000-2600K 참")의
  원인인지는 **측정 전이다.**  내부 경계 광도가 총 에너지를 이미 나르고 있으므로
  틀린 것은 **국소 침착 분포**이지 총량이 아닐 수 있다.
- `TE_TABLE` 로 T_e 를 핀한 twin 런들은 radeq 를 안 썼으므로 **영향 밖**이다.

**처분**: 별도 단으로 세운다(사전등록 + 음성대조 + 기대 변경집합).
그래야 감마의 효과를 **단독 귀속**으로 잴 수 있다.  R7 에 섞으면 영영 못 잰다.

---

## 2. 확인한 것 — 위상 이동이 입력을 바꾸지 않는다

패치는 A2-10 을 반복 **앞쪽**(복사장 commit 직후)으로 옮긴다.  옮기면 입력이 달라지는지
전수 확인했다.

| 의심 | 실측 | 판정 |
|---|---|---|
| A2-10 이 `nlte->J_nu` 를 읽는가? 옮기면 정규화 전 값을 먹는다 | `a210_production_solve` 는 `a210_rebin_checked_J(&nlte->radfield_view, ...)` — **정본 view 에서 재빈닝**한다. `J_nu` 를 읽지 않는다 | 무해 |
| a208/a209 는? | 둘 다 `J_nu`·`j_nu_estimator` 참조 **0건** | 무해 |
| 삭제된 `nlte_normalize_j_nu`(624행)가 필요한가 | 656행의 같은 호출이 같은 게이트로 남아 있고, 그 사이 구간에 `J_nu` 소비자가 없다 | 무해 |
| MC 감마를 수송 **앞**으로 올린 것 | 옛 위치도 새 위치도 입력 plasma 는 **직전 반복이 commit 한 같은 상태**다(수송은 plasma 를 변경하지 않는다) | 동등 |
| `radeq_set_line_re_source` 를 5317→5195 로 올린 것 | 이 함수는 **포인터를 저장**한다(`g_lre_chi_line_full = chi_line_full`). 값 복사가 아니므로 시점 무관. 게다가 그 사이 블록은 `Jsave` 로 `cs.J` 를 저장→변경→**정확히 복원**한다 | 동등 |
| 삭제된 `te_publication.{population,opacity,emissivity}_generation` 대입 | `lumina_plasma.c:12108-12110` 이 A2-10 트랜잭션 **안에서** 이미 같은 값을 찍는다 | 중복 제거 |

---

## 3. 기재만 하는 것 (수리 아님)

### 3-1. `enable_nlte=0` · `bf_opacity_enabled=0` 런이 **불가능해졌다**

R7 헬퍼의 입구:

```c
if (!opacity || !bf || !atom || !plasma || !nlte || !plasma->T_e || n_shells <= 0)
    ... return 5;   /* R7_INVALID_PHASE_INPUT */
```

MC 호출부는 `bf_opacity_enabled ? &bf : NULL`, `enable_nlte ? &nlte : NULL` 을 넘긴다.
⟹ 둘 중 하나라도 끄면 즉시 치명.

**이것은 정직한 fail-closed 다** — a209 는 BF 격자와 `eta_bf` 를 요구하고, 정본 복사장
view 는 nlte 구조체 안에 산다.  없으면 동세대 삼중항이 **원리적으로** 성립하지 않는다.
다만 **돌던 설정이 안 도는 것으로 바뀌었다**는 사실은 대장에 남긴다.
생산 런은 full-NLTE + BF 이므로 실무 영향 없음.

### 3-2. 새 침묵 경로 3건

`compute_radiative_equilibrium_te` 래퍼가 메시지 없이 `return 0` 하는 지점이 셋이다.

```c
if (!plasma || plasma->T_e_generation == UINT64_MAX) return 0;   /* 침묵 */
if (!qualified) return 0;                                        /* a210 이 대신 말함 */
if (plasma->te_publication.committed_te_generation != old_generation + 1)
    return 0;                                                    /* ★침묵 — 이게 문제 */
```

세 번째가 진짜다: a210 이 **성공했는데** 세대가 `old+1` 이 아니면 조용히 실패로 바꾼다.
호출부는 `r7_a210_block_reason()` 의 마지막 폴백 `RADEQ_UNQUALIFIED_TE` 를 찍게 되어
**원인이 오표기**된다.  침묵 금지 규약 위반.  R6 발주에 함께 실어 보낸다.

### 3-3. `radeq_te` OFF 경로의 세대 의미가 바뀌었다

옛 코드: radeq 를 안 쓰면 `plasma.T_e_generation = 0` (⟹ 하류 계약이 전부 막힘).
새 코드: 그 대입이 사라져 **seed 세대 1 이 유지**된다.

L1-1 이 seed 발행자를 세운 뒤로는 **새 쪽이 정직하다** — T_e 가 갱신되지 않았다면
그 T_e 는 실제로 seed 가 발행한 세대 1 이 맞다.  다만 "radeq OFF 는 죽는다" 가
"radeq OFF 는 T_e 고정으로 돈다" 로 바뀐 것이므로 기재한다.

---

## 4. 전달 형식의 결함 1건

발주서는 "아래 unified diff 에는 생략 표기가 없으며 **그대로 적용할 수 있다**" 고 적었으나
**적용되지 않았다**.

- `@@` 행 카운트가 손으로 쓴 값이라 부정확 → `git apply --recount` 필요
- `lumina_cmfgen.c` 삭제 블록에서 주석 **6줄이 누락** → 문맥 불일치로 거부

기준선 자체는 정확했다(스테이징 src 4파일 md5 = 저장소 src 와 동일).  누락분을 복원해
적용했다.  다음 발주부터 **패치는 `git apply --check` 를 통과한 것만 제출**을 요건에 넣는다.

---

## 5. 적용한 것

```
part_00 lumina.h          선언
part_01 lumina_plasma.c   lumina_r7_publish_and_solve_te() 신설 + R8 세대 소유권 래퍼
part_02 lumina_main.c     MC lane 재배선
part_03 lumina_cmfgen.c   DET lane 재배선  (★감마 신설 2줄 제외)
```

빌드 OK(OMP 심볼 확인), 판정 런 `scripts/r7_verdict_chain.sh` 진행 중.

**사전등록된 기대**:
- DET: a209 에서 차단 — R7 실패가 아니라 R6 경계.
  ★R7 성립의 증거는 **차단 지점의 이동**이다.
  수리 전 `A2-10 blocked_stale`(자격 실패, 원인 불명) → 수리 후 `A2-09 blocked_stale_line`(원인 지목).
- MC: 전 위상 성립 `o=e=r`, `t:1→2`.  **이쪽이 진짜 판정이다.**
