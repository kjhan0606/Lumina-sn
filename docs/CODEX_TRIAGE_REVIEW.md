# 독립 검증 판정

요약하면:

| 항목 | 판정 | 핵심 |
|---|---|---|
| E8 재순환 이득 폐합 | (c) 혼합 | `J_ours/CMFGEN`은 독립 비교지만, 두 gain은 `S_fixed`를 공유하고 `S_total` 자체가 `J_ours`의 함수다. 0.00152%는 독립 인과 검증이 아니다. |
| nonpositive 28,949행 | (c) 혼합 | 28,949와 이온별 순위는 캡처 내부의 데이터 의존 census지만, `outside_BALL=0`은 writer가 BALL만 기록했기 때문에 생기는 항등식이다. |
| N9 에너지 가중 분율 | (c) 혼합 | shell-8의 0.99563은 post-EPAY 출력의 데이터 의존 비율이지만 “폐기된 pre-EPAY `eta_line`” 비율이 아니다. B1–B4의 정확한 1은 마스크가 전 셀을 포함한 뒤의 항등식이다. |

추가 provenance 제약이 있다. 캡처 manifest는 payload SHA-256은 봉인하지만, 실행 바이너리를 정확한 source commit/hash에 결박하지 않는다. 아래는 현재 `src/`와 캡처 스키마가 일치하는 production 경로 판정이며, 캡처 바이너리와 현재 dirty source의 bitwise 동일성은 `UNRESOLVED`다.

## 항목 1 — E8 재순환 이득 폐합

### 1. production에서 실제로 계산되는 양

production이 직접 보유하는 per-cell 양은 `S_fixed`, `J`, `chi_tot`, `chi_es`다.

- 선 opacity와 열적 emissivity는 선별로 `w`, `eps_l`, `S_l`을 만들어 `chi_line`, `chi_line_th`, `eta_line`에 누적한다. [lumina_cmfgen.c:1354](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1354), [lumina_cmfgen.c:1369](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1369), [lumina_cmfgen.c:1375](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1375), [lumina_cmfgen.c:1381](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1381)
- `eps_l`의 production 정의는 `C_ul/(C_ul+A_ul beta_esc)`다. [lumina_plasma.c:8442](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:8442), [lumina_plasma.c:8471](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:8471)
- 조립기는
  `chi_es = chi_e + (chi_line-chi_line_th)`와
  `S_fixed=(chi_abs B+eta_line)/chi_tot`
  를 만든다. [lumina_cmfgen.c:1641](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1641), [lumina_cmfgen.c:1644](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1644)
- EPAY가 켜진 셀은 이 `S_fixed`를 rate-shape 또는 scalar 경로로 다시 쓴다. [lumina_cmfgen.c:1689](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1689), [lumina_cmfgen.c:1704](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1704), [lumina_cmfgen.c:1720](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1720)
- transfer source는 production에서 직접
  `S = S_fixed + (chi_es/chi_tot) J`
  로 계산된다. [lumina_cmf_solve.cu:115](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_solve.cu:115), [lumina_cmf_solve.cu:117](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmf_solve.cu:117)
- `J`는 이 source를 쓴 formal solve 출력이고, 이후 선택적으로 damping된다. [lumina_cuda.cu:8078](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:8078), [lumina_cuda.cu:8097](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:8097)

dump는 다음을 독립 측정하지 않고 같은 배열에서 조립한다.

```text
eta_fixed    = chi_tot * S_fixed
eta_coherent = chi_es  * J
eta_total    = eta_fixed + eta_coherent
```

근거는 [lumina_cuda.cu:8183](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:8183), [lumina_cuda.cu:8186](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:8186), 그리고 실제 직렬화 [lumina_cmfgen.c:340](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:340), [lumina_cmfgen.c:344](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:344), [lumina_cmfgen.c:348](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:348)이다.

따라서 발표된 band 양은 production 변수가 아니라 오프라인 투영이다.

```text
S_fixed,band = Σ[(eta_fixed/chi_tot) Δν]
S_total,band = Σ[((eta_fixed+eta_coherent)/chi_tot) Δν]
J_ours,band  = Σ[J Δν] / ΣΔν
eps_eff      = S_fixed,band / S_total,band
```

그 계산은 [emiss_e8_recycling.py:328](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/emiss_e8_recycling.py:328), [emiss_e8_recycling.py:334](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/emiss_e8_recycling.py:334), [emiss_e8_recycling.py:339](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/emiss_e8_recycling.py:339), [emiss_e8_recycling.py:344](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/emiss_e8_recycling.py:344)에 있다.

### 2. 두 gain의 공유 인자와 항등식

발표된 두 gain은 다음이다.

```text
required gain = (J_ours/CMFGEN)/(S_fixed/CMFGEN)
              = J_ours/S_fixed

measured gain = S_total/S_fixed
              = 1/eps_eff
```

따라서:

- CMFGEN은 required gain에서 완전히 소거된다.
- 두 gain은 동일한 `S_fixed` 분모를 공유한다.
- `measured gain=1/eps_eff`는 정의상 항등식이다.
- `required eps=S_fixed/J_ours`와 `measured eps=S_fixed/S_total`도 같은 분자를 공유한다.

둘의 상대 일치는 정확히 다음 한 문장으로 축약된다.

```text
measured_gain / required_gain = S_total / J_ours = 1.00001521
```

실제로 스크립트도 같은 방식으로 required gain과 `S_total/J`를 만든다. [emiss_e8_recycling.py:383](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/emiss_e8_recycling.py:383), [emiss_e8_recycling.py:414](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/emiss_e8_recycling.py:414), [emiss_e8_recycling.py:421](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/emiss_e8_recycling.py:421)

`S_total/J≈1`은 완전한 항등식은 아니다. transfer가 비국소적이므로 일반적으로 local `J=S`가 강제되지는 않는다. 그러나 여기서 `S_total=S_fixed+(chi_es/chi_tot)J`이므로 `S_total`은 이미 같은 `J`를 포함한다. 산란 albedo가 1에 가깝고 `S_fixed/J`가 작으면 `S_total/J≈1`은 구성상 매우 쉽게 나온다. 이는 fixed-point/source 자기일관성 점검이지 독립된 인과 실험이 아니다.

### 3. 유효하게 남는 결론과 폐기할 결론

유효하게 남는 것:

- `J_ours/CMFGEN=11.9771`은 서로 다른 Lumina와 CMFGEN field 경로의 비교다. CMFGEN 입력 provenance가 맞다는 전제 아래 독립 측정이다.
- s8 BALL에서 post-EPAY fixed source가 매우 작고 coherent-return 항이 source를 지배한다는 decomposition은 유효하다.
- `eps_eff=1.90567e-4`는 “band total source 중 fixed source의 비율”로 유효하다.
- `S_total/J≈1`은 해당 iteration의 local source와 solved field가 가까움을 보여주는 수치적 자기일관성 지표다.

폐기해야 하는 것:

- `eps_eff`를 독립적으로 측정된 미시적 파괴확률이라고 부르는 것.
- `1/eps_eff`와 `S_total/S_fixed`의 일치를 검증이라고 부르는 것. 둘은 같은 정의다.
- 0.00152% 일치가 11.977배 과잉의 원인을 “완전히 설명”했다는 결론. 실제 비교는 `J`를 포함해 만든 `S_total`과 그 `J` 자체의 비교다.
- CMFGEN equivalent epsilon이나 그 배율을 이 폐합에서 역산하는 것.

### 4. 4억 cap의 전파

cap은 event file에만 적용된다. 각 event는 atomic reservation index를 받고, index가 cap 이상이면 drop된다. [lumina_cuda.cu:4677](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4677), [lumina_cuda.cu:4688](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4688) 실제 파일도 `min(event_count,cap)`만 쓴다. [lumina_cuda.cu:9018](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:9018), [lumina_cuda.cu:9058](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:9058)

따라서 cap이 전파되지 않는 수치는:

- `S_fixed/CMFGEN`
- `J_ours/CMFGEN`
- required/measured gain과 두 epsilon
- `S_total/J_ours`
- emissivity payload 기반 coherent fraction
- 별도 `lumina_ma_line_destruct.csv` 기반 thermal destruction rate

cap이 직접 전파되는 것은 event pairing에서 얻은 다음 수치다.

- 1,856,667 absorption-terminal 쌍
- same-line/different-line 분율
- same/different coarse-bin 분율
- 95.1164% coarse-bin coherence destruction

이들은 atomic reservation 순서의 400,000,000-record prefix뿐이다. 무작위 표본이 아니므로 전체 iteration에 대한 불편추정량으로 취급할 수 없다. 반면 per-shell census는 cap 검사 전에 누적된다. [lumina_cuda.cu:4682](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cuda.cu:4682)

**항목 1 판정: (c) 혼합 — Lumina/CMFGEN field 비는 독립 측정이지만, gain/epsilon 폐합과 0.00152% 일치는 공유 인자 및 `S_total(J)`에 의존한 대수·fixed-point 자기폐합이다.**

## 항목 2 — “nonpositive 28,949행이 전부 BALL 내부”

### 1. LCMFLP01 writer의 실제 기록 술어

writer는 먼저 선택 자체를 제한한다.

- shell은 `LUMINA_CMF_LINEPOP_SHELLS`의 명시 목록만 허용한다. [lumina_cmfgen.c:543](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:543), [lumina_cmfgen.c:557](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:557)
- 파장 창 기본값은 600–3000 Å이고 환경변수로 정한다. [lumina_cmfgen.c:545](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:545), [lumina_cmfgen.c:578](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:578)

행 기록 술어는 다음의 conjunction이다.

```text
shell ∈ selected_shells
tau_sobolev > 1e-12
nu_min < nu_line < nu_max
computed bin in [0, NB)
600 Å <= lambda_line <= 3000 Å
```

pass-1 count는 [lumina_cmfgen.c:731](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:731), [lumina_cmfgen.c:736](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:736), [lumina_cmfgen.c:742](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:742)에 있고, 실제 row serialization도 같은 선택을 거친다. [lumina_cmfgen.c:776](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:776), [lumina_cmfgen.c:804](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:804), [lumina_cmfgen.c:862](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:862)

따라서 LCMFLP01 row table에는 애초에 BALL 밖 행이 존재할 수 없다.

### 2. `outside_BALL_rows: 0`의 정보량

오프라인 bad 술어는 다음이다.

```text
tau_from_pops <= 0
or n_lower <= 0
or n_upper <= 0
or S_l_pop <= 0
```

[uv_t2n9_offline.py:600](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/uv_t2n9_offline.py:600)

그 뒤 같은 row의 wavelength를 다시 600–3000 Å로 검사한다. [uv_t2n9_offline.py:625](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/uv_t2n9_offline.py:625)

그러므로 `outside_BALL_rows=0`은 writer 선택 술어를 재확인한 항등식이다. payload corruption이나 reader layout 오류를 잡는 sanity check 정도의 정보량은 있지만, BALL 밖 production population에 관한 정보는 0이다.

말할 수 있는 것:

- 선택 shell `[0,8,16,20,45]`과 600–3000 Å, `tau>1e-12`인 캡처 row 1,169,145개 중 28,949개가 bad 술어를 만족한다.
- 이 28,949개는 모두 미정의 `-1` population sentinel이고, 실제 mapped solver-negative row는 0으로 집계됐다.
- 이 조건부 집합 안에서 원소·이온별 row 수와 recorded-A opacity 순위를 낼 수 있다.

말할 수 없는 것:

- BALL 밖에는 nonpositive row가 없다는 것.
- 선택되지 않은 45개 shell에도 없다는 것.
- `tau<=1e-12`로 writer가 누락한 선에도 없다는 것.
- 28,949가 실제 음수 population solver 결함이라는 것. lookup 실패 시 writer가 `-1`을 쓴다. [lumina_cmfgen.c:811](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:811), [lumina_cmfgen.c:815](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:815)
- row 수만으로 에너지·opacity 영향이 크다는 것. 한 row는 한 line-shell entry이며 고유 선, 셀 또는 에너지 단위가 아니다.

### 3. 28,949와 이온별 순위의 판정

28,949 자체는 단순 항등식이 아니다. 캡처된 row의 population mapping과 값에 따라 달라지는 실제 census다. 그러나 캡처 모집단이 writer에 의해 BALL와 5개 shell로 잘려 있으므로 전 production 모집단 측정도 아니다. 즉:

- 측정 부분: 어떤 captured row가 bad인지, 그 수가 28,949인지, 이온별로 몇 개인지.
- 항등식 부분: 그 28,949개가 전부 BALL 안이라는 것.
- 미관측 부분: BALL 밖과 나머지 shell의 bad population.

또한 “Fe III 5,924행”이라는 이온명은 코드 convention상 틀렸다. 출력은 `Z=26, ion=3`이다. 이 코드베이스는 `ion=1`을 II로 사용하고 [lumina_atomic.c:670](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:670), Fe III를 명시적으로 `(26,2)`로 확인한다. [lumina_atomic.c:1287](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:1287) 따라서 5,924행은 **Fe IV**, 다음 5,662/5,571행도 각각 Ni IV/Co IV다.

**항목 2 판정: (c) 혼합 — 28,949와 조건부 이온 순위는 캡처 내부의 데이터 의존 census이고, `outside_BALL=0`은 BALL-only writer 선택에서 나오는 항등식이다.**

## 항목 3 — N9 에너지 가중 분율

### 1. production EPAY와 `eta_fixed` 조립 경로

pre-EPAY line emissivity는 per-line `w*eps_l*S_l`의 합이다. [lumina_cmfgen.c:1375](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1375), [lumina_cmfgen.c:1382](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1382)

첫 pass의 fixed source는:

```text
S_fixed_pre = (chi_abs*B + eta_line_pre)/chi_tot
```

[lumina_cmfgen.c:1644](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1644)

얇은 bin에 대해 EPAY 장부는:

```text
acc_emit += (chi_abs*B + eta_line_pre) dnu
acc_abs  += (chi_abs + chi_line_th) J_lagged dnu
acc_w    += (eta_bf_Milne + chi_line_th B) dnu
acc_dep  += kappa_dep B dnu
```

[lumina_cmfgen.c:1663](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1663)

실제 rate-shape branch는 정확히:

```c
if (epay >= 2 && acc_w > 0.0 && hot_regime)
```

이며, 여기서

```text
wn = (acc_abs+acc_dep)/acc_w
S_fixed_post = wn*(eta_bf_Milne + chi_line_th*B)/chi_tot
```

로 pre-EPAY `eta_line` 형상을 덮어쓴다. [lumina_cmfgen.c:1704](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1704), [lumina_cmfgen.c:1708](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1708), [lumina_cmfgen.c:1720](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:1720)

최종 dump의 `eta_fixed`는 이 post-EPAY `chi_tot*S_fixed`다. [lumina_cmfgen.c:342](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:342)

### 2. disposition writer는 실제 branch를 완전히 기록하지 않는다

LCMFLP01 writer는 actual branch site에서 disposition을 쓰지 않는다. 나중에 다음으로 재구성한다.

```text
thick ? thick_exempt
      : (epay>=2 && hot ? rate_shape_replaced : scalar_rescaled)
```

[lumina_cmfgen.c:904](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:904), [lumina_cmfgen.c:912](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:912)

여기에는 actual branch의 `acc_w>0`이 없다. 따라서 허용된 `acc_w==0` 상태에서는 production은 scalar branch를 타지만 writer는 rate-shape라고 기록한다.

결과적으로:

- `34,304/45,000=0.7623111`은 writer가 재구성한 술어의 정확한 census다.
- 이것이 실제 branch-site rate-shape count와 같다는 것은 현재 artifact로는 `UNRESOLVED`다.
- manifest count와 offline count의 일치는 같은 writer loop가 만든 count와 disposition bytes를 두 방식으로 다시 센 것이다. serialization integrity 검사이지 독립적인 branch 검증이 아니다.

### 3. post-EPAY energy definition이 실제로 재는 것

오프라인 코드는 dump의 `eta_fixed_post`에 frequency overlap과 volume을 곱하고, disposition==2 마스크 안의 합을 전체 합으로 나눈다. [uv_t2n9_offline.py:461](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/uv_t2n9_offline.py:461), [uv_t2n9_offline.py:465](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/uv_t2n9_offline.py:465), [uv_t2n9_offline.py:493](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/uv_t2n9_offline.py:493)

이는 다음 질문에는 맞다.

> “최종 post-EPAY fixed emissivity 중 writer가 rate-shape로 분류한 셀에 위치한 몫은 얼마인가?”

그러나 다음 질문에는 맞지 않는다.

> “조립된 pre-EPAY `eta_line` 에너지 중 EPAY가 폐기한 몫은 얼마인가?”

이유는 세 가지다.

- 가중량이 폐기된 `eta_line_pre`가 아니라 덮어쓴 뒤의 `eta_fixed_post`다.
- `eta_fixed_post`에는 선뿐 아니라 bf/ff continuum와 deposition도 포함된다.
- disposition 자체도 실제 branch의 `acc_w>0` 조건을 보존하지 않는다.

따라서 “조립된 eta가 UV 선 방출 에너지의 99.563%에서 폐기됐다”는 문장은 현재 수치가 측정한 대상을 넘어선다. 정확한 폐기율에는 pre-EPAY line emissivity를 분모로 하고 actual rate-shape branch가 폐기한 동일 line emissivity를 분자로 써야 한다.

### 4. pre/post 선택이 답을 얼마나 바꾸는가

현재 확정된 값은 post-EPAY 정의뿐이다.

- shell 8 BALL: `0.9956303809148374`
- s≥5 B1–B4: 각 shell에서 `1.0`

pre-EPAY 수치는 보고되지 않았다. LCMFLP01에 pre-EPAY `eta_line`이 저장된 shell은 `[0,8,16,20,45]`뿐이므로 shell 8의 pre/post 차이는 추가 산술로 낼 수 있지만, s≥5 전 45 shell의 pre-EPAY 비율은 현재 캡처로 계산할 수 없다. 따라서 정량 변화는:

- shell 8: `UNRESOLVED-PENDING-READONLY-ARITHMETIC`
- s≥5 전 shell: `UNRESOLVED-NOT-CAPTURED`

요청 규율에 따라 실행하지 않은 shell-8 비교 명령은 한 줄이다.

`cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn && python3 -c 'from pathlib import Path; import numpy as np; import scripts.uv_t2n9_offline as u; lp=u.parse_linepop(Path("/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/linepop_iter10")); a=u.check_artifact(Path("/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/emiss_ab_iter10.A")); e,_,_=u.bench.canonical_grid(); post=np.asarray(a.arrays[5]).reshape(50,1000)[:,::-1]; k=list(map(int,lp.shells)).index(8); d=lp.disposition[8]==2; print({n:{"pre":float(np.sum(lp.eta_line[k]*u.bench.band_weights(e,lo,hi)*d)/np.sum(lp.eta_line[k]*u.bench.band_weights(e,lo,hi))),"post":float(np.sum(post[8]*u.bench.band_weights(e,lo,hi)*d)/np.sum(post[8]*u.bench.band_weights(e,lo,hi)))} for n,lo,hi in u.BANDS})'`

이 명령도 기존 두 payload를 조합하는 오프라인 산술이지 독립 production 경로는 아니다.

### 5. B1–B4의 정확한 1.0000000

캡처에서 s≥5 B1–B4의 모든 bin이 disposition 2로 기록됐다. 그러면 오프라인 식은 같은 `emitted` 배열에 대해:

```text
numerator   = sum(emitted[all cells])
denominator = sum(emitted[all cells])
```

가 되므로 정확히 1이다. [uv_t2n9_offline.py:493](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/uv_t2n9_offline.py:493), [uv_t2n9_offline.py:496](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/uv_t2n9_offline.py:496)

분해하면:

- B1–B4의 모든 셀이 thin/replaced 마스크가 된 것은 opacity, shell 두께, threshold에 따른 데이터 의존 결과다.
- 그 조건이 성립한 뒤 에너지 분율이 정확히 1인 것은 항등식이다.
- 다만 이 mask가 actual branch와 정확히 같은지는 `acc_w` 누락 때문에 미해결이다.

### 6. 에너지 분율의 독립 재계산 경로

현재는 존재하지 않는다.

- LCMFLP01은 disposition과 pre-EPAY replay를 만든다.
- LCMFCE01/A payload는 post-EPAY `eta_fixed`를 만든다.
- 오프라인 N9는 이 둘을 결합한다.
- manifest와 CSV는 같은 disposition producer를 재소비할 뿐이다.

독립 경로를 만들려면 actual EPAY branch 내부에서 직접 다음을 계측해야 한다.

1. actual branch enum을 `acc_w>0`까지 포함해 branch site에서 기록한다.
2. branch 진입 직전 `eta_line_pre*dnu`를 disposition별로 누적한다.
3. branch 후 `chi_tot*S_fixed*dnu`도 별도 누적한다.
4. shell volume과 정확한 band-edge overlap을 적용한 numerator/denominator 및 closure residual을 production sidecar에 봉인한다.
5. 이 production counter와 현재 offline artifact join을 비교한다.

**항목 3 판정: (c) 혼합 — 0.99563은 post-EPAY fixed-output 위치의 데이터 의존 비율이고, B1–B4의 정확한 1은 전 셀 마스크 뒤의 항등식이다. pre-EPAY 폐기율과 actual branch count는 현재 계측으로 UNRESOLVED다.**

코드·문서·산출물은 수정하지 않았고, 모델/GPU/무거운 계산 및 위 추가 명령은 실행하지 않았다.