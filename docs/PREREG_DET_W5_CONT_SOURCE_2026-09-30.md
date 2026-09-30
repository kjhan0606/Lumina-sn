# 사전등록 — DET-W5-CONT-SOURCE: fine 연속 방출 항을 bin 소비 Milne 원천에 맞춘 단일 변경의 인구 응답 (2026-09-30, 기안)

기안자 = astra, 개정25-r의 이번 발주에 따른 **단계 설계·사전등록 기안**. 구현·발주·실행 승인·판정이 아니다. 산출은 이 마크다운 본문이며 저자는 파일을 쓰지 않았다. 운전자 파일화 목적지: `docs/PREREG_DET_W5_CONT_SOURCE_2026-09-30.md` — **신설**.

## §0. 선행 사실·설계 판단

### 설계 판단

- **물리 변경은 하나다.** fine 조립기의 연속 방출 항을, 같은 물질로 bin 조립기가 실제 소비한 bf+ff 방출률의 보간값으로 교체한다. 기존 fine 흡수계수와 같은 bin 중심·가중치·끝점 처리를 사용한다.
- **bin 중심에서 소비한 값의 재보간을 택한다.** fine 주파수에서 `bf_get_eta()`를 바로 호출하거나 η/χ 자체를 보간하지 않는다. 기존 fine χ_abs의 이중 보간 경로와 짝이 맞지 않기 때문이다.
- **INIT에도 적용한다.** 연속 원천 계약은 선 연산자와 독립이다. 초기 Gaussian의 선 η와 일반 반복 Sobolev의 선 합성은 그대로 두고, 두 경로의 연속 항에 같은 변경을 적용한다.
- **ff의 기존 소비 의미를 보존한다.** 실제 `bf_get_chi/eta` 배열에는 생산자 ff가 이미 포함되고, bin 조립기는 별도의 열적 ff를 더한다. 이 단은 그 소비식을 맞추며, 기존 ff 중복 가능성을 함께 수리하지 않는다.
- **자체시험 뒤 판정런은 한 번이다.** C1 입력·설정에 신규 env 한 키만 더한다. 효과 문턱은 사전에 고정하고 방향은 등록하지 않는다.
- **선 응답의 대조자료 공백이 있다.** C1에는 요청 셸·이온의 J_c·J̄·a_l 원장이 없다. 새 런에서는 직접 기록할 수 있지만, C1 대비 변화까지 기존 자료로 완성할 수는 없다. 아래 결정 항목을 숨기거나 bin J로 대체하지 않는다.

### 운전자가 user에게 확인할 결정

**D-LINE — 선 응답의 비교 범위.**

제안하는 최소 집행안은 C1 대비 인구·bin 창 J/B·방출 스펙트럼 변화와 함께, **W5 런 내부의 같은 선에 대한 반복별 J_c·J̄·a_l 변화**를 기록한다. C1 대비 선 응답은 `BASE_LINE_FIELD_UNAVAILABLE`로 남긴다.

user가 **C1 대비 J_c·J̄·a_l 변화도 필수**라고 결정하면, 정확한 C1 생산자 물질·장 복원 또는 추가 대조 계산의 범위·자원을 별도로 정해야 한다. 이는 현재의 판정런 한 번에 조용히 포함하지 않는다. 이 선택 전에는 선 판정량 요구를 충족했다고 표시하지 않으며, **판정런 제출은 보류**한다. 코드 설계·오프라인 자체시험 준비는 이 결정과 독립적으로 구체화할 수 있다.

그 밖에 W5 단일 변경, `FINE_BF_OPAC` 제외, FIXED_TE 유지, h200 금지는 이미 주어진 결정이므로 재승인을 요구하지 않는다.

### 경로와 증거 정체성

아래 약칭은 서로 다른 역할의 경로이며, 저장소와 동결 소스를 같은 트리로 간주하지 않는다.

```text
P = /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
T = /gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source
S = /gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src
D = /gpfs/kjhan/lumina/cmfgen_map/diag_fable_20260930
C = /gpfs/kjhan/lumina/cmfgen_map/fixall_20260928/runC1
W = /gpfs/kjhan/lumina/cmfgen_map/w5_cont_source_20260930
```

W는 **신설 산출 루트**다. 이미 존재하면 덮어쓰지 않고 운전자가 충돌 없는 경로를 집행 기록에 확정한다.

| 선행 사실 | 확인 상태·근거 |
|---|---|
| 프로젝트 정체성 | **[저자 실행]** `pwd`로 P 확인. 공유 문맥 전문과 현행 역할 문서를 읽었다. 이번 직접 발주에 따라 astra는 기안만 수행한다. |
| 진단 정본 | **[저자 실행]** D의 DIAGNOSIS와 astra 감사 r1–r4 전문 판독. DIAGNOSIS SHA-256 = `1b8be5191c274e10a41ca466b5ac4e9e5545dc3caf9b2991c8a2bd5c66c4476d`. r4 인계 항목을 이 계약의 제한으로 반영했다. |
| C1 신원 | **[저자 실행]** C의 exports·runner·footer·빌드 신원 기록 판독. job 408652, H100 두 장, CPU 16, 반복 6, `terminal_state=SUCCESS`, 기록 경과 40,131초. 이는 기록 확인이며 Slurm 독립 재조회가 아니다. |
| C1 바이너리 | **[저자 실행]** footer와 `input/binary.sha256`의 기록 일치: `f7fe0f23805a6ffe065a521fe63c3ebf539dbf84412d6490968026dd2c178884`. **[미실행]** 바이너리 실물 재해시·전체 빌드 manifest 대조. |
| 동결 소스 | **[저자 실행]** S의 `lumina_cmfgen.c` SHA-256 = `26b2a3ef5a4fef30a5bfd7600c360ca822306b9bbf6d5af7357c8d14e87e55fd`. `env_universe.h` = `42250348e1015373ad748d8bfb218f3e5c89a19294957cc4462374056316dbb0`, COUNT 533. |
| C1 설정 | **[저자 실행]** `CONT_NLTE_SOURCE=1`, `BF_MILNE=2`, `LINERES_JBAR=2`, FIXED_TE、ITER 6、bin ALI 24、fine ALI 128。`FINE_BF_OPAC`·OMEGA_CMFGEN·NLTE_COLL_ION은 미설정이다. C2는 대조로 사용하지 않는다. |
| 소비 보간 | **[저자 실행]** S의 `lumina_plasma.c:8773–8788,8820–8837`, `lumina_cmfgen.c:2415–2437,2514–2520,5755–5766,6386–6406` 판독. bf accessor 보간 뒤, fine에서 조립된 bin 상태를 다시 보간한다. |
| ff 포함 관계 | **[저자 실행]** `lumina_plasma.c:9822–9850`에서 bf 배열에 ff χ·η를 더함을 확인. `lumina_cmfgen.c:2423–2431`은 별도 Kramers ff를 더한다. 함수·변수 이름을 순수 bf의 증거로 사용하지 않는다. |
| 선 응답 경로 | **[저자 실행]** `lumina_cmfgen.c:7048–7118,8983–8987`과 `line_net_rate.c`의 생산 경로 확인. INIT Gaussian과 일반 Sobolev를 구분한다. |
| 기존 계측의 범위 | **[저자 실행]** C/work 파일 목록, exports와 선 진단 소스 판독. SPRODUCER_CAPTURE=0이며 기존 saturation 진단은 shell 0의 철족을 고른다. 요청한 중간 셸 S/Si/Ca의 대조 원장이 아니다. |
| CMFGEN 대조 | **[저자 실행]** `/gpfs/kjhan/cmfgen_src/cur_cmf/subs/genopaeta_v6.f:202–203,228–229`의 bf χ·η 구성 확인. 모든 선·연속 알고리듬이 단일하다는 주장은 하지 않는다. |
| 실행 검증 | **[미실행]** 기계 프리플라이트·컴파일·자체시험·NC·GPU 실행·새 판독. 기안자가 시행한 것은 read-only 조회와 명시한 소형 파일 해시 확인뿐이다. |

위 조회의 재현 명령:

```bash
pwd
cat /home/kjhan/.codex/SHARED_CONTEXT.md
cat CLAUDE.md
cat "$D/DIAGNOSIS.md"
cat "$D/astra_audit_r1.md" "$D/astra_audit_r2.md" \
    "$D/astra_audit_r3.md" "$D/astra_audit_r4.md"
sha256sum "$D/DIAGNOSIS.md" "$S/lumina_cmfgen.c" "$S/env_universe.h"
cat "$C/input/resolved_lumina.exports" "$C/RUN_FOOTER.txt" \
    "$C/input/binary.sha256"
sed -n '2415,2437p;2514,2520p;5755,5766p;6386,6406p;8983,8987p' \
    "$S/lumina_cmfgen.c"
sed -n '8773,8788p;8820,8837p;9822,9850p' "$S/lumina_plasma.c"
```

## §1. 물리 계약 하나

**질문:** C1과 같은 입력·초기 조건·온도 고정·반복 수에서, fine 연속 방출 항만 bin이 소비하는 Milne bf+ff 방출률과 일관되게 바꾸면 sh10/15/19/24/30의 S/Si/Ca III/II가 어느 인구 권위에서 사전 선택 효과 크기 이상 움직이는가?

변경 대상은 **연속 방출 항과 그 전달에 필요한 보간·수명 관리**다. 다음은 변경하지 않는다.

- bin 전달의 물리식과 bf 광이온화율의 canonical 소비 경로.
- fine χ_abs, 선 η, 선 불투명도·산란 분리, Sobolev 합성식, 준위 행렬.
- `FINE_BF_OPAC`, NT, W2 인구 권위, 충돌률, 원자자료, FIXED_TE.
- 반복 수, 수렴 허용오차, 파장 지원, 내부 경계, warm start.

이 단의 성과는 **W5 변경의 등록된 유한 반복 응답과 한계가 판독 가능한 상태**다. 자기시험 통과나 코드 교체만으로 단을 종료하지 않는다. 공개 CMFGEN 재현이나 W5 주원인 확정은 수용 기준이 아니다.

## §2. 설계

### 2.1 실제 소비량과 bf·ff의 구별

생산자 원시 격자에서 다음을 구별한다.

\[
\chi^{pack}=\chi^{bf}+\chi^{ff,builder},\qquad
\eta^{pack}=\eta^{bf,Milne}+\eta^{ff,builder}.
\]

여기서 ff는 그 생산 주파수에서 열적 관계를 갖는다.

\[
\eta^{ff,builder}_j=\chi^{ff,builder}_j B_\nu(\nu_j,T_e).
\]

`bf_get_chi/eta`는 이름과 달리 이 **pack**을 보간한다. bin 중심 \(\nu_b\)에서의 실제 연속 소비량은 다음이다.

\[
\begin{aligned}
C_b&=\operatorname{bf\_get\_chi}(\nu_b)+\chi^{ff,extra}_b
     =csb\!\to\!chi\_abs[b],\\
E_b&=\operatorname{bf\_get\_eta}(\nu_b)
       +\chi^{ff,extra}_b B_\nu(\nu_b,T_e).
\end{aligned}
\]

따라서 \(E_b\)에는 Milne bf와 열적 ff 항들이 들어 있다. **pack 전체를 순수 η_bf로 부르지 않는다.** 기존 ff 구성의 중복 가능성은 별도 관측으로 남기며, 이번에 제거하거나 재정규화하지 않는다. bin의 소비식 변경까지 포함하면 W5 단일 변경이 깨진다.

ff에 Kirchhoff를 적용하는 것과, 보간 뒤 임의 주파수에서 다시 \(B_\nu\)를 곱하는 것은 다르다.

\[
I(\chi^{ff}B)\ne I(\chi^{ff})B(\nu_f)
\]

가 일반적이다. 본 계약은 **bin에서 실제 소비한 방출률을 보간**한다. ff의 보간 오차를 숨기기 위해 오른쪽 식으로 바꾸지 않는다.

### 2.2 bin 소비값 저장과 fine 대입식

신규 gate가 ON일 때만 bin 조립기의 기존 지역 변수로 \(E_b\)를 저장한다. `S_fixed*chi_tot`에서 선·침적 항을 빼서 역산하지 않는다.

아래는 구현해야 할 대입식이다. `w5_cont_eta`는 bin 셀별 임시 배열이고, `w5_cont_on`은 §2.4의 gate다.

```c
/* bin 조립기: 기존 소비 변수와 같은 호출·물질·주파수 */
if (w5_cont_on) {
    w5_cont_eta[idx] = cmf_cont_src_numerator(
        1, chi_bf, chi_ff, B, eta_bf_cont, 0.0);
    /* = eta_bf_cont + chi_ff * B
       eta_bf_cont는 bf_get_eta가 반환한 bf+builder-ff pack */
}

/* 기존 bin S_fixed 대입은 그대로 유지한다. */
```

fine χ_abs 보간 블록에서 **기존 b·f·ia·ib를 공유**한다.

```c
/* 기존 log-nu 중심 좌표 및 끝점 처리를 그대로 사용 */
double x = log(fs.nu[i] / csb->nu_min) / csb->d_log_nu - 0.5;
int b = (int)floor(x);
double f = x - b;
if (b < 0) { b = 0; f = 0.0; }
if (b >= csb->n_bins - 1) {
    b = csb->n_bins - 2;
    f = 1.0;
}
size_t ia = (size_t)s * csb->n_bins + b;
size_t ib = ia + 1;
size_t k = (size_t)s * NF + i;

/* 기존 대입: 변경하지 않는다. */
fs.chi_abs[k] =
    (1.0 - f) * csb->chi_abs[ia] + f * csb->chi_abs[ib];

/* ON에서만 S_fixed 버퍼를 조립 전 연속 방출률 scratch로 사용 */
if (w5_cont_on) {
    fs.S_fixed[k] =
        (1.0 - f) * w5_cont_eta[ia] + f * w5_cont_eta[ib];
}
```

최종 fine 조립에서 `:6395`의 대입을 다음처럼 분기한다.

```c
double Bnu = cm_planck(fs.nu[i], Te);
double numerator;

if (w5_cont_on)
    numerator = fs.S_fixed[idx] + eta[idx];
else
    numerator = fs.chi_abs[idx] * Bnu + eta[idx];

/* 이하 ct, exact-zero 처리, 오류 반환, fs.J warm start는 기존대로 */
```

이에 따라 비교할 연속 성분은

\[
E_f=(1-f)E_b+fE_{b+1},\quad
C_f=(1-f)C_b+fC_{b+1},\quad
S_{cont,f}=E_f/C_f
\]

이다. **비율 \(E_b/C_b\)의 보간이 아니다.** 선·산란을 포함한 총소스 또는 두 솔버의 J 일치를 요구하는 식도 아니다.

허용되는 구현 변형은 같은 값·연산 순서를 보존하는 지역 변수·헬퍼 분리뿐이다. 중간 scratch가 선 적재 과정에서 덮이지 않는지 직접시험으로 확인한다. 오류 로그의 `source_numerator`도 실제 ON 분자를 보고하도록 바꾼다.

### 2.3 수명·호환 범위·초기 경로

임시 원장은 파일 정적 소유 구조로 두어 헤더·공개 ABI 변경을 피한다. 기록에는 소유 `CMFGENState` 포인터, NS/NB, 격자 값, 조립 호출 순번과 유효 상태를 둔다.

- 매 조립 진입에서 무효화하고 해당 조립이 성공한 뒤 유효화한다.
- 같은 `csb`의 fine 호출에서만 소비한다. bin 조립과 fine 사이 물질 갱신이 없는 기존 호출 순서를 유지한다.
- 크기·격자·소유 호출 불일치, 할당 실패, 비유한 방출률은 기존 실패 반환 사슬로 종료한다.
- 모든 정상·오류 종료에서 해제한다. 이전 호출 원장의 재사용이나 B로의 폴백은 없다.
- ON은 C1의 `CONT_NLTE_SOURCE=1`, `BF_MILNE=2`, `FINE_BF_OPAC=0/미설정`, EPAY=0 소비 경로를 대상으로 한다. 방출 재가중이나 다른 opacity 정제가 개입하는 조합은 지원한 것으로 가장하지 않는다.
- INIT Gaussian과 일반 Sobolev 모두 연속 항을 교체한다. INIT의 선 η는 그대로 더한다. 일반 Sobolev의 선 η 적재가 비활성이라는 기존 경로도 유지한다.
- INIT의 Gaussian 프로파일 평균을 Sobolev \(J_c\)·\(\beta\)·\(a_l\)로 표시하지 않는다.

### 2.4 env와 수치 처리

신설 키:

```text
LUMINA_CMF_FINE_CONT_NLTE_SOURCE
```

미설정 또는 문자열 `"0"`은 OFF, `"1"`은 ON이다. 그 밖은 명시적 오류다. `env_universe.h`에 정렬 순서대로 등록하고 COUNT를 **533→534**로 갱신한다. 다른 env는 신설하거나 활성화하지 않는다.

OFF에서는 신규 원장 할당·출력 파일 접근·새 로그를 하지 않고 기존 산술 경로를 그대로 실행한다. env 목록·빌드 신원처럼 의도적으로 달라지는 메타데이터와 **물리 배열의 byte 동일성**은 구분한다.

새 clamp·floor·부호 교정·지수 상한·작은 분모 대체값을 두지 않는다. 기존 격자 끝점 연장은 그대로 유지하는 보간 경계 조건이며, 물리량의 크기를 자르는 처방으로 확대하지 않는다.

Planck는 현재 `cm_planck`의 `expm1`/대수적으로 동등한 Wien 표현을 사용한다. 진단 스크립트의 `np.clip(...,700)`, `min(x,700)`, `x>=700이면 B=0`을 승계하지 않는다. λ<531 Å를 변경·검증·판독에서 제외하지 않는다. 실제 표현 범위를 벗어난 비율은 로그 표현이나 명시적 미정의 상태로 보고하며 값을 채워 넣지 않는다.

### 2.5 최소 선 계측

기존 SPRODUCER 전체 배열을 활성화하면 env 델타와 메모리가 늘어나므로 사용하지 않는다. 이미 계산되는 `line_value[s]`, `material`, `radiation`에서 선택한 행만 기록한다.

선 집합은 **런 전에 C1로 고정**한다.

- 셸: 10, 15, 19, 24, 30.
- 이온: S II/III, Si II/III, Ca II/III.
- 파장 묶음: [100,531), [531,912), [912,2000), [2000,20000] Å.
- 각 셸·이온·묶음에서 C1 최종 준위 인구로 계산한 \(n_l f_{lu}\)가 큰 mapped 선을 최대 네 개 선택한다. 양 끝 준위가 존재하고 \(f_{lu}>0\)인 선만 대상으로 하며, 동률은 원자 선 인덱스 순이다.
- 빈 묶음은 EMPTY로 남긴다. 결과를 본 뒤 선을 교체하지 않는다.

이 선택은 흡수 강도의 표본 선정 지표이며, 광이온화 기여도나 전체 선 집합의 대표성 보증이 아니다.

신설 `w5_lines.csv`에는 호출·연산자·복사 생산 세대·물질 세대·shell·line_id·Z·ion·ν·J_c·J_c 오차상한·β·국소항 L_l·J̄·J̄ 오차상한·τ_eff·SRCE_CHK 상태를 기록한다. \(a_l\)은 판독기에서 계산한다. INIT에서는 Sobolev 전용 열을 NOT_APPLICABLE로 둔다.

일반 반복에서

\[
\bar J_l=\beta_lJ_{c,l}+L_l,\qquad
a_l=\frac{\beta_lJ_{c,l}}{\bar J_l}.
\]

분모가 양수이고 두 방출 기여가 비음수일 때 \(0\le a_l\le1\)이다. J̄=0이면 비율을 만들지 않고 원래 합성항을 보고한다. β 자체를 1 이하로 자르지 않는다.

고정 물질에서만

\[
\frac{\bar J'_l}{\bar J_l}
=1+a_l\left(\frac{J'_{c,l}}{J_{c,l}}-1\right)
\]

을 사용할 수 있다. 실제 반복 사이에는 β와 L_l도 변하므로 각 시점의 값을 따로 보고한다.

## §3. 변경집합·산출물·집행 범위

**src 편집은 단일 태스크**로 묶는다. 운전자가 다른 소스 변조 태스크가 없음을 확인한 뒤 구현 담당자에게 발주한다. 이 문서는 구현자·감사자를 새로 배정하지 않는다.

### 변경집합 — DET-W5-CONT-SOURCE (표)

| 경로 | 분류 | 허용 변경 |
|---|---|---|
| `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/lumina_cmfgen.c` | 생산·직접 계측 | §2의 gate, bin 방출률 원장, 동일 보간, fine 연속 분자 분기, 수명·오류 처리, 선택 선 기록, 자체시험용 컴파일 조건부 진입점 |
| `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/env_universe.h` | 등록 | 신규 키와 COUNT 갱신 |
| `/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/tests/w5_cont_source_selftest.c` | **신설** CPU 자체시험 | §4의 실제 생산 함수·조립 경로 시험과 결함 주입 |
| `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_w5_cont_selftest.py` | **신설** 시험 실행기 | 동결 전·변경 후 소스의 CPU 시험 빌드, OFF 비교, NC 실행·rc 수집. 명시적 독립 출력 루트 필수 |
| `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/w5_cont_readout.py` | **신설** 판독기 | C1 referent·선 집합 생성, 기존 판독 로직의 범위 지정 재사용, §5 표·분기·판독 NC |

### 변경집합 끝

`lumina_plasma.c`, bf 생산자, `line_net_rate.c`, CUDA 커널, 공개 헤더, 원자자료와 기존 진단 산출은 변경하지 않는다. 신규 소스 심볼은 `w5_cont_` 접두사를 사용한다.

기존 도구는 다음 범위로 재사용한다.

- `diag_fable_20260930/ionmap.py`: 공개 분율의 속도 보간과 단계·준위합 구별.
- `fixall_20260928/traj.py`, `ab_readout.py`: 반복 궤적·방출 대역 적분.
- `lamp_20260930/window_by_iter.py`: shell/spectral 결합 구조.
- `diag_fable_20260930/recon_bf_source.py`, `w5_ratio.py`: 소비 보간·시점 구별의 참고.
- `euv_budget.py`, `lamp_check_euv_it5.py`: publication 방출 장부의 보조 판독.

하드코딩된 398939 대조, Planck 지수 제한, `maximum(...,1e-300)` 같은 분모 대체는 가져오지 않는다. 기존 원본은 보존한다. publication 방출률을 전달 솔버의 총소스로 인증하지 않는다.

신설 산출은 W 아래에 둔다.

| 산출 | 소비 목적 |
|---|---|
| `impl/`의 변경 전 파일 사본·패치·시험 명령·로그·배열 비교 결과 | 변경 범위와 OFF 동일성 |
| `referents/`의 C1 판독 표·공개 분율·선 선택 manifest | 고정 대조·선 집합 |
| `build/`의 소스 manifest·명령·환경·바이너리 해시 | 소스–실행 신원 |
| `run1/input`, `run1/work`, stdout/stderr·footer | 판정런 증거 |
| `run1/work/w5_lines.csv` | 직접 생산한 선 응답 |
| `readout/`의 인구·창·선·스펙트럼·수렴 표와 결과 JSON | §5 판정 |
| `execution_record.json` | 게이트 rc·역할·결정·중단 사유의 집행 기록 |

### 판정런과 자원

base는 **C1의 봉인 입력과 기존 산출**이다. 신규 런은 C1의 최종 상태에서 restart하지 않고 C1과 같은 초기 조건에서 시작한다.

물리·알고리듬 exports 델타:

```text
+ LUMINA_CMF_FINE_CONT_NLTE_SOURCE="1"
```

입력·출력 경로는 신규 런 루트로 치환하되, 경로별 대응표와 입력 내용 해시를 검사한다. 경로 치환을 물리 키 변경으로 세지 않으며, 그 예외로 임의 값을 바꾸지도 않는다.

- h100 또는 a100, 같은 종류의 80 GB급 GPU 두 장.
- 단일 노드·태스크, CPU 16, `--mem=64G`, `--time=30:00:00`, 재큐잉 금지.
- 기대 실행시간은 C1 기록 약 11.1시간을 참고한 추정이며 보장이 아니다.
- INIT와 논리 반복 0–5. 추가 반복·OFF GPU 런·C2 대체·자동 재실행 없음.
- bin 원장 추가 메모리는 double×NS×NB이며 C1 크기에서 약 0.5 MB다. fine 전체 크기의 새 배열은 만들지 않는다.
- 오프라인 시험은 GPU 0, 단일 CPU를 기본으로 grammar-debug에서 실행한다. 로그인 노드에서는 조회만 한다. 예상 범위를 넘는 계산은 운전자가 적절한 계산 자원을 정한다.

C1 runner의 파일을 원본 위치에서 수정하지 않는다. 신규 사본에서 h200 허용 문자열·프로필도 제거해 자원 검사와 제출 partition을 일치시킨다. `module purge`와 빌드에 맞는 명시적 module load를 기록한다. C1 runner의 오래된 selftest 상수를 그대로 통과 증거로 사용하지 않는다.

원본 C1·진단·봉인 입력은 보존한다. 수정 파일은 편집 전 바이트 사본과 해시를 남기고, 판정 실패가 원본 덮어쓰기나 재런 권한을 만들지 않는다.

## §4. 게이트·오프라인 자체시험·음성대조

검사는 아래 묶음으로 집행한다. 작은 항목마다 별도 감사를 만들지 않는다.

| 게이트 | 수용 기준 | 소비 |
|---|---|---|
| **G-ID** | §0의 증거 정체성·D-LINE 결정, §3 변경집합·C1 입력 내용·exports 델타·빌드 신원·GPU 자원·FIXED_TE 일치 | 단일 변경 비교의 성립 |
| **G-SELF** | 아래 연속 성분 시험, OFF byte 비교, 유효한 결함 주입, 판독 분기 시험 통과 | 구현과 잣대의 직접 검증 |
| **G-RUN** | 모델 정상 종료, 필요한 반복·세대·파일 완비, 신규 경로 실제 소비, 기존 fine solve의 성공 조건 충족, 판정 입력 유효 | §5 분기 가능 여부 |

### G-SELF: 같은 물질·주파수의 연속 성분

생산 함수가 있는 translation unit을 CPU로 컴파일해 실제 bin 저장·fine 보간·분자 분기를 호출한다. production 식을 시험 파일에 복제한 것만 검사하지 않는다. 독립 reference는 원시 bf·ff fixture에서 accessor 규칙과 bin 중심을 명시적으로 계산한다.

시험점은 내부 bin 중심, 비중심 주파수, bf 문턱 양쪽, 끝점 연장, λ<531 Å와 Wien 꼬리를 포함한다. 평탄한 배열만 쓰지 않고 이웃 값이 다른 비선형 표본을 사용한다.

각 점에서 다음을 확인한다.

1. 같은 bin 조립 호출의 χ·η가 짝지어졌는가.
2. fine χ_abs가 기존 bin 상태 보간과 같은가.
3. bf Milne, builder ff, 별도 열적 ff를 분리한 fixture의 합이 실제 bin 소비 연속량을 재현하는가.
4. fine 연속 \(E_f/C_f\)와 동일 주파수로 옮긴 bin 소비 연속 \(E_{\rm ref}/C_{\rm ref}\)가 **모든 유효 시험점에서 상대 \(10^{-3}\) 이내**인가.
5. bf-only, ff-only, bf+ff, 비LTE, LTE fixture와 정확한 zero 성분을 모두 처리하는가.
6. INIT와 일반 경로에서 연속 항만 바뀌고 선 η·산란·χ·warm start가 보존되는가.

상대오차는 양의 reference에 대해 \(|q-q_{\rm ref}|/|q_{\rm ref}|\)다. reference가 정확히 0이면 결과도 정확히 0이어야 한다. \(C=E=0\)인 연속 성분에는 비율을 정의하지 않고 exact-zero 상태를 비교한다. \(C=0,E>0\)인 성분은 η·χ를 각각 비교하고 source ratio를 UNDEFINED로 기록한다. 임의 작은 수를 분모에 넣지 않는다.

**이 \(10^{-3}\) 기준은 연속 성분의 소비 일관성에만 적용한다.** 선·산란을 포함한 총소스나 bin/fine J의 일치를 요구하지 않는다.

### G-SELF: OFF byte 동일성

피연산자는 §3 `impl/`에 보존한 **변경 전 동결 소스**와 **변경 후 소스**를 같은 컴파일러·옵션으로 빌드한 CPU fixture 출력이다.

미설정과 명시적 0을 각각 실행하여, fixture의 bin/fine χ 배열·연속/총 분자·S_fixed·warm-start J·선 합성 출력을 바이트로 비교한다. 부호 있는 zero도 바이트 비교에 포함한다. OFF에서 신규 파일·태그·원장 접근이 없어야 한다.

코드 diff 확인은 보조 증거이며 이 실행 비교를 대신하지 않는다. 이 결과를 전체 GPU 실행의 byte 동일성으로 확대하지 않는다.

### 결함 주입

같은 시험 하네스에서 주입한 값·분기와 실제 차이를 기록한다.

| NC | 주입 | 검출해야 할 실패 |
|---|---|---|
| NC-INTERP | fine에서 원시 `bf_get_eta(ν_f)`를 직접 쓰도록 잘못 연결 | 이중 보간 reference와 불일치 |
| NC-FF | 열적 extra-ff를 누락하거나 중복 가산 | ff-only와 혼합 fixture의 성분 불일치 |
| NC-THERMAL | ON에서도 χ_abs·B 사용 | 비LTE 연속 소스 불일치 |
| NC-OFF | OFF 분자의 비영 표본을 실제 representable 값으로 변경 | byte 비교 실패 |
| NC-STALE | 다른 조립 호출·격자의 원장을 제공 | 원장 정체성 실패 |
| NC-READOUT | 판정용 행 누락·NaN과 문턱 직전/정확 경계/양방향 이동 입력 | X/NONE/MOVE의 등록된 분류 |

결함을 넣은 실행은 rc≠0과 해당 사유를 보여야 한다. 하네스의 “NC 통과”는 **결함이 성공적으로 거부됐음**을 뜻한다. 성한 입력의 통과도 함께 확인한다.

### G-RUN의 한계

새 경로는 INIT와 일반 반복에서 각각 소비 횟수·조립 호출 신원을 간결한 태그로 증언한다. env가 1이라는 사실만으로 발화를 인정하지 않는다.

기존 fine exact solve가 실패하거나, 필수 입력이 누락·비유한이면 X다. 반면 **외곽 인구의 반복 표류는 측정 결과**이며, 이를 숨기거나 자동 수치 실패로 바꾸지 않는다. `[CONV]`의 작은 갱신량만으로 전체 물리 수렴을 인증하지 않는다.

## §5. 기대치·판정량·문턱·분기

### 5.1 인구 권위와 공개 비교

대상 셸은 \(\mathcal S=\{10,15,19,24,30\}\), 원소는 Z=14,16,20이다.

각 반복 \(k=0,\ldots,5\)에서 다음을 별도로 보고한다.

\[
x^{stage}_{Z,s,k}
=\frac{n^{stage}_{q2}}{n^{stage}_{q1}},\qquad
x^{levels}_{Z,s,k}
=\frac{n^{NLTE}_{q2}}{n^{NLTE}_{q1}}.
\]

출처는 `physics_DET_iter*.ion.csv`의 해당 stage·nlte 열이다. 최종 준위합은 `nlte_levels_final.csv`와 같은 세대를 대조한다.

공개 \(x_{\rm pub}\)는 toy06 19.480 d의 II·III **분율을 각각** 셸 중심 속도로 보간한 뒤 비를 만든다. 비 자체를 보간하거나 파일 행번호를 shell_id로 간주하지 않는다. 원자료 속도 범위 밖은 외삽하지 않고 미정의로 남긴다.

W2의 세 번째 권위인 사다리 r은 `lumina_ion_channels.csv`에서 따로 보고한다. 채널 gen을 논리 반복으로 고정 상수 매핑하지 않고 해당 런의 물질 sweep 기록과 결합한다. r을 stage나 준위합으로 대체하지 않는다.

필수 인구 표:

- 각 권위의 \(x\), \(x/x_{\rm pub}\), \(\log_{10}(x/x_{\rm pub})\).
- 같은 k에서 W5와 C1의 차이.
- stage·준위합 간 차이와 사다리 r.
- k=3→4, 4→5의 각 궤적 변화.

### 5.2 효과 크기와 분기 통계

끝 두 반복의 로그 평균으로 비교한다.

\[
\delta_{Z,A,s}
=\frac12\sum_{k=4}^{5}
\left[\log_{10}x^{W5}_{Z,A,s,k}
-\log_{10}x^{C1}_{Z,A,s,k}\right],
\]

여기서 \(A\in\{stage,levels\}\)다.

\[
m_{Z,A}=\operatorname{median}_{s\in\mathcal S}|\delta_{Z,A,s}|,\qquad
M=\max_{Z,A}m_{Z,A}.
\]

**문턱은 \(M\ge0.30\) dex**다. 대략 두 배의 변화를 구별하기 위해 선택한 효과 크기이며, 잡음 추정·통계적 유의성·예상 W5 진폭이 아니다. 같은 원소·권위에서 대상 셸의 과반이 문턱 이상의 절대 변화를 보여야 해당 중앙값이 문턱에 도달한다.

원소·권위별 \(m\)과 각 셸의 부호를 모두 보고한다. Si/Ca만 움직였으면 황의 이동으로 부르지 않는다. 서로 다른 권위를 합쳐 한 개선량으로 만들지 않는다.

### 5.3 선·복사장·스펙트럼·수렴 판독

**선:** §2.5의 고정 집합에 대해 실제 J_c·J̄·β·L_l·a_l와 인구 세대를 보고한다. D-LINE의 최소안이 선택되면 일반 반복 k=0→5와 k=3→5의 변화를 기록하며, 이를 C1 대비 W5 인과 효과로 부르지 않는다. C1 선 대조 열은 명시적으로 unavailable이다.

**창 J/B:** 각 대상 셸·반복에서 다음 파장창을 사용한다.

- 저장 격자의 청색 끝부터 531 Å.
- 531–760 Å, 760–912 Å.
- 912–1500 Å, 1500–2000 Å.
- 기존 비교창 500–620 Å.

창별 값은

\[
R_{\mathcal W}
=\frac{\sum_b J_b\,\Delta\nu_{b\cap\mathcal W}}
       {\sum_b B_\nu(\nu_b,T_e)\,\Delta\nu_{b\cap\mathcal W}}.
\]

bin 중심·창 경계와 겹치는 폭을 명시하고, C1과 W5에 같은 규칙을 적용한다. \(\langle J/B\rangle\)와 혼용하지 않는다. λ<531 Å의 큰 비를 언더플로라는 이유로 버리지 않는다.

**방출:** 최종 observer spectrum의 기록 격자 전체 적분 \(L_{\rm grid}\), λ<2500 Å 분율, 912–2000 Å 분율을 같은 적분기로 비교한다. 분율은 해당 파일의 전체 기록 파장 적분을 분모로 한다. 이를 격자 밖까지 포함한 bolometric luminosity로 부르지 않는다. 음의 flux나 비유한 값을 절단하지 않는다.

**외곽:** sh19/24/30의 stage·준위합·n_e 궤적과 마지막 bin 갱신량, fine solve 상태를 함께 보고한다. 고정 반복 수의 종료와 수렴해를 구분한다. 이 단에서 수렴하지 않은 셸의 최종 차이는 **유한 반복 응답**이다.

### 5.4 등록 분기

| 분기 | 판정식 | 허용 해석 | 다음 걸음 |
|---|---|---|---|
| **W5-MOVE — 움직임 있음** | G-ID·G-SELF·G-RUN 통과, \(M\ge0.30\) | 등록된 원소·인구 권위에서 어느 방향이든 문턱 이상 응답 | 원소·권위·셸·공개 gap·수렴 상태를 보고하고, W5를 후속 기준선 후보로 둘지 user 결정. 원천비·선 응답·인구 변화를 분리해 설명 |
| **W5-NONE — 움직임 없음** | 게이트 통과, \(M<0.30\) | 이 조건·반복창에서 등록 크기의 응답 없음 | W5 무관함을 기각 결론으로 만들지 않는다. 기존 자료로 남은 W2/W3/W4/H1–H4 후보를 정리하고 다음 단일 표적을 user에게 제안 |
| **W5-X — 수치 실패·미시연** | 필수 게이트 실패 또는 판정 입력 불완전 | 효과 크기 판정 불가 | 구현·신원·실행·자료 공백을 구분하고 증거 보존. 판정런을 이미 사용했으면 재실행은 별도 user 결정 |

준위·분모가 정확히 0인 경우를 작은 양수로 바꾸지 않는다. 그로 인해 등록 로그 통계를 정의할 수 없으면 X로 두고 원값을 보고한다.

**개선 방향을 사전등록하지 않는다.** 공개 gap의 감소·증가·교차는 결과 표의 관측이다. 기대한 개선이 없다는 이유로 셸·권위·파장·문턱·반복창을 변경하지 않는다.

## §6. 주장 상한

- W5는 **확인된 연속 원천 처방 차이이자 결함 후보**다. 주원인, 원인별 dex 몫, 개선 방향은 미확정이다.
- H1–H4를 유지한다. 원천비·연속장비·선 J̄비·인구비는 서로 다른 양이다.
- 소비 연속 성분 자체시험이 통과해도 선 처리·격자·총소스·두 솔버의 J가 동등하다는 뜻이 아니다.
- W2의 사다리 r·damping된 stage 배열·준위합은 별도 권위다. 단계 Γ의 LTE 준위 가중을 실제 NLTE 인구 가중으로 바꿔 해석하지 않는다.
- W4는 run2/398939의 **bin 선 열적 채널**에 관한 관측이다. C1의 설정이나 산란을 포함한 총 선 소스에 그대로 확장하지 않는다.
- 외곽 미수렴, 전하 폐합의 영향, 원자자료·준위 대응 오류 미배제를 유지한다.
- 독립 σ 전수 적분, 동일 조건의 fine/bin 복사장·bf 적분 비교, CMFGEN 내부 J 비교의 **재현 BLOCKED**를 유지한다. 이번 연속 성분 시험은 이들의 대체물이 아니다.
- C1 대비 선 원장 공백도 별도로 유지한다. 새 런의 반복 변화나 bin J를 그 대조값으로 채우지 않는다.
- bf pack과 bin 별도 ff의 중첩을 보존하는 것은 이번 개입의 범위 제한이다. ff 처방의 물리적 정확성을 인증한 것이 아니다.
- 한 번의 DET 응답으로 최종목적인 MC 수송 스펙트럼 재현을 달성했다고 주장하지 않는다.

## §7. Q-GOAL·Q-LEAN 자기답·집행 기록

### Q-GOAL 자기답

**자기답: PASS. 독립 감사 판정은 아니다.**

최종목적은 MC 수송으로 CMFGEN 수준의 SN Ia 스펙트럼을 재현하는 것이다. 이번 단은 그 전제인 물질·복사장 계산에서, 같은 물질의 두 소비 경로에 서로 다른 연속 원천을 공급하는 W5 후보를 통제하고 중간 셸 이온화 응답을 측정한다.

이를 건너뛰면 향후 fine/bin 차이를 순수 격자 효과로 읽기 어렵고, 원천 처방 차이를 안은 채 인구·스펙트럼 해석을 이어가게 된다. 다만 이 단은 W5의 우선순위나 주원인 지위를 입증하는 단이 아니다.

### Q-LEAN 자기답

**자기답: APPROPRIATE. 독립 감사 판정은 아니다.**

새 물리 키는 하나이며, 소스 편집은 하나의 태스크다. bin 크기 임시 배열과 선택 선 기록만 추가한다. 전체 fine J 덤프·전체 SPRODUCER 배열·새 광이온화 적분·OFF GPU 재런·fine opacity 정제는 포함하지 않는다.

각 증거의 소비자는 명확하다.

| 증거 | 소비 |
|---|---|
| 연속 성분 자체시험·NC | η·χ 보간과 실제 ON/OFF 경로 |
| 신원·exports·FIXED_TE | C1 단일 변경 비교 |
| 인구 표 | MOVE/NONE |
| 선택 선 기록 | 응답 전달의 제한된 해석 |
| 창 J/B·UV 분율·L | 장·스펙트럼 동반 변화 |
| 외곽 궤적·기존 solve 상태 | 유한 반복 응답의 한계 |
| D-LINE 결정 | 충족하지 못한 대조 요구의 처리 |

새 계측을 지우면 사용자의 선 응답 요구를 측정할 수 없다. 반대로 C1에 없는 선 대조를 만들기 위한 추가 solve를 조용히 넣으면 한 번의 판정런·최소 변경 계약을 벗어난다. 그 경계만 user 결정으로 남긴다.

### 집행 기록

| 단계 | 담당·상태 | 기록할 내용 |
|---|---|---|
| 기안 | astra, read-only 본문 제출 | 모델 ID, 저자 실행/미실행, 증거 경로·해시 |
| 요구 범위 확정 | 운전자, 미집행 | D-LINE의 user 선택과 근거 |
| 파일화·기계 검사 | 운전자, 미집행 | prereg·referents·canonical·dispatch의 명령·rc |
| 구현·직접시험 | 운전자가 현행 배분으로 발주, 미집행 | 실제 구현자, src 변조 단일 태스크, 패치·OFF·NC |
| 코드 검토·빌드·스테이징 | 운전자 집행, 미집행 | 실제 담당자, 봉인 접촉 가부, 신원·runner·입력 확인 |
| 판정런 | 운전자, 미집행 | job·노드·GPU·자원·시작/종료·rc |
| 판독·판정 | 운전자, 미집행 | 게이트·분기·미충족 요구·주장 상한 |
| 종료 처리 | 현행 배분에 따라 운전자 기록 | 감사 시행 여부·실제 저자·재현 상태·보존 위치 |

프리플라이트는 파일화 후 P를 root로 실행한다. 선언 블록의 변경집합은 **절대경로**로 동결 소스와 저장소를 구분한다.

```bash
timeout 5m python3 -B scripts/check_prereg_preflight.py \
  docs/PREREG_DET_W5_CONT_SOURCE_2026-09-30.md --root .

timeout 5m python3 -B scripts/check_gate_referents.py \
  docs/PREREG_DET_W5_CONT_SOURCE_2026-09-30.md --root .

timeout 5m python3 -B scripts/check_doc_canonical.py \
  docs/PREREG_DET_W5_CONT_SOURCE_2026-09-30.md \
  docs/PREREG_DET_BB_ALI_2026-09-25.md \
  CLAUDE.md \
  /gpfs/kjhan/lumina/cmfgen_map/diag_fable_20260930/DIAGNOSIS.md \
  /gpfs/kjhan/lumina/cmfgen_map/diag_fable_20260930/astra_audit_r1.md \
  /gpfs/kjhan/lumina/cmfgen_map/diag_fable_20260930/astra_audit_r2.md \
  /gpfs/kjhan/lumina/cmfgen_map/diag_fable_20260930/astra_audit_r3.md \
  /gpfs/kjhan/lumina/cmfgen_map/diag_fable_20260930/astra_audit_r4.md
```

rc≠0이면 기안 재발주 대상으로 기록한다. 운전자가 의미·문턱·범위를 임의 수정하지 않는다. 이 블록의 분기 코드는 **X=0, MOVE=1, NONE=2의 분할 검사**다. 실제 인구 입력에서 통계를 계산하는 판독기 검증은 NC-READOUT이 담당한다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-W5-CONT-SOURCE (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/(?:src|tests)/[A-Za-z0-9_/.]+\\.(?:c|h)|/home/kjhan/BACKUP/Eunha\\.A1/Claude/Lumina-sn/scripts/[A-Za-z0-9_/.]+\\.py)",
    "symbol": "w5_cont_",
    "roots": [
      "/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src",
      "/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/tests",
      "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts"
    ],
    "expected_extra": [
      "/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/lumina_cmfgen.c",
      "/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/env_universe.h",
      "/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/tests/w5_cont_source_selftest.c",
      "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/run_w5_cont_selftest.py",
      "/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/w5_cont_readout.py"
    ]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
    "metrics": {
      "count": "len(v)",
      "bad": "sum(1 for x in v if x not in (0.0, 1.0, 2.0))",
      "failed": "sum(1 for x in v if x == 0.0)",
      "moved": "sum(1 for x in v if x == 1.0)",
      "none": "sum(1 for x in v if x == 2.0)"
    },
    "rules": [
      {
        "name": "W5-X",
        "predicate": "count == 0 or bad > 0 or failed > 0 or (moved > 0 and none > 0)"
      },
      {
        "name": "W5-MOVE",
        "predicate": "count > 0 and bad == 0 and failed == 0 and moved == count"
      },
      {
        "name": "W5-NONE",
        "predicate": "count > 0 and bad == 0 and failed == 0 and none == count"
      }
    ],
    "adversarial_fixtures": [
      {"name": "empty", "mix": []},
      {"name": "failed", "mix": [[0.0, 1]]},
      {"name": "moved", "mix": [[1.0, 1]]},
      {"name": "none", "mix": [[2.0, 1]]},
      {"name": "conflicting-results", "mix": [[1.0, 1], [2.0, 1]]},
      {"name": "failure-overrides", "mix": [[0.0, 1], [1.0, 1]]},
      {"name": "invalid-code", "mix": [[3.0, 1]]}
    ],
    "residual": "W5-X"
  },
  "references": [
    {"path": "CLAUDE.md"},
    {"path": "docs/PREREG_DET_BB_ALI_2026-09-25.md"},
    {"path": "scripts/check_prereg_preflight.py"},
    {"path": "scripts/check_gate_referents.py"},
    {"path": "scripts/check_doc_canonical.py"},
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag_fable_20260930/DIAGNOSIS.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag_fable_20260930/astra_audit_r1.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag_fable_20260930/astra_audit_r2.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag_fable_20260930/astra_audit_r3.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag_fable_20260930/astra_audit_r4.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/lumina_cmfgen.c",
      "flags_existing": [
        "cmf_cont_src_numerator",
        "cm_planck",
        "cmfgen_fine_jbar",
        "CMF_FINE_LINE_OPERATOR_INIT_SHARED_GAUSSIAN",
        "CMF_FINE_LINE_OPERATOR_CMFGEN_NONOVERLAP_SOBOLEV",
        "LUMINA_CMF_FINE_BF_OPAC"
      ],
      "flags_planned": ["LUMINA_CMF_FINE_CONT_NLTE_SOURCE"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/env_universe.h",
      "flags_existing": ["LUMINA_ENV_UNIVERSE_COUNT 533"],
      "flags_planned": ["LUMINA_CMF_FINE_CONT_NLTE_SOURCE"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/lumina_plasma.c",
      "flags_existing": ["bf_get_chi", "bf_get_eta", "eta_milne_ok"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/lteunmap_eps_20260917/source/src/line_net_rate.c",
      "flags_existing": ["line_net_sobolev_radiation", "continuum_term", "local_emission_term"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/fixall_20260928/runC1/input/resolved_lumina.exports",
      "flags_existing": [
        "LUMINA_FIXED_TE_PROFILE",
        "LUMINA_CMF_CONT_NLTE_SOURCE",
        "LUMINA_CMF_BF_MILNE",
        "LUMINA_PURE_CMFGEN_ITER"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/fixall_20260928/runC1/job.slurm"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/fixall_20260928/runC1/RUN_FOOTER.txt",
      "flags_existing": ["job_id=408652", "terminal_state=SUCCESS"]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/omega_port_20260929/APPROVED_SHA.txt"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag_fable_20260930/ionmap.py"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag_fable_20260930/recon_bf_source.py"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/diag_fable_20260930/w5_ratio.py"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/fixall_20260928/traj.py"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/fixall_20260928/ab_readout.py"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/lamp_20260930/window_by_iter.py"
    }
  ]
}
```
