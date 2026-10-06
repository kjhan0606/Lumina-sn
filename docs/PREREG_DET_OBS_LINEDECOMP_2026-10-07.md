# 사전등록 — DET-OBS-LINEDECOMP: 매핑 선의 이온·셸별 탈출 기여와 E′−R′ 차이 분해

2026-10-07 · 기안  
정본 예정 경로: `docs/PREREG_DET_OBS_LINEDECOMP_2026-10-07.md`

## §0. 목표·승인·주장 상한

### Q-GOAL — 최종목적에 직접 기여하는가?

이 단의 목표는 봉인된 E′·R′ 최종 view에서 **FUVb·UV2500의 `mapped_line` 탈출 광도를 원소·이온·셸별로 정확히 분해하고, 그 차이를 주입항 변화·소비 집합 변화·하류 수송 변화로 구분하는 것**이다.

선행 DET-OBS-CAPTURE는 두 대역의 E′ 출력과 E′−R′ 차이에서 매핑 선 원천이 과반임을 확인했다. 그러나 이 결과만으로 어느 이온·셸을 조사해야 하는지, 차이가 `S_used` 증가 때문인지 알 수 없다. 이 단을 생략하면 국소 원천의 크기와 실제 탈출 기여를 혼동한 채 물리 수리 후보를 선택하게 된다.

관측 가능한 완료 산출은 이온·셸별 광도표, 부호를 보존한 차이표, 폐합하는 요인 분해표와 수치 구간이다. 이 산출은 **다음 물리 조사·수리 후보를 좁히는 결정**에 소비한다.

**기안자 자기답: PASS.** 사전감사자의 결론은 별도로 기록한다.

### Q-LEAN — 불필요한 계측·게이트를 추가하는가?

기존 캡처와 선 목록만 읽는다. 새 GPU 계산, 인구·온도·J 재계산, 전 반복 dump, 원천 제거 실험, 격자 수렴 연구를 추가하지 않는다. 선행 캡처 무결성·scalar 바이트 재생·일곱 레인 폐합 시험도 반복하지 않는다.

새 검사는 다음 두 묶음뿐이다.

1. 선 ID·원소·이온·셸 및 E′/R′ 대응의 정확성.
2. 새 탈출 분해·요인 분해의 폐합과 판정 구간의 유효성.

선·셸 쌍별 수송 계수는 요인 분해에 직접 소비한다. 모든 ray·공명 이력을 영구 저장하거나 이온마다 별도 replay를 실행하지 않는다. 요인은 전체 묶음에 대해 같은 누적 과정에서 계산하므로, “큰 묶음”을 본 뒤 선택적으로 재실행하지 않는다.

**기안자 자기답: APPROPRIATE.** 사전감사는 Q-GOAL·Q-LEAN에 한정한다.

### 승인과 범위

승인 근거:

`/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/evidence/USER_DECISION_2026-10-07.txt`

user 2026-10-07 「승인」에 따라 기존 캡처로 선·이온·셸의 `S_used`·τ·소비 집합·공통 감쇠 차이를 오프라인 분해한다.

이 단은 **고정된 두 상태의 출력 경로 귀속**이다. 물리 오류의 원인 확정, NRB·인구·J의 인과효과 측정, 공개 CMFGEN 잔차의 이온별 배분 또는 잔차 감소를 주장하지 않는다. `S_used` 증가가 원인이라는 결론을 전제하지 않는다.

생산 소스·생산 바이너리·입력·선행 증거는 변경하지 않는다. 추가 GPU 생산 런은 없다.

## §1. 기준선·입력·보존

```text
P = /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn
C = /gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006
D = /gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007

T2 = C/source
EC = C/Eprime/run1
RC = C/Rprime/run1

새 도구 = D/tools
새 빌드 = D/build
새 산출 = D/results/<실행 식별자>
새 증거 = D/evidence
```

### 1.1 승계하는 검증

선행 폐합 기준은 커밋 `bf92ecf`와 다음 기록이다.

- `docs/PREREG_DET_OBS_CAPTURE_2026-10-06.md`
- `docs/VERDICT_DET_OBS_CAPTURE_2026-10-07.md`
- `C/astra_closeout.md`
- `C/astra_closeout2.md`
- `C/readout_b1/E`, `C/readout_b1/R`, `C/readout_b1/paired`

캡처 무결성·소비 coverage·live 무변경·scalar 바이트 재생·일곱 성분 폐합은 선행 PASS를 승계한다. 이 단의 선행 기준점은 **수송 오차상한이 반영된 `readout_b1/`**이다. 첫 판독 `readout/`의 분기 구간으로 되돌아가지 않는다.

운전자는 입력 경로와 선행 manifest·판독 증거의 신원을 새 실행 manifest에 연결한다. 대형 payload 전체 재해시, scalar 재실행 또는 선행 시험 배터리 반복을 새 통과 조건으로 만들지 않는다.

### 1.2 실제 입력

| 입력 | 경로·의미 |
|---|---|
| E′ 캡처 | `EC/work/det_obs_capture/` |
| R′ 캡처 | `RC/work/det_obs_capture/` |
| 선 분류 권위 | 각 런의 `input/model/line_list.csv` |
| 선 목록 봉인 | 각 런의 `input/deck.sha256` |
| 선행 수송 기준 | `C/readout_b1/{E,R}/replay/components.csv` |
| 선행 대역 기준·오차 | `C/readout_b1/{E,R}/result.json` |
| 재사용 알고리듬 | T2의 sparse reader·공명 순회·연속체 분할·ray 적분 |
| 재사용 적분 규약 | `scripts/det_obs_capture_readout.py`의 대역 적분·오차 전파 |

두 `deck.sha256`에 기록된 `line_list.csv` SHA-256은 다음과 같다.

```text
cbd81c4f9b3bd85d723b3b1eee6367735e8f52145920412aaf68388b465554b5
```

선 목록은 이번에 새로 연결하는 분류 입력이므로 그 봉인과 실제 읽은 파일의 일치는 이번 연결 검사에 포함한다.

### 1.3 보존·복구

C와 T2는 읽기 전용 기준선이다. D에는 새 도구와 새 산출만 만든다. D가 승인문 때문에 이미 존재하는 것은 허용하지만, 기존 build/result 파일을 조용히 교체하지 않는다.

실패 산출은 실패 상태와 함께 보존한다. 수정 후 실행에는 새 식별자를 부여한다. 일반 파일 읽기·쓰기를 사용하며 GPFS 전용 호출, 특수 rename 보장, 파일시스템별 우회 코드를 추가하지 않는다.

복구는 새 산출을 부적격으로 표시하고 봉인 입력에서 재생하는 방식이다. 선행 자료를 수정하여 새 결과에 맞추지 않는다.

## §2. 선·이온·셸의 식별

### 2.1 선 ID

캡처의 `line_id`는 **0-based opacity 행 번호**다. CSV 헤더를 제외한 행 순서로 연결한다.

CSV의 `line_id` 열, 파장 정렬 결과, 전이명 또는 주파수만으로 다시 색인하지 않는다. 동주파수 선도 별개 행으로 유지한다. 다음을 연결 검사에서 확인한다.

- CSV 행수와 캡처 `NL`의 대응.
- 행 순서의 `nu`와 캡처 `line_nu`의 대응.
- 소비 레코드의 `line_id`·`shell_id`가 해당 배열 범위 안에 있음.
- 분류표의 원소·이온·하위/상위 준위가 그 행에서 직접 읽힌 값임.

CSV `line_id` 열은 원자료 식별자로 보존할 수 있지만 결합 키로 사용하지 않는다.

### 2.2 묶음

기본 묶음은

\[
g=(Z,\mathrm{ion\_number},s)
\]

이다. 여기서 \(s\)는 주입한 셸의 0-based ID다. 마지막 상호작용 셸이나 감쇠시킨 셸로 바꾸지 않는다.

기본표에서 다음 주변합을 만든다.

- 원소·이온별: 셸에 대해 합산.
- 셸별: 원소·이온에 대해 합산.

두 주변표는 같은 광도의 다른 분류다. 두 표를 서로 더하지 않는다. 이온 표기는 CSV의 정수 `ion_number`를 권위값으로 유지한다. 로마 숫자·전하 표시는 해당 관례가 확인된 경우에만 보조 열로 붙인다.

셸 표에는 `r_inner`, `r_outer`를 함께 싣는다. 특정 원소나 셸을 사전에 표적으로 선택하지 않는다.

### 2.3 E′·R′ 대응과 소비 집합

상태 \(X\in\{E,R\}\)에서 전체 소비 집합을 \(C_X\), 그중 `mapped=1`인 집합을 \(A_X\)라 한다. 원소는 \(p=(l,s)\)다.

\[
A_c=A_E\cap A_R,\quad
A_+=A_E\setminus A_R,\quad
A_-=A_R\setminus A_E.
\]

`mapped`는 캡처 레코드에서 읽는다. `S_used>0`으로 추정하지 않는다.

진입·이탈은 다음을 구분하여 기록한다.

- 반대 상태의 \(C_X\)에도 없음: 실제 소비 집합의 진입·이탈.
- 반대 상태에서도 소비되지만 `mapped=0`: 매핑 레인 분류의 이동.

두 경우 모두 mapped 광도의 집합항에는 포함하지만, 후자를 “새로 소비된 선”이라고 부르지 않는다. 비소비 상태의 `S_used`·τ는 알려져 있지 않으므로 0이나 상대 상태 값으로 채우지 않는다.

같은 셸·ray·공명을 비교하기 위해 관측 격자, 선 주파수와 순서, 셸 기하, `inv_ct`, ray·substep·공명 소유 규칙의 대응을 확인한다. 이 대응이 깨지면 현재 계약의 **하류 감쇠 비교**로 승인하지 않고 `INPUT_ALIGNMENT_FAILED`로 기록한다.

## §3. 탈출 기여의 정의와 계산

### 3.1 대역

| 대역 | 파장 범위 |
|---|---|
| FUVb | 1500–2000 Å |
| UV2500 | 2500–3000 Å |

선행 판독기의 파장 경계 보간과 사다리꼴 적분을 그대로 사용한다. 대역 경계에 필요한 양쪽 격자점도 계산에 포함한다.

관측 대역을 선의 정지계 파장으로 잘라 선 목록을 제한하지 않는다. Doppler 이동 후 해당 관측 격자점에 공명하는 선이 표적이다. 그 광자를 감쇠시키는 다른 선·연속체도 모두 유지한다.

### 3.2 정확한 선형 분해

소비 쌍 \(p=(l,s)\)의 공이동계 주입항을

\[
f_X(p)=1-e^{-\tau_X(p)},\qquad
j_X(p)=S_X(p)f_X(p)
\]

로 정의한다. \(S_X\)는 캡처의 실제 fallback 선택 후 `S_used`다. 수치 계산은 선행과 같은 `-expm1(-tau)`를 쓴다.

한 관측 격자점·ray에서 쌍 \(p\)의 공명 사건을 \(e\)라 할 때, 그 주입이 출구까지 받는 전달계수를

\[
T_{Xe}=\prod_{\text{사건 }e\text{ 이후}} a_X
\]

로 정의한다.

- 연속체 구간: 선행 연산으로 구한 \(a_X=e^{-\tau_c}\).
- 모든 후속 선 공명: 매핑 여부와 무관하게 \(a_X=e^{-\tau_l}\).
- 이후 내경계 교체로 지워지는 주입: 전달계수 0.

주입 사건 자신의 \((1-e^{-\tau})\)는 \(j_X\)에 들어간다. 같은 사건의 \(e^{-\tau}\)를 자신의 방출에 한 번 더 곱하지 않는다.

ray 적분·\(L_\nu\rightarrow L_\lambda\) 변환·대역 적분의 선형 가중치를 \(w_{be}\), 선행 Doppler 인자를 \(D_{Xe}\)라 하면

\[
H_{Xpb}=\sum_{e\in p}w_{be}D_{Xe}^{3}T_{Xe},
\qquad
L_{Xpb}=j_X(p)H_{Xpb}.
\]

따라서

\[
L_{Xgb}=\sum_{p\in A_X,\;g(p)=g}L_{Xpb},
\qquad
\sum_gL_{Xgb}=L_{X,\mathrm{mapped},b}.
\]

광도 단위는 `erg s⁻¹`이다. \(H\)는 광도를 공이동계 `S_used`의 단위로 나눈 수송 계수다.

이 등식은 **선행의 공유 계산 계수와 고정 수송에 대한 정확한 선형 분해**다. 구현의 부동소수 결합 순서 차이는 §5의 상한으로 처리한다.

다른 원천의 주입을 이 묶음에 넣지는 않지만, 그 원천과 연결된 opacity를 제거하지도 않는다. 전자산란과 미매핑 산란의 고정 J 주입을 특정 원자의 최초 방출로 재귀 귀속하지 않는다.

### 3.3 계산 방식

새 하네스는 **ray·관측 격자점별 역방향 가중치 누적**으로 \(H\)를 계산한다.

1. T2와 같은 순방향 사건 순서로 연속체 구간·선 공명·내경계 교체를 구성한다.
2. 출구 가중치 1에서 역순으로 진행한다.
3. 선 사건에서는 먼저 현재 하류 가중치를 해당 쌍의 \(H\)에 반영하고, 그 뒤 해당 선의 감쇠를 상류 가중치에 곱한다.
4. 연속체 사건에서는 감쇠만 곱한다.
5. 내경계 교체를 넘어 상류로 진행할 때 가중치를 0으로 만든다.

사건 기록은 처리 중인 ray·격자점 범위에서만 유지한다. 전체 ray·주파수 사건 dump는 저장하지 않는다. 선·셸 쌍별 \(H\)와 오차상한은 두 대역만 누적한다.

이온·셸별 수백·수천 레인을 매 사건마다 갱신하거나 묶음마다 수송을 다시 실행하지 않는다.

## §4. E′−R′ 차이의 요인 분해

### 4.1 공통 소비 쌍

\(p\in A_c\)에 대해

\[
\Delta j=j_E-j_R,\quad
\bar j=\frac{j_E+j_R}{2},
\]
\[
\Delta H=H_E-H_R,\quad
\bar H=\frac{H_E+H_R}{2}
\]

라 두고

\[
L_{Epb}-L_{Rpb}
=
\underbrace{\Delta j_p\,\bar H_{pb}}_{\text{주입항 변화}}
+
\underbrace{\bar j_p\,\Delta H_{pb}}_{\text{하류 수송 변화}}
\]

로 분해한다.

이는 두 끝점의 곱 차이에 대한 정확한 항등식이다. 한 상태에서 다른 상태로 먼저 S를 바꾸거나 먼저 감쇠를 바꾸는 순차 분해 두 가지를 평균한 배분과 같다. E와 R을 바꾸면 모든 차이항의 부호가 반전된다.

### 4.2 주입항 안의 S·τ 구분

공통 쌍에서만 다음 보조 분해를 계산한다.

\[
\Delta j
=
\underbrace{(S_E-S_R)\frac{f_E+f_R}{2}}_{J_S}
+
\underbrace{\frac{S_E+S_R}{2}(f_E-f_R)}_{J_\tau}.
\]

탈출 광도 단위의 두 항은

\[
F_{S,pb}=J_{S,p}\bar H_{pb},
\qquad
F_{\tau,pb}=J_{\tau,p}\bar H_{pb}
\]

이다. 합은 §4.1의 주입항 변화다.

여기서 \(F_\tau\)는 **자기 공명의 주입 효율 변화**다. 같은 τ가 다른 주입을 가리는 효과는 그 다른 주입의 하류 수송항에 들어간다.

이 계약은 `S·f(τ)`를 주입항으로 먼저 묶은 뒤 H와 대칭 분해하는 계층을 고정한다. 세 인자 S·f·H의 모든 순열을 평균한 배분과 같다고 주장하지 않는다. 교차항 배분은 이 선택에 의존하지만, 합계는 정확하며 결과를 본 뒤 순서를 바꾸지 않는다.

### 4.3 집합 진입·이탈

\[
F_{+,gb}=\sum_{p\in A_+,\;g(p)=g}j_EH_E,
\]

\[
F_{-,gb}=-\sum_{p\in A_-,\;g(p)=g}j_RH_R.
\]

진입은 E′의 실제 탈출 기여, 이탈은 R′의 실제 탈출 기여의 음수다. 존재하지 않는 반대 상태의 S·τ·H를 만들어 추가 요인으로 나누지 않는다.

소비 집합 변화는 두 곳에 나타날 수 있다.

- 해당 쌍 자체의 주입이 생기거나 사라진 효과: \(F_+\), \(F_-\).
- 그 선의 감쇠가 다른 공통 쌍의 탈출을 바꾼 효과: 해당 공통 쌍의 하류 수송항.

둘은 서로 다른 주입에 대한 기여다. 집합항을 소비 predicate 변경의 총 인과효과라고 해석하지 않는다.

### 4.4 묶음별 폐합식

\[
\Delta L_{gb}=L_{Egb}-L_{Rgb}
\]

에 대해

\[
\boxed{
\Delta L_{gb}
=
F_{S,gb}+F_{\tau,gb}+F_{H,gb}+F_{+,gb}+F_{-,gb}
}
\]

를 요구한다. 여기서

\[
F_{H,gb}
=\sum_{p\in A_c,\;g(p)=g}\bar j_p\Delta H_{pb}.
\]

보고용 상위 세 항은 다음과 같다.

| 요인 | 정의 |
|---|---|
| 주입항 변화 | \(F_S+F_\tau\) |
| mapped 소비 집합 진입·이탈 | \(F_++F_-\), 두 부호 항도 별도 표시 |
| 하류 감쇠·수송 변화 | \(F_H\) |

### 4.5 하류 수송항의 해석 범위

\(F_H\)에는 전체 연속체 소광, 다른 매핑·미매핑 선의 τ, 하류 소비 사건의 출현·소멸이 포함된다. 다른 원천의 방출 세기 자체가 감쇠 계수는 아니다.

선 공명으로 나뉘는 연속체 구간이 상태에 따라 달라지면, 선행 이산 수송에서의 구간 분할 효과도 H 차이에 포함된다. 이를 연속 수송의 물리 opacity 변화만으로 순화하여 해석하지 않는다.

이 단에서는 \(F_H\)를 “어느 이온의 opacity 때문인가”까지 추가 배분하지 않는다. \(F_H\)가 크게 나오면 **그 주입을 가리는 하류 수송을 다음 조사 후보로 지정**한다. 이를 위해 새로운 혼합 상태 수송이나 opacity 제거 실험을 추가하지 않는다.

## §5. 수치 상한과 폐합

### 5.1 새 계산의 상한

선행과 같은 binary64 최근접 반올림 전제에서

\[
u=2^{-53},\qquad a_{\min}=2^{-1074},\qquad
\rho(x)=\frac{u|x|}{1-u}+a_{\min}
\]

를 사용한다.

새 상한은 다음을 포함한다.

- 하류 감쇠 곱과 역방향 가중치 누적.
- Doppler 계수와 주입항의 곱, 연산 재결합.
- ray 가중치·단위변환·대역 적분 가중치의 구성과 합산.
- 쌍→기본 묶음→주변합의 합산.
- E′−R′ 차이와 다섯 요인항의 사칙연산.
- 분율, 양·음 기여 합, 순위·과반 분기.

상한 계산은 바깥쪽으로 반올림한다. `exp`, `expm1`, 기하·보간 계수는 선행과 같은 계산값을 조건으로 삼는다. 그 함수들의 전체 수학적 오차나 물리·격자 오차까지 인증하지 않는다.

**선행 일곱 레인의 상한을 새 세분화 레인·역방향 계산의 상한으로 그대로 복사하지 않는다.** 새 연산의 상한은 별도로 유도한다. 선행 상한은 선행 mapped 기준값과 비교할 때 그 기준값의 오차로만 소비한다.

연산식·합산 순서·오차 전파식을 전체 E′/R′ 판독 전에 직접시험 증거에 고정한다. 관측된 폐합 잔차에 맞춘 배수, 상대오차 문턱, floor, 음수 보정은 금지한다.

### 5.2 필수 폐합

다음 모두를 각 대역에서 확인한다.

1. E′·R′ 각각의 기본 묶음 합과 선행 mapped 대역 광도의 일치.
2. 기본 묶음 합과 이온별·셸별 주변합의 일치.
3. 묶음별 다섯 요인 합과 해당 \(\Delta L_{gb}\)의 일치.
4. 전체 묶음 차이 합과 선행 mapped E′−R′ 차이의 일치.

예를 들어 첫 비교는

\[
\left|\sum_g\widehat L_{Xgb}
-\widehat L^{\,\mathrm{prior}}_{X,\mathrm{mapped},b}\right|
\le
B^{\,\mathrm{new}}_{Xb}
+B^{\,\mathrm{prior}}_{Xb}
+B^{\,\mathrm{compare}}_{Xb}
\]

로 판정한다.

다른 폐합도 양쪽 산출의 전파 상한과 비교 연산 상한을 사용한다. 차이가 작다는 이유로 상한 밖 실패를 승인하지 않는다.

폐합 상한과 별도로 **개별 묶음·개별 요인항의 오차상한**을 결과에 전달한다. 성분합 폐합만 통과한 것을 개별 순위 인증으로 사용하지 않는다.

상한이 비유한하거나 계산 전제가 깨지면 부적격이다. 유한하지만 구간이 넓어 순위를 가리지 못하면 계산 실패와 구별하여 `UNRESOLVED`로 보고한다.

## §6. 판정량·분기·필수 산출

### 6.1 광도와 순위

각 대역, E′·R′ 각각에 대해 기본 묶음·이온별·셸별 표에 다음을 싣는다.

- \(L_E\), \(L_R\), \(\Delta L\)와 각각의 수치 구간.
- 상태별 mapped 광도에 대한 분율.
- 차이의 부호.
- §4의 다섯 요인항과 상위 세 요인합.
- 공통·진입·이탈 쌍수와 분류 이동 여부.

분율은 분모의 구간 하한이 양수일 때만 정의한다. 이는 전체 N 광도에 대한 분율이 아니라 **mapped 레인 내부의 분율**이다.

각 분류 수준에서 다음을 적용한다.

- 분율 하한이 \(1/2\)를 넘는 단일 묶음: `MAJORITY`.
- 과반 경계가 해결되어 있고, 한 묶음의 광도 하한이 나머지 모든 상한보다 큼: `MIXED_LARGEST`.
- 최대 후보 또는 과반 경계가 겹침: `UNRESOLVED`.

중심값으로 정렬한 표는 허용하되, 구간이 겹치는 순위를 확정 순위로 표시하지 않는다. “상위 여러 묶음의 합”을 새 단일 지배 이온으로 부르지 않는다.

### 6.2 부호가 있는 차이

각 분류 수준에서

\[
P_b=\sum_g\max(\Delta L_{gb},0),\qquad
M_b=\sum_g\max(-\Delta L_{gb},0)
\]

를 계산하여

\[
\Delta L_{\mathrm{mapped},b}=P_b-M_b
\]

와 함께 보고한다.

양의 차이와 음의 차이를 각각 P·M에 대해 분류한다. 작은 순차이를 분모로 삼아 거대한 “기여율”을 만들지 않는다. `max`는 양·음 부분의 정의이며 결과를 0으로 보정하는 clamp가 아니다. 구간에도 같은 함수를 적용한다.

P·M은 분류 수준에 따라 달라질 수 있다. 이온 내부의 셸 상쇄와 셸 내부의 이온 상쇄를 숨기지 않으며, 서로 다른 분류 수준의 P·M을 같은 양처럼 비교하지 않는다.

### 6.3 요인 귀속

요인 분석은 모든 기본 묶음에 대해 계산한다. 요약에서는 §6.1·§6.2에서 식별한 최대·과반 묶음과 수치적으로 분리되지 않은 최대 후보를 다룬다.

요인항도 양·음 기여를 분리하여 같은 구간 규칙을 적용한다.

- \(F_S\) 우세: 공통 소비 쌍의 `S_used` 차이에 연결된 주입 변화가 큼.
- \(F_\tau\) 우세: 공통 소비 쌍의 자기 공명 주입 효율 차이가 큼.
- \(F_H\) 우세: 하류 감쇠·수송 차이가 큼.
- 진입·이탈 우세: mapped 소비 집합 차이가 큼.
- 큰 반대 부호 항 공존: 상쇄 구조를 보고하며 단일 요인으로 축약하지 않음.
- 구간 중첩: `UNRESOLVED`.

이 표현은 등록한 대칭 배분에 따른 수치 귀속이다. “S 생산자가 틀렸다”, “이 이온의 opacity가 오류다”라는 원인 판정으로 올리지 않는다.

### 6.4 필수 산출

| 산출 | 내용 |
|---|---|
| `ion_shell.csv` | 모든 기본 묶음의 E′·R′·차이·요인·구간 |
| `ions.csv` | 원소·이온별 주변합과 분기 |
| `shells.csv` | 셸별 주변합·반경과 분기 |
| `closure.json` | §5의 폐합 잔차·상한·유효성 |
| `result.json` | 대역별 분기·상쇄·수치 미분리·한계 |
| `manifest.json` | 입력·도구·빌드·실행·산출의 신원과 재현 명령 |
| 쌍별 계수 자료 | 키 `(line_id,shell_id)`, 두 대역 H와 상한; 요인 분해 재현용 |
| 판정문·종료감사 | 목표 달성 여부와 다음 후보 |

쌍별 자료는 binary 등 간결한 형식으로 저장하고 스키마를 manifest에 명시한다. 전 쌍의 문자열 CSV를 만들지 않는다.

개별 전이 목록은 필수 산출이 아니다. 필요하면 저장된 쌍별 기여를 선 ID별로 합산하여 요약에 최대 전이와 구간이 겹치는 후보를 추가할 수 있다. 이때 원래 행 번호·CSV 식별자·준위·정지계 파장과 선정 규칙을 함께 기록한다. 추가 수송은 하지 않는다.

## §7. 변경집합과 선행 replay 연결

새 독립 오프라인 트리를 D 아래에 둔다. T2 사본의 생산 소스를 수정하지 않는다.

T2의 검증된 reader·공명 순회·기하·감쇠 계산을 읽기 전용으로 재사용하거나 필요한 부분을 새 도구에 명시적으로 이식한다. 이식한 경우 원 함수·원본 해시와 새 계산의 차이를 구현 기록에 남긴다. 생산 바이너리에 hook을 추가하지 않는다.

**scalar 바이트 재생은 이미 확정된 기준점으로 재사용한다.** 새 하네스가 scalar를 다시 실행하는 것을 필수 단계로 삼지 않는다. 새 알고리듬의 검증 대상은 저장된 mapped 레인에 대한 추가 세분화다.

### 변경집합 — DET-OBS-LINEDECOMP (표)

신규 C·Python 심볼은 `det_obs_linedecomp_` 접두를 사용한다. 아래 경로는 추가 허용 파일이며 기존 생산 파일의 수정 허가가 아니다.

| 경로 | 함수·항목 | 허용 변경 |
|---|---|---|
| `/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/tools/det_obs_linedecomp.c` | `det_obs_linedecomp_*` | sparse 입력 연결, 사건 구성, 역방향 H·상한, 쌍별 출력 |
| `/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/tools/det_obs_linedecomp_selftest.c` | `det_obs_linedecomp_*` | 작은 직접시험·음성대조 |
| `/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/tools/build_linedecomp.sh` | 오프라인 빌드 | 명시적 소스 목록·컴파일러·옵션·원본 연결 기록 |
| `/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/tools/run_linedecomp.slurm` | CPU 실행 | 자원·모듈·시험·E′/R′·짝 판독·종료 상태 기록 |
| `scripts/det_obs_linedecomp_readout.py` | `det_obs_linedecomp_*` | 선 분류 연결, 묶음·요인·구간·분기, manifest |

### 변경집합 끝

T2의 `src/`·기존 replay·생산 빌드·기존 판독기는 변경하지 않는다. 새 사전등록·판정문·증거 manifest와 결과표는 문서·실행 산출이며 소스 변경집합과 구분한다.

허용 파일만으로 구현할 수 없으면 필요한 변경을 특정하여 계약 개정으로 넘긴다. 변경집합을 넓히기 위해 기존 생산 함수를 조용히 고치지 않는다.

## §8. 최소 게이트·직접시험·음성대조

| 게이트 | 실제로 막는 실패 | 통과 조건 |
|---|---|---|
| G-JOIN | CSV 식별자를 행 번호로 오인, 동주파수 선 혼동, 셸·상태 대응 오류, mapped 분류 이동을 신규 소비로 오기 | §2 연결과 분류가 권위 입력에 일치 |
| G-DECOMP | 자기 선 감쇠 중복, 다른 선 opacity 누락, core 이전 주입 생존, 요인 누락·부호 오류, 수송 오차의 분기 구간 미전파 | §3–§6의 직접시험·폐합·구간 연결 충족 |

직접시험은 작은 인공 문제 하나의 필요한 변형으로 묶는다. 서로 다른 원소·이온, 동주파수 선, 두 셸, 중간 연속체, 미매핑 감쇠 선, core 교체를 포함하여 각 항을 따로 확인할 수 있게 한다.

시험의 정답은 새 역방향 루틴 자체가 아니라 작은 독립 순방향 선형 행진 또는 직접 대수식으로 구한다.

필수 음성대조는 다음과 같다.

1. **잘못된 행 연결:** 행 번호 대신 다른 선 식별자를 사용하거나 분류 행을 교환한 시험 입력이 G-JOIN에서 실패해야 한다.
2. **잘못된 수송:** 미매핑 선 감쇠 누락 또는 core 교체 무시가 G-DECOMP에서 실패해야 한다.
3. **잘못된 차이·구간:** 공통 쌍에서 j와 H를 함께 바꾼 문제의 교차항 누락, 집합 이탈 부호 반전, 수송 상한의 분기 입력 누락을 각각 검출해야 한다.

정상 대조에는 동일 상태 비교를 포함한다. 이때 차이·요인항은 0과 일치해야 한다. E/R 교환 시 차이항의 부호 반전과 진입·이탈 교환도 작은 자료에서 확인한다.

대형 캡처를 훼손하거나 실제 입력에 결함을 주입하지 않는다. 음성대조는 이 두 게이트의 검출력을 확인하는 직접시험이며 별도 감사 단위를 만들지 않는다.

기계 프리플라이트는 현행 `check_prereg_preflight.py`, `check_gate_referents.py`, `check_doc_canonical.py`, `check_dispatch_preflight.py`를 사용한다. 실행 명령·rc·문서 신원을 기록하며, 그 통과를 Q-GOAL·Q-LEAN 사전감사로 대신하지 않는다.

## §9. 실행·비용·재현

### 9.1 실행 환경

CPU 시험·오프라인 계산·판독은 a10 Slurm 작업으로 실행한다. 로그인 노드에서는 파일 조회만 한다.

```text
--partition=a10
--nodes=1
--ntasks=1
--cpus-per-task=16
--mem=64G
--no-requeue
```

a10에서 GPU를 요청하지 않는 CPU 작업으로 실행하며, GPU 예약량은 0이다. GPU 생산 실행·CUDA 수송은 없다.

러너는 `module purge` 후 고정한 모듈을 로드하고 `OMP_NUM_THREADS=16`을 명시한다. 실제 모듈·컴파일러·libm·옵션을 기록한다. 계산 출력은 D 아래에 둔다. 작업은 한 번에 하나만 실행한다.

### 9.2 비용 상한

선행 실측 기준은 상태당 scalar 약 7분, 일곱 성분 vector 약 2.5분이다. 한 쌍의 scalar+vector 기준량은 약 19분이다. 이번에는 scalar를 반복하지 않지만 역방향 사건 처리·쌍별 누적·상한 연산의 비용은 아직 실측되지 않았다.

본 기안의 **전체 집행 상한**은 다음과 같다.

| 항목 | 상한 |
|---|---:|
| 동시 CPU | 16 thread |
| 작업별 메모리 | 64 GB |
| 빌드·직접시험·본 계산·허용 수정을 합한 Slurm 할당 경과 | 120분 |
| CPU 예약량 | 32 core·h |
| a10 GPU 예약량 | 0 GPU·h |
| 새 GPU 생산 런 | 0회 |

120분은 선행 한 쌍 전체 replay 약 19분의 약 6.3배를 잡은 **지출 한도**다. 새 알고리듬의 예상 완료시간이나 실측 배수라는 뜻은 아니다.

제출별 `--time`은 남은 총예산 이내로 정하고, 실패 작업의 사용량도 차감한다. 자동 재제출·재시작은 하지 않는다. 원인이 특정된 도구 오류의 수정·필요 범위 재실행은 총예산 안에서 수행할 수 있다. 상한 증가는 user 결정 사항이다.

메모리는 sparse 자료와 두 대역의 쌍별 H·상한, 제한된 thread별 작업 배열로 구성한다. 구현 시 명시적 배열 크기로 64 GB 이내임을 확인한다. 전 ray 사건이나 전체 이온 레인의 dense 배열을 만들지 않는다.

입력 payload는 복제하지 않는다. 저장 예약은 명시적인 쌍수·레코드 크기·thread 임시배열·출력 스키마로 계산하여 기록한다. 새 결과를 잘라 저장해 예약량에 맞추지 않는다.

### 9.3 계획 인터페이스와 재현 기록

다음은 구현할 오프라인 인터페이스다. 아래 명령은 Slurm 작업 안에서 실행한다.

```bash
python3 -B scripts/det_obs_linedecomp_readout.py selftest \
  --c-selftest "$D/build/det_obs_linedecomp_selftest" \
  --out "$D/results/$RUN_ID/selftest"

python3 -B scripts/det_obs_linedecomp_readout.py decompose \
  --capture "$C/Eprime/run1/work/det_obs_capture" \
  --line-list "$C/Eprime/run1/input/model/line_list.csv" \
  --deck "$C/Eprime/run1/input/deck.sha256" \
  --prior "$C/readout_b1/E" \
  --exe "$D/build/det_obs_linedecomp" \
  --threads 16 \
  --out "$D/results/$RUN_ID/E"

python3 -B scripts/det_obs_linedecomp_readout.py decompose \
  --capture "$C/Rprime/run1/work/det_obs_capture" \
  --line-list "$C/Rprime/run1/input/model/line_list.csv" \
  --deck "$C/Rprime/run1/input/deck.sha256" \
  --prior "$C/readout_b1/R" \
  --exe "$D/build/det_obs_linedecomp" \
  --threads 16 \
  --out "$D/results/$RUN_ID/R"

python3 -B scripts/det_obs_linedecomp_readout.py paired \
  --e "$D/results/$RUN_ID/E" \
  --r "$D/results/$RUN_ID/R" \
  --out "$D/results/$RUN_ID/paired"
```

`P`, `C`, `D`, `RUN_ID`의 실제 값, 작업 디렉터리, 빌드 명령, 전체 실행 명령과 산출 신원을 manifest에 기록한다. 재현 기록을 위해 같은 대형 계산을 한 번 더 실행하지는 않는다.

CLI가 달라지면 운전자가 실제 명령과 차이를 집행 기록에 남긴다. 물리 정의·입력·산출·게이트가 달라지는 변경은 단순 CLI 차이로 처리하지 않는다.

## §10. 완료 기준과 판정

다음을 모두 갖추면 운전자는 목표 달성 여부를 판정한다.

1. G-JOIN과 G-DECOMP 통과.
2. 두 대역의 E′·R′ 기본 묶음표와 이온별·셸별 주변표.
3. 부호와 상쇄를 보존한 E′−R′ 차이 및 다섯 요인항.
4. 선행 mapped 광도와 차이에 대한 폐합.
5. 개별 광도·요인·분율·순위에 수송 오차가 전파된 구간.
6. 지배 후보 또는 수치 미분리의 명시.
7. 자원·입력 보존·이탈·이해관계 기록.
8. 운전자 판정과 종료감사 기록.

판정은 다음과 같다.

- **PASS:** 등록 산출·폐합·구간이 갖춰졌다. 과반 이온이 없거나 요인이 수치적으로 분리되지 않는 결과도 정직하게 보고했다면 PASS가 가능하다.
- **REWORK:** 새 도구의 연결·수송·요인·구간 처리 오류로 목표 산출이 아직 성립하지 않는다.
- **BLOCKED:** 봉인 입력으로 대응을 정의할 수 없거나 자원 상한 등으로 완료하지 못했다. 부족한 산출과 필요한 결정을 특정한다.

`UNRESOLVED`는 유효한 계산의 과학적 분기다. 계산 오류·결손·상한 초과를 이 이름으로 감추지 않는다. 실패 후 산출 목표를 더 쉬운 중간 목표로 바꾸어 폐합하지 않는다.

다음 후보는 식별된 이온·셸과 우세 요인에 따라 좁히되, 실제 수리·재계산의 승인으로 해석하지 않는다.

## §11. 집행 기록·분장 장부

현행 분장은 이번 발주의 user 지시를 따른다. 코드 감사는 user 2026-10-06 지시로 해제되어 있으며 새 행·단계로 복원하지 않는다.

| 업무 | 담당 | 이해관계·규칙 |
|---|---|---|
| 기안·계약 개정·종료감사 | astra | 계약 기안자가 종료감사를 수행함을 명시 |
| Q-GOAL·Q-LEAN 사전감사 | Fable | 호출 실패 시 gpt-6.1-sol 대행, 실패·대행 신원 기록 |
| 구현·직접시험·기계 게이트·실행·판독·판정 | 운전자 | 구현자와 판정자 겸임 |
| 범위·비용 증가·후속 물리 결정 | user | 현재 승인 범위 밖 결정 |

빈칸은 미집행이다. 집행 기록으로 계약 수용 기준을 바꾸지 않는다.

| 집행 항목 | 담당 | 상태·경로·신원·결과 |
|---|---|---|
| user 승인 | user | 2026-10-07 「승인」, §0 경로 |
| 본문 기안 | 기안자 | 본 문서; read-only; 실제 모델 식별 상태는 첫 줄 |
| 정본 파일화·기준선 연결 | 운전자 | e578af1(기안 + 개정 1); 기준선 = DET-OBS-CAPTURE bf92ecf, `readout_b1/` |
| 기계 프리플라이트 | 운전자 | lageunha: prereg·gate_referents·doc_canonical PASS; 사전감사 발주서 dispatch PASS |
| Q-GOAL·Q-LEAN 사전감사 | Fable 또는 기록된 대행 | Fable 429(`/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/evidence/fable_preaudit_failed.txt`) → gpt-6.1-sol 대행 `/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/preaudit_sol.md`: 부합·적정·차단 0 |
| 오프라인 구현·빌드 신원 | 운전자 | `/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/tools/`, 빌드 `/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/build/f3/`(도구 240b6fb6…, 경고 0) |
| G-JOIN·G-DECOMP 직접시험·음성대조 | 운전자 | C 6/6, 판독기 18/18 (`/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/results/f3/selftest_b3/`) |
| E′·R′ 수송 계수와 묶음 분해 | 운전자 | 정본 `/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/results/f3/{E,R}` VALID, 폐합 1·2 통과 (f2 는 판독기 분기 결함 전 판) |
| 짝 요인 분해·구간·분기 | 운전자 | 정본 `/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/results/f3/paired_b3` VALID, 폐합 3·4 통과, 양·음 쪽 분류 |
| Slurm 상태·CPU 예약 비용·메모리 | 운전자 | a10 job 415012·415066·415067·415069·415075·415080, 누적 32:07, 약 8.57 core·h, MaxRSS 4.3 GB(f2) |
| 목표 달성 판정 | 운전자 | PASS `docs/VERDICT_DET_OBS_LINEDECOMP_2026-10-07.md`; 구현자 겸임 |
| 종료감사 | astra | 1회차 `/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/astra_closeout.md`: 폐합 불가·차단 2(B1 과반 경계, B2 음의 쪽 분류) → 판독기 수리·f3 재판독; 2회차 `/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/astra_closeout2.md`: B1·B2 해소, 신규 차단 B3 → 수리·`paired_b3`; 3회차 대기 |
| 이탈·미해결·후속 후보 | 실제 담당 | 판정문 §4(9건)·§6 |

대행이나 실측 모델 ID 차이는 숨기지 않는다. 다른 모델의 산출을 astra 또는 Fable의 실제 수행으로 재명명하지 않는다.

## §12. 기계 선언

아래 분기는 유효한 광도·양/음 차이·요인 분류 각각에 적용한다. `v`는 실제 구간 계산 뒤 판독기가 내는 단일 코드다.

- `0`: 과반 식별.
- `1`: 혼합·최대 식별.
- `2`: 수치 경계·0과 미분리·분모 부적격 등 유효한 미분리 결과.
- 그 밖·결손: 입력·계산·증거 부적격.

`2`에는 원인을 별도 필드로 기록한다. 전체 묶음 PASS와 개별 과학 분기를 구별한다. 이 선언의 형식적 통과만으로 물리 계산·폐합을 검증했다고 주장하지 않는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-OBS-LINEDECOMP (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/tools/(?:det_obs_linedecomp(?:_selftest)?\\.c|build_linedecomp\\.sh|run_linedecomp\\.slurm)|scripts/det_obs_linedecomp_readout\\.py)",
    "symbol": "det_obs_linedecomp_",
    "roots": [
      "/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/tools/det_obs_linedecomp.c",
      "/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/tools/det_obs_linedecomp_selftest.c",
      "/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/tools/build_linedecomp.sh",
      "/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/tools/run_linedecomp.slurm",
      "scripts/det_obs_linedecomp_readout.py"
    ],
    "expected_extra": [
      "/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/tools/det_obs_linedecomp.c",
      "/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/tools/det_obs_linedecomp_selftest.c",
      "/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/tools/build_linedecomp.sh",
      "/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/tools/run_linedecomp.slurm",
      "scripts/det_obs_linedecomp_readout.py"
    ]
  },
  "branches": {
    "regimes": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [9.0, 9.0]],
    "metrics": {
      "valid": "len(v) == 1 and all(x in (0,1,2) for x in v)",
      "majority": "sum(1 for x in v if x == 0)",
      "largest": "sum(1 for x in v if x == 1)"
    },
    "rules": [
      {"name": "X", "predicate": "not valid"},
      {"name": "MAJORITY", "predicate": "valid and majority == 1"},
      {"name": "MIXED_LARGEST", "predicate": "valid and largest == 1"},
      {"name": "UNRESOLVED", "predicate": "valid and majority == 0 and largest == 0"}
    ],
    "adversarial_fixtures": [
      {"name": "majority", "mix": [[0,1]]},
      {"name": "mixed-largest", "mix": [[1,1]]},
      {"name": "boundary-or-zero", "mix": [[2,1]]},
      {"name": "invalid", "mix": [[9,1]]},
      {"name": "missing", "mix": []},
      {"name": "extra", "mix": [[0,1],[1,1]]}
    ]
  },
  "references": [
    {"path": "CLAUDE.md"},
    {"path": "docs/PREREG_DET_OBS_CAPTURE_2026-10-06.md"},
    {"path": "docs/VERDICT_DET_OBS_CAPTURE_2026-10-07.md"},
    {"path": "scripts/check_prereg_preflight.py"},
    {"path": "scripts/check_gate_referents.py"},
    {"path": "scripts/check_doc_canonical.py"},
    {"path": "scripts/check_dispatch_preflight.py"},
    {
      "path": "scripts/det_obs_capture_readout.py",
      "flags_existing": [
        "def band_with_bound(",
        "def band_of_bound(",
        "def component_bands("
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c",
      "flags_existing": [
        "det_obs_capture_march_vec",
        "det_obs_capture_cont_vec",
        "det_obs_capture_bin_vec"
      ]
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/tools/det_obs_capture_replay.c"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/astra_closeout.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/astra_closeout2.md"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/readout_b1/E/result.json"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/readout_b1/R/result.json"
    },
    {
      "path": "/gpfs/kjhan/lumina/cmfgen_map/obs_linedecomp_20261007/evidence/USER_DECISION_2026-10-07.txt"
    }
  ]
}
```

## 기안 메모

- 전체 입력의 기하·선 순서 대응은 구현 시 G-JOIN으로 확인한다. 본 기안에서 대형 배열 순회나 계산을 실행하지 않았다.
- 새 역방향 계산의 실행시간은 미측정이다. 120분은 선행 비용을 참고한 등록 지출 상한이며 성능 예측이 아니다.
- 하류 수송항의 opacity 주체별 재분해는 의도적으로 제외했다. 이번 결과로 필요한 후속 질문을 특정한다.
- 본 기안의 실제 세부 모델 ID는 확인되지 않았다. 운전자는 호출 헤더의 실제 ID를 장부에 기록하고, 요청된 astra 수행 여부와 구별한다.