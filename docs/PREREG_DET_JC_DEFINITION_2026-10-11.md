# 사전등록 — DET-JC-DEFINITION: Sobolev 브래킷 입사장 정의와 선 포함 여부 점검

2026-10-11 · 읽기 전용 기안  
운전자 파일화 대상: `docs/PREREG_DET_JC_DEFINITION_2026-10-11.md`  
상태: **기안. 소스·기존 기록 조회 완료. 신규 계산·구현·게이트·판정·종료감사 미집행.**

## §0. Q-GOAL·Q-LEAN 자기답

**Q-GOAL — 최종목적에 직접 기여하는가? 자기답: PASS.**

최종목적은 공개 CMFGEN과 비교 가능한 SN Ia 이온화 상태와 스펙트럼을 얻는 것이다. 직전 단은 대표선의 생산 입력 연쇄와 전달을 확인했지만, Sobolev 브래킷의 \(c=\beta J_c\)에 들어가는 장의 정의까지 판별하지 못했다.

이번 단은 **실제로 소비된 생산 분기의 자기 선·다른 선 포함 여부와 CMFGEN 대응 정의를 대표선 여섯 행의 표 하나로 확정**한다. 정의 차이가 있으면 기존 자료로 크기를 식별할 수 있는지까지 판정한다. 다음 결정은 정의 수리를 별도 발주할지, 다른 기전을 조사할지, 필요한 장 자료의 확보를 발주할지다.

생략하면 비활성 선 침적 코드를 실제 생산 경로로 오인하거나, 거친 장과의 차이를 자기 선의 이중 처리로 잘못 해석할 수 있다.

**Q-LEAN — 불필요한 측정·게이트를 더하는가? 자기답: APPROPRIATE.**

남기는 것은 실행 분기·상류 불투명도 계보, CMFGEN 정의 대응, 대표선의 기존 차이와 거친 불투명도 맥락이다. 모두 주 표의 판단에 소비된다. 소스 확정에는 새 코드를 만들지 않는다.

복사전달·정상해·전체 선 감도·추가 원소·추가 셸 계산은 0이다. 저장 산출 재검·쓰기 결함 모사·결정표 재계산 장치를 만들지 않는다. 직접시험은 추가 판독 코드의 음성 대조로 한정한다. 종료감사의 1차 대상은 **과학 표의 수치·대응·주장 한정**이다.

기안 조회에서 실제 생산 호출12의 `CMFGEN_NONOVERLAP_SOBOLEV` 경로와 직접 선 침적 제외를 확인했다. 이를 숨기고 자기 선 포함 여부를 완전히 미지인 가설처럼 등록하지 않는다. 상류 유입·CMFGEN 대응·크기 식별 한계는 별도로 확인한다.

## §1. 권한·입력·봉인 기준

사용자 결정은 「J_c 정의 점검 — 소스·오프라인」이다. 기안·개정·모든 감사는 astra, 구현·직접시험·게이트·실행·판정은 운전자다. GPU 0·생산 변경 0이므로 경량 사전감사는 생략한다(25-ab). 코드 감사는 없다(25-z). 종료감사는 astra가 수행하며 계약 저자 이해관계를 명시한다.

| 기호 | 대상 |
|---|---|
| P | `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn` |
| B | `/gpfs/kjhan/lumina/cmfgen_map/jcdef_20261011` |
| S | `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src` |
| T | `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1` |
| V | `/gpfs/kjhan/lumina/cmfgen_map/lineinp_20261011/results/run1` |
| C | `/gpfs/kjhan/cmfgen_src/cur_cmf` |

직전 폐합 기준은 `VERDICT_DET_SI_LINE_INPUT_2026-10-11.md`의 edc058a 판본이다. 과거 판정·저작자는 변경하지 않는다.

S의 생산 소스와 T의 실제 실행 연결은 기존 `STAGE_IDENTITY.txt`·`input/run_build_identity.json` 및 직전 증거를 소비한다. 계측 소스와 봉인 소스 전체가 동일하다고 가정하지 않는다. 차이가 있으면 이번에 소비하는 분기·배열 구성·호출에 영향을 주는 범위만 확인한다.

실제로 소비한 파일의 경로·크기·SHA-256, 판독기 판본, 명령·작업 번호를 보조 증거에 한 번 기록한다. 전체 트리 목록·해시는 만들지 않는다.

## §2. 과학 질문·고정 대상

> 호출36에 전달된 대표선별 \(J_c\)는 자기 선 또는 겹치는 다른 선의 불투명도·방출을 포함한 미세장인가? 해당 선을 제외한 입사장인가? 모든 직접 선 항을 제외한 연속장인가? 실제 실행 분기를 따라 구분하고 CMFGEN `SOBJBAR_SIM`의 입사 항과 대응시킬 수 있는가?

정의 차이가 확인되면 두 번째 질문은 다음과 같다.

> 같은 물질·경계·전달 조건에서 그 차이가 \(J_c\)를 낮추는 양을 기존 자료로 식별할 수 있는가? 식별할 수 없다면 어떤 정보가 없어 부호·크기·상한을 정할 수 없는가?

대표선은 다음 여섯 개로 고정한다. 결과에 따라 교체하거나 순위를 다시 계산하지 않는다.

| 묶음 | `aux` | Si II 하→상 `level_num` | λ(Å), 식별용 | 직전 \(\log_{10}(J_c/J_{\nu,\mathrm{coarse}})\) |
|---|---:|---|---:|---:|
| G1 | 760321 | 1→7 | 1533.4 | −0.530196 |
| G1 | 632403 | 1→10 | 1264.7 | −0.867113 |
| G1 | 757019 | 0→7 | 1526.7 | −0.717288 |
| G2 | 1676767 | 7→12 | 6348.9 | −0.223144 |
| G2 | 1353542 | 6→12 | 3857.1 | −0.097575 |
| G2 | 1679119 | 7→11 | 6373.1 | −0.194388 |

마지막 열은 직전 정본에서 인용한 값이며 새로 계산하지 않는다.

선 신원은 기존 `(call,op)`·`aux`·이온·상하 준위·원자 주파수를 함께 사용한다. 계산에는 반올림 파장을 쓰지 않는다. 생산 원장 호출12·물질12·복사12와 소비 호출36을 구분한다.

공통 맥락은 인용값으로 둔다: 셸10 Lumina F0 III/II=3.559757, 공개 CMFGEN=105.697(약 105.70), 19.480 d, 속도 대응 11,544 ↔ 11,534.59 km/s. 이번 단은 이 비를 재계산하거나 변화량을 예측하지 않는다.

## §3. 잣대 1 — 실제 Lumina 실행 경로

**함수 이름·주석·비활성 코드의 존재는 실행 증거가 아니다.** 다음 연결을 표의 공통 근거로 기록한다.

1. T의 실제 설정, 실행 소스·바이너리 연결, `init_pass` 여부.
2. `cmfgen_fine_jbar`에 전달된 `line_operator`와 `line_net_parity`.
3. 미세장 불투명도·방출의 모든 유입 경로.
4. 장 풀이 후 가우스 평균과 `line_net_sobolev_radiation` 소비.
5. 생산 원장 호출12에서 소비 호출36까지의 직전 전달 근거.

기안 조회에서 확인한 사항은 다음과 같다.

- 실행 설정은 `LUMINA_CMF_LINERES_JBAR=2`, `LUMINA_CMF_FINE_UNMAPPED_EPS=1`, `LUMINA_CMF_FINE_CONT_NLTE_SOURCE=1`, `LUMINA_CMF_FINE_VDOP=1000000`이다.
- `lumina_cmfgen.c` 약 11770행은 `init_pass`이면 `INIT_SHARED_GAUSSIAN`, 이후에는 `CMFGEN_NONOVERLAP_SOBOLEV`를 전달한다. 설정값 하나만으로 모든 호출의 연산자가 같다고 하지 않는다.
- `stderr.log`의 생산 호출12는 `operator=CMFGEN_NONOVERLAP_SOBOLEV`, 물질12·복사12를 명시한다.
- `lumina_cmfgen.c` 약 8535행은 요청 연산자와 parity 조건으로 `sobolev_operator`를 정한다. 해당 연산자에서 parity가 없으면 정상 생산으로 계속하지 않는다.
- 같은 파일 약 8804행과 8883행의 두 선 침적 경로는 모두 `!sobolev_operator`를 요구한다. 따라서 호출12에 직접 가우스 선 침적 루프를 적용해 설명할 수 없다.
- 같은 파일 약 9780행은 `fs.J`를 가우스 평균하여 `line_value`를 만들고, 약 9810행은 이를 Sobolev 방사 항의 입사장으로 전달한다.

**직접 침적 제외와 상류 선 기여 제외를 구분한다.** `fs.chi_line=0`만 확인하고 모든 선 기여가 없다고 판정하지 않는다.

상류 확인은 다음에 한정한다.

- `chi_abs`: 약 2887행의 bf+ff 구성과 약 2963행의 대입, 약 8399행의 보간을 연결한다.
- `chi_es`: 약 2962행의 선 산란 몫 혼입, 약 2530행 부근의 `g_chi_line_sc_unmapped` 누적, 약 8394행의 차감을 연결한다.
- 실제 mapped/unmapped 분류와 활성 분기에서 차감 대상이 전체 혼입 몫을 덮는지 확인한다. 배열 존재·크기·같은 조립 호출의 연결이 빠지면 잔여 기여를 미확정으로 남긴다.
- 연속 방출은 약 2980행 부근의 선 방출을 제외한 생산과 약 9040행 부근의 `S_fixed` 조립을 확인한다.
- `LUMINA_CMF_FINE_BF_OPAC` 등 미기재 선택지는 봉인 기본값과 실제 호출을 함께 확인한다. 미기재를 일률적으로 0으로 해석하지 않는다.

인구·온도·전자밀도가 이전 선 상호작용의 영향을 받았다는 사실은 **고정 물질에서 전달 연산자에 자기 선을 직접 넣었다는 뜻이 아니다.** 간접 자기일관 영향은 이번 대상에서 제외한다.

## §4. 잣대 2 — CMFGEN 대응 정의

CMFGEN은 **명시된 소스 경로의 정의 비교**에 사용한다. 공개 모델의 여섯 선이 실제로 `SOBJBAR_SIM`으로 계산되었다는 실행 증거는 이번 입력에서 확보되지 않았다.

확인할 호출 계보는 다음과 같다.

- `cmfgen_sub.f` 약 3504행: `SECTION='LINE'`에서 `COMP_OPAC`·`COMP_J_BLANK`, 이후 `SOURCE=ZETA+RJ*THETA`.
- 같은 파일 약 3535행: 동시 처리 선들의 `CHIL`·`ETAL` 합산.
- 같은 파일 약 3638행: `SOBOLEV` 분기의 `SUB_SOB_LINE_V3`.
- `sub_sob_line_v3.f` 약 131행: `SOBJBAR_SIM` 호출.
- `sobjbar_sim.f` 약 100–165행: `CHI`·`SOURCE`·경계조건으로 광선별 `AV`를 구한다.
- 같은 파일 약 170–210행: `EXPONX(GAM)`을 곱해 각도 적분하고 선별 순복사 항으로 환산한다.

입사 항의 비교 대상은 다음과 같다.

\[
C_{\mathrm{CMF}}=\sum_k w_k\,\beta_k\,AV_k,
\qquad
\beta_{\mathrm{CMF}}=\sum_k w_k\,\beta_k.
\]

\(\beta_{\mathrm{CMF}}>0\)이면 대응 입사장을 \(J_{c,\mathrm{eff}}=C_{\mathrm{CMF}}/\beta_{\mathrm{CMF}}\)로 **정의할 수 있다**. 이는 이번에 공개 모델에서 추출한 수치가 아니다.

`AV`는 광선별 대칭 강도 변수이며 Lumina의 각도 평균 `fs.J`와 같은 배열이 아니다. 각도별 β가 일정할 조건에서만 β와 각도 평균을 분리할 수 있다. 일반 속도장에 Lumina의 스칼라 \(\beta J_c\)를 그대로 동일시하지 않는다.

`CHIL`·`ETAL`이 연속 전달의 `CHI`·`SOURCE`에 직접 더해지는지와, 선 군의 Sobolev 광학깊이·국소 방출에 들어가는지를 분리해 확인한다. `COMP_OPAC`·`COMP_J_BLANK`의 연속장 생성과 `IMPURITY_CODE`의 저장장 사용 경로도 구분한다.

`MOMJBAR`는 다른 경로의 대조 근거다. `momjbar.f` 약 119–124행은 `CHI+CHIL*PROF`를 구성한다. 이 전체 선장 정의를 `SOBJBAR_SIM`의 `AV` 정의로 옮기지 않는다.

겹침은 세 가지로 나눈다: 직접 미세 프로파일 중첩, 상류 평균 불투명도에 남은 다른 선 기여, `NUM_SIM_LINES`의 동시 Sobolev 처리. 주파수가 가까운 선 목록만으로 실제 중첩 처리를 인정하지 않는다.

## §5. 잣대 3 — 기존 자료로 허용되는 크기 판별

우선 기존 \(J_c\)·거친 \(J_\nu\)·dex 차이를 인용하고, 선 중심이 속한 비교 구간의 불투명도 성분을 연결한다.

구간 선택은 `[nu_lo,nu_hi)`이며 마지막 끝점만 포함한다. 같은 구간에 여러 대표선이 있으면 같은 거친 값을 공유했음을 표시한다. 외삽·임의 보간은 하지 않는다.

거친 구간에서 다음만 계산한다.

\[
\chi_{\mathrm{cont},b}=\chi_{\mathrm{es},b}
+\chi_{\mathrm{bf},b}+\chi_{\mathrm{ff},b},
\qquad
q_{\mathrm{bb},b}=
\frac{\chi_{\mathrm{bb},b}}{\chi_{\mathrm{total},b}}.
\]

성분합과 저장 `chi_total`의 대응도 확인한다. 이 \(q_{\mathrm{bb},b}\)는 **비교 구간의 성분 비율**이며 미세장 자기 선의 기여율이 아니다. 산란 몫의 분류를 소스로 설명하고 중복 합산하지 않는다.

`physics_comparison.c` 약 43행은 공개 방사장을 적분 보존 구간상수 방식으로 목표 불투명도 구간에 재격자화한다. 약 596행은 그 \(J_\nu\)와 불투명도 뷰의 성분을 함께 기록한다. 따라서 CSV의 모든 열이 `fs` 원배열을 같은 방법으로 평균한 값이라고 가정하지 않는다.

미세장은 상류 연속 성분의 log-ν 위치에서 선형 보간을 사용하고, 가우스 평균은 선 중심 ±4 Doppler 폭에서 수행된다. 거친 선 불투명도는 선별 침적을 구간에 집계한 양이다. 비교 manifest의 물질13·복사12·불투명도23과 생산 물질12의 차이도 함께 적는다.

**허용하지 않는 추론:**

- `chi_bb/chi_total`, 불투명도 역비 또는 국소 \(e^{-\tau}\)를 \(J_c\) 감소율로 치환.
- 구간의 모든 `chi_bb`를 해당 대표선 하나에 배정.
- 거친 \(J_\nu\)를 자기 선 제거 후의 \(J_c\)로 간주.
- 선 방출·산란·경계·공간 전달을 고정하지 않고 선 추가가 항상 장을 낮춘다고 단정.
- 미세장 원배열이 없는데 거친 값을 보간하여 생산 가우스 평균을 복원했다고 보고.

정량적 정의 효과는 동일 물질·경계·전달·격자에서 포함 규칙만 다른 두 장 또는 이를 결정하는 충분한 기존 입력이 있을 때만 인정한다.

\[
\Delta_{\mathrm{def}}=
\log_{10}\!\left(J_c^{\mathrm{actual}}/
J_c^{\mathrm{matched\ incident}}\right).
\]

등록 입력에는 이 대응 장 쌍이 확인되지 않았다. 새 전달 계산은 하지 않는다. 따라서 소스에서 차이가 발견되어도 원칙적으로 **감소량의 부호·크기·유한 상한 미식별**을 허용한다.

직접 자기 선 유입 경로가 배제되면 “그 직접 중복 기전은 해당 차이의 설명에서 제외”라고 쓴다. 이를 전체 정의 차이 0 dex 또는 모든 선 영향 0으로 표시하지 않는다.

## §6. 결함 주장 성립·반증 조건

“자기 선 이중 계산”의 확정에는 다음 조건이 모두 필요하다.

1. 실제 생산 호출의 \(J_c\)에 자기 선의 전달 효과가 들어가는 경로가 확인된다.
2. 하류가 그 효과를 제거·보정하지 않고, 이미 포함한 국소 포획 또는 탈출 효과를 다시 적용한다.
3. 비교하는 입사장·국소 방출 분해가 같은 물질·선·정규화에서 성립하며, 단순한 다른 근사나 합법적인 분해가 아님을 식으로 보인다.

자기 선 포함 사실만으로 조건2·3을 충족했다고 하지 않는다. β가 두 위치에 보인다는 이유만으로도 확정하지 않는다.

**반증 조건:** 실제 분기가 직접 선 침적을 제외하고 상류 자기 선 유입도 배제되거나, 확인된 보정이 중복 효과를 제거하면 해당 이중 계산 가설은 반증된다. 비활성 초기화 경로에서만 자기 선을 찾은 경우도 실제 소비 호출에 대한 증거가 아니다.

상류 잔여 기여가 미확정이면 결함 확정도 반증도 유보한다. 다른 선의 배경 기여는 자기 선 이중 계산과 별도 판정한다.

결함의 구조적 성립과 관측된 dex 차이의 설명 능력은 별개다. 결함을 특정해도 감소량이 미식별이면 그대로 남긴다.

## §7. 주 산출·기대치

정본은 `jc_definition_table.csv`, 표시본은 같은 여섯 행의 `jc_definition_table.md`다. 표를 여러 독립 산출로 나누지 않는다.

| 열 묶음 | 필수 내용 |
|---|---|
| 신원 | 묶음·`aux`·`call`·`op`·상하 준위·원자 주파수·생산/소비 세대 |
| Lumina 정의 | 실제 연산자·직접 자기 선 포함 여부·상류 잔여 여부·다른 선/겹침 처리·소스 근거 |
| CMFGEN 대응 | `AV`·입사 적분·동시 선 처리·동일한 점/다른 점·공개 실행 대응의 확보 여부 |
| 기존 크기 | 생산 \(J_c\)·거친 \(J_\nu\)·직전 dex·구간 경계·불투명도 성분·\(q_{\mathrm{bb},b}\) |
| 효과·판정 | \(\Delta_{\mathrm{def}}\) 또는 미식별 사유·직전 차이와의 관계·이중 계산 주장 상태·선별 결론 |

`미확정`, `미식별`, `해당 없음`, 정확한 0을 구분한다. 공통 소스 근거는 보조 증거의 짧은 식별자로 참조할 수 있다.

사전 기대는 실제 호출12에서 직접 자기 선·다른 선 프로파일 침적이 제외된다는 것이다. 상류 차감까지 확인되면 자기 선 직접 중복 가설은 지지되지 않을 것으로 예상한다.

CMFGEN Sobolev 경로와는 입사장·국소 선 항의 분리에서 대응할 것으로 예상한다. 각도 처리·선 군·연속 전달·가우스 평균까지 동일할 것으로 예상하지 않는다.

거친 선 불투명도 비율의 크기와 대표선 순서, 남은 정의 차이의 감소 효과는 예측하지 못한다. 미세장 미보존 때문에 정량적 원인 귀속은 미식별로 남을 가능성을 사전등록한다.

수치 연결 허용오차는 \(M=\max(|a|,|b|)>0\)일 때 \(|a-b|/M\le10^{-10}\)이다. \(M=0\)이면 두 값이 정확히 0이어야 한다. 이산 신원은 정확히 일치해야 한다. 비·로그의 분모와 양수 조건을 만족하지 않으면 사유를 남기고 계산하지 않는다. floor를 넣지 않는다.

## §8. 최소 변경집합·보존

### 변경집합 — DET-JC-DEFINITION (표)

| 파일 | 허용 변경 |
|---|---|
| `scripts/det_si_line_input.py` | `jcdef`·`jcdef-negative` 부명령 추가: 기존 여섯 행 연결, 거친 성분 판독·등록 비율·주 표 출력만 |
| `/gpfs/kjhan/lumina/cmfgen_map/jcdef_20261011/tools/run_det_jc_definition_offline.slurm` | 제한 CPU 실행·입력 신원·비용 기록 |

### 변경집합 끝

기존 `LINES`·CSV 판독·`det_si_line_input_match`를 재사용한다. 기존 `analyze`·`negative`의 동작과 과거 출력은 바꾸지 않는다. 생산 연쇄·정상해·대표선 순위는 다시 계산하지 않는다.

소스 정의·결함 판정은 근거를 읽어 표에 작성한다. 문자열 검색 결과를 과학 판정으로 바꾸는 새 장치는 만들지 않는다.

보조 산출은 주 표에 소비되는 소스 계보·설정·미식별 사유·입력 신원의 `evidence.json`, 실행 로그, 새 코드 음성 대조 결과뿐이다.

생산 소스·바이너리·실행 입력·CMFGEN 소스 변경은 0이다. 신규 출력은 B의 새 실행 경로에만 쓴다. 기존·실패 산출은 보존한다. 일반 파일 읽기·쓰기만 사용하며 하드링크·파일시스템 고유 기능·clamp/floor는 도입하지 않는다.

## §9. 직접시험·게이트

새 코드의 음성 대조는 다음으로 한정한다.

- 다른 선·셸 또는 중복 행을 연결해 거부되는지 확인.
- 구간 경계·Hz 단위 오연결을 거부하는지 확인.
- 비유한 성분·잘못된 성분합·0 분모를 정상 비율로 출력하지 않는지 확인.

각 시험은 유효 입력과 실제 결함 주입을 함께 확인한다. 소스 결론·기존 수식·저장 파일 쓰기·과학 분기를 시험 장치로 복제하지 않는다.

| 게이트 | 막는 실재 실패 |
|---|---|
| G-PATH | 초기화·비활성 분기 또는 다른 생산 세대를 실제 소비 경로로 판정하는 실패 |
| G-MAP | 선·셸·구간·불투명도 분류를 잘못 연결해 수치를 귀속하는 실패 |
| G-CLAIM | 거친 성분 비율을 장 감소량으로 바꾸거나 소스 대응을 공개 실행 동일성으로 확대하는 실패 |
| G-RUN | GPU·전달 재실행·생산 변경·비용 초과를 등록 결과에 섞는 실패 |

과학적 정의 차이나 정직한 미식별은 게이트 실패가 아니다. 필요한 근거가 없는데 확정했다고 표시한 경우가 실패다. 기존 기계 프리플라이트는 기존 도구로 집행한다.

## §10. 비용·실행·재현

| 항목 | 상한 |
|---|---|
| GPU·생산 실행·복사전달·정상해 재계산 | 모두 0 |
| 실행 위치 | a10 CPU Slurm, 작업명 `LUMINA_` 접두 |
| 작업당 | 1 task·1 CPU·4 GiB·벽시계 180 s |
| 누적 | 할당 벽시계 600 s·\(1/6\) CPU core·h |
| 스레드·신규 저장 | OpenMP·BLAS 각 1, 10 MiB |

직접시험·기계 검사·수정 실행·감사 재계산을 누계에 포함한다. 로그인 노드에서는 조회만 하며 수치 계산·해시·시험을 실행하지 않는다.

Slurm 요청은 `--partition=a10 --ntasks=1 --cpus-per-task=1 --mem=4G --gres=gpu:0 --time=00:03:00 --exclude=syn102,syn03`이다. 실제 GPU 할당 0을 확인한다. 러너는 `module purge` 후 필요한 모듈을 명시하고 Python 판본을 기록한다.

작업 내부의 예정 인터페이스:

```bash
python3 -B scripts/det_si_line_input.py jcdef-negative
python3 -B scripts/det_si_line_input.py jcdef \
  --run /gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1 \
  --baseline /gpfs/kjhan/lumina/cmfgen_map/lineinp_20261011/results/run1 \
  --out /gpfs/kjhan/lumina/cmfgen_map/jcdef_20261011/results/run1
```

재현은 같은 입력·코드·명령과 새 출력 경로를 사용한다. 소스 논증의 확인과 수치 실행 재현을 구분한다. 상한 내 완료하지 못한 항목은 미완료 사유를 남기며 권한을 자동 확대하지 않는다.

## §11. 과학 분기·폐합·주장 상한

| 코드 | 분기 | 조건 |
|---:|---|---|
| 0 | `DEFINITION_DIFFERENCE` | 실제 경로의 선 포함 또는 입사 항 정의 차이를 소스로 특정함 |
| 1 | `INCIDENT_SEPARATION_CONFIRMED` | 여섯 선의 자기 선 유입이 배제되고 CMFGEN Sobolev 경로와 입사장·국소 선 항의 분리가 대응함 |
| 2 | `INDETERMINATE` | 실제 분기·상류 유입·CMFGEN 대응 중 필수 근거가 부족함 |

분기0은 결함 확정과 동의어가 아니다. 어떤 정의가 다르고 어떤 조건에서 비교했는지 적는다. 가우스·각도·선 군 처리의 차이도 수치적 중요성이 입증된 것처럼 표시하지 않는다.

분기1은 전체 연산자의 수치적 동등성을 뜻하지 않는다. 조건부 각도 대응 등 남은 차이를 표에 명시한다. 일부 선만 미확정이면 전체 분기1을 선언하지 않는다.

크기 판정은 분기와 별도로 `식별`, `미식별`, `직접 중복 기전 배제` 중 하나를 행별로 기록한다. 정의 확정·크기 미식별을 동시에 보고할 수 있다.

폐합에는 여섯 행의 과학 표, 운전자 판정, astra 종료감사가 필요하다. 정의 미확정이면 **정의 판별 목표 미달**을 명시한다. 크기 미식별이면 그 이유와 부족한 입력을 적으며, 감소량을 설명했다고 하지 않는다.

종료감사는 수치·선 대응·상류 계보·CMFGEN 식 대응·결함 주장 조건을 우선 확인한다. 기안자와 종료감사자가 같은 astra라는 이해관계를 명시한다.

생산 승격·물리적 정답·자기일관 효과·스펙트럼 개선·철족 일반화를 선취하지 않는다. 정의 차이 또는 결함을 찾더라도 수리는 이 단 밖이다.

기안자는 문서·설정·소스·로그·기존 CSV만 조회했다. 파일 작성·신규 계산·직접시험·제출·게이트·판정은 하지 않았다. 운전자는 집행 항목·실제 담당·작업 번호·누적 비용·판정·종료감사 상태를 추가한다.

### 집행 기록 (운전자 추가 — 정본 장부, 개정16 ①; 위 기안 기록은 그대로 둔다)

증거 접두 `B=/gpfs/kjhan/lumina/cmfgen_map/jcdef_20261011`. 작업 번호·경과는 `B/JOBS.txt`·`B/evidence/sacct_offline_jobs.txt`, 게이트 집행은 `B/evidence/GATE_RECORD.json`.

| 항목 | 담당(실제) | 상태·증거 |
|---|---|---|
| user 결정 | user | 「J_c 정의 점검 — 소스·오프라인」(`B/evidence/USER_DECISION_2026-10-11.txt`) |
| 기안·개정 1 | astra(자기보고 「gpt-6-astra」·「GPT-6-astra」) | a1c4ed7 · ada2109(OLD/NEW 2쌍, 비한국어 잔재 2곳 정리, 행 단위 2행 치환·의도 외 소실 0) |
| 기계 게이트(기안·개정) | 운전자 | 기안 브리프 canonical(a10 419391), 기안 prereg·referents·canonical rc 0(419395), 개정 브리프 1차 dispatch rc 1(뒷겹에 계약 커밋 해시 재서술 — 발주서 결함, 고침; 419397) → 2차 rc 0(419398), 개정 후 rc 0(419400) |
| 소스 확정(§3–§4) | 운전자(판독) | 생산 분기·미세장 직접/상류 선 유입·연속 방출·CMFGEN `COMP_OPAC`·`COMP_J_BLANK`·`SOBJBAR_SIM` 정의를 봉인·CMFGEN 소스에서 읽고 `B/results/run1/evidence.json` 의 `source` 에 기록(문자열 검색 판정 장치 없음) |
| 구현 | 운전자(Opus 5.5) · 코드 감사 없음(25-z) | 3a8e5c2 — `scripts/det_si_line_input.py` 에 `jcdef`·`jcdef-negative` 추가(기존 명령 무변경), 러너 `B/tools/run_det_jc_definition_offline.slurm` |
| 직접시험(§9) | 운전자 | a10 419404 **11/11**(양성 대조 3·결함 주입 거부 8), 러너 위생 rc 0 |
| 등록 계산 run1 | 운전자 | a10 419406(2 s) `B/results/run1/`: G-PATH·G-MAP·G-CLAIM·G-RUN 지적 0; **분기 1 INCIDENT_SEPARATION_CONFIRMED** — 6/6 자기 선 유입 배제·크기 「직접 중복 기전 배제」, 비교 구간 q_bb 0.62–0.98 |
| 판정 | 운전자(구현·판정 겸임) | `docs/VERDICT_DET_JC_DEFINITION_2026-10-11.md` — 종료감사 대기 |
| 종료감사 | astra | 대기 |
| 비용 | 운전자 | GPU 0; 오프라인 a10 7 작업 할당 67 s · 0.019 core·h / 600 s · 1/6 core·h(기안·개정 게이트 5 작업 65 s 포함); 신규 저장 59 KiB / 10 MiB |

**이탈(기재):** 없음. 개정 브리프의 1차 발주 게이트 거부는 발주서 결함이며 계약 문면 변경이 아니다.

## §12. 기계 선언

참조에는 실재 파일만 둔다. 예정 파일은 변경집합에만 둔다. 아래 fixture는 기존 프리플라이트용이며 새 과학 결정 장치를 요구하지 않는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-JC-DEFINITION (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_si_line_input\\.py|/gpfs/kjhan/lumina/cmfgen_map/jcdef_20261011/tools/run_det_jc_definition_offline\\.slurm)",
    "symbol": "det_si_line_input",
    "roots": [
      "scripts/det_si_line_input.py",
      "/gpfs/kjhan/lumina/cmfgen_map/jcdef_20261011/tools/run_det_jc_definition_offline.slurm"
    ],
    "expected_extra": [
      "scripts/det_si_line_input.py",
      "/gpfs/kjhan/lumina/cmfgen_map/jcdef_20261011/tools/run_det_jc_definition_offline.slurm"
    ]
  },
  "branches": {
    "regimes": [[0,0],[1,1],[2,2],[9,9]],
    "metrics": {
      "valid": "len(v) == 1 and all(x in (0,1,2) for x in v)",
      "c0": "sum(1 for x in v if x == 0)",
      "c1": "sum(1 for x in v if x == 1)"
    },
    "rules": [
      {"name":"DEFINITION_DIFFERENCE","predicate":"valid and c0 == 1"},
      {"name":"INCIDENT_SEPARATION_CONFIRMED","predicate":"valid and c1 == 1"},
      {"name":"INDETERMINATE","predicate":"not valid or (c0 == 0 and c1 == 0)"}
    ],
    "adversarial_fixtures": [
      {"name":"difference","mix":[[0,1]]},
      {"name":"separation","mix":[[1,1]]},
      {"name":"indeterminate","mix":[[2,1]]},
      {"name":"invalid","mix":[[9,1]]},
      {"name":"missing","mix":[]},
      {"name":"multiple","mix":[[0,1],[1,1]]}
    ]
  },
  "references": [
    {"path":"docs/PREREG_DET_SI_LINE_INPUT_2026-10-11.md"},
    {"path":"docs/VERDICT_DET_SI_LINE_INPUT_2026-10-11.md"},
    {"path":"scripts/det_si_line_input.py","flags_existing":["det_si_line_input_match","det_si_line_input_ledger"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/jcdef_20261011/evidence/USER_DECISION_2026-10-11.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/lineinp_20261011/results/run1/si10_line_input_table.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/input/resolved_lumina.exports"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/stderr.log"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/work/w5_lines.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/work/physics_comparison/physics_DET_iter0010.spectral.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1/work/physics_comparison/physics_DET_iter0010.manifest.json"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_jbar.c"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/line_net_rate.c"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/physics_comparison.c"},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f"},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/comp_opac.f"},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/comp_j_blank.f"},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/sub_sob_line_v3.f"},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/subs/sobjbar_sim.f"},
    {"path":"/gpfs/kjhan/cmfgen_src/cur_cmf/subs/momjbar.f"}
  ]
}
```
