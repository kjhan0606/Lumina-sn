# 사전등록 — DET-SI-CONSUMPTION-TRACE: Si 셸10 이중 전리 권위의 소비 시점 귀속

2026-10-10 · 읽기 전용 기안  
운전자 파일화 대상: `docs/PREREG_DET_SI_CONSUMPTION_TRACE_2026-10-10.md`  
상태: **기안. 기계 게이트·경량 사전감사·구현·직접시험·빌드·실행·판정·종료감사 미집행.**

## §0. Q-GOAL·Q-LEAN 자기답

**Q-GOAL — 최종목적에 직접 기여하는가? 자기답: PASS.**

최종목적은 공개 CMFGEN과 비교 가능한 SN Ia 이온화 상태와 스펙트럼을 얻는 것이다. 이 단은 O 물질13의 Si 셸10에서 committed NLTE III/II=3.56369885와 stage III/II=213.664025가 갈리는 약 1.78 dex를 실제 소비 시점의 율·행렬·인구로 조사한다. 다음 수정의 대상이 정식화인지, solve·SL·CE·후처리인지, 둘 다인지 결정하는 것이 관측 가능한 끝점이다.

공개 CMFGEN III/II≈105.7과 두 권위의 거리를 같은 표에 둔다. 비교 가능한 값을 다시 숨기지 않되, **계측 재실행은 CMFGEN 대비 값을 개선하는 개입이 아니다.** 이 단을 생략하면 서로 다른 내부 권위 중 어느 계산 경로를 수정할지 수치 근거가 부족하다. 잔차 취득 자체만으로 단을 성공 처리하지 않는다.

**Q-LEAN — 각 측정이 어디에 소비되는가? 자기답: APPROPRIATE.**

| 유지하는 측정 | 주 표의 소비 칸·결정 |
|---|---|
| 실제 solve 입력·raw 해·선택 경로 | solve 잔차, 알고리듬 귀속 |
| 보존행 교체 전 율 항과 적용 인구 | 물리율 잔차·채널 순수지, 정식화 귀속 |
| SL 매핑·분배 가중치·FL 전이율 | SL/FL 잔차와 조건부 비 차이, 묶음 근사 귀속 |
| CE·n_e 내부 순서와 실제 상대 인구 | 같은 소비 상태의 대응, 반복·감쇠 귀속 |
| 감쇠·정규화·commit 전후 인구 | III/II 변화와 후처리 잔차, 알고리듬 귀속 |
| Si 관련 BB-NRB 입력·출처 | FL 연산자와 준위별 광전리 수지의 입력 식별 |
| 같은 세대 저장량과 O의 대조 | O의 1.78 dex에 귀속할 자격 |
| 공개 시각·속도·Si II/III | 공개 행과 두 권위의 dex |
| 입력·빌드 신원·비용·완료 표지 | 표의 증거 신원과 실행 범위 |

삭제 대상은 전 원소 trace, 전 스펙트럼 재비교, 새로운 저장 산출 재검 장치, 쓰기 결함 모사 배터리, 결정표 재계산 장치, 판독기 자체 검증 체계다. 새로운 과학 산출은 표 하나이며, 보조 자료는 그 표의 수치·대응·한정에 소비되는 것만 남긴다.

물질7로 줄이면 더 싸지만 O 물질13의 동일 질문을 대신하지 못한다. 따라서 한 번의 물질13 재취득과 작은 CPU 판독으로 묶는다. 구현·시험·게이트·판독을 각각 별도 단으로 닫지 않는다.

## §1. 기준선·경로·역할

| 식별자 | 고정 대상 |
|---|---|
| B — 작업 기반 | `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010` |
| T — 새 계측 런 | `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/TRACE/run1` |
| O — 기준 런 | `/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1` |
| 봉인 소스 | `/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source` |
| 작업 소스 사본 | `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/source` |
| 진단 근거 | `/gpfs/kjhan/lumina/cmfgen_map/diag8_dualauth_20261010/DIAGNOSIS.md` |
| 공개 Si 자료 | `data/standart_data1/toy06/ionfrac_si_toy06_cmfgen.txt` |

사용자 선택은 B의 `evidence/USER_DECISION_2026-10-10.txt`에 기록된 「trace 재취득 GPU 런」이다. diag8의 폐합은 진단 주장 범위의 폐합이며 이번 귀속이나 재취득 성공의 승인이 아니다.

기안·개정·경량 사전감사·종료감사는 astra, 계측 구현·직접시험·러너·판독기·게이트·빌드·스테이징·제출·판정은 운전자다. 코드 감사는 두지 않는다.

경량 사전감사는 생산 소스 변경과 GPU 실행 때문에 빌드·구현 발주 전에 Q-GOAL·Q-LEAN 대상으로 1회 실시한다. **astra는 계약 기안자라는 이해관계를 감사문에 명시하며 독립 계약 감사로 표기하지 않는다.** 본문 자기답은 그 감사의 집행 기록이 아니다.

## §2. 과학 질문·포착 세대·중단점

질문은 다음 하나다.

> O와 동등한 물질13을 만드는 Si 셸10의 실제 소비 경로에서, stage와 committed NLTE의 약 1.78 dex 차이를 채널 정식화와 알고리듬 중 어느 쪽의 수치 증거가 설명하는가?

물질13을 선택한다. diag8에서 O 물질7의 stage/NLTE 비는 239.4782/5.89634이며 물질13의 213.664025/3.56369885와 다르다. E′·R′ 물질7의 갈림도 설정·상태가 다른 선행 증거다. 이른 상태는 갈림의 존재를 보이지만 O 물질13의 귀속을 증명하지 않는다.

동등한 재시작 경로가 입증되지 않았으므로 O와 같은 초기 입력에서 시작한다. `LUMINA_PURE_CMFGEN_ITER=12`를 유지하여 마지막 반복 판별 등에 영향을 주지 않는다. 물질14 final이나 복사13 관측 capture를 물질13 갱신의 입력으로 대신하지 않는다.

포착 창은 **물질13 후보 생성 시작부터 그 후보의 최종 commit과 기존 DET 비교 덤프 완료까지**다. 이 창 안의 Si 셸10 관련 n_e·CE 반복은 모두 기록한다. 마지막 호출만 남겨 앞선 감쇠·상대 인구 계보를 잃지 않는다.

중단 위치는 lumina_cmfgen.c 약 11856행의 기존 비교 덤프 호출 이후다. 목표 commit=13, 대응 덤프의 물질13/복사12, trace 종결 기록을 확인한 뒤 기존 정리 경로로 빠져나온다. 다음 복사 풀이·물질14 갱신·최종 스펙트럼 생성은 수행하지 않는다. 이 조기 종료는 trace 모드에서만 작동한다.

`TRACE_TARGET_COMPLETE`는 이 제한 실행의 완료 표지다. O의 12회 전체 정상 완료나 관측 스펙트럼 완료와 혼동하지 않는다. 중단 연결에 필요한 호출자 전달만 허용하며, 대상 상태를 계산하는 순서·조건은 바꾸지 않는다.

## §3. 상태 동등성·계측의 비개입

O의 바이너리 SHA-256은 `8ef03fb37b3e0de986f9c9ba15f0b7bbf8f98800d62ab18fb666ec518914f5d8`이다. T는 계측 때문에 재빌드하므로 바이너리 해시 일치를 요구하지 않는다. 봉인 파일 목록·해시, 계측 패치, 실제 컴파일 명령·버전, 새 바이너리 해시를 함께 기록한다.

O의 `input/run_build_identity.json`과 `impl/build_b2.sh`의 실제 빌드 처방을 승계한다. 봉인 소스·O·diag8 산출은 변경하지 않는다. 작업 사본은 일반 파일 복사로 만들고 하드링크를 사용하지 않는다.

O의 입력·원자 자료·고정 온도·씨앗·물리 exports를 유지한다. 허용 차이는 기존 경로 키의 T 이식과 새 trace 활성화·출력 경로뿐이다. 상속된 `LUMINA_*`는 제거하고 등록 exports를 읽는다. Ω·NRB·CPU 행렬 조립·n_e 폐합·SL·감쇠·충돌 전리 gate를 바꾸지 않는다.

계측은 기본 꺼짐이다. 켜져도 기존 계산 결과를 복사·기록할 뿐, 율·인구·행렬·분기 조건·합산 순서·난수 상태에 쓰지 않는다. 채널 분해는 실제 항이 생성되는 곳에서 별도 버퍼에 기록하며 생산 행렬을 분해 버퍼로 다시 조립하지 않는다.

다음 두 증거를 함께 사용한다.

1. §8의 음성 대조에서 동일 소형 입력의 trace 꺼짐/켜짐 사이 물리 배열·상태 전이를 바이트 단위로 대조한다. 기록 동작 외 변화는 실패다.
2. T의 기존 물질13 덤프를 O의 `physics_DET_iter0010` 자료와 대조한다. Si 셸10의 stage·NLTE 이온합, 원소 총량, T_e, n_e, 같은 셸의 저장 CE 상대 이온합과 binned J를 비교한다.

저장 상태의 수치 동등성은 각 필드에 대해 \(|a-b|\le10^{-8}\max(|a|,|b|)\)로 등록한다. 양쪽 0은 같음, 한쪽만 0은 불일치다. 저장 표기 정밀도가 더 거칠면 그 반올림 구간도 함께 제시하고 그 한계를 넘는 동등성을 주장하지 않는다.

이 검사는 **저장 가능한 상태의 동등성**이다. O에 없던 내부 소비 인구·행렬의 완전 일치를 소급 증명하지 않는다. 입력 계보·음성 대조·저장 상태 재현을 합쳐 O 동등 재취득의 근거로 사용한다.

불일치하면 최종 분기는 **판정 불가 — O 상태 재현 실패**다. T의 표는 실제 취득 상태의 조건부 결과로 남기되 O의 1.78 dex 귀속으로 승격하지 않는다. 허용오차 확대·목표 세대 교체·추가 GPU 실행은 하지 않는다.

## §4. 최소 소비 trace

레코드 키는 런, 출력 예정 물질 세대, 입력 복사 세대, 후보 식별자, n_e 반복, CE 반복, 호출 순번, Si 셸·이온쌍, 처리 단계다. 후보 폐기와 최종 채택을 구별한다. 동일 물질 태그만으로 내부 시점을 결합하지 않는다.

세대·호출 순번별 레코드를 새 파일에 기록하고 덮어쓰지 않는다. 일반 MATDUMP의 세대 없는 출력은 사용하지 않는다. 전용 파일시스템 기능이나 파일 교체의 특수 의미에 의존하지 않는다.

| 포착 위치 | 필요한 내용 |
|---|---|
| stage 사다리 소비 | T_e·n_e·J 출처, 준위 가중 광전리 항, NT·RR·DR·충돌율, 보정 전 비, dJ/dN 입력·보정 후 비·stage 밀도 |
| Si pair 조립 | 실제 FL 전이별 채널·율·적용 인구, SL 매핑·within-SL fraction, 보존행 교체 전 생성자·우변 |
| solve 경계 | 보존행·변수변환 정보, 실제 solve 입력·정규화, raw 출력·반환 상태·선택된 알고리듬 |
| SL→FL·후처리 | raw SL, 분배 직후 FL, 정규화 전후, CE 감쇠 전후 인구 |
| CE·n_e 소비 | 실제 accessor 반환값·권위·상대 원소/이온, n_e trial·반환값·채택 상태와 반복 종료 사유 |
| 최종 commit | 채택 후보와 연결된 Si FL·stage 인구, 관련 상대 인구, 물질13 commit 및 덤프 대응 |

CE 상대는 실제 Si 연산자가 읽은 원소·이온에 한정한다. Ca II라는 예상만으로 다른 실제 소비 상대를 버리지 않으며, 상대 원소 전체 행렬은 취득하지 않는다.

BB-NRB는 Si pair가 읽는 관련 선의 ID, continuum/local-emission, τ·β·Jbar 및 provenance를 포착한다. 전 선 census와 표본 선만의 대체를 모두 피한다. 실제 비활성 항은 gate와 0을 기록하고 결손과 구분한다.

lumina_plasma.c 약 20970–21090행에는 `population_generator_stationary_gth`와 일반 선형 풀이의 분기가 있다. **실제로 사용한 경로를 기록한다.** GTH이면 생성자와 정규화가 실제 solve 대상이며, 사용하지 않은 보존행 교체 행렬만으로 solve 잔차를 대표하지 않는다.

FL 연산자는 실제 생산 항의 기록으로 구성한다. 실제 소비에 없던 전이나 물리율을 추가하지 않는다. SL 수준에서만 정의되어 FL 대응을 식별할 수 없는 항은 그 사실을 남기고 해당 FL 귀속을 미정의로 둔다.

## §5. 네 잔차·채널 순수지·비 차이의 연결

실제 solve 잔차는 입력 계와 raw 해의 성분별 후방오차다.

\[
\eta_i=\frac{|(Ax-b)_i|}{\sum_j|A_{ij}x_j|+|b_i|}.
\]

GTH는 실제 생성자 잔차와 정규화 잔차를 분리한다. 보존행·물리행처럼 단위가 다른 행을 단일 절대 노름으로 합치지 않는다. 분모와 분자가 모두 0이면 0, 분모만 0이면 미정의다.

물리율 잔차는 **보존행 교체 전** 각 FL·SL 행의 부호 있는 유입−유출이다. 채널 \(c\)의 Si II→III 순수지는 \(F_c=I_c-R_c\), 단위는 \(\mathrm{cm^{-3}\,s^{-1}}\)로 통일한다. \(I_c,R_c,F_c\)를 함께 남겨 큰 상쇄를 숨기지 않는다.

정규화 물리 잔차는 해당 행의 총 유입+유출로 나눈다. Si 전리 순수지는 모든 채널의 전리+재결합 총량으로 나눈다. 영 분모를 작은 수로 치환하지 않는다. 원량도 표에 병기한다.

| 잔차 종류 | 평가 대상·해석 |
|---|---|
| 실제 solve | 실제 입력과 raw 해. 작은 값은 그 계의 풀이 정확도만 지지한다. |
| 물리율·채널 | 원래 물리행과 동일 소비 인구. 보존행으로 제거된 행도 포함한다. |
| SL 대 FL | 같은 FL 인구의 개별 잔차와 SL별 합을 나란히 둔다. 상쇄되는 FL 불균형을 구분한다. |
| 감쇠·후처리·commit | 단계별 인구와 그 단계에 연결된 소비율. 고정율 대입과 갱신율 평가는 별도 표기한다. |

기계적 수치 분해의 기준은 정규화 잔차 \(10^{-8}\), III/II 변화 0.01 dex다. 이는 계산상 분해능이며 물리적 허용오차·수렴 기준이 아니다. 저장 정밀도·산술 오차가 기준을 가로지르면 미분리로 둔다.

잔차 크기만으로 1.78 dex의 원인을 확정하지 않는다. 같은 trace의 마지막 채택 소비 상태에서 필요할 때만 CPU로 **고정 SL 계 1개와 고정 FL 계 1개**의 정상해를 계산한다. 총량·외부장·n_e·CE 상대를 고정하며 반복 갱신·채널 삭제 실험·새 물리 모델은 수행하지 않는다.

고정해는 유일성·정규화·유한성·비음성 및 잔차를 만족해야 한다. 조건이 충족되지 않거나 CPU 상한 안에 계산할 수 없으면 해당 비 차이는 미정의다. 작은 후방오차를 해 비의 정확성 보증으로 대신하지 않는다.

stage 보정 전 비 \(R_0\), 보정 후 \(R_s\), 고정 FL·SL 해 \(R_F,R_L\), raw 비 \(R_{\rm raw}\), committed 비 \(R_c\)가 같은 소비 조건으로 대응할 때 다음 연결을 사용한다.

\[
\log_{10}\frac{R_s}{R_c}
=\log_{10}\frac{R_s}{R_0}
+\log_{10}\frac{R_0}{R_F}
+\log_{10}\frac{R_F}{R_L}
+\log_{10}\frac{R_L}{R_{\rm raw}}
+\log_{10}\frac{R_{\rm raw}}{R_c}.
\]

각 항은 차례로 stage 보정, 채널·인구 정식화, SL 근사, solve, 감쇠·후처리의 **고정 소비 조건부 연결**이다. 마지막 항은 저장된 처리 단계별 비 변화로 다시 대응한다.

stage와 pair의 n_e·J·CE 소비 조건이 다르면 이 항등식의 정식화 항으로 시간차를 흡수하지 않는다. 반복 trace로 조건 차이와 비 이동을 명시하고, 분리할 수 없는 부분은 미귀속으로 남긴다. 다른 시점율을 commit 인구에 대입한 값은 고정 연산자 진단이지 commit 시점의 새로운 자기일관 물리 잔차가 아니다.

## §6. 주 산출 — 표 하나·기대치

기계 원본은 `si10_trace_table.csv`, 표시본은 같은 행의 `si10_trace_table.md`다. 각 수치 행에는 소비 키·단위·정의 여부·원량·정규화량·증거 레코드를 둔다.

| 행 묶음 | SE·NLTE 칸 | stage 칸 | 공개·대응·판정 칸 |
|---|---|---|---|
| 물질13 III/II | T와 O의 committed 비 | T와 O의 stage 비 | 공개 105.6974≈105.7, 양쪽 공개 대비 dex |
| 실제 solve | 선택 경로·최대 잔차·정규화 잔차 | 해당 없음 | raw 비·정의 여부 |
| 물리율 잔차 | SL·FL 최대 및 Si 순수지 | 사다리 순수지 | 동일 소비 조건 여부 |
| 광전리 | 하위 ground와 각 비자명 SL 묶음의 \(I,R,F\) | 같은 준위 묶음의 LTE 가중 기여 | 묶음 ID·적용 인구 |
| NT·RR·유도 역과정·DR | 채널별 \(I,R,F\) | 채널별 \(I,R,F\) | 식·적용 인구 차이 |
| 충돌 전리·3체·CE | gate와 \(I,R,F\) | gate와 \(I,R,F\) | 실제 부재·0·결손 구별 |
| BB·충돌 여기 | 준위 잔차 기여·이온 내 상쇄 | 직접 전리項 해당 없음 | 전리 순수지와 준위 재분배 구분 |
| SL→FL·감쇠·후처리·commit | 단계별 잔차와 III/II·이동 dex | dJ/dN 전후 비·이동 dex | 알고리듬 연결 |
| 조건부 연결·최종 귀속 | §5의 비 차이 항 | 총 격차 | 정식화/알고리듬/둘 다/판정 불가 |

공개 대응은 19.480 d, Lumina 셸10 vel_mid=11544 km/s, 공개 vel_mid=11534.59 km/s다. 공개 `si1`,`si2`를 II·III로 읽는 기존 가정을 유지하고 diag8의 공개 판독·대응 함수를 재사용한다.

공개 거리는 \(\log_{10}(R/R_{\rm public})\)다. O의 참고값은 stage +0.3057 dex, committed NLTE −1.4722 dex다. 실제 표는 원자료로 산출하며 이 반올림값을 계산 입력으로 쓰지 않는다. 공개 CMFGEN 채널율·SE 잔차는 확보하지 않았으므로 해당 칸은 미확보다.

기대치는 다음과 같다. 입력·계측 비개입이 성립하면 저장 물질13과 두 비는 O를 재현해야 한다. 갈림이 없어지면 물리 개선으로 읽지 않고 재현 문제를 먼저 판정한다. **지배 채널, 네 잔차의 크기, 조건부 비 이동의 부호·크기는 예측하지 못한다.** 충돌 전리 gate 차이·DR·NT·SL·CE 어느 하나도 사전에 주원인으로 지정하지 않는다.

## §7. 최소 변경집합·보존

### 변경집합 — DET-SI-CONSUMPTION-TRACE (표)

| 파일 | 허용 변경 |
|---|---|
| `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/source/src/lumina_plasma.c` | 실제 소비 항·solve·분배·CE·n_e·commit trace만 추가 |
| `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/source/src/lumina_cmfgen.c` | 포착 창·기존 덤프 뒤 완료 및 정리 연결 |
| `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/source/src/lumina_cuda.cu` | 제한 실행 완료를 호출자에 전달하여 후속 계산 중단 |
| `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/source/src/det_si_trace.h` | 기본 꺼짐 trace 공통 선언·기록 보조 구현 |
| `scripts/det_si_trace_readout.py` | trace 판독·잔차·제한 고정해·주 표·기존 공개 함수 연결 |
| `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/tools/det_si_trace_negative.c` | §8의 소형 음성 대조 |
| `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/tools/run_det_si_trace_gpu.slurm` | O 러너 승계·계측 완료와 비용 기록 |
| `/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/tools/run_det_si_trace_offline.slurm` | CPU 빌드·준비·판독 연결 |

### 변경집합 끝

새 식별자는 `det_si_trace_` 접두를 사용한다. 새 환경 키는 `LUMINA_SI_TRACE`와 `LUMINA_SI_TRACE_DIR` 두 개이며, 대상은 물질13·Si·셸10으로 고정한다. 기존 엄격 환경 검사와 연결되는 등록은 같은 변경 파일 안에서 처리하고 검사를 끄지 않는다.

추가 생산 파일 변경이 필요하면 조용히 범위를 넓히지 않고 계약을 개정한다. clamp/floor·물리율·solver·감쇠·종료 허용오차·원자 자료 수정은 금지한다. 기존 문제를 발견해도 이번 패치로 수리하지 않는다.

보조 산출은 trace 원자료, `STAGE_IDENTITY.txt`, `RUN_FOOTER.txt`, `OWNER.txt`, 입력·빌드 신원, `evidence.json`, 제출·비용 기록이다. 실패 자료를 보존하고 기존 출력 위치를 재사용하지 않는다. 별도 결정 재계산 파일은 만들지 않는다.

## §8. 게이트·음성 대조·귀속 분기

| 게이트 | 막는 실제 실패 |
|---|---|
| G-SCOPE | 기존 문서·발주 검사와 변경·입력 신원 대조로 계측 외 물리 변경이나 잘못된 기준 런 사용을 막는다. |
| G-TRACE | 소비 키·채택 계보·필수 항·실제 solve 경로를 확인하여 서로 다른 내부 시점이나 누락 연산자를 같은 잔차로 결합하는 실패를 막는다. |
| G-EQUIV | §3의 비개입·동등성으로 계측 또는 재빌드가 바꾼 상태를 O의 갈림으로 귀속하는 실패를 막는다. |
| G-TABLE | 행·단위·부호·공개 대응·채널 합과 실제 물리행의 대응으로 다른 양의 비교와 누락 채널 귀속을 막는다. |
| G-RUN | 목표 완료·정상 정리·할당·비용으로 미완료 trace나 상한 초과 실행을 완료로 보고하는 실패를 막는다. |

새 코드 직접시험은 음성 대조로 한정한다. 소형 입력에서 trace 활성화·비대상 세대·비대상 셸이 물리 배열과 제어 흐름을 바꾸지 않는지 검사하고, 대상 완료 전에는 중단하지 않아야 한다. 시험용 물리 배열 변경과 잘못된 소비 키를 각각 주입하여 해당 검사가 거부함을 시연한다. 파일 쓰기 실패 모사·저장 표 재독해·결정 재계산 배터리는 추가하지 않는다.

최종 과학 분기는 다음 네 개뿐이다.

| 코드 | 분기 | 필요한 증거 |
|---:|---|---|
| 0 | 정식화 | 같은 소비 조건의 채널·적용 인구 차이가 비 격차에 연결되고, 알고리듬 기여는 등록 분해능 안이다. |
| 1 | 알고리듬 | solve·SL·반복·감쇠·후처리의 비 이동이 격차에 연결되고, 정식화 기여는 등록 분해능 안이다. |
| 2 | 둘 다 | 정식화와 알고리듬 각각에 분해능을 넘는 비 연결이 있고 미귀속 부분이 결론을 바꾸지 않는다. |
| 3 | 판정 불가 | 상태·소비 대응·연산자·유효 고정해가 부족하거나 잔차에서 1.78 dex로의 연결이 닫히지 않는다. |

단독 귀속은 반대 축의 미측정을 0으로 취급하지 않는다. §5의 연결 잔여가 0.01 dex를 넘거나 오차가 분기를 바꿀 수 있으면 판정 불가다. 반대 방향 기여·상쇄도 모두 보존한다. 연결 항을 독립적인 비선형 인과 기여율로 표시하지 않는다.

## §9. 비용·실행 규약

| 항목 | 등록 상한 |
|---|---|
| GPU 실행 | 모델 진입 1회, h200 ×2 우선·할당 전 h100 ×2 대체 가능 |
| GPU 작업 | 16 CPU·64 GiB, 작업당 `--mem=64G` |
| GPU 할당 벽시계 | 75,600 s=21시간, `--time=1260` |
| 누적 GPU 비용 | 42 GPU·h |
| 할당 후 재제출·재시작·requeue | 0회 |
| 오프라인 | a10 CPU Slurm, 2 CPU·8 GiB, 동시 1작업 |
| 오프라인 누적 | 1,800 s·1 core·h, 빌드·시험·준비·판독·감사 재현 포함 |

O 실측 67,575 s×2=37.5417 GPU·h를 근거로 약 12% 여유를 둔다. 물질13 덤프 뒤 종료로 후속 작업은 절약하지만 절약량을 사전 실측한 것은 아니다. 42 GPU·h는 **이번 기안의 상한 제안**이며 사용자 원문의 정확한 수치 승인으로 표기하지 않는다.

GPU 요청은 `--gres=gpu:h200:2` 또는 `--gres=gpu:h100:2`다. 모든 작업은 `LUMINA_` 접두, `--exclude=syn102,syn03`, 명시적 메모리·시간을 사용한다. a10 CPU 작업은 GPU 0을 명시하고 GPU를 할당받지 않는다. 로그인 노드에서는 계산하지 않는다.

모듈은 작업 안에서 `module purge` 후 O의 gnu13/13.2.0·cuda/13.0.2를 적재한다. Slurm에 저장된 정수 분 TimeLimit과 실제 자원 요청을 증거에 남긴다. 대기시간은 별도 보고하며 할당 비용과 합치지 않는다.

CPU·GPU 상한에 닿으면 중단하고 미완료 항목을 보고한다. 할당 전 대기 작업의 취소·대체 제출만 허용한다. 할당 후 실패는 남은 예산이 있어도 추가 제출하지 않는다. h100에서도 상한을 늘리지 않는다.

## §10. 집행·재현 인터페이스

아래는 구현할 인터페이스이며 기안자가 실행한 명령이 아니다. 각 작업 종료와 필요한 게이트 확인 뒤 순서대로 실행한다.

```bash
TRACE_BASE=/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010
sbatch --exclude=syn102,syn03 --time=20 "$TRACE_BASE/tools/run_det_si_trace_offline.slurm" prepare
sbatch --partition=h200 --gres=gpu:h200:2 --cpus-per-task=16 --mem=64G --time=1260 --exclude=syn102,syn03 "$TRACE_BASE/tools/run_det_si_trace_gpu.slurm"
sbatch --exclude=syn102,syn03 --time=10 "$TRACE_BASE/tools/run_det_si_trace_offline.slurm" analyze
```

prepare는 빌드·음성 대조·입력 신원·빈 런 준비를 수행한다. analyze는 실제 trace에서 표를 산출하며 저장 표의 판정을 다시 계산하는 검증기를 호출하지 않는다. 고정해 계산 여부·사용 시간·미정의 사유도 같은 증거에 기록한다.

## §11. 종료감사·주장 상한

종료감사의 **1차 대상은 과학 표의 수치·소비 시점 대응·공개 대응·귀속 주장 한정**이다. 판독기 내부 구조는 과학 수치나 행 귀속·판정을 바꿀 수 있는 구체적 결함만 «폐합 차단»이다. 추상적인 방어 강화·내부 구조 선호는 새 폐합 조건으로 만들지 않는다.

운전자는 네 분기 중 하나와 부족한 증거를 기록한다. astra는 계약 저자라는 이해관계를 명시하고 그 판정문을 감사한다. 판정 불가는 유효한 보고 결과지만 목표 귀속 달성으로 표시하지 않는다.

결론은 **Si 셸10·등록 물질13·등록 소비 조건**에 한정한다. 철족 전체·청색 절대 플럭스·전체 수렴·물리적 정답·채택할 권위·생산 승격을 선취하지 않는다. 공개에 가까운 비라는 이유만으로 해당 권위를 정답으로 선택하지 않는다.

## §12. 기계 선언

직전 계약의 changeset·branches·references 형식을 따른다. 분기 fixture는 기존 사전등록 검사 입력이며 새 결정 재계산 장치가 아니다. 참조는 기존 실재 파일만 지정한다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-SI-CONSUMPTION-TRACE (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_si_trace_readout\\.py|/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/(?:source/src/(?:lumina_plasma\\.c|lumina_cmfgen\\.c|lumina_cuda\\.cu|det_si_trace\\.h)|tools/(?:det_si_trace_negative\\.c|run_det_si_trace_(?:offline|gpu)\\.slurm)))",
    "symbol": "det_si_trace_",
    "roots": [
      "/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/source/src/lumina_plasma.c",
      "/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/source/src/lumina_cmfgen.c",
      "/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/source/src/lumina_cuda.cu",
      "/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/source/src/det_si_trace.h",
      "scripts/det_si_trace_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/tools/det_si_trace_negative.c",
      "/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/tools/run_det_si_trace_gpu.slurm",
      "/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/tools/run_det_si_trace_offline.slurm"
    ],
    "expected_extra": [
      "/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/source/src/lumina_plasma.c",
      "/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/source/src/lumina_cmfgen.c",
      "/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/source/src/lumina_cuda.cu",
      "/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/source/src/det_si_trace.h",
      "scripts/det_si_trace_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/tools/det_si_trace_negative.c",
      "/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/tools/run_det_si_trace_gpu.slurm",
      "/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/tools/run_det_si_trace_offline.slurm"
    ]
  },
  "branches": {
    "regimes": [[0,0],[1,1],[2,2],[3,3],[9,9]],
    "metrics": {
      "valid": "len(v) == 1 and all(x in (0,1,2,3) for x in v)",
      "c0": "sum(1 for x in v if x == 0)",
      "c1": "sum(1 for x in v if x == 1)",
      "c2": "sum(1 for x in v if x == 2)"
    },
    "rules": [
      {"name":"FORMULATION","predicate":"valid and c0 == 1"},
      {"name":"ALGORITHM","predicate":"valid and c1 == 1"},
      {"name":"BOTH","predicate":"valid and c2 == 1"},
      {"name":"INDETERMINATE","predicate":"not valid or (c0 == 0 and c1 == 0 and c2 == 0)"}
    ],
    "adversarial_fixtures": [
      {"name":"formulation","mix":[[0,1]]},
      {"name":"algorithm","mix":[[1,1]]},
      {"name":"both","mix":[[2,1]]},
      {"name":"indeterminate","mix":[[3,1]]},
      {"name":"invalid","mix":[[9,1]]},
      {"name":"missing","mix":[]},
      {"name":"multiple","mix":[[0,1],[1,1]]}
    ]
  },
  "references": [
    {"path":"docs/PREREG_DET_COLL_PRESC_2026-10-09.md"},
    {"path":"docs/VERDICT_DET_COLL_PRESC_2026-10-09.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/evidence/USER_DECISION_2026-10-10.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag8_dualauth_20261010/LOOP_CLOSED.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag8_dualauth_20261010/DIAGNOSIS.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag8_dualauth_20261010/astra_audit_r1.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag8_dualauth_20261010/astra_audit_r2.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag8_dualauth_20261010/tools/r2_crosscheck.py","flags_existing":["public_ions"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/STAGE_IDENTITY.txt","flags_existing":["outer_iterations=12","run_binary_sha256"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/RUN_FOOTER.txt","flags_existing":["elapsed_seconds=67575","allocated_gpus=2"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/job.slurm"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/SUBMIT_COMMAND.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/resolved_lumina.exports"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/input/run_build_identity.json"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/work/physics_comparison/physics_DET_iter0010.manifest.json"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/work/physics_comparison/physics_DET_iter0010.ion.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/work/det_obs_capture/meta.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/tools/run_det_coll_presc_gpu.slurm"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/impl/build_b2.sh"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/Makefile"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_plasma.c","flags_existing":["population_generator_stationary_gth","population_dense_solve_equilibrated"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cmfgen.c","flags_existing":["physics_comparison_dump_if_requested"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/obs_capture_20261006/source/src/lumina_cuda.cu"},
    {"path":"data/standart_data1/toy06/ionfrac_si_toy06_cmfgen.txt","flags_existing":["si1","si2","19.480"]}
  ]
}
```
