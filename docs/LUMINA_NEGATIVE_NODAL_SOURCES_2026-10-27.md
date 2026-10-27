# 새 물질의 음수 총소광 원인 추적

## 실제 결과

완료된 전주파수 계수1224468을 읽기 전용으로 전90점에서 조사했다.
수송 입력이 사용하는 합산 순서로 총소광이 음수인 표본은 **8275개**이며
계수 생산 보고서의 개수와 같다. 49개 반경점, 원0기준 노드0–58에서
나타났고 파장 범위는 약 **0.522–70.88㎛**다. 이 범위는 연속된 파장
구간이라는 뜻이 아니다.

가장 음수인 표본은 노드54, 파장 **17.258867㎛**에서
총소광 **−4.20604e-14 cm⁻¹**다. BF는0, FF는9.79161e-17,
전자산란은5.55173e-16, 선 소광 합은 −4.27135e-14 cm⁻¹이다.
전 선의 기여를 독립적으로 합한 값은 저장 선 소광과 해당 표본에서
정확히 일치했다. 가장 큰 음수 선 기여는 **Si II FL124→125**,
중심파장17.258943㎛, **−4.06961e-14 cm⁻¹**이다. 인접한 Ca II
FL58→59도 −2.01741e-15 cm⁻¹을 기여한다. 가장 음수인 대표12표본에서
Si II 선이 가장 큰 음수 기여자였으나, 이것을 전체8275표본의 주원인으로
확대하지 않는다.

Si II 두 준위는 각각 원0기준 SL67과68에 속한다. 해당 노드에서 새 물질의
`n_lower/g_lower = 2.04604e-5`, `n_upper/g_upper = 2.54738e-5`이고,
원70에서 같은 방식으로 복원한 C full 인구는 각각2.04076e-5와2.53968e-5다.
두 상태 모두 통계중량당 상위 인구가 약24% 높다. 따라서 새 물질 변환이
처음 만든 역전이라고 볼 근거는 없다. 원70 비교값은 **저장된 native FL
원본이 아니라 native SL에서 C로 복원한 인구**다. 원70 자체도 미수렴
참조이므로 그 역전이 물리적으로 확정됐다는 뜻은 아니다.

음수 최댓값에 인접점 간 거리9.31561e13cm를 그대로 곱하면3.92이지만,
이는 해당 소광이 그 거리에서 일정하다는 가정의 크기일 뿐 실제 광선
광학깊이가 아니다. 팽창 시간1.683072e6초와 고정 Doppler 폭13.5115km/s의
곱인 길이는2.27409e12cm이고, 여기에 같은 국소 소광을 곱하면0.0956이다.
이 또한 **차원 추정**이며 실제 주파수 이동·기하·소광 변화가 포함된 수송
증폭률이나 엄밀한 상한이 아니다. 음수값은 임의로0으로 바꾸지 않았다.

이 결과만으로 Si/S 이온화 불일치의 원인, 실제 방출 스펙트럼, 광선 증폭,
수송 수렴이나 CMFGEN 동등성을 판정할 수 없다. 기존 원70/J71 국소 대조는
다른 파장3.4953㎛에서 raw 총소광과 native 처리값의 부호 차이 및 native
`SRCE_CHK` 처방을 이미 기록했다. 그 처방이 **이번17.26㎛ 표본에서도**
작동했는지는 여기서 비교하지 않았다.

## 기존 원70 장에서 직접 광이온화에 기여하는 정도

문제의 두 준위가 이온화에 직접 얼마나 관여하는지, 저장된 **원70 물질·원70
복사장**의 전체 BF 결과에서 `n_lower × 경로확률 × 총 BF 광이온화율`을
합산했다. 원70은 고정 온도의 미수렴 상태이며 새 자체장 결과가 아니다.
총 BF율에는 bound와 dissolution 항이 포함된다. Si II 하위준위인 것을
실제 원자주소와 대조한250개 경로가 분모다.

가장 음수인 노드54에서 Si II 전체 직접 광이온화는4.43602e5 cm⁻³ s⁻¹이고,
FL124의 두 경로는54.0528 cm⁻³ s⁻¹(**0.0122%**), FL125의 두 경로는
602.941 cm⁻³ s⁻¹(**0.1359%**)다. 둘의 합은 **0.1481%**다. 전90점에서
두 준위의 합은 중앙값0.0349%, 최댓값0.6545%(노드89)다. 노드54의
최대 기여는 FL8(16.87%), FL11(16.14%) 등의 낮은 준위에서 왔다.

이 수치는 **기존 장에서의 직접 BF 기여만** 뜻한다. 문제 선이 다른 준위에
미치는 BB 연쇄, 새 자체장으로 인한 변화, 열평형과 최종 이온화 분포에 대한
민감도를 배제하거나 정량화한 것은 아니다. 따라서 직접 BF가 작다는 이유로
역전이 무해하다고 단정하지 않는다.

최초 탐색에서 내부 stage3을 Si II로 잘못 선택해 Si III를 집계했으나,
원자주소 대조에서 발견해 해당 수치는 폐기했다. 최종 계산은 lower의
`ion_number=1`과 roster `stage=2`를 모두 검증한다. 명령 인수 이름 오류도
수정한 뒤 실물 자료를 정상 처리했다. 새 적분이나 수송은 수행하지 않았다.

재현은 grammar-debug CPU1에서 다음 명령으로 한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=scripts \
python3 scripts/analyze_lumina_inverted_line_photoionization.py \
  --root "$RUNS/lumina_material_bf90_20261027-B64JUq/run" \
  --levels "$RUNS/lumina_accepted_input_20261027-79u5Sp/material/atomic/levels.csv"
```

BF 보고서 SHA256은 `482b18c4d884984787c108575e096cd7ac18665aa5e85fda63a66f83f8d286c6`,
물질 snapshot은 `35b41808ee1e0527216db6d880987da313185a6ed58dbf3fe66ff5625480cc9f`,
경로 목록은 `967e09c692143e0dafb9ee0f19a5ab71f09278b3fafaeba861d1e2188a9eb886`,
총 BF율은 `921409cfea2d7dee95921ddaf9ac850a23008960e0f2ea1b6d83680c67b59484`다.

## 증거와 재현

- 계수 생산 보고서 SHA256: `73e434837f41b0044cd100f6b1f904949f3366adf6b7f85a48eba5e909fea655`.
- 계수 본문 SHA256: `652636d01e6965b69ea39239a9beb6aa70ec5f0a1e7ffb36a0c1bfd59974b3f1`.
  생산 기록에서 계승했으며 이번 순회에서 본문 전체 해시를 다시 계산하지 않았다.
- 선 목록 SHA256: `b005ab99613579793bd72266e53bc604dc8b00aff3684775291795c4beac4a9e`.
- 새 C full 인구 SHA256: `2ce239bd5d39e994e2cca0a46b283d8ccb68fc17a60bbaabcc58c9291de22ea6`.
- 원70 C full 복원 인구 SHA256: `27df293f279d487396f9c33509424861ef30c15a897925bcd6066d1e2f72b358`.
- 두 독립 판독 결과 JSON의 SHA256은 모두
  `3264b81060e833de97a77bdd7a1bae4f20fb14fe216de872a77e29e8a1e17820`이다.
  결과는 grammar-debug의 임시 디렉터리에 있으며 영구 보존을 보장하지 않는다.
- 합성 자료 직접시험은 `tests/test_lumina_negative_nodal_sources.py` 1개 통과.
  처음 판독 시 파일 전체의 가상주소 매핑이4GiB 한도에서 실패했고, 실제
  본문을 읽기 전에 중단됐다. 이후 반경점별 읽기로 수정해 두 번 재현했다.

grammar-debug에서 CPU1, 스레드1, 가상주소4GiB, 최대540초로 다음을 실행한다.
기존 출력은 덮어쓰지 않는다. `RUNS`는 작업 자료의 상위 경로를 가리킨다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts \
python3 scripts/analyze_lumina_negative_nodal_sources.py \
  --sources "$RUNS/lumina_accepted_sources_20261027-b5eCNv/run/sources" \
  --material "$RUNS/lumina_accepted_input_20261027-79u5Sp/material" \
  --c-full "$RUNS/lumina_accepted_input_20261027-79u5Sp/material/C-imported-full.bin" \
  --reference-full "$RUNS/lumina_signed_source_trace_20261020-YbcirG/full_population90.bin" \
  --output NEW_OUTPUT.json
```

새 RT·SE·BF 적분이나 계수 수정은0이다. GPU 수송437451은 제출 후
사용자 GPU 자원 한도로 대기 중이었으며 실제 시작과 결과는 별도로 확인한다.
