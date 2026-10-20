# Si/S 광이온화 채널과 CMFGEN 분포 일치 묶음

최종 목표는 **스펙트럼과 물리량의 CMFGEN 일치**다(사용자 명확화,2026-10-20). 이 묶음의 조건부 이온분포 비교·결합 안정화는 중간 증거이며, 원래 자기일관 상태와 최종 스펙트럼 수용 기준을 대체하지 않는다.

## 다음 목적 묶음 — 실제 Lumina 물질·장·스펙트럼 연결

완료 감쇠분기 비채택 뒤 Astra의 다음 계획과 입력phase정정을 수신했다. **원래물질70/J71의동일물질대조 → Lumina자체장/전원소물질·ne 두번갱신 → 같은최종물질의관측자계절대스펙트럼**이다. 원래J71은물질70의장이지물질71의장이아니므로초기물질을70으로맞췄다. 최대자체장3/물질갱신2/추가CMFGEN0이며고정T개발출력을최종물리합격으로표시하지않는다. 새생산코드구현·물리실행은아직없다. 실제분장·소스확인·다음구현과보존규칙은[오늘인수인계서](HANDOVER_2026-10-20.md), 상세계획은[생산브리지계획](LUMINA_STATE_SPECTRUM_BRIDGE_2026-10-20.json)에있다.

## 최신 실행 인수 — 감쇠/복원 분기 종료·채택조건 미달·재현 완료

**429145도11:44:00 KST, COMPLETED/0·31분43초로 끝났다.** 전체native2/RT1 계약을 지켰고 자동후속은 없다. 원래step을복원한 실제반응에서 Si V깊이10의국소이동은1.231597→0.046903%p로줄었지만, 양원소P 질량p95/최대와ne 총5성분은모두원래71보다증가했다. ne는0.411490→0.536719%, Si전체P최대는2.288973→2.872792%p, S는2.066647→2.565017%p다. 따라서 이번공통반-step/복원방법은채택하지않는다.

완료71B의실제ne/J71B C1도전90깊이에서인증·채택됐지만조건부비교일뿐이다. frozen최종판독은 `CRITERIA_NOT_MET`이며별도출력에서전체JSON·수치·결정을재현했다. 최종스펙트럼과물리량의일치는미완료다. Astra에완료묶음평가와실제Lumina자기일관출력으로연결되는다음목적계획을한번요청했다. 상세증거·해시·현재의뢰는[오늘인수인계서](HANDOVER_2026-10-20.md)의최신절에있다. 아래실행중기록은과거상태다.

## 이전 실행 인수 — 반-step70B/C1 완료·원래 step 복원71B 실행

Astra의 완료70/71 평가를 받았으며 원래 관찰 목표 미달은 유지했다. 다음 묶음은 같은69/J70에서 첫 공통 step만1/2로 줄인70B를 만들고, 새 J71B 뒤 원래 step을 복원한71B 응답을 기존 경로와 대조하는 것이다. 최대native2/RT1, 첫 부모20분·다음 부모45분/CPU16/OMP16/32GiB/GPU0·노드미고정/retry0이다. 첫 적용량 감소 자체를 안정화나 최종 성공으로 보지 않는다.

private 감쇠 실행파일·실제 입력 검증과117개 회귀시험을 마친 뒤 **429131을10:55:09 KST 한 번 제출했고10:55:10 시작**했다.10:55:19 확인은RUNNING9초이며 기존 J70을 읽기 전용 참조하므로 새 RT0이다. 첫 분기의 전 상태 중간값·raw correction·갱신 전 witness 일치가 확인되기 전 두 번째 실행은 하지 않는다. 원본·기존69/70/71·기본 full-step 적격성은 보존한다. 현재 계획·해시·재현 명령과 인수는 [오늘 인수인계서](HANDOVER_2026-10-20.md)와 [유한 분기 계획](CMFGEN_DAMPING_BRANCH_2026-10-20.json)에 있다.

**429131은11:05:21 KST, COMPLETED/0·10분11초로 끝났다.** 전 상태의 사전 중간값, 원래 raw correction, 전90깊이 배율과 갱신 전 witness byte 동일성이 모두 확인됐다. 실제ne변화0.351987824%/charge3.4644e-15이며 반-step의 기본 full-step 적격성을 참으로 바꾸지 않았다. 실제70B/ne70B/J70의 전90깊이 C1은 새root에서 진행 중이고 추가RT/native0이다. 후속 복원 runner와143개 회귀시험은 준비됐으나 실제70B/C1 통합 확인 전에는 새 J71B를 제출하지 않는다. 작은 반-step 자체가 안정화·물리 동질성 성공이라는 뜻은 아니다.

그 뒤 실제70B C1도 Si/S각90/90 인증·채택/900행/support불일치0으로 끝났다. 반-step 조건부III 최대차이는Si1.949927%p/S1.641086%p로 원래70보다 크며, 작은 ne/P 자체를 수렴으로 읽지 않는다. 전체 중간 상태·raw correction/witness를 실제 입력에서 다시 재현하고 C1/native70B 상태 동일성을 확인한 뒤 **429145를11:12:16 KST 한 번 제출·11:12:17 시작**했다.11:12:33은RUNNING16초·syn06/a40·CPU16/OMP16/32GiB/GPU0·노드미고정·45분/공유40분/retry0이다. 새J71B와 원래 witness/omega1의71B 응답으로만 효과를 판별하며 자동 후속은 없다. 상세 신원·입력 통합 재현·다음 C1 명령은 오늘 인수인계서에 있다.

## 이전 실행 인수 — 실제71 완료·CMFGEN 조건부 분포 비교 재현

**429062는10:34:36 KST에 COMPLETED/0·33분26초로 끝났다.** 실제70→새J71→물질71이며 추가 RT/native는 없다. 실제71의 T/ne/J71을 공통으로 사용한 Lumina 이온화 풀이(C1)는 Si/S 전90깊이 각각 인증·실제 채택을 완료했다. II–VI900행의 support 불일치0이다. 최대 절대 분율차는 Si II0.0000208114%p/III0.0918574793%p, S II0.0103110860%p/III0.2670580968%p이며 전단계 최대는Si0.1451314764%p/S0.2670602578%p다. 이는 독립적인 복사장·온도·전자밀도 계산이나 스펙트럼 일치의 증거가 아니다.

전자밀도 최대 이동은69/70/71에서1.074104%/0.703976%/0.411490%로 줄었다. 실제 물질 최대 이동도71에서Si2.288973%p/S2.066647%p로 줄었지만 조건부 차이보다 크다.70의0.5% 미달과 S 장응답 질량p95의 기준69 대비 미감소 때문에 사전 관찰 목표는 **미달**이다. frozen 최종 판독은 `LIMIT_REACHED_EVALUATE`, `objective_observed=false`, `automatic_extension=false`이고 별도 출력에서 전체 JSON을 재현했다. Si V 깊이10의 두-step 근접 후보 진폭1.231597%p도 남아 집계 최대값 감소만으로 수축성이나 공동수렴을 주장하지 않는다.

원래 released-T 앵커·전체SE/RE·에너지·자기일관 분포/스펙트럼·3-grid 목표는 미완료다. Astra에 완료 묶음 평가와 다음 목적 계획을 한 번 요청했으며 중복 호출하지 않는다. 자동72/T해제/물리율 조정이나 추가 물리 제출은 없다. 정확한 비교표·해시·재현 명령과 현재 인수 사항은 [오늘 인수인계서](HANDOVER_2026-10-20.md)의 최신 절에 있다. 다음 절의 실행 중 기록은 과거 상태로 보존한다.

## 이전 실행 인수 — 실제70 완료·마지막71 장 계산 진행

실제70의429038도COMPLETED/0으로끝났고 실제ne70/J70 C1 전90점비교를완료했다. 조건부III 최대차이는Si.089306%p/S.269578%p, ne최대이동은.703976%다.최대이동의감소와별개로S의질량p95는커진혼합추세이며70은사전.5%목표미달이다. 공동수렴·원래released-T앵커·전체SE/RE·에너지·자기일관분포/스펙트럼·3-grid목표는미완료다.

사전규칙의 `CONTINUE71_WITHIN_LIMIT`과실제70선택·build를확인한뒤 **Slurm429062를10:01:08 KST 한 번제출·10:01:10시작**했다.현재새J71을계산중이며71결과는아직없다.70의.5%미달을소급통과로바꾸지않고마지막71로감쇠/반전추세를확인한다.최대71뒤자동72/T해제/별도동일장native는없다.기존68/69·70계약과결과는보존한다.

현재 상태, 정확한 실행 경로와 해시, 재현 결과 및 다음 명령은 [오늘 인수인계서](HANDOVER_2026-10-20.md)의 최신 절을 정본으로 참조한다. 기존 계획과 과거 실패는 아래에 보존한다. 현행 역할은 운전자 구현·직접시험·실행·재현, Astra의 의미 있는 묶음 기획·계획·평가이며 Fable 의뢰는 취소 상태다.

## 목표와 판정

Si II/III 및 S II/III의 실제 CMFGEN 광이온화 채널을 보존해, 공통 원자자료·복사장·온도·전자밀도 조건에서 Lumina의 준위율과 최종 이온분포가 CMFGEN과 일치하는지 판정한다. 단순한 코드 실행 성공이나 고정장 진단만으로 묶음을 종료하지 않는다. 최종 이온분율 허용치는 활성 shell에서 절대차 `≤0.05`, 비율 차이 `≤0.10 dex`이며, [동등성 계획 v2](CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md)의 나머지 M/E/S-동등 기준도 적용한다.

## 범위·입력·제외

- 범위: `phot_data_A/B`별 하위 준위 단면적, 실제 상위 이온 도착 준위와 문턱, 같은 phot grid 안의 통계 가중 분할, 채널 합산 연속 불투명도, 채널별 광이온화·Milne 역재결합 및 pair 풀이 연결. 기존 element-wide 경로가 활성일 때도 동일한 채널 자료를 소비하도록 한다.
- 출발 입력: Si II A/B, Si III A, S II A, S III A는 실행별 봉인 `atomic_links.txt`와 `MODEL_SPEC`를 사용한다. 활성 준위 수는 링크된 oscillator 파일 전체 길이가 아니라 해당 CMFGEN 실행의 `MODEL_SPEC` NF로 정하고, 활성 superlevel 수는 같은 F_TO_S의 NF 접두부에서 확인한다. PHOT 경로 및 checksum도 그 실행의 링크 원본을 따른다. 입력 선택을 묵시적으로 바꾸지 않는다.
- B 헤더의 여기 에너지 표기는 상위 준위 에너지의 권위값으로 쓰지 않는다. 상위 이온이 존재하는 비교에서는 실제 `osc_data`의 준위 에너지로 문턱을 만든다. `g_target / Σg`는 같은 phot grid 내부 단면적 분할에만 쓰며 상위 이온 점유율로 대용하지 않는다.
- 제외: CMFGEN 소스 수정, MC 추정기의 권위화, 전 원소 solver 전환, 무관한 계측·게이트 추가, 기준 원자 vintage 변경, 명시 요청 없는 commit/push. 기존 Slurm 427702는 이 묶음과 분리해 계속 보존한다.

## 변경·실행 책임

운전자(Codex)가 계획 통합, 코드 구현, 직접시험, 실행 준비, 결과 재현과 종료판정을 맡는다. Astra는 이번 채널/풀이 설계의 단일 물리 자문을 완료했으며 추가 정례 호출은 하지 않는다. Fable·Sol·Opus 등 다른 모델에는 이 묶음의 일을 위임하지 않는다.

구현 우선순위는 채널별 원자자료를 복원한 뒤, 이미 상위 이온 준위를 포함하는 pair 행렬의 실제 도착 준위로 BF 전이 목적지를 연결하는 것이다. 그 경로가 공통 점유율과 보존을 만들지 못한다는 재현 증거가 있을 때만 Si 한정 element-wide 대안을 검토한다. 채널 합계를 재결합률에 그대로 재사용하는 축약은 허용하지 않는다.

## 수용 기준과 증거

1. 원자자료 회계: Si II A/B, Si III A, S II A, S III A의 각 하위-상위 채널, 목표 준위, `g` 분할, 문턱, 단면적 원본이 완전하게 식별되고 checksum이 기록된다. 미해결 도착 준위·중복·누락은 0건이어야 한다.
2. 전달 일관성: 각 채널의 흡수 단면적이 총 연속 불투명도에 한 번만 합산되고, 광이온화와 Milne 역재결합은 그 채널별 단면적·문턱·상위 준위 통계가중치·상위 준위 점유율을 사용한다. 정·역율은 공통 `σ dν/ν` 로그가중치를 써야 하며, 전체/문턱 부분 bin의 Planck 양성대조에서 LTE 상세평형 `≤10⁻⁸`, 원소/이온 보존 잔차 `≤10⁻¹⁰`을 만족해야 한다.
3. 직접 비교: 동일 bin/문턱/복사장에 대해 채널별 단면적과 양방향 율을 CMFGEN 경로와 대조한다. 단면적 비교는 수치 적분 오차 범위 내에서 일치하고, 율은 고정된 동결상태 rate 허용치(median `≤0.03 dex`, p95 `≤0.10 dex`, max `≤0.20 dex`) 안에 있어야 한다. 잘못된 단면적을 양방향에 함께 넣어 LTE 검사를 통과시키는 것은 실패다.
4. 통합 비교: pair 연결과 인접 pair의 공통 상위 준위 population이 일관되고, 수렴한 matched-physics CMFGEN self-run과 최종 이온분포가 위 허용치를 만족해야 한다. 고정온도 Stage A 산출물은 staging 전용이며 최종 판정 자료가 아니다.

예상 증거는 코드/바이너리 및 원자료 hash, 채널 manifest, 직접시험 로그, 고정장 율 비교, 수렴 self-run의 입력·출력 hash와 shell별 Si/S 이온분율 비교표다. 증거는 이 문서 또는 최종 완료 기록에 한 번에 묶어 기록한다.

## 자원·보존·재현

로그인 노드에서는 코드·소규모 단위/통합시험만 수행한다. 원자 덱 생성은 노드 미고정 Slurm `a10` job(CPU 10, RAM 100 GiB, GPU 0, 최대 1시간)에서만 수행하며, 현 427702를 취소·재시작하지 않는다. 변환기 출력은 `/gpfs/kjhan/cmfgen_runs/lumina_model_spec_<jobid>`의 새 격리 경로에 생성하고, `atomic_links.txt`, `MODEL_SPEC`, `LEVEL_SL_STEQ_LINKS`, S III·Co II 보정 PHOT 및 세 소스 파일의 제출 시 SHA256이 달라지면 실행을 거부한다. 기존 기준 입력·raw output을 덮어쓰거나 삭제하지 않는다. 실패 시 부분 출력과 정확한 입력·소스 hash를 보존하고, 원인 수정 전 결과를 acceptance에 사용하지 않는다.

덱 재생성 재현: `CMFGEN_EXACT_HYD=1 sbatch -p a10 --export=ALL,... scripts/submit_cmfgen_model_spec_deck.slurm`에 제출 시점 SHA256 환경변수(`EXPECT_EXPAND`, `EXPECT_PARSER`, `EXPECT_DRIVER`, `EXPECT_LINKS`, `EXPECT_MODEL_SPEC`, `EXPECT_LEVEL_MAP`, `EXPECT_SIII_PHOT`, `EXPECT_COII_PHOT`, `EXPECT_HYD_L_DATA`, `EXPECT_GBF_N_DATA`)를 전달한다. Slurm 산출물은 run별 `regen_runtime.txt`, `regen_inputs.sha256`, `regen_outputs.sha256`, HYD/GBF table provenance 및 활성 원자 manifest 검증을 포함한다. 이 단계는 입력 스테이징이며 물리 검증 PASS가 아니다.

재현 기본 명령(실제 S III·Co II 입력은 해당 sealed run의 보정 PHOT 경로를 지정):

```bash
CMFGEN_LINKS=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/atomic_links.txt \
CMFGEN_MODEL_SPEC=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/MODEL_SPEC \
CMFGEN_LINK_FTOS=1 CMFGEN_VINTAGE_MATCH=0 \
LUMINA_SIII_PHOT_PATH=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/atomic_local/SUL/III/19apr23/phot_data_A \
LUMINA_COII_PHOT_PATH=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/atomic_local/COB/II/19apr23/phot_data_A \
SI_ROUTE_TEST_TMP=/gpfs/kjhan/cmfgen_runs \
python3 -m pytest -q tests/test_cmfgen_bf_channels.py tests/test_cmfgen_zero_route.py
make -j4
```

실행에서는 `LUMINA_CMFGEN_SIGMA_BF=$REF/cmfgen_sigma_bf.bin`과
`LUMINA_CMFGEN_BF_CHANNELS=$REF/cmfgen_bf_channels.bin`을 같은 참조 덱에서
지정한다. 채널 경로를 명시하면 파일 누락·손상은 fail-closed다. 환경변수가
없을 때의 기본 탐색도 실행 인자 `ref_dir/cmfgen_bf_channels.bin`이므로,
다른 덱이나 저장소 전역 자산을 잘못 섞지 않는다.

최종 계산의 재현 명령, Slurm ID, source/binary/input hash는 수용 가능한 출력이 생긴 뒤 이 기록에 추가한다.

## 2026-10-19 운전자 구현·직접시험 기록

- 실제 Si/S 원자료 경로시험은 Si II A/B, Si III A, S II A, S III A의 route map을 만들고, 상위 이온 준위·`g` 분할·원시 PHOT checksum·가중 채널합 보존을 확인했다. 실행은 S III 보정 PHOT 파일을 `LUMINA_SIII_PHOT_PATH`로 지정하고 활성 S III 256준위 접두부를 사용했다.
- 실제 자료시험 중 `_route_sigma_model`의 type 2 호출 인자 수 오류가 발견되어 수정했다. 이후 전체 채널/ν⁻³ 꼬리 회귀시험은 `10 passed`.
- 전·역율이 서로 다른 bin 가중치를 쓸 수 있음을 찾아 pair/EW 채널 경로에 공용 로그가중치 함수를 연결했다. A2-05 직접시험에 전체·문턱 부분 bin과 Planck LTE 율/flux 양성대조를 추가했고 `A2_05_BF_RATE_SELFTEST PASS`.
- 명시적 PHOT 채널에서는 역재결합에만 ground-core `REC_SPINGATE`를 적용하면 정율 단면적과 불균형해져 LTE 상세평형을 깨므로 그 단방향 gate를 제거했다. 구형 ground-only 경로의 기존 gate는 유지한다.
- C 채널 로더 자가시험은 `SH_GRID_LOADER_SELFTEST PASS`(손상 확률 그룹 거부 포함). CPU 전체 바이너리 `make -j4 TARGET=/tmp/lumina-bfch.9fTHY3/lumina-cpu` 링크 성공. C 소스 문법검사 성공; 기존 OpenMP 미사용·들여쓰기·미사용 변수 경고는 남아 있다.
- CUDA 컴파일은 `grammar-debug`에서 `cuda/12.8.1` 모듈을 찾지 못해 수행하지 못했다. GPU 바이너리 검증으로 표시하지 않는다.
- 채널 역 Milne의 자극항은 정방향율이 쓰는 세대검증 `RadiationFieldView`가 아니라 legacy `nlte->J_nu` 배열을 읽고 있었다. 결정론적 CMFGEN 재생 field에서는 두 배열이 동일하다고 보장되지 않는다. `bf_rate_view_jnu`를 추가해 explicit channel pair/EW 경로가 공통 view의 bin 값만 소비하고, `UNSAMPLED`·`STALE`·`OUT_OF_GRID` bin은 자극항에서 제외하며 `EXACT_ZERO`는 유효한 0으로 유지하도록 수정했다. `a2_05_bf_rate_selftest`가 published field를 legacy 값과 다르게 두고 이 선택 및 유효성 분기를 검증한다.
- 이 수정 뒤 직접시험 `A2_05_BF_RATE_SELFTEST PASS`, 실제 Si/S channel 원자료 회귀 `10 passed`, 전체 CPU 바이너리 `/tmp/lumina-bf-commonview-20261019` 빌드 성공. CUDA 컴파일 및 신규 Lumina 물리 실행은 아직 하지 않았다.
- 실제 CMFGEN 19.48d 입력의 S III·Co II `phot_data_A`는 run-local 복구본이며 저장소 기본 PHOT와 해시가 다르다. 기존 `parse_all_ions()`는 환경변수 오버라이드를 반영하지 않아 채널 시험만 복구본을 읽고 변환기 본 실행은 저장소 기본본을 선택할 수 있었다. `(16,3)`에는 `LUMINA_SIII_PHOT_PATH`, `(27,2)`에는 `LUMINA_COII_PHOT_PATH`를 적용하고, 설정된 경로가 없거나 일반 파일이 아니면 실패하도록 수정했다. 원자 빈티지 manifest에 `phot_override_env`를 기록한다. 변환기 본 함수에 링크된 기본 경로와 run-local 오버라이드를 함께 넣은 회귀시험을 추가했고, 실제 Si/S 경로시험도 동일 선택 helper를 사용한다.
- 실제 링크 입력에서 CMFGEN `MODEL_SPEC`가 S III의 활성 원자를 256 FL/79 SL로 지정하지만 원본 `osc_data`와 F_TO_S는 380 FL/127 SL임을 확인했다. 기존 `use_superlev` 분기는 항상 oscillator 전체를 보존해 이 실행의 활성 원자보다 124 FL/48 SL을 더 포함했다. 변환기에 `CMFGEN_MODEL_SPEC` 입력을 추가해 링크 순서와 이온 label을 함께 대조하고 NF를 결합한다. 같은 실행의 `LEVEL_SL_STEQ_LINKS`에서 각 활성 준위의 이름과 최종 SL ID를 F_TO_S 접두부와 하나씩 대조하며, 활성 superlevel 수까지 검증하도록 수정했다. 선택 NF 및 활성 superlevel 수, MODEL_SPEC 경로는 원자 빈티지 manifest에 기록한다. 같은 경로가 Si II/S II/S IV/Co II에도 적용된다.
- 기존 50-shell 링크 덱의 활성 level 행 수는 Si II/III/IV=157/147/66, S II/III/IV=324/380/194, Co II=1000이었다. 현재 현대 CMFGEN `MODEL_SPEC`는 각각 125/147/61, 322/256/176, 2558을 요구한다. 이 덱을 현재 CMFGEN 모델의 동질성 판정에 쓰면 안 된다. 이번 실데이터 회귀는 정확한 `atomic_links.txt`+`MODEL_SPEC`+run-local PHOT를 변환기에 연결해 6개 Si/S 이온 및 Co II의 NF·활성 F_TO_S 개수와 채널 경로를 확인했다.
- 두 회귀 파일은 `grammar-debug`에서 실행해 `13 passed`를 확인했다. 시험은 변환기 본 경로의 MODEL_SPEC/osc-link 이온 label 일치, 활성 NF, F_TO_S와 `LEVEL_SL_STEQ_LINKS`의 준위별 구성원 일치, S III·Co II 오버라이드 선택, 누락 경로 fail-closed, manifest 기록, 실제 Si/S route 보존을 포함한다. 전체 atomic 패키지 재생성이나 Lumina 물리 계산은 수행하지 않았다.
- 아직 새 Lumina 계산/고정장 율표/수렴 자기일관 비교를 실행하지 않았다. 따라서 이 기록은 구현 완료나 물리 목표 달성이 아니며, 427702 및 기존 입력·출력도 건드리지 않았다.

## 2026-10-19 운전자 진행 기록

- 활성 `MODEL_SPEC` 원자 덱 생성 job `1196247`은 `normal` 파티션에서 2분 17초 후 exit 0으로 끝났고 MaxRSS는 2,415,488 KiB였다. 입력·핵심 출력 checksum은 재현 확인했으나, 산출물은 비교용으로 **기각**한다. `parse_all_ions()`가 `MODEL_SPEC` 바깥의 자동 선택 종(Si I, S I 등)을 추가했고, 복사 템플릿의 오래된 `DECK_PROVENANCE.json`도 그대로 따라왔다. 이 덱은 보존만 하고 어떤 물리 비교에도 사용하지 않는다. 기존 427702는 취소·변경하지 않았다.
- 제출 시 고정한 구현 SHA256: `expand_atomic_data_cmfgen.py` `469bec1bac0d300dcd2c098b483426a3b3b4b1f0a918af4ca80ad5a7a0868fa1`; `cmfgen_parser.py` `d13173ce8ca7604ec38f9e24f79fb1f9d36a9e0b40ba46e16b1c51fb14640b11`; `deck_regen_r1_vintage_driver.py` `e9f068522af063bb2a8a7fb208bc19f2b81a795332134a2f7804a8045571431c`.
- 봉인 실행 입력 SHA256: `atomic_links.txt` `f395a07c887330d54dfc34271b1d9688c288e85c8487896778f3ba18ff039f52`; `MODEL_SPEC` `3e6f53861f16e2dac822d68e1412f91813c7a80f939fdbd0c4b1394e9055f2fb`; `LEVEL_SL_STEQ_LINKS` `7689a8b1cf6f0db0d6c0c9fc8dc99983318987f6a137785ac5a81eca25455b84`; run-local S III PHOT `3bd4d13837612696dc7ccd676281c3a056ca1d4006ac861c3e99171d4965306a`; Co II PHOT `bcddd512e234b4406d0cc6e3cefd37885ec96c8744535a75346d364f894002f4`.
- 기각 원인을 고치기 위해 MODEL_SPEC 모드에서는 모델에 없는 자동 선택 이온을 건너뛰고, 생성된 전체 이온 key 집합이 MODEL_SPEC과 정확히 같지 않으면 실패하도록 구현했다. 템플릿의 `DECK_PROVENANCE.json`을 재사용하지 않고 새 run links/MODEL_SPEC/준위맵 hash와 모든 active `osc/F_TO_S/PHOT/col` 원자료의 SHA256을 담은 `DECK_PROVENANCE.json` 및 `atomic_source_identity.json`을 생성하도록 수정했다. 제출 스크립트는 manifest·source identity·provenance의 ion set/count와 봉인 hash를 교차 확인한다.
- 수정 후 `tests/test_cmfgen_bf_channels.py tests/test_cmfgen_zero_route.py`는 `14 passed in 6.49s`였다. 별도 새 덱 생성 job으로 corrected output을 만들기 전까지 이전 output은 기각 상태다.
- 작업 종료만으로는 수용하지 않는다. 수정 덱의 정확한 이온 집합, NF/SL 및 PHOT 선택, 원자료·산출 checksum을 확인한 다음 공통 상태 채널율 비교를 진행한다. 수렴 물리 실행은 채널별 율 비교와 shell 대응이 준비되기 전 시작하지 않는다.

## 2026-10-19 운전자 공통상태 광율 사전진단

- 수정 덱 생성 job `1196288`은 `grammar038`에서 `COMPLETED`, exit `0:0`, elapsed `00:01:38`, MaxRSS `2,008,608K`로 끝났다. `regen_inputs.sha256`와 `regen_outputs.sha256` 전체 항목을 재검산해 모두 `OK`였다. 정확한 27개 `MODEL_SPEC` 이온 집합과 Si II/III·S II/III의 NF/NS를 진단기가 대조했다. 덱 출력은 입력 스테이징 자격만 갖추며 물리 기준해는 아니다.
- 같은 상태 forward 광율 진단기 `scripts/compare_cmfgen_bf_superlevel_rates.py`와 회귀시험을 운전자가 작성·수정했다. CMFGEN OSC의 `E_eV`는 이미 `E_ion-E_level` 문턱이고, `E_cm`은 여기 에너지다. `levels.csv`의 준위·SL 번호는 0-based, CMFGEN OSC/F_TO_S/런타임 준위맵은 1-based임을 명시해 일치 검증한다. PRRR는 MODEL_SPEC N_SL 및 10-depth 블록을 엄격히 파싱한다.
- 기존 후보 `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern`은 문서상 이미 비수렴·앵커 부적격이다. 실제 사전진단에서도 `POPSIL/POPSUL`과 각 `*PRRR` 이온밀도가 같은 상태가 아님이 확인됐다. depth 43에서 POP 합/PRRR 이온밀도(cm⁻³)는 Si II `114.50 / 2.8773e7`, Si III `2.8827e7 / 2.6995e4`, S II `9.366e3 / 1.4519e7`, S III `1.6032e7 / 4.4849e4`였다. PRRR 전자밀도와 RVTJ 전자밀도도 최대 상대차 `4.7255%`였다.
- 진단 산출물 `/gpfs/kjhan/cmfgen_runs/lumina_bf_superlevel_rates_1196288_stateaudit`의 `same_state_gate=NOT_PASS`, `interpretation_status=RATE_DIFFERENCE_NOT_ATTRIBUTABLE_TO_KERNEL_ALONE`이다. 따라서 그 안의 superlevel/총 광율 차이는 Lumina 커널 실패나 물리 불일치의 증거로 쓰지 않는다. 기존 출력이 공통상태 oracle이 될 수 없다는 점만 재확인했다. 기존 입력·출력은 보존했다.
- 새 fixed-T Stage A job은 이미 별도로 `427702`로 제출되어 실행 중이다(아래 Stage A 기록). 중복 제출이나 취소는 하지 않았다. 다음 비교는 Stage A 출력의 세대 일관성을 먼저 확인하고 수행한다.
- 위 비교기 및 atomic-channel·zero-route·Stage A 설정 회귀시험은 `26 passed in 8.16s`였다. 이 결과는 도구 직접시험이며 물리 수용판정이 아니다.

## 2026-10-19 운전자 BF 채널 fail-closed 수정

- 코드 추적에서 explicit 채널 pair 조립이 `nlte_bf_gamma_canonical()`의 오류/비소비 가능 상태를 무시하고 광이온화율 `0`으로 계속 진행할 수 있음을 확인했다. 또한 역 Milne 자극항의 `bf_rate_view_jnu()` 반환값을 무시해, 복사장 bin이 비소비 가능해도 0으로 대체할 수 있었다.
- `bf_rate_result_is_consumable()`을 추가했다. 유한·비음수율, 유한·허용범위 내 `w_miss`, 유효한 분산을 확인하고, `EXACT_ZERO`에는 정확히 0인 율·누락가중치·분산만 허용한다. 활성 단면적 지지가 있는 route의 canonical 율이 이 계약을 만족하지 않거나 역 Milne 자극항의 공통 `RadiationFieldView` bin을 읽을 수 없으면 pair 조립을 실패시킨다. 이 오류는 `nlte_solve_ion_shell()`에서 `ASSEMBLY_FAILED`로 전파된다. 단면적 지지가 없는 route는 정확한 물리적 0으로 건너뛴다. JEQB의 Planck 양성대조 경로는 그대로 유지한다.
- 직접시험: `make selftest_a2_05_bf_rate && ./selftest_a2_05_bf_rate` → `A2_05_BF_RATE_SELFTEST PASS`(소비 가능/불가 상태 경계 포함). `python3 -m unittest tests/test_cmfgen_bf_channels.py` → `7 tests`, `1 skipped`(봉인 실자료 경로 환경변수 미설정). `git diff --check` 통과.
- 전체 CPU 바이너리는 `make -j4 TARGET=/tmp/lumina-bf-channel-failclosed-20261019`로 링크 성공했다. 바이너리 SHA256 `325bbcc3e4fac919c24e38eb49eafc4a30e3521713806afc199a4b8e6d516e16`; `src/lumina_plasma.c` SHA256 `486d53711fa7e4a8042517464b28fa577542b843e1d8d355e58456a4dcc81f88`; `src/bf_rate_jnu.c` SHA256 `dd7bacaa575be7fd89d92b24b35d72414485bc4e33893c5b9e69138b0eaac2cc`. `ldd`에서 누락된 공유 라이브러리는 없었다. 기존 컴파일 경고는 남아 있으며 이번 변경으로 경고를 숨기거나 정리하지 않았다.
- 이것은 잘못된 0 대체를 막는 구현·직접시험 결과이지 이온화 분포의 물리 검증이 아니다. 신규 물리 계산은 아직 실행하지 않았고 Slurm 427702 및 기존 기준 입력·출력은 건드리지 않았다. 다음은 봉인된 동일 상태에서 채널 정·역율과 최종 stage-density/전자밀도 폐합을 확인하는 일이며, 이 기록만으로 묶음 목표를 PASS 처리하지 않는다.

## 2026-10-19 운전자 NLTE 단계·전하 폐합 및 Si 소유권 재추적

- private-candidate CE sweep에 `population_nlte_stage_charge_close()`를 연결했다. 각 NLTE 이온의 준위합을 단계밀도에 반영하고, 원소별 총수 보존을 확인한 뒤 전하합에 맞춰 `n_e`를 감쇠 갱신한다. 물리 율 sweep마다 단계합 변화와 잔여 전하 오차가 모두 1% 미만일 때만 candidate core가 수렴한다. 실패 시 private transaction은 폐기한다.
- 현재 소스 기준 직접시험 `make selftest_a2_07_population`은 최신 대상이 up-to-date였고 `./selftest_a2_07_population`은 `A2-07 population selftest: PASS`로 끝났다. 이는 합성 입력에서의 보존·폐합 시험이며 실제 Si/S 해 검증은 아니다.
- 통합 candidate 회귀 `make selftest_nlte_population_candidate`도 up-to-date였고 실행 결과 `status=PASS`, `public_mutations=0`, `solver_core=SHARED_ENTRYPOINT`였다. 이는 private transaction 통합 확인이지 CMFGEN 동질성 증거는 아니다.
- 전체 CPU 실행파일은 `make -j4 TARGET=/tmp/lumina-nlte-charge-closure-20261019`에서 최신 상태(up-to-date)였고, SHA256은 `694259c69754dfd874b541711568972b4628ecf7845d2c7c9a5652bff40c5499`이다. `ldd`에 미해결 라이브러리는 없고 `git diff --check`도 통과했다. 관련 현재 소스 SHA256: `population_contract.c` `69bdec0e1475941cde7306f202841f080240c82aa0499f2abd124cdbedff120b`, `population_contract.h` `20d8746edc5a453f26c4989d3512d1dccb54fb72da27d5de25bb21000295dd20`, `lumina_plasma.c` `ea49b9812804ce3de0c92b4c69bf393c8825d01369771e30a4cbf25f71ae41ca`, selftest `1b4c7a10cff74783cb96e93e877b29e8698907fb38ff1dff26cec3f26a9a33a2`.
- 소유권 감사와 현 코드를 대조한 결과 Stage4의 Si II–III와 Si III–IV는 Si III 슬롯을 공유한다. `pair_shares_slot`이 양쪽 solve의 각 이온 총량을 기존 단계밀도로 고정하고, 후속 쌍 뒤 공유 Si III 준위인구를 선행 쌍 값으로 복원한다. 따라서 새 단계합/전하 폐합은 `n_e` 정합에는 기여하지만 Stage4 Si II/III/IV의 총 이온분율을 갱신하지 못한다. 이 상태에서 CE 반복횟수 증가나 수렴 로그는 Si 분포의 물리 개선 증거가 아니다.
- 기존 element-wide 코드에 Si를 단순 allow-list로 추가하는 것도 충분하지 않다. 현재 private view에는 Si II/III만 있고, 실행부는 II/III/IV 세 슬롯을 요구한다. 원소-wide gate는 Z=16,26만 허용하고 Stage4와 동시 사용을 거부한다. Si IV 포함·공유단계 단일 소유권과 Si IV→V 경계 처리까지 설계해야 한다.
- 실행 중인 Stage A의 봉인 `MODEL_SPEC` SHA256은 계획값과 같은 `3e6f53861f16e2dac822d68e1412f91813c7a80f939fdbd0c4b1394e9055f2fb`이며, 해당 모델에는 Si의 `Sk2/SkIII/SkIV/SkV`와 황의 `S2/SIII/SIV/SV`가 모두 선언되어 있다. 따라서 V단을 포함한 실제 activity가 작다고 확인되기 전에는 Si IV나 S IV를 닫힌 상단 단계로 가정하면 안 된다. 기존 element-wide 행렬은 II–IV까지만 구성하고 Fe V 경계만 별도 처리하므로, Si/S에 대한 상단 처리의 동질성은 아직 입증되지 않았다.
- 이 구조 선택은 기존 S/Fe element-wide 시험의 범위를 넘어선다. multi-agent 경로는 thread 한도로 실패했으나, 운전자가 read-only Codex CLI로 Astra에 한 차례 설계평가를 전달해 결론 `REVISE`를 받았다. Astra는 II–V 단일 소유권이 분포 목표에 필요하다고 보았지만, 미모델 단계 경계·Si V 표현·최종 전자밀도 결합 계약을 먼저 고정하라고 권고했다. 제안 최소안은 private candidate 안에서 II/III/IV/V를 하나의 원소 보존계로 조립하고, 공유 Si III 인구 벡터 하나만 두며, pair별 총량 pin/restore와 중복 내부율을 없애는 것이다. synthetic 수용시험에는 분율 변화 가능성, 원소/전하 잔차, 유의한 IV↔V 왕복, 실패 시 public state byte 불변이 포함된다. Astra 평가는 물리 실행 승인이나 달성 판정이 아니며, 아직 설계 경계가 미확정이므로 신규 Si 물리 run은 시작하지 않는다. 다른 모델로 대체하지 않았다.
- Slurm CMFGEN fixed-T staging job `427702`은 마지막 상태 확인에서 `RUNNING`, elapsed `3:27:09 / 14:00:00`, node `syn01`이었다. 취소·중복 제출하지 않았다. Stage A 종료 산출물과 동일세대 상태가 확인되기 전에는 최종 Lumina 비교에 쓰지 않는다.
- 묶음 결론은 계속 미달성이다. S 단계 폐합도 동일상태율·분포 비교와 자기일관 실행으로 검증되지 않았고, Si 단계 총량은 현재 Stage4 owner 구조에서 고정되어 있다.

## 2026-10-19 운전자 exact-Hyd 입력·대용 금지 수정

- 보존된 덱 생성 job `1196288`의 실제 로그를 재확인했다. `CMFGEN_EXACT_HYD`가 꺼져 있어 27종 active deck 중 7,223개 준위에 fit type 2/3/8/9의 `params[0]-as-sigma_0` 대용값이 들어갔다. 경고 집계에는 Si IV 29, Si V 17, S V 26개 준위가 포함된다. 기존 산출물은 덮어쓰지 않으며 이 값이 들어간 sigma 파일은 matched-physics 비교 입력으로 사용하지 않는다.
- 같은 evaluator는 이미 CMFGEN `SUB_PHOT_GEN`에서 이식·검증된 code path다. `validation/layer1/L1_I3_EXACTHYD_VERDICT.md`의 fit별 A/B 및 `docs/SH_GRID_MIGRATION_VERDICT_2026-08-08.md`의 raw CMFGEN 적분 비교를 기존 근거로 재사용한다. 이는 evaluator 근거이지 최종 Si/S 분포 근거가 아니다.
- 운전자는 `CMFGEN_EXACT_HYD=1`에서 fit 입력 누락·형식 오류·물리값 거부 시 대용으로 떨어지지 않고 즉시 실패하도록 했다. 모델 provenance는 `hyd_l_data.dat`와 `gbf_n_data.dat`의 경로·크기·SHA를 기록하고 제출부는 봉인한 표 SHA와 exact mode를 재검증한다.
- 표 입력 SHA256: `hyd_l_data.dat` `f2f0b01ad95d75c81ee29063c6e81d040c50bc9bae4f1479d6e184a6dd074f0a`; `gbf_n_data.dat` `89a9fefcd46ac35cd53d0002a33e3b80acfd6e744e60e0fd39371d8e51431d91`. 구현 SHA256: `expand_atomic_data_cmfgen.py` `8fd57ec6abc980cebf5850db89e0a3820ee394b51f5b2ceeb5ffcb6bce7c819e`; `deck_regen_r1_vintage_driver.py` `0ac773b1a0aa52b32c0d78df627d29ccefaf470deee2eaf1002f6bf15e704984`; `submit_cmfgen_model_spec_deck.slurm` `666bf265abea086d21a01cfa39b9e54c9c256a7c659ae91cd1e1ec3fc02e48f2`.
- 직접시험: `tests/test_cmfgen_bf_channels.py` 9개 통과, 실자료 경로시험 1개는 봉인 환경변수 미설정으로 skip. `SI_ROUTE_TEST_TMP=/tmp python3 -m unittest tests/test_cmfgen_zero_route.py` 7개 통과. 새 exact fit 알려진-answer 테스트는 CMFGEN 수소 바닥상태 `6.3034e-18 cm²`를 확인했고 malformed fit은 대용 없이 거부했다. `bash -n` 및 `git diff --check` 통과.
- 다음 staging은 새 job-ID 전용 출력으로 동일 27종 deck을 exact-Hyd로 재생성한다. 이는 원자료/바이너리 준비이며, 기존 비수렴 seed를 CMFGEN oracle로 승격하거나 이온분포 물리 PASS를 뜻하지 않는다.

## 2026-10-19 exact-Hyd 덱 재생성 결과

- 처음 두 제출 시도는 실행 전에 거부됐다. 기존 스크립트의 `normal` 파티션과 `grammar072,078,080` 제외 목록이 현재 클러스터의 `a10`·`synNN` 구성과 맞지 않았다. 스크립트를 노드 미고정 `a10`, CPU 10/RAM 100 GiB/GPU 0/1시간으로 맞추고, 오래된 제외 목록을 제거했다. 이 설정 교정은 계산 물리 범위를 바꾸지 않았다.
- 재제출 job `427843`은 `a10` 파티션 `syn01`에서 `COMPLETED`, exit 0, elapsed `00:01:41`로 종료했다. 로그는 fit type 2/3/8/9의 7,223개 준위를 정확 evaluator로 계산했다고 기록하며, 활성 원자 검증은 `ACTIVE_ATOM_MANIFEST_CHECK PASS`였다. 오류 대용 `params[0]-as-sigma_0` 경고 없이 `DECK_REGEN_COMPLETE`에 도달했다.
- 새 덱 `/gpfs/kjhan/cmfgen_runs/lumina_model_spec_427843`은 활성 이온 27개를 포함한다. runtime provenance의 `CMFGEN_EXACT_HYD=1`; `hyd_l_data.dat` SHA256 `f2f0b01ad95d75c81ee29063c6e81d040c50bc9bae4f1479d6e184a6dd074f0a`, `gbf_n_data.dat` SHA256 `89a9fefcd46ac35cd53d0002a33e3b80acfd6e744e60e0fd39371d8e51431d91`로 입력 봉인값과 일치한다. 입력·출력 `regen_*.sha256` 전체 검산은 모두 `OK`였다.
- 생성 산출물 SHA256: `DECK_PROVENANCE.json` `eb664361f5b0e807eca54bf6ff1db045a36d9941c58c281fdf335fa9da85e772`; `atomic_data_cmfgen.h5` `2fec4d73e157e2daf24d1dea972ad965768e647207077344e37647f10e84a120`; `cmfgen_sigma_bf.bin` `a0c9e516036b7e89a46df41bd37f6288722b7ccaedb5ea9e5893da9ff99884c8`; `cmfgen_bf_channels.bin` `b34dcdae4e3d9fa593f52e54398ed380e856e1a4430e740e1d351f00e65d9592`. 갱신된 제출 스크립트 SHA256은 `24368e07de40098f32aec9eab61c5ebfa6fe12ecbb748949f969b909791e071f`이다.
- 이 결과는 원자 덱 스테이징 기준만 통과했다. 광율 동결상태 비교나 Si/S 이온분포 물리 판정은 하지 않았다. 기존 fixed-T Stage A job `427702`는 마지막 확인에서 `RUNNING`, `syn01`, elapsed `03:45:27 / 14:00:00`이었으며 취소·재시작하지 않았다.

## 2026-10-19 Stage A live-step 회계 확인

- job `427702`와 실행 step `427702.0`은 계속 `RUNNING`이며 step 이름은 `cmfgen_dev.exe`, 노드는 `syn01`이다. 경과 `03:52:39`; `sstat`는 step CPU 누적 `1-13:21:10`, 최대 RSS `97,819,832K`(약 93.3 GiB), 최대 VM `106,795,896K`를 보였다. 따라서 batch step의 CPU 0/RSS 12M만 보였던 앞선 조회는 CMFGEN 본 step의 상태가 아니며, 이 작업은 실제 계산 중이다.
- CMFGEN stdout/stderr가 실행 이후 비어 있고 아직 `metrics/` 결과가 없지만, 이는 무진행 판정 근거가 아니다. active step과 CPU/RSS 사용이 확인되므로 기다리며 보존한다. 수렴이나 물리 PASS를 추론하지 않는다.

## 2026-10-19 운전자 Si/S II–V 공동 소유 구현 착수

- 첫 Astra 설계평가의 `REVISE`는 역사로 보존한다. 그 후 고정된 Stage A `MODEL_SPEC`, CMFGEN 소스 분기, Si/S 단계별 PHOT 경계 자료를 함께 제공해 범위가 좁혀진 후속 평가를 받았고, 결론은 `PROCEED_TO_IMPLEMENT`였다. 허용 범위는 구현과 합성시험뿐이며 물리 실행이나 all-shell production 승인은 아니다.
- 초기 설계 당시 `MODEL_SPEC`에는 full atomic model VI가 없어 임의의 VI 원자나 자료 없는 V→VI 경로를 발명하지 않는 것으로 기록했다. 아래의 2026-10-19 후속 source/input 확인은 이 문구의 blanket한 `V→VI 율 미포함` 결론을 수정한다. CMFGEN은 `DI_S` terminal population을 보존식·전하식에 포함하고, 봉인된 Si V/S V `PHOT` 입력에도 VI ground로 이어지는 실제 photoionization route가 있다. VI full atom과 VI terminal scalar는 구분해야 하며, scalar/율 경로의 정확한 Lumina 대응은 여전히 검증 대상이다.
- 후속 read-only 소스 추적은 정규화의 식을 좁혔다. `/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/set_abund_clump.f`에서 `POP_SPECIES=AT_ABUND*POP_ATOM/ABUND_SUM`을 구성하고, 질량분율 입력은 원자수 분율로 변환한다. `cmfgen_sub.f`는 이를 `STEQ_MULTI_V10`의 `POP` 인자로 전달한다. `generate_full_matrix_v3.f`가 ion별 SE 행을 합치고 ground-state ionization row를 교체한다.
- 단, 상단 경계는 단순히 활성 II–V 합이 원소 총수라는 뜻이 아니다. `STEQ_MULTI_V10`은 `NEXT_PRES`가 거짓일 때 보존 행에 `DI_S-POP`을 넣고, `SUM_STEQ_SOL`은 그 추가 변수를 다음 `ION_ID(ID+1)`의 첫 준위로 표기한다. 따라서 Lumina에서 VI full atom이나 임의 atomic data를 만들지는 않되, CMFGEN의 실제 VI terminal `DI`와 V단 PHOT forward/reverse rate를 분리해 재현해야 한다. 이 값이 Lumina의 외부 ion-population stage와 같은 물리량인지 및 해당 rate route가 현재 owner에 포함되는지는 아직 수치로 입증되지 않았다.
- 봉인 소스 식별 SHA256: `cmfgen_sub.f` `092f8526661b1f9a5eaeb7a875f07f55623f427ad07c383f2659f2fb67143374`; `steq_multi_v10.f` `3fa18172b5eab3490ef159d314fc91289badd302d32788801fc982c155d3d2e7`; `generate_full_matrix_v3.f` `4b754d2bad82604c29aa70850f1b6bfc0f721dc62af6463fc8107a7b2f834065`; `set_abund_clump.f` `6decd7bc0c697b61a4bd9169c4d4718c3362a6f656e6cd588aee13be0a221057`; `sum_steq_sol.f` `759aee1f1a885d01e97c09de6ffb320d51446804d871a2f04116508b79df8caa`.
- 미해결 acceptance evidence는 이 식을 봉인된 실행 binary/deck 세대와 연결하고, 같은 세대의 shell별 Si/S active II–V population 합, `DI`/외부 stage 항, element `POP_SPECIES` 총수와 잔차를 대조하는 것이다. 이를 확보하기 전에는 물리 run을 금한다. 현재 코드는 외부 Lumina stage mass를 보존식에서 제외하고 비영 외부 mass/opacity를 fail-closed하지만, 그 stage와 CMFGEN `DI`의 대응은 아직 가정으로 남아 있다.
- 운전자가 code-only 묶음 구현을 진행 중이다. EW layout을 Si II–V와 S II–V, Fe II–IV로 확장하고 21개 인접 pair를 만들었다. 공동 owner에서는 각 활성 stage의 BB를 한 번, II→III→IV→V 경계율을 한 번씩 조립하도록 일반화했으며, 외부 stage population은 활성 원소 총수에서 보존해 제외하고 비영 외부 mass/opacity는 fail-closed한다. 선택기 `SHELL=-1`은 Si/S에만 all-shell을 의미한다.
- production CE 경로는 commit된 Si/S 공동 population을 현재 `n_e`에서 매 CE iteration 재계산하고, 이후 charge closure가 `n_e`를 갱신하도록 변경했다. Si/S는 outer CE 감쇠에 포함된다. 반복 실행 진단 파일이 서로 덮어쓰지 않도록 CE iteration별 파일명을 분리했다. 테스트 전용 layout 회귀는 36 ion slots/21 adjacent pairs와 all-shell 선택을 확인한다.
- 직접시험 현재 결과: `selftest_nlte_element_wide_layout`에서 전용 synthetic 빌드 매크로가 빠져 있던 것을 Makefile에서 보완했다. `make selftest_nlte_element_wide_layout` 및 `./selftest_nlte_element_wide_layout` 통과(36 slot capacity, 21 adjacent pairs, Si/S all-shell 선택, 4단계 원소수·전하 폐합, 합성 II–V birth/death chain의 인접율 균형과 IV↔V 율 perturbation 응답, BB 단일-owner topology helper). 기존 `./selftest_a2_07_population`도 통과했고 `git diff --check`도 통과했다. 다만 rate 응답은 synthetic chain solver 검증이지 production owner의 실제 matrix assembly/rollback 검증이 아니다. 실제 복사장 rate 정확성이나 CMFGEN 일치 증거도 아니다.
- 앞서 production CPU 전체 링크는 `/tmp/lumina-si-s-owner-20261019`로 성공했다(SHA256 `df67bcabd9cf58fbeacbe4dfec733a24dbd6c057bd1d727b087ce712db1c3e09`; `ldd`상 누락 라이브러리 없음). 이 빌드 성공은 실행 결과의 물리적 정확성을 보증하지 않는다. production owner assembly synthetic/rollback 시험은 남아 있다.
- CMFGEN fixed-T staging job `427702`는 마지막 조회에서 `RUNNING`, `syn01`, elapsed `04:45:59 / 14:00:00`이었다. 작업을 취소·재시작하지 않았다. 이 변경으로 인한 새 물리 run은 아직 제출하지 않았다.
- **현재 판정: REWORK/미완료.** II–V owner code와 기초 합성시험은 구현했지만 production owner assembly/rollback 시험 및 CMFGEN 보존식의 동일 세대 수치 증거가 미완료다. 물리 run은 아직 시작하지 않았다. Astra 평가는 계획/설계 자문으로만 사용했고, 다른 모델은 호출하지 않았다.

## 2026-10-19 CMFGEN terminal-DI 보존 연결 및 exporter 보강

- 고정 Stage A의 CMFGEN 소스와 입력을 다시 연결했다. `STEQ_MULTI_V10`은 `NEXT_PRES`가 거짓이면 보존식에 `DI_S-POP`을, 전하식에 `ZION*DI_S`를 더한다. `SUM_STEQ_SOL`은 각 활성 이온의 추가 STEQ 변수를 다음 `ION_ID(ID+1)`의 첫 준위로 표기한다. `eval_lte_v5`는 다음 이온이 없을 때 `ATM(ID)%DXzV_F`를 그 가상 다음 이온의 ground population으로 복사한다. 종료 `POP{species}` 파일은 독립 `POP_SPECIES` 총수와 각 활성 이온의 `XzV_F`, `DXzV_F`를 출력한다. 따라서 terminal `DI`는 원소수·전하 폐합에 포함되는 다음 이온 ground-state reservoir로 식별된다.
- 같은 실행 `MODEL_SPEC`에는 Si V와 S V가 각각 52/47 full level로 활성화되어 있고, V단 photoionization 입력도 실제 연결되어 있다. Stage A `PHOTSkV_A` SHA256 `8d6609658da76f99b8aee903e852b4684dd8c47f92efcb81185e356f9bcbc263`, `PHOTSV_A` SHA256 `664aa143032728a4b28d32ed0fa6beeeef883c1d765b4c14e102b28e000edaca`; 두 파일 모두 photoionization route 1개와 다음 이온의 최종 상태를 선언한다. 파일은 Stage A `linked-inputs.sha256` 기록과 대조했다.
- 소스 SHA256 추가 식별: `eval_lte_v5.f` `f76fe569ef3a739abbea5ca9d41aec76e6cb9293e2df4882ed004227181e72c0`; `rite_asc.f` `b9f94909539cf56ecf27076948f32519f1daf897bc61db7828231c3c4eb7c8e4`. 기존 `steq_multi_v10.f`, `sum_steq_sol.f` 봉인 해시는 바로 위 기록의 값과 일치한다.
- 운전자는 `scripts/export_cmfgen_ion_fractions.py`를 보강했다. 완료 실행의 `POP{species}`에서 독립 총수와 마지막 활성 이온의 `DXzV_F`를 읽어, 최종 `SCRTEMP`의 전체 표현 population 합 및 inferred terminal 행과 각각 대조한다. 두 출력 모두 동일 실행에서 생성된 필수 입력으로 해시한다. 상대 허용오차 `1e-7`은 CMFGEN `E16.7` 출력 정밀도(8 유효숫자)의 반올림 경계에 맞춘다. 총수 또는 DI가 불일치하면 결과 파일을 만들기 전에 fail-closed한다.
- 회귀시험 fixture를 실제 Stage A 형태인 Si/S II–V 활성 + VI terminal closure로 확장했다. 성공 경로, 독립 `POP_SPECIES` mismatch, top-stage `DXzV_F`/terminal DI mismatch를 포함한다. `python3 -m unittest tests/test_export_cmfgen_ion_fractions.py tests/test_ptf_gated_metrics.py tests/test_configure_cmfgen_stageA.py`는 `15 passed`; `git diff --check`는 통과했다. 이는 output parser/보존 identity 시험이지 Lumina 물리 결과 검증은 아니다.
- 현재 Lumina owner는 Si/S II–V만 행렬에 넣고 그 밖의 stage mass를 owner 보존계에서 빼며, 비영 경계 질량은 gate에서 차단한다. CMFGEN terminal DI가 owner 입력의 외부 ion-population slot과 같은지, V단의 기존 PHOT를 이용해 scalar terminal을 rate matrix에 포함해야 하는지는 숫자·율 경로 검증 전까지 미해결이다. Astra의 이 좁은 물리 계약 교차평가는 agent thread limit으로 시작되지 않아 `CONSULTATION_PENDING`이다. 다른 모델의 의견을 대신 만들어 넣지 않았다.
- `427702`는 마지막 상태 조회에서도 `RUNNING`, `syn01`, elapsed `04:45:59`였다. 현재 run root에는 `SCRTEMP`, `RVTJ`, `OUTGEN`이 보이지만 계산·후처리가 끝나지 않았다. 이 단계에서 Lumina 물리 계산은 여전히 시작하지 않았으며 목표는 미달성이다.

## 2026-10-19 Stage-V 말단 PHOT 상태의 통계중량 추적

- CMFGEN `GET_NUM_PHOT_ROUTES_V1` (SHA256 `5ee0e057557bf8b7a2ef8cf861e6fe34d89449cec77aca768addf97c7271d6d5`)에서 다음 이온 원자모형이 없으면 각 photo-route를 `FIN_LEV_ID=1`로 보내고, PHOT 입력의 `EION`과 `GION`을 그대로 route excitation/statistical weight로 사용한다. Stage A Si V/S V 입력은 각각 route 하나, excitation 0 eV, `GION=6`/`2`다. 입력 SHA256은 위 terminal-DI 기록의 두 값과 동일하다.
- 이 때문에 Si V→VI의 현재 CMFGEN 계약은 `topion_levels.csv`의 Si VI level 1 (`g=4`) 하나와 동일시할 수 없다. 그 catalog에는 level 2 (`g=2`, 5090 cm⁻¹)도 있지만 `src/lumina_atomic.c`는 top-ion catalog를 partition 계산에만 사용하고 shared NLTE level 배열에는 넣지 않는다. Stage V의 CMFGEN PHOT는 실제로 unresolved ground term을 `g=6`, `E=0`인 단일 말단 route로 취급한다. S V→VI는 양쪽 자료 모두 ground statistical weight 2다.
- Lumina의 일반 BF grid 생성기는 단일 Stage-V route의 단면을 만들 수 있지만, route-aware channel 범위는 Si/S II·III에만 있고 EW 행렬의 scalar boundary는 현재 Fe만 활성화되어 있다. 따라서 Si/S V→VI의 terminal `DI`는 여전히 Lumina owner 행렬에 들어가지 않는다. 구현 후보는 PHOT 원자료에서 봉인한 유효 `(E=0, g=6/2)` terminal target을 별도 scalar unknown으로 연결하고 그 동일 target으로 Milne 역율을 계산하는 것이다. 이는 소스에서 도출한 후보이지 아직 승인된 설계나 구현 사실이 아니며, 실제 동일 상태 율·population 대조가 필요하다.
- 최신 Slurm 확인: `427702 RUNNING`, `syn01`, elapsed `04:55:34`. Stage A의 최종 `SCRTEMP`/`POP` 폐합 증거는 아직 없고 Lumina 물리 계산도 시작하지 않았다. Astra의 좁은 물리 계약 평가는 agent thread limit으로 `CONSULTATION_PENDING`; 다른 모델로 대체하지 않았다. 묶음은 계속 `REWORK/미완료`다.

## 2026-10-19 운전자 terminal BF 구현 및 코드 추적 보정

- 코딩과 직접시험은 운전자가 수행한다. Stage-V→VI 말단 scalar BF 경로를 추가하며, Python BF-channel v2는 virtual target `-1`과 PHOT 유래 `target_energy_eV`/`target_g`를 보존하고 C 로더는 실재 target level의 metadata 및 terminal metadata를 구분 검증한다.
- 구현 검토에서 두 오류를 찾아 수정했다. 첫째, level-table의 `level_ion`은 `ion_number`이므로 Si/S V lower level은 `4`지만, 별도 population ladder의 terminal VI 슬롯은 `ion_pop_stage=5`다. 로더와 EW 조립기는 lower level을 `n_stages=4`로 검사하고, terminal population은 stage 5에 유지한다. 회귀시험은 Si의 PHOT `g=6`, S의 `g=2`를 수용하고 잘못된 lower stage 및 0 통계중량을 거부한다.
- 둘째, terminal route의 초기 적분안은 route-aware 구현과 달리 문턱 아래 bin을 포함하고 문턱 부분 bin을 자르지 않았으며, 역율 자극항에서 `nlte->J_nu`를 직접 읽었다. 이를 `bf_rate_sigma_log_weight()`의 같은 부분-bin 지지/정규화와 `bf_rate_view_jnu()`의 세대검증 복사장으로 통일했다. JEQB 경로는 부분 bin에서 `B_nu(T_e)`를 쓰고, 일반 경로는 canonical view의 같은 bin을 쓴다. 문턱 지지 밖 bin은 정·역 적분에서 제외된다.
- 직접시험: `selftest_sh_grid_loader` PASS (v1 실재 target 호환, v2 Si/S terminal 허용, 잘못된 stage 및 g 거부); `selftest_a2_05_bf_rate` PASS (기존 공통 partial-bin/kernel 시험); `selftest_nlte_element_wide_layout` PASS (production terminal assembler를 시험 전용 fixture로 직접 호출해 Si/S의 아래·부분·위 문턱 bin 처리와 독립 JEQB/Milne 율을 대조, 이어 36 slot·21 adjacent pair·합성 4단계 폐합·율응답 확인). 이는 3-bin 합성 field 검증이지 CMFGEN 고정장의 실제 shell별 결과나 전체 물리 정확성을 증명하지 않는다. `git diff --check` 통과.
- 전체 CPU 바이너리 `/tmp/lumina-terminal-scalar-20261019` 링크 성공. SHA256 `80effdccc6a5fb42b961397861a629e3e3e3640dc917a35057e072332db87644`; `src/lumina_atomic.c` `b8ac01c57eec1518f543cd65d3f8aef7095b3d072a445435790940018c3844cf`; `src/lumina_element_wide.c` `9329f95e7c42af3e191566cbf422aaacf2f034f2d10fdad106c7c9381b590e3e`; `tests/sh_grid_loader_selftest.c` `43f135614785f3d411b925181586cb820ae6a921a87bf030fe5773623879cc3c`. 빌드의 기존 경고는 남아 있으며 새 물리 계산에 사용하지 않았다.
- 마지막 Slurm 조회에서 `427702`는 `syn01`에서 `RUNNING`, elapsed `05:32:35`였다. 취소·재시작·입력 변경하지 않았다. 고정장 공통상태 비교와 Si/S shell별 population/율 검증은 아직 없고, Lumina 물리 실행도 시작하지 않았다. 따라서 scalar 경로 구현은 진행됐지만 묶음은 계속 `REWORK/미완료`다.

## 2026-10-19 운전자 Si/S owner 실패 시 pair fallback 차단

- 상위 CE 루프를 추적해 Si/S owner가 `FAIL`이면 EW가 아무것도 commit하지 않는데도 기존 pair solver가 대신 돌아가고, 그 뒤 candidate가 정상 수렴으로 표시될 수 있음을 확인했다. 이는 실제 계산이 Si/S 원소-wide 결과가 아닌데도 그렇게 오인하게 만드는 경로다.
- Si/S가 선택된 shell에서 EW commit 모드를 명시했는데 owner 상태가 `PASS`가 아니면 이제 `POP_FORBIDDEN_FALLBACK`으로 candidate transaction을 중단한다. shadow 모드, 선택되지 않은 shell, Fe 경로의 동작은 변경하지 않았다. 진단에는 Z/shell/CE 반복과 차단 상태를 남긴다.
- 직접 회귀시험은 Si와 S의 commit 실패 차단, commit 성공 허용, shadow 모드 허용, 비선택 shell 허용, Fe 동작 불변을 확인했다. `make selftest_nlte_element_wide_layout && ./selftest_nlte_element_wide_layout` 통과. 전체 CPU 링크 `/tmp/lumina-owner-failclosed-20261019` 성공, SHA256 `15fd4560c2b3d0ee78f4d6bc653ca1c2ef49ec4da7a2b95e795520c4866177a9`; `ldd` 미해결 라이브러리 없음. 수정된 `lumina_plasma.c` SHA256 `c4240e530f59a119022134157c4fe2f974e5c6bb4aa938169467d9035a2a1ec2`, 회귀시험 소스 SHA256 `5f83c8e2d058a11a3ef367d0e7ed992bb161e2d0013e0e67e5554ddcb3f1e08f`. 기존 컴파일 경고는 남아 있다. `git diff --check` 통과.
- 이는 실패한 owner 결과가 pair fallback으로 위장되는 것을 막는 구현 검증이다. 동일 복사장에서의 실제 shell별 Si/S 분포와 CMFGEN 동질성은 여전히 미검증이며, 이 수정만으로 묶음 목표를 완료 처리하지 않는다.

## 2026-10-19 운전자 CMFGEN 전자충돌 이온화 연결

- 봉인 CMFGEN 소스에서 `OMEGA_GEN_V3`는 PHOT edge 단면적으로 diagonal 충돌강도를 만들고, `STEQ_MULTI_V10`은 이를 collisional ionization으로 원소 평형식에 더한다. CMFGEN 변수는 `T/10^4 K`, `EDGE/10^15 Hz`, PHOT 단면적은 `10^10` 배 저장이며 `ZION=ion_number+1`이다. `OMEGA_GEN_V3`의 상수 `3.23`과 `STEQ_MULTI_V10`의 `8.63e-8`식을 합치면 Lumina가 쓰던 `1.55e13/sqrt(T)` 계수가 약 `0.1%` 이내에서 나온다. 출처 SHA256: `omega_gen_v3.f` `c9ecf51f4a27ab3fa11784ac4bfc1cbadef3b6f1ba63882fc3cd2b1d3d580ee5`, `subcol_multi_v6.f` `d2e6855d3197f66691705ba0c4eae4312c945e4da71936eef6af1355a5ac0015`, `steq_multi_v10.f` `3fa18172b5eab3490ef159d314fc91289badd302d32788801fc982c155d3d2e7`.
- Lumina의 기존 collisional BF 계수는 이 CMFGEN 근사식과 같은 계열이었지만, 활성화가 `LUMINA_ARTIS_PARITY` 전체 네트워크에 종속돼 있었다. `LUMINA_NLTE_BF_COLLISION` override를 추가했다. 값이 주어지면 BF 전자충돌만 독립 제어하고, 미설정이면 과거 동작대로 ARTIS parity 값을 따른다. pair route와 Si/S terminal scalar는 새 공통 helper를 사용하며 legacy route도 같은 helper를 소비한다. 그러므로 CMFGEN 비교에서 다른 ARTIS 충돌률을 무심코 켜지 않고 해당 과정만 활성화할 수 있다. Si/S owner가 commit PASS를 내려면 요구되는 collisional-BF inventory도 충족할 수 있게 됐다.
- `grammar-debug`에서 현재 소스로 회귀시험을 다시 빌드해 두 환경을 재현했다: override 미설정(ARTIS parity off) 및 `LUMINA_NLTE_BF_COLLISION=1`(ARTIS parity off) 모두 `EW owner PASS` 시험 종료. 시험에는 CMFGEN 원식의 `OMEGA_GEN_V3`→`STEQ_MULTI_V10` 대수 환산을 ion number 0–4에 대해 독립 계산해 비교하는 검사가 포함되며 상대차 한계는 `1e-3`이다. override/default 상속 정책도 직접 검사한다. 테스트 바이너리 SHA256 `bf0da9e7c1222c81443f88c7d0067941c7c81bf2d65bcdba4f87b4f8a7eec864`.
- 전체 CPU 링크는 `grammar-debug`에서 성공했고 미해결 공유 라이브러리는 없다. 바이너리 SHA256 `0f8cf7c00c0f72ef1612f74f41ad24080b8bb296d2854175c24945f77caed2be`; 현재 소스 SHA256: `lumina_plasma.c` `3076c106709f0646607f9f95d84cb8fcd66c15868661dd705c2976bcd565a3d6`, `lumina_element_wide.c` `1380623addd42e2757aefa11ef74b0d479ed501d96291e9e6f94c383bebb24b5`, test `fb75c35eca6f769bbc8afcd3b258a5fea2394d12f6b149784f08f28047bd6110`. 기존 빌드 경고는 유지됐고 새로 숨기거나 정리하지 않았다.
- 이 검증은 CMFGEN collisional-ionization **계수의 대수 등가성**과 enable 경로를 다룬다. PHOT edge 단면적 추출, full-level/SL population projection, 실제 shell별 율 및 이온분포는 아직 비교하지 않았다. `427702`는 최신 확인에서 `RUNNING`, `syn01`, elapsed `05:48:56`이며 fixed-T Stage A staging이라 최종 oracle이 아니다.

## 2026-10-19 운전자 CMFGEN PHOT_ID 충돌채널 선택 수정

- CMFGEN 원문에서 `OMEGA_GEN_V3`는 `PHOT_ID=1`로 `SUB_PHOT_GEN`을 호출한다. `GET_NUM_PHOT_ROUTES_V1`은 다음 이온이 존재할 때 `DO_PHOT(K,K)=TRUE`로 두며, 없을 때는 `DO_PHOT(1,K)=TRUE`를 모든 K에 설정한다. `SUB_PHOT_GEN`은 이 선택에 맞는 K만 합산하고 `PHOT_GRID(K)`의 격자를 읽는다. `RDPHOT_GEN_V2`에서 PHOT 파일 ID는 A부터 1-based로 매기며, Lumina sidecar는 같은 정렬을 0-based `phot_grid`로 기록한다. 따라서 활성 Si/S 인접 이온 전이는 첫 route인 `phot_grid=0`만 collisional BF에 기여하고, CMFGEN에 다음 이온이 없는 말단 전이는 모든 route가 ground reservoir에 기여한다.
- 이 mapping은 Stage A 출력 `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_stageA_fixedT_427702/PHOT_SUMMARY_FILE`에서도 직접 확인했다. `Sk2`(Si II) grid 1은 `T`, grid 2는 `F T`이고, `SkIII`(Si III)와 S II–V는 해당 단일 route가 `T`다. 따라서 해당 CMFGEN 모델에서 활성 Si II→III collisional route는 A만이며 B는 제외된다. 요약 파일 SHA256 `c10c5a928818660e2b10c2690934e62e1c47b058865e3dbf1c6774ed9e7da79a`.
- 이전 구현은 활성 pair에서도 각 route의 확률을 곱한 collisional BF를 모든 격자(A, B, …)에 더했다. 각 격자의 route 확률 합이 1이므로 다중 격자 이온에서는 CMFGEN이 선택하지 않는 B 이후 격자 충돌률이 추가될 수 있었다. `nlte_bf_cmfgen_ci_route_included()`를 추가해 pair owner는 grid 0만, terminal scalar는 모든 grid를 소비하게 했다. 복사 BF와 Milne 역율은 이전대로 모든 실제 route를 유지했다.
- 출처 코드 SHA256: `omega_gen_v3.f` `c9ecf51f4a27ab3fa11784ac4bfc1cbadef3b6f1ba63882fc3cd2b1d3d580ee5`; `sub_phot_gen.f` `e25e8e375bbf0381d6b6376e8c2a624032f6c789296a0d273adb1081b57bf537`; `get_num_phot_routes_v1.f` `5ee0e057557bf8b7a2ef8cf861e6fe34d89449cec77aca768addf97c7271d6d5`; `rdphot_gen_v2.f` `acf70de299e68bd8aabf78e181e56965ba5d5d22a5ddc97aa218df38ad3c97d1`.
- `grammar-debug` 회귀시험 바이너리 `/tmp/lumina-ew-ci-route-20261019`를 현재 소스로 빌드했다. `LUMINA_NLTE_BF_COLLISION` 미설정 및 `=1` 두 프로세스에서 모두 `EW owner PASS`. 시험은 CMFGEN 식과 Lumina 충돌률 계수 비교, override 상속, 활성 상부 이온일 때 grid 0만 선택(합계 1), 말단에서 모든 grid 선택(합계 2)을 포함한다. 바이너리 SHA256 `7067397c4152d8720640a8ee1c8e92c561e7aba6d378ce2e954baa18c4a675ab`.
- 전체 CPU 링크도 성공했다: `/tmp/lumina-cmfgen-ci-route-20261019`, SHA256 `8404e80bbcadfecf73eb30426c65eb781f7929781dfaf55796e664913b0f0762`; `ldd`에서 미해결 공유 라이브러리는 없었다. 기존 컴파일 경고는 계속 남아 있다. route builder 회귀는 `python3 -m unittest tests.test_cmfgen_bf_channels`에서 `11 tests`, `OK (skipped=1)`. `git diff --check` 통과.
- 운전자 소스 SHA256(이번 수정 후): `lumina_plasma.c` `09b234a30a3fe3bebafe986fdd80955da8523e2be9abb653d9e8f80098126b7d`; `lumina_element_wide.c` `301efb167685bc31b3aac5b8eac3e8b94f51f290bb3ab6d264527cb0d70a253a`; 시험 `nlte_element_wide_layout_selftest.c` `0d1a291c75fff110fd4f2ebf9c087d425372c03e0bd71ad6e2a349ff406d859b`.
- 최신 Slurm 확인에서 `squeue -j 427702`는 유효하지 않은 job ID를 반환했다. `sacct -j 427702` 결과는 `422019_548 COMPLETED`로 표시되어 해당 실행 ID와의 대응이 확인되지 않는다. 알려진 run root에서 `OUTGEN`은 iteration 51 도중 끝나며 `MOM_J_REL_V9` 과도 iteration 오류와 최대 변화 `71433%`를 기록한다. `SCRTEMP`, `OUTGEN`, `RVTJ`, `CFDAT_OUT`은 있으나 top-level `POP*` 완료 출력은 없다. 따라서 종료/성공을 판정할 수 없고 수렴한 oracle로 취급하지 않는다.
- 다음 물리 묶음의 평가를 Astra에 요청했으나 agent thread limit으로 호출이 시작되지 않아 `CONSULTATION_PENDING`이다. 다른 모델로 대체하지 않았고, 새 run 또는 계획/자원 계약 변경은 시작하지 않았다.
- 물리적 인수기준은 여전히 미달성이다. CMFGEN 원자 route별 충돌율 선택을 맞춘 소스/합성시험을 확인했을 뿐, 실제 동일 복사장 shell별 Si II/III·S 이온분포 비교는 아직 수행하지 않았다. `427702`가 fixed-T Stage A라면 최종 비교 oracle이 될 수 없으며, 완료된 수렴 released-T CMFGEN 결과와 동일 상태의 Lumina 율/분포 대조가 필요하다. 이번 변경을 이유로 CMFGEN 작업을 취소·재시작하거나 새 대규모 물리 run을 제출하지 않았다.

## 2026-10-19 운전자 production owner 조립·커밋 fixture

- 기존 `nlte_element_wide_layout_selftest`는 실제 terminal assembler를 호출하지만, public Si/S owner 실행부의 pair 조립·terminal 결합·최종 커밋/실패 비변경은 거치지 않았다. 이를 확인하기 위해 Si와 S 각각에 대해 4개 이온 슬롯, 준위별 BB 선, BF route, terminal PHOT scalar 및 유효 감마 publication을 갖춘 작은 합성 입력을 추가했다. 복사장은 상세평형 양성대조 `J=B_ν(T_e)`이며 원자율은 fixture 값이다.
- fixture는 `nlte_element_wide_run_status()`에서 실제 `ew_run_impl → nlte_assemble_rate_matrix`를 거친다. Si/S 정상 경로 모두 `EW_PASS`, rank `9/9`, condition number `6.295`, scaled residual 각각 `8.767e-16`/`7.376e-16`으로 커밋됐고, 이온 준위합+terminal VI가 원소 총수 `1e10`을 보존하며 population과 terminal density가 초기값에서 바뀌는 것을 확인했다. 이어 Si/S의 BF route 하나를 손상시킨 음성 대조는 둘 다 `EW_FAIL_FALLBACK_BASELINE`으로 차단됐고, public populations와 terminal density가 byte/value 불변이었다.
- 일반 빌드와 AddressSanitizer/UBSan 빌드에서 전체 `selftest_nlte_element_wide_layout`가 종료코드 0으로 통과했다. 시험 바이너리 SHA256: 일반 `f438e8452846728881ed0f538e6e93cf25702b354545ce390b6418984f5c88ff`, ASan/UBSan `dc660d59a9e38e8b71c120eb2341c86ae6aed0f3c248cb5b24826b1d9f874b9a`. 시험 소스 SHA256 `f8cd850bbde56596d40f78aeff952e1c78bcb0aee72e47773cde29e193111164`.
- 이 변경에서 현재 `lumina_element_wide.c`가 사용하는 `a208_tau_interaction_measure()` 때문에 Makefile 시험 타깃에 `src/opacity_publication.c` 의존성을 추가했다. `make -n selftest_nlte_element_wide_layout`로 대상 소스 연결을 확인했고, `git diff --check -- Makefile tests/nlte_element_wide_layout_selftest.c` 통과. 기존 경고는 그대로 두었다.
- **증거 범위 제한:** 이것은 production 조립·커밋·fail-closed 경로의 작은 합성 시험이다. 고정 CMFGEN 복사장 율, 실 원자자료, shell별 Si/S 분율 또는 CMFGEN 동질성을 입증하지 않는다. 실제 물리 묶음은 계속 미달성이며, Stage A가 수렴 oracle이 아니라는 판정도 바뀌지 않는다.
- 다음 물리 묶음(수렴 released-T CMFGEN oracle 및 동일상태 율·분율 비교) 계획을 현행 배정에 따라 Astra에 요청했지만, sub-agent 생성이 `agent thread limit reached`로 시작되지 않았다. 상태는 `CONSULTATION_PENDING`; 다른 모델로 대체하거나 이 상태에서 신규 물리 계산을 제출하지 않는다. 새 계획평가는 thread 자리가 생긴 뒤 한 번에 묶어 재요청한다.

## 2026-10-19 운전자 CMFGEN PHOT edge 단면적 경로 수정

- CMFGEN `OMEGA_GEN_V3`는 `PHOT_ID=1`, `FREQ=0`, `SET_TO_EDGE=.TRUE.`로 `SUB_PHOT_GEN`을 호출해 충돌 이온화 계수의 `PHOT_CROSS`를 얻는다. `SUB_PHOT_GEN`은 edge 빈도를 문턱에 `EQUAL_COR_FAC=1+1e-14`를 곱한 값으로 설정한다. 따라서 첫 전체 주파수 bin의 평균 단면적은 CMFGEN edge 단면적과 같지 않다.
- 기존 Lumina 구현은 첫 양수 bin 평균을 `sigma_edge`로 재사용했다. 변환기는 이제 각 채널의 CMFGEN PHOT 단면적 evaluator를 정확한 edge 빈도에서 평가해 별도로 보존하며, sidecar를 v3로 올렸다. C 로더는 v1/v2 복사 채널을 계속 읽되, 전자충돌을 켠 경우 edge 값이 없는 구형 sidecar는 pair/EW 경로에서 fail-closed한다. pair 및 Si/S terminal 충돌 이온화율은 더 이상 bin 평균을 대체값으로 사용하지 않는다. checksum에도 edge 값이 포함된다.
- `grammar-debug`에서 `SI_ROUTE_TEST_TMP=/tmp python3 -m unittest tests.test_cmfgen_bf_channels tests.test_cmfgen_zero_route`를 실행해 `18 tests, OK (skipped=1)`을 확인했다. C 로더 시험은 v1/v2 호환, v3 edge 값 왕복, 음수 edge 거부를 포함해 `SH_GRID_LOADER_SELFTEST PASS`; Si/S element-wide 합성 시험은 새 edge 값으로 충돌 정·역율을 대조해 `EW owner PASS`였다. 전체 CPU 빌드 `make -j4 TARGET=/tmp/lumina-bf-edge-full-20261019`도 성공했고 `ldd`에 미해결 라이브러리가 없다. 기존 컴파일 경고는 남겨 두었다.
- 재현 바이너리 SHA256: 전체 CPU `/tmp/lumina-bf-edge-full-20261019` `90bcc8bf12b5317e366f30f40e1900b6c80623037a0d2bf832c03e7932314aae`; 로더 `/tmp/lumina-bf-edge-loader-20261019` `5637f7c22dc94aaaeee6cd9d50af947add1b9d29e9ca45dc70deafb4731bf1f9`; EW `/tmp/lumina-bf-edge-ew-20261019` `67aae51c4ac4434980b90c5cb1728c10bdf16f6893a33b2784aea9084f22fe26`. 핵심 소스 SHA256: 변환기 `0b196963738094fec9992adc5585e4a669f8918c9333b0139c80845bed661c4f`, `lumina_atomic.c` `33607b5ff5fc7dc40a47ffdbf67dfa55e9f817ef9196f5fc66a5bd87f09678ef`, `lumina_plasma.c` `04b0855c0d2eebfe0dc9dd7cb62fba06085ee5fbd9d99bdc707afd84ec2eed23`, `lumina_element_wide.c` `8f06eba333c4533dd6022b9a0f6a8cef920b6045d62a5be73ecdbcc6d797a7d8`, channel test `45bb934d1d800746a8effc7a83df7416cffbe0a1b2b65924d61520195d3b5470`.
- 이들은 코드·합성 입력 검증이다. 실제 동일 복사장에서의 shell별 율/이온분율 비교는 아직 없고, Si II/III 및 S 분포의 CMFGEN 동질성 목표는 미달성이다.

## 2026-10-19 운전자 실원자자료 회귀 및 v3 덱 재생성

- 사용자의 현행 배정에 따라 구현과 직접시험은 운전자가 수행한다. CMFGEN 실제 `toy06_19p48d_modern` 원자입력·`MODEL_SPEC`·정확 Hyd/PHOT 파일을 연결한 Si/S II–V 및 Co II 경로 회귀시험이 `grammar-debug`에서 통과했다. 시험은 실제 입력에서 2,512개 route channel을 생성하고 활성 준위·SL 대응과 v3 edge 단면적 sidecar를 검증했다. 이는 원자자료 경로 확인이지 shell별 물리율/이온분포 비교는 아니다.
- 기존 `lumina_model_spec_427843`은 sidecar v2여서 새 충돌율 경로의 적격 입력이 아니다. 이를 보존하고, v3 sidecar 및 manifest SHA256/바이너리 차원 검증을 포함하도록 제출 스크립트를 갱신했다. `bash -n` 및 `git diff --check` 통과; 스크립트 SHA256 `73442029c09c641583c561a4ae726a9158257fd568b755431e10a42945b9fb03`.
- 새 exact-Hyd 덱 생성 job `427962`는 `a10/syn03`, CPU 10, RAM 100 GiB, GPU 0, 제한 1시간으로 제출되어 확인 시 `RUNNING`이었다. 출력은 새 job-ID 전용 `/gpfs/kjhan/cmfgen_runs/lumina_model_spec_427962`를 대상으로 하며 기존 덱을 덮어쓰지 않는다. 완료 후 v3 schema, manifest와 실제 바이너리 해시/차원을 확인한다.
- 같은 시각 Stage A fixed-T CMFGEN job `427702`도 `a10/syn01`에서 `RUNNING`(경과 7:43:15/14:00:00)이었다. 이것은 고정 온도 staging이며, 현재 로그에서 보인 비정상 대규모 반복 변화 때문에 수렴 released-T oracle로 승격할 수 없다. 취소·변경하지 않고 보존한다.
- 따라서 지금 진행 중인 것은 CMFGEN 입력 경로와 충돌율 입력 덱을 재현 가능하게 봉인하는 구현/staging이며, 물리 비교 계산은 아직 시작하지 않았다. Released-T 계산은 별도 온도 안정화·체크포인트·롤백 기준이 미정인 `docs/CODEX_RELT_RECIPE.md`의 No-Go를 해소하기 전에는 시작하지 않는다. Si II/III 및 S 이온분포의 CMFGEN 동질성 목표는 미달성이다.

## 2026-10-19 운전자 PRRR 상태 계측 의미 수정

- 소스 추적에서 기존 `compare_cmfgen_bf_superlevel_rates.py`의 총 광율 정규화에 잘못된 CMFGEN 변수를 쓴 것을 발견했다. `cmfgen_sub.f`는 `ATM(ID)%DXzV`를 `WRRECOMCHK_V5`의 `DHYD`로 전달하고, `WRRECOMCHK_V5`는 이를 PRRR의 `Ion Density`로 출력한다. 현재 이온의 총밀도는 해당 이온 full-level `CI` 합으로 계산해야 한다. 다만 아래에서 정정하듯, `DXzV`를 다음 이온의 **개별 full-level 1**로 해석했던 후속 매핑은 별도의 오류였다. 원본 산출물은 재현 이력으로 보존한다.
- `SUM_STEQ_SOL`은 추가 `DXzV` 변수를 `ION_ID(ID+1)` 및 `VEC_SL=1`로 표기한다. 2026-10-19 초기 수정은 이를 다음 활성 이온의 첫 full-level CI와 비교했으나, 이는 SL1 그룹 전체와 단일 full level을 혼동한 것이었다. source hash: `sum_steq_sol.f` `759aee1f1a885d01e97c09de6ffb320d51446804d871a2f04116508b79df8caa`; `cmfgen_sub.f` `092f8526661b1f9a5eaeb7a875f07f55623f427ad07c383f2659f2fb67143374`; `wrrecomchk_v5.f` `d1449a28f06fe636f53bd377d7d053aaff8822e0227be77406358cbb80336b84`; `rite_asc.f` `b9f94909539cf56ecf27076948f32519f1daf897bc61db7828231c3c4eb7c8e4`.
- 실제 `toy06_19p48d_modern` 출력과 덱 `lumina_model_spec_427962`에서 원소 population closure는 Si `6.95e-8`, S `4.37e-8`로 `2e-6` 기준을 통과했다. 당시 기록한 PRRR next-stage 대 POP ground 오차는 DXzV↔full-level1 혼동에 근거하므로 무효이며 아래 SL1 재비교로 대체한다. PRRR 전자밀도/RVTJ 전자밀도 `4.725%` 오차는 별도로 남지만, 곧 확인한 출력 단계 차이 때문에 물리 오류 판정에는 쓸 수 없다.
- 올바른 현재-이온 밀도로 계산한 광이온화 총율 차이(중앙값/P95/최대, dex)는 Si II `0.0217/0.1479/0.2866`, Si III `0.00558/0.2157/0.2835`, S II `0.0235/0.8394/0.9419`, S III `0.0353/0.2843/0.3716`이다. superlevel rate-density P95는 각각 `0.2698/0.3098/0.6919/0.3956 dex`로 남아 있다. 이 snapshot은 비수렴·상태 불일치라 이 차이를 Lumina 커널의 물리 오류로 단정할 수 없고, 기준 수용도 통과하지 않았다. 재생 결과는 `/gpfs/kjhan/cmfgen_runs/lumina_bf_superlevel_rates_427962_semanticfix`; 보고서가 모든 입력 hash를 기록한다.
- 직접시험 `tests/test_cmfgen_bf_superlevel_rates.py`: `grammar-debug`에서 `13 passed`; `git diff --check` 통과. 비교기/시험 SHA256은 각각 `8f0ddb9f75a0adb1c42b348ac96a0827220a166db0cc8407aabdc92a534d481e`, `3ddea61da2750c6d05f7f5c6cc172625d8529cdc9cd6a587572b650f134c5b3e`.
- 현재 Stage A fixed-T job `427702`는 확인 시에도 `RUNNING`, `syn01`, 경과 `08:05:38/14:00:00`이었다. OUTGEN은 iteration 51에서 최대 보정 `1e7%`를 기록하고 iteration 52에 들어간 상태이며 최종 `POPSIL/POPSUL`은 아직 없다. 작업은 취소·변경하지 않았다. 따라서 수렴한 CMFGEN anchor 및 동일 상태 물리 비교는 아직 미확보이고, 목표는 미달성이다.
- 수정된 물리 계측 의미와 다음 물리 번들에 대한 Astra 자문을 요청했지만, agent 생성이 다시 `agent thread limit reached`로 시작되지 않았다. `CONSULTATION_PENDING`으로 남기며 다른 모델로 대체하지 않았다. 이 자문 제한과 별개로 운전자는 코드 수정·직접시험 및 기존 승인범위의 진단을 계속한다.

## 2026-10-19 운전자 CMFGEN F_TO_S photo-route v4 일치

- CMFGEN `GET_NUM_PHOT_ROUTES_V1` 원식과 비교해 종전 변환기의 route 단위가 잘못됐음을 수정했다. CMFGEN은 대체 목적지 이름 중 먼저 일치하는 이름을 선택하고, 일치한 상위 fine level을 `F_TO_S`로 묶어 상위 superlevel 하나의 route로 취급한다. route 에너지는 첫 일치 fine member 값이며, route 통계중량과 grid 확률은 묶인 fine member들의 `g` 합에서 계산한다. 별칭들의 목적지를 합집합으로 중복 추가하지 않는다.
- 변환 sidecar를 v4로 올리고 CMFGEN의 target superlevel anchor, 합산 `target_g`, route 확률을 기록한다. 한 lower/grid row 안의 중복 target superlevel은 거부하고, 실제 활성 lower level 각각에 모든 선언된 PHOT grid가 있어야 한다. 일부 누락을 조용히 건너뛰던 동작은 fail-closed로 바꿨다. C loader는 v4의 anchor `g`보다 큰 group `target_g`를 허용하면서, v1–v3의 기존 개별 fine-level 계약은 유지한다.
- pair와 element-wide 조립에서 v4 역 Milne/충돌 재결합율은 목적 초준위의 합산 `target_g`를 쓰고 해당 초준위로 전량 도착시킨다(`within_sl_frac`로 다시 분할하지 않음). 구형 v1–v3 sidecar는 이전 fine-level 분할 규약을 유지한다. Si II/III 및 S II/III에서 발생한 채널 수 감소는 F_TO_S 초준위 병합 결과이지 단면적 누락이 아니다.
- 실 봉인 원자입력 회귀는 `grammar-debug`에서 `18 passed`로 재통과했다. 결과는 총 `1,362`개 route였으며 Si II A/B 250, Si III 147, Si V terminal 203, S II 322, S III 256, S V terminal 184개다. 이 회귀는 alias 우선순위, 실제 superlevel grouping, route-grid completeness와 sidecar 생성만 검증한다.
- 로더의 v4 메타데이터 직접시험은 `SH_GRID_LOADER_SELFTEST PASS`였고, element-wide owner 통합 fixture는 grouped upper superlevel을 넣어 Si/S commit 경로와 손상 route rollback 모두 통과했다. 추가 율 단위 regression은 production pair BF assembler에 lower level 1개와 fine member 2개를 같은 상위 superlevel로 주고 v3/v4를 비교한다. 전방 광·충돌 이온화율은 동일하고, 역 광·충돌 재결합율은 각각 `4/3` 비율을 만족했다(구형: anchor `g=2`, fine fraction `0.25`; v4: group `g=6`, 전체 초준위 도착). 합성 owner 실행파일은 `/tmp/lumina-ew-superlevel-v4-selftest-20261019`에서 빌드·실행했다. 이 직접시험은 route 소비 의미만 검증한다.
- v4 반영 전체 CPU 바이너리 빌드는 `/tmp/lumina-cmfgen-bf-superlevel-v4-20261019`에서 성공했으며, 기존 경고는 별도 정리하지 않았다. grouped-superlevel pair rate 및 public Si/S owner fixture는 `/tmp/lumina-ew-superlevel-v4-selftest-20261019`에서 다시 빌드·실행되어 `EW owner PASS`였고, 최종 시험 바이너리 SHA256은 `966c1755e2bd361c2190444f5aa13f956e2b46b7a54fffa67edbd27112e87aeb`이다. 이 턴의 핵심 파일 SHA256: 변환기 `scripts/expand_atomic_data_cmfgen.py` `8b987c90e4d2db575f9585ca216bd1f69d805206c9a70d6120435b9f407550c2`; `src/lumina_atomic.c` `2aed6b25229c3c9edd7ba169d7acfefeebd7cbda9b32904c01335be522c4e05e`; `src/lumina_plasma.c` `84b10cab2917887bdd81c45ab7bddadc2425c64eecc609d5f74d9a85f541bafe`; `src/lumina_element_wide.c` `b5a897b7139e998f4351811495e235e28ff6070660fb7b80c08e995d3972e961`; 원자자료 시험 `tests/test_cmfgen_bf_channels.py` `a1e021867fcd7378f50c50b973c5027853ab8b2afe49d146049ab7dfaeaf2a64`; owner 직접시험 소스 `tests/nlte_element_wide_layout_selftest.c` `8231f79c94d673984f6042f60b6de4eeddddaa88a4156d29b280eff1c7a8c8cb`.
- 과거 owner 합성 fixture 최종 population에 임의의 최소 차이를 요구한 비교는 유효하지 않아 폐기했다(실제 차이가 `1.9e-6`에 불과). 이를 물리효과로 해석하지 않고, rate assembler 출력에서 직접 `4/3` inverse-rate 관계를 확인하는 시험으로 바꿨다.
- **판정 한계:** 수정은 CMFGEN의 F_TO_S photo-route 해석과 Lumina 소비 규약의 소스/단위시험 일치다. 아직 v4 sidecar를 사용한 새 fixed-state CMFGEN 율 비교 또는 수렴 Lumina run은 없으며, shell별 Si/S 이온분포의 CMFGEN 동질성 목표는 미달성이다. 427702의 운용상 보호 여부도 유지하고, 새 물리 계산은 제출하지 않았다.
- 후속 출력 확인에서 기존 deck job `427962`의 sidecar는 schema v3, 2,512 channels로 확인됐다. 해당 디렉터리는 보존하고 재사용하지 않는다. 봉인된 동일 CMFGEN 원자입력의 현재 SHA256은 이전 제출 manifest와 모두 일치했다. v4 전용 새 덱 staging job `428083`을 `a10`, 노드 미고정, CPU 10, RAM 100 GiB, GPU 0, 최대 1시간으로 제출했다. 생성 예상 경로는 `/gpfs/kjhan/cmfgen_runs/lumina_model_spec_428083`이며 출력 경로는 job-ID 전용이다. 제출 당시 변환기 SHA256 `8b987c90e4d2db575f9585ca216bd1f69d805206c9a70d6120435b9f407550c2`, parser `d13173ce8ca7604ec38f9e24f79fb1f9d36a9e0b40ba46e16b1c51fb14640b11`, driver `0ac773b1a0aa52b32c0d78df627d29ccefaf470deee2eaf1002f6bf15e704984`, submit script `75f2e1a4783ea843be9a0839dc9b160042c580b72a4709d2b2843d31b2077e43`; 봉인 입력은 `atomic_links` `f395a07c887330d54dfc34271b1d9688c288e85c8487896778f3ba18ff039f52`, `MODEL_SPEC` `3e6f53861f16e2dac822d68e1412f91813c7a80f939fdbd0c4b1394e9055f2fb`, `LEVEL_SL_STEQ_LINKS` `7689a8b1cf6f0db0d6c0c9fc8dc99983318987f6a137785ac5a81eca25455b84`, run-local S III PHOT `3bd4d13837612696dc7ccd676281c3a056ca1d4006ac861c3e99171d4965306a`, Co II PHOT `bcddd512e234b4406d0cc6e3cefd37885ec96c8744535a75346d364f894002f4`, HYD-L `f2f0b01ad95d75c81ee29063c6e81d040c50bc9bae4f1479d6e184a6dd074f0a`, GBF-N `89a9fefcd46ac35cd53d0002a33e3b80acfd6e744e60e0fd39371d8e51431d91`. 아직 Slurm 상태를 다시 조회하지 않았다. 이 job은 입력 덱 생성만 하며 CMFGEN/Lumina 물리 계산을 시작하지 않는다.
- v4 전체 CPU 실행파일 `/tmp/lumina-cmfgen-bf-superlevel-v4-20261019` 존재 및 SHA256 `fd8fc3d1511d8cf4e0152fad98816ea4d04fa8b5f183b8b7d53f8167febc4308`을 확인했다. 새 physical comparison run은 계속 미제출이다.

## 2026-10-19 운전자 양방향 BF 고정상태율 비교기

- 사용자의 최신 배정대로 코드 구현과 직접시험은 운전자가 수행했다. `scripts/compare_cmfgen_bf_superlevel_rates.py`를 확장해 CMFGEN PRRR의 `Recombination Rates` superlevel 행을 읽고, sealed v4 route binary의 route별 sigma·문턱·합산 상위 `g`·확률, 같은 CMFGEN `EDDFACTOR` 복사장, `RVTJ` 온도/전자밀도, 상위 이온의 실제 F_TO_S superlevel population으로 Lumina v4 Milne 역율 밀도를 재구성한다. 상위 SL은 sealed `LEVEL_SL_STEQ_LINKS`와 덱 준위표에 대조한다. 3866-bin canonical 광율 격자와 1234-bin route Milne 격자는 별도 보존장 재격자로 처리한다. route binary schema/hash/version을 fail-closed 확인하고, 유효하지 않은 활성 Jν bin을 0으로 대체하지 않는다.
- `grammar-debug`에서 `python3 -m pytest -q tests/test_cmfgen_bf_superlevel_rates.py`는 `19 passed`였다. 시험에는 PRRR RR 블록, v4 CSR 바이너리, 잘못된 schema/후행 바이트, 부분 문턱 bin Milne 적분, 누락 복사장 차단, 두 상위 fine member를 한 SL로 집계하는 경로가 포함된다. 실제 비교 명령은 아래 sealed 입력으로 `grammar-debug`에서 실행했다:

  ```bash
  python3 scripts/compare_cmfgen_bf_superlevel_rates.py \
    --cmfgen-run /gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern \
    --deck /gpfs/kjhan/cmfgen_runs/lumina_model_spec_428083 \
    --fixture /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/l1bf_fixture \
    --out /gpfs/kjhan/cmfgen_runs/lumina_bf_superlevel_bidirectional_428083_20261019
  ```

- 결과 요약은 `/gpfs/kjhan/cmfgen_runs/lumina_bf_superlevel_bidirectional_428083_20261019/summary.json`에 보존했다. forward와 reverse 둘 다 `NOT_PASS`이며 같은 상태 gate도 `NOT_PASS`다. 원소 population closure는 Si 최대 상대잔차 `6.95e-8`, S `4.37e-8`로 통과했지만, PRRR의 다음 이온 ground와 POP 출력 ground 최대 상대차는 Si II `1.031`, Si III `0.564`, S II `17.10`, S III `3.312`; PRRR 전자밀도와 RVTJ 차이는 네 이온 모두 `4.725%`였다. 이 입력 쌍은 동일 실행/동일 상태가 아니므로 편차를 Lumina 커널의 물리 오류로 판정하지 않는다.
- 관측된 RR 밀도 편차(중앙값/P95/최대, dex)는 Si II `0.00718/0.185/0.622`, Si III `0.00371/0.123/0.220`, S II `0.0341/0.250/0.391`, S III `0.0326/0.185/0.281`이다. 모든 이온에서 P95 또는 최대가 고정상태율 허용치를 넘으나, 앞의 상태 불일치 때문에 원인 분류·물리 수용 판정에는 사용할 수 없다. 이는 처음으로 역재결합 출력까지 비교기에 연결한 forensic 결과이지 수렴된 이온분포 결과가 아니다.
- 재현 식별자 SHA256: 비교기 `fcd803820fe529d092a202f9033884ba559333607c385f3848fe934467f1440b`; 시험 `3baad20a547766c18e16a442073afe5d780bbadd61abda10656aa12a3449b995`; fixture `b6eb45aa6357578ddbf8af7cb75ad6a7f5ec702c158fe5d76cc9fdf2ed067314`; `summary.json` `1f708ed18a386235138feaf9323351dc3e13eff5a51453a31136a19067d73961`; detail CSV `77c04e5ec8d71dc56c58c8cbee918e9562cfc050fd5ea2e4ae834928516a8212`. 보고서의 `input_sha256`에는 run/deck/atomic-source/PRRR/Jν 및 도구 입력이 각각 기록돼 있다.
- `427702`의 현재 Slurm 식별은 미해결이다. `squeue`와 `scontrol show job 427702`는 `Invalid job id specified`였고, `sacct --jobs=427702`는 무관한 2026-08-10 job `422019_548`를 반환했다. 해당 run root에는 `EDDFACTOR`, `SCRTEMP`, `OUTGEN`은 있으나 `POPSIL`, `POPSUL`, Si/S `*PRRR`가 없어 이번 비교 입력으로 사용할 수 없다. 이 작업은 취소하거나 재제출하지 않았다. 우선 실제 job/accounting identity와 동일 실행 POP/PRRR 산출물 경로를 확인해야 한다.
- 결론: 코드 경로·synthetic rate시험 및 고정장 양방향 comparator 구현은 진행됐지만, 동일 상태 비교도 수렴 oracle도 아직 확보하지 못했다. Si II/III·S 이온분포의 CMFGEN 동질성 목표는 미달성이고, 새 Lumina 물리 계산은 제출하지 않았다.

## 2026-10-19 운전자 PRRR/RVTJ 온도 상태 일치 gate

- 기존 역 Milne 재구성은 `RVTJ` 온도를 사용하면서 `same_state` gate에서 PRRR의 자체 온도 프로필을 대조하지 않았다. 비교기에서 PRRR `Temperature [1.0E+4K]`의 모든 10-depth 블록을 엄격히 파싱하고 K로 변환하도록 구현했다. 온도 누락·깊이 누락/중복·비연속 블록·폭 불일치는 실패한다. 동일 상태 판정은 next-stage SL1, 전자밀도, 원소 폐합과 함께 PRRR↔RVTJ 온도 상대차 `<=5e-5`도 요구하며, 깊이별 값을 CSV에 기록한다.
- 직접시험: `ssh grammar 'ssh grammar-debug "cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn && python3 -m pytest -q tests/test_cmfgen_bf_superlevel_rates.py"'` → `22 passed`. `git diff --check` 통과.
- 상태 불일치의 소스 원인도 확인했다. 실행 소스 `cmfgen_sub.f`는 최종 반복 플래그 `LST_ITERATION` 안에서 line 3033에 `WRRECOMCHK_V5`를 호출하지만, `SOLVE_FOR_POPS`는 line 4002에서 그 뒤에 수행된다. 이어 line 4036의 `POPTOION`은 solved `POPS` 벡터를 이온 population에 복사하고 `ED=POPS(NT-1,:)`, `T=POPS(NT,:)`로 갱신한다(`poptoion.f` lines 18–26). 최종 `RVTJ`는 line 4398, 종별 `POP*`는 line 4528 이후 기록된다. 따라서 이 run의 PRRR는 population solve 전 상태이고, 비교기가 읽은 `RVTJ`/`POP*`는 solve 후 상태다. 파일들은 같은 실행에서 생성됐지만 같은 solver phase가 아니다. `T`가 fixed였기에 PRRR↔RVTJ 온도가 거의 같은 사실은 이 판정과 양립한다.
- 해당 source identity: `cmfgen_sub.f` SHA256 `092f8526661b1f9a5eaeb7a875f07f55623f427ad07c383f2659f2fb67143374`; `wrrecomchk_v5.f` `d1449a28f06fe636f53bd377d7d053aaff8822e0227be77406358cbb80336b84`; `poptoion.f` `099c8f232b8a3a55dffb70eae392bf02a0d120019378c4f1efbddb5d84f51d30`. 기존 OUTGEN 감리도 이 run의 마지막 반복을 비수렴으로 판정했다(`MAXCH=3.46e3%`, `docs/CODEX_EQUIV_PLAN_C_REVIEW.md`). 즉 현재 오프셋은 Lumina 율 커널의 오류로 귀속할 수 없으며, 이 CMFGEN fixed-T 출력 묶음은 수렴 oracle도 아니다.
- 새 읽기 전용 진단은 이전과 동일한 CMFGEN 실행 및 v4 덱으로 수행했고, 덮어쓰기를 막기 위해 `/gpfs/kjhan/cmfgen_runs/lumina_bf_superlevel_temperatureaudit_428083_20261019`를 사용했다. PRRR↔RVTJ 온도 최대 상대차는 `4.79724e-5`다. 당시 next-stage ground 수치는 이후 SL1 매핑 오류로 무효 판정됐다. 전자밀도 차이 `4.725%`와 pre/post-solve 단계 불일치 때문에 `same_state_gate=NOT_PASS`이며, 이 결과는 물리 동질성을 판정하지 않는다. 새 물리 계산은 실행하지 않았다.
- 다음 유효 비교는 PRRR와 같은 **pre-`SOLVE_FOR_POPS` 상태**의 full populations·`ED/T`를 함께 보존해 대조하거나, 실제 수렴을 입증한 모델에서 pre/post-solve 차이가 허용치 안에 있음을 먼저 보여야 한다. source phase가 정렬되지 않은 현 진단을 고치는 명목으로 gate를 완화하거나 현재 산출물을 물리 acceptance로 승격하지 않는다.
- 새 출력 SHA256: `summary.json` `46d6516d15002d4746ebd0a61ca6c11f22f62539de5f247b066c562e1cadf0c4`; `snapshot_consistency.csv` `0856137565af0717b91f183245c7bcb59452df8a43c9e3f1e64f744e65f959d7`. 현재 비교기 SHA256 `1cce237cde9083552c2b001148980d94cea2e57041fa4ad0ea8db49359830f50`; 시험 SHA256 `d3097f4341e2d4c489a82d890546ee3c06f3592f4a7751be6d3f196fbc2484f5`.

## 2026-10-19 운전자 PRRR DXzV→SL1 매핑 및 pre-solve state 대조

- 코드 재감리에서 `SUM_STEQ_SOL`의 `VEC_SL(J)=1`을 확인했다. PRRR `DXzV`는 다음 이온의 **superlevel 1 전체 인구**다. 반면 `POPSIL/POPSUL`의 `CI(:,1)`은 그 SL에 속한 개별 첫 full level일 뿐이다. 비교기를 `F_TO_S`/sealed runtime map에 따라 모든 full-level CI를 SL별 합산한 뒤 SL1과 대조하도록 고쳤다. 비교기 결과 schema는 v4로 올리고 관련 CSV/JSON 필드 이름도 SL1로 바꿨다. 기존 raw 출력은 수정하지 않았으며, 이전 SL1 오차 수치는 폐기한다.
- `grammar-debug` 직접시험 `tests/test_cmfgen_bf_superlevel_rates.py`: `23 passed`. 새 진단 출력 `/gpfs/kjhan/cmfgen_runs/lumina_bf_superlevel_sl1audit_428083_20261019`의 schema v4 결과에서 PRRR DXzV↔최종 POP 기준 SL1 최대 상대차는 Si II `1.031`, Si III `0.564`, S II `1.172`, S III `0.584`다. 이 비교도 서로 다른 solver phase의 상태를 대조하므로 원인 판정에는 아직 사용하지 않는다. PRRR↔RVTJ `n_e` 차이는 `4.725%`, 온도 차이는 `4.798e-5`이며 same-state gate는 `NOT_PASS`다.
- 더 직접적인 상동 단계 대조로 `SCRTEMP` 40개 보존 상태를 훑었다. `toy06_19p48d_modern`의 model dimensions는 `ND=90`, `NT=1800`; record 39에서 네 PRRR 모두 `n_e` 최대 상대차 `3.8898e-5`, `T` 최대 `4.7980e-5`, DXzV↔해당 다음 이온 SL1 최대는 Si II `4.2502e-5`, Si III `4.0054e-5`, S II `4.0138e-5`, S III `4.0001e-5`였다. 이는 PRRR와 같은 pre-`SOLVE_FOR_POPS` 상태가 기록되어 있음을 출력 반올림 수준에서 확인한다. 기존 PRRR/POP 차이는 단순 파일 혼합이 아니라 pre-solve 대 post-solve 출력 차이다.
- 재현 한계와 다음 구현: 현재 comparator의 광율 가중에는 final POP의 full-level 분포를, RR 재구성에는 final POP/RVTJ를 사용하므로 아직 이 matched pre-solve state를 rate 재생에 소비하지 않는다. RR은 SCRTEMP의 상위 SL 인구·`n_e/T`로 같은 상태 재구성이 가능하다. Photo rate density는 CMFGEN `SUP_TO_FULL_V3b`의 full-level 분배까지 pre-solve 상태로 재현하거나, 같은 위치에서 full-level pre-solve population을 보존해야 한다. 이 fixed-T run은 `MAXCH=3.46e3%` 비수렴이므로 어떤 경우에도 최종 물리 acceptance 기준이 아니다.
- 새 SL1 진단 `summary.json` SHA256 `19d372e6a7858dd7e7e14d51cd5d799523df3eb1e75b1969a1ea146f50782419`; `snapshot_consistency.csv` SHA256 `8616b1cb0aa2837a0807c88b950c931afb2d792c4761dbc098593bac22d08455`.

## 2026-10-19 운전자 SCRTEMP 동상태 RR 재생

- 직전 절에서 확인한 SCRTEMP record 39를 비교기가 직접 찾도록 확장했다. PRRR 네 이온의 `DXzV`↔다음 이온 SL1, 전자밀도, 온도를 모든 보존 SCRTEMP 상태와 대조하고, 허용오차 내 후보가 **유일하게 하나**일 때만 진행한다. 결과는 record 39 하나를 선택했으며 네 이온 모두 PRRR↔SCRTEMP 최대 상대차가 SL1 `4.00–4.25e-5`, `n_e` `3.890e-5`, `T` `4.798e-5`였다. 두 개 이상 후보 또는 후보 부재는 실패 처리한다.
- Lumina RR 재생은 이제 matched SCRTEMP의 상위 이온 superlevel population, `n_e`, `T`를 사용한다. CMFGEN `cmfgen_sub.f`에서 PRRR 호출은 3033행, EDDFACTOR 기록/종료는 3922행 이후, `SOLVE_FOR_POPS`는 4002행, 사후 `POPTOION`은 4036행이다. 따라서 같은 실행의 EDDFACTOR 복사장과 matched SCRTEMP 상태를 이용한 RR 재생은 pre-solve phase에 정렬된다. 이 소스의 SHA256은 `092f8526661b1f9a5eaeb7a875f07f55623f427ad07c383f2659f2fb67143374`다. forward photo-rate density는 여전히 사후 POP의 full-level population을 사용한다.
- RR 밀도 차이(중앙값/P95/최대, dex): Si II `0.00214/0.01566/0.03315`, Si III `0.000540/0.00519/0.01181`, S II `0.02794/0.04005/0.04613`, S III `0.03186/0.04585/0.07865`. 세 이온은 기존 median/P95/max 기준을 통과했고, S III는 중앙값 기준 `0.03 dex`를 `0.00186 dex` 초과해 `NOT_PASS`다. 전체는 `0.00599/0.04232/0.07865 dex`; 광이온화는 기존처럼 `NOT_PASS`이며 이 run의 비수렴 상태도 그대로다. 따라서 CMFGEN 동질성/최종 물리 목표는 미달성이고 기준 완화는 하지 않았다.
- `grammar-debug` 직접시험 `tests/test_cmfgen_bf_superlevel_rates.py`: `25 passed`. 수정된 solver-phase 보고를 포함한 새 읽기 전용 재생 산출물은 `/gpfs/kjhan/cmfgen_runs/lumina_bf_superlevel_phaseaudit_428083_20261019`이다. `summary.json` SHA256 `1a1d2b75222fb251c86af02e560f0ae2241af0cb36dd679557d4744e22995463`; `si_s_bf_superlevel_rates.csv` `7a66ee55e00ed94cd4a94d2c54bb53d7b3d685a88da655f290c14e16d2c65f5d`; `snapshot_consistency.csv` `a0094c9c075b9a4f6bd0f92d2367026835eb8f12b720fb2ff723afc3e4d33077`.
- 구현/시험 SHA256: 비교기 `6d722800a9c14f5e51cea7557d883a163efa6a55afdf52d4a7faf2fd9ccadc65`; 시험 `fdfd38ebb38d6e4cf53f45739f0080e46a0efdb8c1148c6c7bbea742acc7a934`. 다음 유효 구현은 CMFGEN `SUP_TO_FULL_V3b`의 full-level 재분배를 사후 POP fixture에 먼저 대조해 검증한 뒤, 이를 pre-solve SCRTEMP에 적용하여 forward photo-rate 상태를 맞추는 것이다. 물리 run은 아직 제출하지 않았다.
- 다음 묶음에서 실제 `SUP_TO_FULL_V4` 계측과 Python 재현 중 경로 선택·물리평가를 Astra에 요청했으나 `agent thread limit reached`로 호출되지 않았다. `ASTRA_PENDING`; 자문을 받은 것으로 간주하지 않는다. 운전자는 가역적인 소스 추적만 계속하며, CMFGEN 소스 수정이나 새 물리 run은 Astra 계획평가가 가능해질 때까지 시작하지 않는다.

## 2026-10-19 운전자 실제 PHOT route 소스 대조

- 사용자 최신 지시에 따라 코딩과 직접시험은 운전자가 맡았다. 실제 CMFGEN `GET_NUM_PHOT_ROUTES_V1` route 열거와 Lumina grouped route 목록을 대조하려던 읽기 전용 비교는, `parse_f_to_s_int_seq`가 모든 원자자료에 8번째 열 FL ID가 있다고 잘못 가정해 중단됐다. 봉인된 Si IV `f_to_s_split`은 7열 구형 형식이며 CMFGEN은 데이터 행 순서를 FL ID로 소비한다. 파서를 구형 행순서와 명시적 FL-ID 형식 모두에 맞추고 혼합 형식은 거부하게 했다. `INT_SEQ` 부호는 CMFGEN 독자에 부호 제한이 없어 거부하지 않도록 수정했다. 이는 비교기 입력 파서/진단 변경일 뿐 물리 코드나 원자자료는 수정하지 않았다.
- `grammar-debug` 직접시험은 `tests/test_cmfgen_bf_superlevel_rates.py` 단독 `38 passed`, 두 관련 시험 파일 통합 `40 passed`였다. 기존 수정된 `a2_05_l1bf_gate.py` 회귀도 함께 통과했다. 재생 명령은 앞 절의 동일한 run/deck/fixture를 사용하고 출력만 새 경로 `/gpfs/kjhan/cmfgen_runs/lumina_bf_superlevel_routeaudit_retry2_428083_20261019`로 격리했다. 비교기는 기존 물리율 게이트가 미통과라 exit 1을 반환했지만 `summary.json`과 CSV를 정상 작성했다. 이는 계산 실행 실패가 아니라 기존 `NOT_PASS` 판정의 종료 코드다.
- 실제 대상 PHOT grid 5개(Si II 2개, Si III 1개, S II 1개, S III 1개) 모두에서 route 개수·목적 초준위·anchor 순서·목표 에너지·통계중량·확률이 일치했다. 따라서 `GET_NUM_PHOT_ROUTES_V1`의 `F_TO_S_SAVE` 갱신 의심은 이 실행에 연결된 실제 PHOT 채널에서는 재현되지 않았고, 현재 큰 율 잔차의 원인으로 지지되지 않는다. 이 비교는 조사된 5개 grid에 한정되며 다른 원자자료 일반에 대한 증명은 아니다.
- 같은 재생에서 기존 결과는 변하지 않았다. 광율 최대 절대편차는 Si II `0.38410 dex`, Si III `0.05396`, S II `0.44055`, S III `0.17439`로 게이트는 계속 `NOT_PASS`; 재결합 전체 중앙값/P95/최대는 `0.00599/0.04232/0.07865 dex`이며 S III 중앙값 기준 초과도 그대로다. fixed-T CMFGEN 입력은 여전히 비수렴(`MAXCH≈3.46e3%`)이므로 이는 수렴 이온분포 동질성 판정이 아니다. 새 물리 계산이나 Slurm 제출은 없었다.
- 결과 경로: `/gpfs/kjhan/cmfgen_runs/lumina_bf_superlevel_routeaudit_retry2_428083_20261019`. `summary.json` SHA256 `5a6936cb7950190b0be5ff9ee9b82d236b7aef84bdf55f744851af1dd13c38e8`; `si_s_bf_superlevel_rates.csv` `76582e9faf7c92f057174197b7d28c84aab866204880086125d86306f47a258b`. 비교기 SHA256 `845b2f45d19a759ed4ad611e8f4cc62fad63f607daf64c5740de22ca74a0daa8`; 시험 SHA256 `dcce3debd141e1ae8a86de64fa43ff0d701ed1dc6caf2c375810bcd0758a8e03`. 통합 직접시험: `ssh grammar 'ssh grammar-debug "cd /home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn && python3 -m pytest -q tests/test_a2_05_l1bf_gate.py tests/test_cmfgen_bf_superlevel_rates.py"'` → `40 passed`. Astra 평가 요청은 여전히 `agent thread limit reached`로 `ASTRA_PENDING`; 다른 모델을 대체 호출하지 않았다.

## 2026-10-19 운전자 PHOT 평활화 누락 원인 재현과 생산 덱 수정

- **큰 국소 광율 잔차의 원인을 실제 CMFGEN 루틴으로 재현했다.** 봉인 실행 `MODEL`은 `SIG_GAU_KMS=3000`이고 Si/S PHOT 헤더는 입력 평활화 폭 `0.0`이다. `RDPHOT_GEN_V2`는 `SM_PHOT_V3`와 `CUT_POINTS_V3`를 적용해 유효 단면적을 만들지만, 기존 Lumina 변환기는 원본 테이블을 그대로 적분했다. 파일 hash가 같아도 실제 소비한 단면적은 달랐다. 선택 설정은 `FRAC_SIG=.25`, `CUT_ACC=.02`, `ABV_EDGE=T`이며 세 선택값은 해당 실행에서 생략됐을 때 `RD_CONTROL_VARIABLES`가 쓰는 기본값이다.
- `grammar-debug`에서 원본 Fortran 세 파일만 `gfortran -O2 -fPIC -shared -ffixed-line-length-none`으로 격리 빌드했다. 라이브러리 `/tmp/lumina-sm-phot-20261019-vIpAcT/libsm_phot.so` SHA256 `90bd55a89c78b947e848969f47abfc250baf2f4ba08dc78c81d5245c3c1cd6de`; `SM_PHOT_V3` 소스 SHA256 `c36aefa4f2fb32901b30ec1e8f347bbf96377d84da13d0233ec6fb8b42efd4d1`, `CUT_POINTS_V3` `e7312ef12dad7be5f68d89ca7d3b268a8b521382cc8b8ff3e98bdc673f497b55`, 실행 `MODEL` `be7c79a3efd8bc13e9f2158aa9ea6f506534444ba5104c98f69b79967107af13`이다. CMFGEN 원본 소스와 원자자료는 수정하지 않았다.
- 동일 SCRTEMP39 개체수와 raw EDDFACTOR를 유지하고 단면적에만 원본 평활화 처리를 적용한 지점 비교는 다음과 같다. 아래는 덱 생성 전 원인 재현이며 전체 shell의 생산 율 통과 판정이 아니다.

  | 이온·지점 | 원본 PHOT/PRRR 율 비 | 평활화 PHOT/PRRR 율 비 | 평활화 후 절대 dex 차이 |
  |---|---:|---:|---:|
  | Si II SL18, depth15, FL32/33 | 0.3942563 | 1.0000683 | 0.00002966 |
  | S II SL40, depth34, FL165/169 | 0.3275665 | 1.0001397 | 0.00006065 |
  | S III SL56, depth27, FL121 | 0.6555123 | 1.0001832 | 0.00007954 |

- Si II `3s2_5g_2Ge`는 원본 1,813개 점에서 평활화 291개 점이 됐고 문턱 단면적은 `4.367588887658677 Mb`였다. 이 값은 wrapper 회귀시험의 독립적인 직접 Fortran 기준값으로 사용했다. `DO_DIS=T`에 따른 CMFGEN 문턱 아래 준위 용해 항도 별도로 계산했으며, 이 지점들에서 합산 율 변화는 각각 약 `4.6e-5`, `3.2e-4`, `1.4e-6`(평활화 율 대비 상대값)이었다. 큰 차이를 제거한 항은 PHOT 평활화다. 생산 경로의 준위 용해 처리는 전체 비교 결과에서 필요한 범위를 계속 확인한다.
- 구현: `cmfgen_parser.py`가 입력 PHOT의 기존 평활화 폭을 노출한다. 새 `cmfgen_phot_smoothing.py`는 원본 Fortran 루틴을 호출하며, 이미 목표 폭의 90% 이상으로 평활화됐거나 헤더가 없는 경우에는 CMFGEN의 생략 규칙을 따른다. `expand_atomic_data_cmfgen.py`는 봉인된 `MODEL_SPEC`의 실행 `MODEL`을 읽어 모든 PHOT 파싱·HDF5·총 단면적·채널별 정/역율 덱 생성에 같은 처리를 공급한다. 요구되는 평활화를 조용히 생략하지 않는다. `deck_regen_r1_vintage_driver.py`는 설정·모델 hash·Fortran 소스 hash를 기존 provenance에 기록하고 평활화 라이브러리를 새 덱에 보존한다. 비교기도 해당 provenance로 같은 PHOT 평가를 재생한다. 생산 수송 자체에는 공유 라이브러리가 필요 없고 생성된 단면적을 소비한다.
- 직접시험: 봉인된 실제 원자자료와 원본 평활화 라이브러리를 사용해 PHOT wrapper·실제 Si/S channel bake·같은 상태 비교기·복사장 재빈닝의 네 시험 파일이 `57 passed`였다. 실제 자료 시험을 포함해 skip은 0이었다. 기존 provenance 시험의 환경변수 기대값은 실행 환경의 실제 값과 비교하도록 고쳤다. 수정한 Slurm 스크립트는 shell syntax 검사도 통과했다. 이는 전체 분포 동질성 판정이 아니다.
- 같은 승인 범위에서 평활화를 반영한 새 덱을 생성하고 전체 고정상태 정/역율을 다시 비교한다. 자원은 기존과 같은 노드 미고정 `a10`, CPU 10, RAM 100 GiB, GPU 0, 최대 1시간이며 기존 덱 `428083`과 모든 raw 결과를 보존한다. 이 작업은 원자 덱 생성이고 자기일관 물리 계산은 후속 근거에 따라 진행한다.
- Astra의 묶음 평가도 이번에는 정상 수신했다(agent `01a15262-e1dd-7302-998d-11dafa6ead57`; 완료한 이전 자문 스레드를 정리해 슬롯을 확보). 판단은 목표 미달성과 조건부 후속 진행이다. 국소율 원인 수정과 별개로 수렴 CMFGEN oracle 확보를 불필요하게 직렬화하지 말고, 검증된 새 checkpoint 복제본에서 안정화 증거를 확보한 뒤 released-T로 이어가도록 권고했다. 작은 중요도나 총 Γ의 근접성으로 기존 율/분율 기준을 면제하지 않는다. 최종적으로 생산 owner 정/역율, 전하·RE·수송 및 마지막 반복 변화와 shell별 Si/S 분포 증거를 함께 요구한다. 제안한 후속 CMFGEN 자원 상한(각 14시간·16 CPU·200 GiB)은 아직 실행한 자원이 아니며, 현 덱 생성 한도와 구분한다. 자문 수신으로 `ASTRA_PENDING`은 해소됐지만 물리 목표 달성 판정은 계속 미완료다.
- 새 덱 job `428222`는 `syn03`에서 실행되어 `DECK_REGEN_COMPLETE`까지 완료했다. 출력 `/gpfs/kjhan/cmfgen_runs/lumina_model_spec_428222`는 27개 이온, 24,542개 준위, 2,588,798개 선, Si/S 채널 1,362개를 포함하고 기존 활성 원자 manifest 검사를 통과했다. 요청 자원은 위 CPU 10/RAM 100 GiB/GPU 0/최대 1시간과 같았다. `regen_inputs.sha256`, `regen_outputs.sha256`, `DECK_PROVENANCE.json`에 실제 모델·코드·원본 Fortran·라이브러리·산출물 신원이 보존돼 있다. 덱 생성 시 코드 SHA256은 변환기 `dc07c062658e5abb96e021bc61e3b2d5aaedc1cc1c58fcad89dce90b9defc22e`, parser `bdadfcf984b0731fa0f38d3692dc54f3e5774317c8320b6b2cc2b0359c5391fc`, 평활화 wrapper `7e1ab034318e7937053c915cea632a580fedb38b11be9b59a5625e2a86eed427`, driver `a6db5fe03a9ba54f586a9f0aaf7544a2865a9a142343c6971e287af0db76f7d4`다.

## 2026-10-19 운전자 상위 superlevel 열적 분배함수 수정과 양방향 율 판독

- 평활화 덱만 반영한 전체 고정상태 재생 `/gpfs/kjhan/cmfgen_runs/lumina_bf_superlevel_smoothed_428222_20261019`에서는 광율 네 이온이 모두 기존 median/P95/max 수치 기준을 통과했다. 남은 실패는 S III 재결합 median `0.0314938 dex`였다. S II RR에도 `0.0277343 dex`의 체계적인 오프셋이 남았다.
- CMFGEN `SUP_TO_FULL_V3b`는 상위 ground superlevel의 실제 미세구조 에너지들로 `GION(T)=Σg_f exp[-(E_f-E_anchor)/kT]`를 계산한다. 반면 Lumina v4 역 Milne/3체 재결합은 PHOT branching에 쓰는 정적인 `Σg`를 Saha 분모로도 사용했다. S II의 목적 S III ground SL은 `g=(1,3,5)`, `E=(0,.03703284,.10328876) eV`, S III의 목적 S IV ground SL은 `g=(2,4)`, `E=(0,.11796229) eV`다. 해당 온도 프로필에서 정적 합/열적 분배함수는 각각 `1.0356–1.0940`, `1.0400–1.1053`으로, 관측한 오프셋의 방향과 크기를 설명했다.
- 실제 코드 `lumina_plasma.c`의 pair와 `lumina_element_wide.c`의 EW BF assembler 모두에서 v4의 Saha 분모를 `g_anchor/f_anchor`로 수정했다. `f_anchor`는 같은 현재 `T_e`에서 이미 계산한 target SL 내부 Boltzmann 비율이므로 위 `GION(T)`와 같다. 전이 도착지는 계속 target SL 전체이며 fine member에 인구를 임의로 나눠 도착시키지 않는다. 정방향 광/충돌 이온화율과 PHOT branching 확률은 같은 원자료를 유지하고, 역방향의 열적 정규화가 바뀐다. terminal scalar는 단일 가상 준위이므로 기존 통계중량을 유지한다. 비교기의 RR 재생도 모든 활성 target SL member의 에너지·g로 같은 분배함수를 계산한다.
- 과거 v3/v4 역율 비 `4/3` fixture는 `g=(2,4)`, 비퇴화 에너지에 대해 임의의 `f_anchor=.25`를 넣어 구현된 규칙을 확인했을 뿐, 물리적인 열적 분배를 검증하지 못했다. 해당 시험의 당시 PASS 기록은 보존하되 이 역율 정규화의 물리 정확성 근거에서 제외한다. 새 C 시험은 실제 Boltzmann 비율을 만들고, v3 anchor 표현과 v4 SL 표현의 동일성, 축퇴 극한 `G=Σg`, 1 eV 분할 시 독립 해석식 `6/(2+4 exp[-1 eV/kT])`에 따른 방사·충돌 역율 변화와 정방향 불변을 검증했다. Si/S owner의 정상 commit과 손상 route rollback도 함께 통과했다.
- 수정 후 같은 27,180개 SL-depth 셀의 양방향 율은 아래 기존 수치 기준을 모두 통과했다. source snapshot은 여전히 fixed-T 비수렴 모델이므로 이 표는 **동결상태 연산 비교**이며 최종 수렴 oracle/이온분포 동질성 PASS가 아니다.

  | 이온 | 광율 median/P95/max (dex) | RR median/P95/max (dex) |
  |---|---|---|
  | Si II | .000819/.009078/.039331 | .000743/.008413/.033152 |
  | Si III | .000923/.008290/.053960 | .000540/.005191/.011810 |
  | S II | .000545/.003331/.028305 | .000842/.003198/.009597 |
  | S III | .000639/.004767/.054507 | .000766/.004482/.015247 |

- 결과 `/gpfs/kjhan/cmfgen_runs/lumina_bf_superlevel_thermal_target_428222_20261019/summary.json`(schema v10)은 photo/RR 수치 gate, PRRR↔SCRTEMP pre-solve 매핑, full-population 사후 재현을 모두 `PASS`로 기록했다. summary SHA256 `326b5a1d974170b15d51bf374ec7113a9fb881a2724d5a5517a008244fecd585`; detail CSV `adcc8cee0a136bb6d8e747c03ad5b86b0eea872c0ae4d3ad81b3e86a1716c2e9`다. GPU 계산은 사용하지 않았다.
- 전체 CPU 빌드와 C 직접시험을 `grammar-debug`에서 수행하고 `/gpfs/kjhan/cmfgen_runs/lumina_upperpartition_build_20261019-gGeegL`에 바이너리와 빌드 로그를 보존했다. `lumina` SHA256 `70a8f9955525906c8538a78619c4e4ad47ef65d869669230bb36578462ba260f`, `selftest_nlte_element_wide_layout` `b846b44cfcff375ef95b6d6316bb45d53b88b05b6b5c421fa6f01c146a51d96c`. 코드 SHA256: `lumina_plasma.c` `e4b1df7dd1bee6d9ac2149e535eaaa6bfefefc77e6709e578df120921170081f`, `lumina_element_wide.c` `b3a1c8896e26743f44d02afa5c46c8bb6ed7ccc26e559992593a16f08714c084`, C 시험 `5256b723b95c1459c884e9de03d80ab9a8ed2dc705c35e050cfbfd43a6d0ff64`, 비교기 `898194f6be5c10c54406da94b97d11615c655b6c848855e4396e2aef02c23396`, Python 율 시험 `84fd9dfea0c7673993f64d626a333199519044e525c8ba22787680540b1ab815`다.
- **남은 목표:** 수렴한 matched-physics released-T CMFGEN oracle 확보, 수정된 실제 생산 owner의 공통장 분포 응답, 전하·RE·수송을 함께 푼 새 Lumina 해와 shell별 Si II/III·S II/III 분율 비교 및 자체 종료감사. 두 원인 수정과 동결상태 수치 통과로 이 목표를 축소하거나 종료하지 않는다.
- 최종 직접시험은 실제 봉인 원자입력·원본 평활화 라이브러리를 함께 설정한 여섯 Python 시험 파일(`test_cmfgen_phot_smoothing`, `test_cmfgen_bf_channels`, `test_cmfgen_bf_superlevel_rates`, `test_a2_05_l1bf_gate`, `test_export_cmfgen_ion_fractions`, `test_ptf_gated_metrics`)에서 **71 passed, skip 0**이었다. 위 C owner 시험과 전체 CPU 빌드도 최종 C 소스에 대해 재수행해 통과했다. 관련 tracked 소스의 `git diff --check`와 제출 스크립트 `bash -n`도 통과했다. 수정사항은 미커밋 상태이며 기존 사용자 변경은 보존했다.

## 2026-10-19 다음 목적 묶음 — 생산 인구응답과 CMFGEN 수렴 재개 방식 확정

- 두 물리 수리와 전 셀 수치 비교가 착지한 뒤 Astra(agent `01a15281-e58f-7e30-9288-ce845087b6ba`)에 다음 실행 결정을 한 번에 의뢰했다. 결론은 **Lumina 고정장 생산 owner 인구응답 진행, 새 CMFGEN iteration은 현재 0회**다. 새로운 정례 감사나 Q-GOAL/Q-LEAN을 추가하지 않았다.
- 목적은 수정된 율이 실제 Si/S 공동 생산 owner에서 어떤 이온분포 응답을 만드는지 확인하고, 같은 묶음 안에서 CMFGEN 재개의 시작 checkpoint·제어를 근거로 하나로 정하는 것이다. 이 산출물은 이후 자기일관 계산을 가능하게 하는 증거이며 최종 분포 동질성 목표를 대체하지 않는다.
- owner 실행은 덱 `428222`, 보존 CPU 바이너리, matched SCRTEMP39와 같은 `J,T,n_e,원소총수`를 사용한다. 90개 깊이에서 II–V와 terminal closure를 포함한 실제 C 생산 owner의 완결된 풀이와 commit/fallback, 전후 이온분율, 원소보존 및 실제 조립계 scaled SE 잔차를 얻는다. CPU 1·RAM 16 GiB·GPU 0·1시간 상한이다. 외부 장·온도·전자밀도를 고정하므로 전하·RE·수송 공동수렴을 주장하지 않으며, 비수렴 CMFGEN population에 가까워졌다는 사실을 새 합격 기준으로 삼지 않는다.
- 구현 진입점은 `nlte_element_wide_run_status[_ce]`와 `nlte_init`이다. 기존 `bench_frozen_oracle.c`는 기록된 Lumina MC `stdout.log`, Jbar·field dump를 요구하고 본래 제한된 shell만 지원하므로 CMFGEN 상태를 그대로 넣을 수 없다. `tests/cmf_frozen_field_probe.c`도 방사 형식해용이며 인구 owner 실행기가 아니다. 기존 생산 owner와 입력 publication을 호출하는 최소 CMFGEN 상태 adapter를 마련하고, Python의 율 재구성을 인구 풀이 대용으로 쓰지 않는다.
- 병렬 CMFGEN 원인분석은 modern30–40 및 StageA41–59의 기존 출력만 사용한다(CPU 1·RAM 8 GiB·30분 상한). 실제 BA/LAMBDA·NG 적용, 증가/감소/반환 correction과 적용 scale, 실제 인구/전자밀도/Si/S 분율/광도 변화, moment 오류 시점, restart의 JEW·행렬 이력을 한 반복 표로 연결한다. 소비자는 다음 단일 restart 선택이며 별도 정례 계측 체계를 만들지 않는다. 저장되지 않은 반복별 Jν를 복원했다고 표시하지 않는다.
- Astra가 확인한 중요한 정정: modern의 `44300→13800→8980→3460%`는 최대 **증가량** 수열이고 40회 `SOLVEBA_V13` 반환 MAXCH는 `1e7%`다. 앞의 `MAXCH≈3460%` 표기는 전체 solver 수렴 지표로 사용하지 않는다. 또한 StageA41회는 population 갱신 전에 moment 오류가 있었으므로 NG 해제나 BA 재진입 하나만을 발산 원인으로 확정하지 않는다. 현재 NG 원인 가설을 근거 없이 변경해 19회를 재제출하거나 T를 해제하지 않는다.
- StageA의 최신 권위 상태는 59회 정상종료 및 사후 상태 일치이지만 active subset p95 `79.2876%`, 전자밀도 마지막 `|Δln|=.0188877`, depth27 Si III 분율 변화 `+26.1361%`로 비수렴이다. 실제 전 59회 Si/S 분율 53,100개 행과 terminal VI 폐합까지 추출했다. 원소총수/terminal DI 상대잔차는 모두 `4.4e-8` 이내였다. 사후 자료는 `/gpfs/kjhan/cmfgen_runs/stageA_427702_recheck_20261019-86Yyb7`이며, `metrics.json` SHA256 `9cf9877ab2ebbdf5c3e6de690c9657eb63020e24ef965e15ab15eee342d95c0b`, 분율 metadata `f2462e9553aeccf5ab8f97c537a38b908cb0d1a5d762609dfd81e4cb7b8f0d78`, 분율 CSV `fc1f9f4581076e5d3ae49796c2b2c3a64f400b2547b6e1d69abd99c8325987f8`이다. 최신 exporter SHA256 `95aa87e067f00feacbc05dc177c46a43449dc7490c2086dd71eb899eea30af3b`.
- 묶음 범위 밖인 원자료/CMFGEN 소스 변경·다중 수치 탐색·수용 기준 완화·원본 출력 덮어쓰기/삭제는 수행하지 않는다. 현재 새 자기일관 계산은 제출되지 않았고, 목표는 계속 미완료다. 다음 실행 준비에서는 위 두 내부 작업을 이어가며 작은 확인마다 Astra를 재호출하지 않는다.

## 2026-10-19 운전자 공통장 생산 풀이 연결과 restart 이력 판독

- `bench_cmfgen_owner.c`와 `scripts/prepare_cmfgen_owner_response.py`를 구현했다. Python은 봉인된 입력의 신원과 PRRR/SCRTEMP phase를 확인하고 입력만 내보낸다. 인구 풀이는 기존 `nlte_init`와 실제 `ew_run_impl`을 호출한다. 90개 깊이, 원래 Jν 197,291개 표본, Si/S II–V 및 terminal VI의 518개 SL/terminal 변수를 입력했다. 실제 매칭 기록은 계속 SCRTEMP39다.
- 증거·빌드·로그의 새 루트는 `/gpfs/kjhan/cmfgen_runs/lumina_owner_response_20261019-oibhpY`다. `input/owner_input.bin` SHA256은 `e39002ae2cc7022aad5bfca87a591da4e82201f97d8ab5ce64b1907aed541fbf`이며 입력 원본은 변경하지 않았다. CMFGEN 깊이점에서 고정장 국소 인구를 푸는 것이며 새로운 수송 격자나 셸 평균 분포 계산으로 표시하지 않는다.
- 입력은 고정 `Jν,T_e,n_e`이고 Si/S 원소총수는 SCRTEMP 원래 SL/terminal 합이다. 초기 fine population은 같은 T에서 생산 코드의 SL 내부 Boltzmann 비율로 분배하되 각 원래 SL 총수와 이온 총수를 보존한다. 이는 CMFGEN SUP_TO_FULL 미세준위 분포를 그대로 복제했다고 주장하는 것이 아니다. 연속광은 기존 보존 재빈닝을 사용하고, 선 Jbar는 같은 native Jν의 주파수 선형 보간과 기존 등록 Gaussian profile의 정확 segment 적분을 사용한다. MC estimator는 존재하지 않으므로 할당된 영값 C2 버퍼를 소비하지 않는다.
- 실제 덱 로더 연결에서 덱428222의 풍부도에는 원자목록 밖 C/O 행과 기존50셸이 남아 있음을 확인했다. 해당 덱은 원자표 수치 비교에는 사용됐지만 이 물질 파일 그대로의 생산 실행은 불가능했다. 원본을 수정하거나 로더 검사를 우회하지 않았다. `input/atomic_view`는 같은 원자 파일들을 원본에 링크하고 첫 matched CMFGEN 점의 실제 6개 원소 풍부도로 물질 초기화 파일만 별도 생성했다. 재정규화하지 않았으며 실제 합 `0.9996042140`의 기존 D6 경고도 숨기지 않는다. 각 풀이점에서 Si/S 원소총수는 다시 실제 입력값으로 설정한다.
- IV 단계의 legacy 목적지 표는 봉인 PHOT/F_TO_S에서 재생성했고, II/III는 새 PHOT CSR 자체를, V는 이미 구현된 terminal scalar를 사용한다. 실제 봉인 collision manifest에서 Si/S 관련 표 7개·4,072개 전이를 별도 import했다. Si V는 원본이 0개 tabulated transition을 선언하므로 원래 CMFGEN 근사식 경로를 유지하며 임의 표를 만들지 않는다. 새 sidecar는 원본 덱에 쓰지 않는다.
- 생산 EW coverage가 실제로 소비하는 PHOT CSR 대신 미사용 legacy MA 목적지 표까지 요구하던 불일치를 수정했다. PHOT 채널이 있는 행은 그 CSR의 실제 목적지·다음 이온·production projection을 확인하고, 채널이 없는 행만 기존 legacy/Kramers 계약을 유지한다. 기존 물리 수용 문턱은 바꾸지 않았다. 직접시험은 v4에서 legacy 표를 제거해도 정상 Si/S 채택이 가능하고 CSR 목적지를 손상하면 채택하지 않는 것을 검증했다.
- 실수 이진 PHOT 목적지 에너지와 `levels.csv`의 `%.10f` 표기가 64-ULP 비교에서 충돌하던 로더 결함도 수정했다. 허용 범위는 해당 파일 형식의 반 decimal quantum(`5e-11 eV`)과 부동소수점 roundoff뿐이다. `2.345e-11 eV`의 정상 직렬화 차이는 허용하고 `1e-7 eV`의 손상은 계속 거부하는 C 회귀를 통과했다. 이는 물리율/이온분율 수용기준 완화가 아니다.
- `NLTEEWResponse`는 같은 생산 풀이에서 후보/채택 여부, 실제 scaled SE·보존 잔차와 후보 이온총수를 반환한다. 대형 행렬 덤프를 추가하지 않았으며 거부된 후보를 채택된 인구로 표시하지 않는다. 현재 직접시험은 Python 43건, 실제 owner 정상·손상 transaction, 등록 profile의 상수/선형 적분, 원자 로더의 정상/손상 입력 회귀가 통과했다.
- 별도 기존 출력 판독 `scripts/analyze_cmfgen_restart_history.py`는 modern30–40과 StageA41–59의 30개 반복을 하나의 `restart_history/iteration_history.csv`로 연결했다(SHA256 `bb150bcded9505fa98409c9b6579b442e776668cc89aa19fb17b080d32cd8fc3`). BA/LAMBDA, raw 증가·감소/반환 MAXCH, 적용 scale, 실제 인구/전자밀도/Si/S 변화, moment 오류와 광도를 같은 행에 기록했다. 저장되지 않은 Jν를 복원하지 않았다.
- modern30–40부터 이미 `L(d1)/L(dND)=1352–3286`이었고 StageA는 `1155–30238`이었다. 고정 T에서는 외부 온도 유지에 따른 에너지 불균형을 허용할 수 있으므로 이 값만으로 수송 코드 오류를 확정하지 않지만, RE 수렴 기준해로 사용할 수 없다는 증거다. 큰 광도 불균형은 restart 이전에도 존재했다. 따라서 JEW/행렬 이력 초기화만을 원인으로 확정하지 않는다.
- 두 실행 POINT1의 마지막 NG 적용 기록은 모두 `-1000`이다. 원본 `solve_for_pops.f`는 MAXCH가 VAL_DO_NG보다 작아진 뒤에야 NG를 예약하고, 적용 전까지 LAST_NG를 그대로 유지한다. 따라서 StageA의 NG 해제만으로 기존 대비 악화가 생겼다는 설명은 현재 증거로 지지되지 않는다. StageA41·42의 moment 오류는 해당 iteration population 갱신보다 앞서 기록됐다. 임의로 NG만 다시 켜거나 같은19회를 반복 제출하지 않았다.
- 이 연결의 초기 실패들은 모두 원본 보존 상태에서 좁혀 수정했다. 외부 T의 세대는 각 입력점의 정체성으로만 부여하며 radiative-equilibrium publication 또는 전하·RE·수송 공동수렴을 주장하지 않는다. 새 소스·입력으로 실제90깊이 production 응답 실행을 계속하며, 아직 그 분포가 산출·수용됐다고 기록하지 않는다. 전체 목표는 계속 미완료다.

### 실제 90깊이 응답 완료 — a6 정본

- 최종 `owner6.log`는 `CMF_OWNER_RESPONSE_COMPLETE depths=90 GPUh=0 final_goal=NOT_JUDGED`, 프로세스 exit0이다. `owner_response_a6.csv` 900행과 `owner_summary.json`을 보존했다. 각각 SHA256은 `3e30835844114d84f6b2badeaaec4fa854db645649affd45c01783bd454e8ad7`, `99f114629cfcf530ef2221ba1d7cb43ac936447b65a26867b34e6e9a498a49b7`다. 실제 실행 바이너리 `bench_cmfgen_owner_a6` SHA256은 `aa1701bb497fc7d9207a752fa9c216ca21b4f08151a849008da452ce70a17a82`이며 별도 이름으로 보존한다.
- 첫 완결 응답 a5의 `BF_OUT_OF_GRID=11250`을 조사하니 Si II gridB107개·gridA18개, 총125개 **명시적인 전구간 영 단면적** route×90깊이였다. J 결측이 아니었다. `bf_rate_gamma_from_view`가 유효한 edge 위 영역까지 표가 있는 영 곡선만 `EXACT_ZERO`로 처리하도록 고쳤다. 표 끝이 물리 edge보다 아래인 기존 SH-GRID 음성대조는 계속 `OUT_OF_GRID`다. 상수/수소형 단면, 실제 결측·stale, 영 복사장과 작은 결측 질량의 분리, SH-GRID 폐합을 포함한 기존 C 율 시험과 새 영 단면·잘린 표 음성대조가 통과했다.
- a6은 BF `rate_terms=143910`, `blocked_stale=0`, `blocked_unsampled=0`, `blocked_out_of_grid=0`이다. 영 율 11250개가 명시적으로 분류됐다. BB `rate_terms=2777220`이며 stale·unsampled·OOG·cache miss·profile·hash·disabled block은 모두0이다. 기존 100–20000Å BB 계약 밖436770개 소비는 명시적으로 제외됐으며 이를 CMFGEN 전체 BB 과정 동등성으로 확대하지 않는다. a5/a6의 전후·후보·채택 분율, SE·보존 잔차, κ는900행 모두 동일했다. 이 변경은 물리율을 조절한 것이 아니라 잘못된 상태 분류를 고친 것이다.
- 총180개 풀이 후보는 모두 기존 `p_elem_valid`(유한/비음수·보존·scaled SE)를 만족했지만 **실제 채택은 Si0/90, S69/90**이다. Si SE 최대 `5.8253e-15`, 보존 최대 `2.0717e-15`지만 raw rank가 모든 깊이에서269로 요구270보다 낮고 κ=`2.6317e12–7.0311e18`로 기존1e12 한도를 초과했다. 무방향 graph component는1이다. 실제 추가 폐쇄 클래스인지 약한 연결의 수치 rank 판정인지 아직 구분하지 않았다.
- S는 raw rank246/247(요구246), graph component1, SE 최대 `1.4799e-15`, 보존 최대 `1.2123e-15`, κ=`2.8296e6–3.9329e10`이다. numerical 판정21점 실패는 κ 한도 때문이 아니며, 해당 점의 permutation·pivot 등 실제 실패항은 후속 확인이 필요하다. 후보 유효성과 채택·해의 정확성을 구별한다. 문턱·물리 연결·약한 율을 바꿔 통과시키지 않았다.
- 아래는 같은19.48일 **비수렴 CMFGEN pre-solve 분포 대 고정장 생산 후보**의 III 분율이다. 공개2일 셸 비교와 다른 실험이다. Si 후보는 모두 거부됐으므로 실제 Si 인구는 원래값을 유지했다. S는 표시한27/34/40/53/61깊이에서 실제 채택됐다.

  | 깊이(1-based) | Si III 원래 / 후보 | S III 원래 / 후보 | Si / S 원소밀도(cm^-3) |
  |---|---|---|---|
  | 27 | .514955 / .977855 | .331118 / .917970 | 1.124e6 / 6.261e5 |
  | 34 | .956806 / .998648 | .469110 / .977811 | 4.743e6 / 2.642e6 |
  | 40 | .994214 / .999734 | .891145 / .995783 | 1.596e7 / 8.891e6 |
  | 53 | .999630 / .997107 | .963787 / .834684 | 1.780e8 / 9.917e7 |
  | 61 | .049833 / .954641 | .323246 / .929649 | 3.250e5 / 1.810e5 |

- S III 최대 변화 깊이62의 원소밀도는 `~1e-20 cm^-3`다. 이를 큰 스펙트럼 효과로 표시하지 않으며, trace라는 이유로 기존 수용기준에서 면제하지도 않는다. Si 깊이61은 위 밀도처럼 이 trace 경우가 아니다. 참조 자체가 비수렴이고 SL 내부 분배도 다르므로 후보와 참조의 큰 차이를 곧바로 CMFGEN 평형해 대비 물리 오류로 단정하지 않는다.

### Astra 통합 평가와 다음 목적 묶음

- 이번 결과와 반복 이력을 한 번에 Astra(agent `01a152a5-1282-7ae3-92a2-9f5f70df0204`)에 의뢰해 평가·다음 계획을 받았다. 파일 편집·실행 제출은 하지 않은 read-only 평가이며 완료 뒤 agent를 닫았다. 판단은 **생산 고정장 응답/거부는 확인, Si/S 동질성 미입증**이다. 운전자 직접시험이나 결과 생성과 작성자를 혼동하지 않는다.
- 다음 묶음 목표는 **기존 기준을 유지한 Si/S 생산해 채택 복구와 단일 CMFGEN 안정화 경로의 유용성 확인**이다. Si 깊이1·61의 추가 저랭크 방향을 관련 SL/율과 연결하고, S 실패1·20과 인접 통과19의 실제 permutation/pivot 등 실패항을 확인한다. 필요한 소형 행렬만 보존·독립 대조하고 큰 전 깊이 덤프나 새 정례 게이트를 만들지 않는다. 물리적 특이성이라면 미해결로 남기며 약한 율 삭제·연결 추가·rank/κ 기준 완화로 수용하지 않는다.
- CMFGEN의 선택안은 검증된 StageA59 사후 상태의 별도 복제본에서 fixed-T와 기존 NG 설정을 유지하고, 기존 `RD_LAMBDA` 경로로 자동 BA 재진입만 억제하는 **단일 continuation**이다. 실제 입력 키와 첫 restart 반복부터 LAMBDA 적용 여부를 먼저 소스·유효 입력으로 확인해야 하며, 첫 회를 BA로 수행해 놓고 LAMBDA 유지 시험으로 표시하지 않는다. CMFGEN 원본 소스/물질/광도/온도 입력을 근거 없이 수리하지 않는다. 새 CMFGEN 계산은 이 기록 시 아직 미제출이다.
- Astra가 제안한 새 자원 상한은 owner CPU1·16GiB·2시간/GPU0, CMFGEN CPU16·200GiB·최대4회 또는4시간/GPU0 중 먼저 도달하는 한도다. 이는 **다음 실행 제안**이며 이번에 사용한 자원이 아니다. 실행 준비에서 기존 사용자 자원 권한·입력 봉인·실제 적용 제어를 다시 확인한다. 현재 묶음은 CPU1·16GiB·1시간의 기존 물리 실행 한도 안에서 수행했다.
- 단일 CMFGEN 안정화 시도의 지속 후보 기준은4회 모두 moment/grey/NaN/Inf 오류0, 마지막 같은 subset correction p95≤39.6438%(Stage59의 절반), 마지막 `|Δln ne|`와 Si III/S III 최대 절대 fraction step이 각각 Stage59보다 커지지 않는 것이다. 이는 안정화 시도의 유용성 판단이며 최종 물리 수용기준이 아니다. moment 또는 비유한 오류 재발 시 조기 중단하고 자동 연장·다중 설정 탐색·온도 해제를 하지 않는다.
- 최종 수렴 CMFGEN 기준해, Lumina 전하·RE·수송 공동수렴, 같은 물리의 전 셸 분포 비교와 자체 종료감사는 계속 미완료다. 이번 생산 후보·국소 잔차·Astra 평가를 최종 목표 달성으로 승격하지 않는다.
- 최종 소스의 전체 CPU 빌드와 C owner 정상·손상 회귀가 다시 통과했다. `lumina_final` SHA256 `f4bf71d1ac04f8db3694f196ecc239d45a469688f3de5e57372734f3f8cf69f0`, `selftest_ew_final` `d32f520d72e59cba401f95778be27957870a63e1850c54cf48eabf126cfcf5e1`, `selftest_a2_05_bf_rate` `4b911a09eccb0c0ad77d2bdbf598ecd1424ae9debf8e3bd98fc1b1c55fab2c6d`다. 봉인 원자료·원본 Fortran smoothing library를 지정한7개 Python 시험 파일은 **76 passed, skip0**이었다. `git diff --check`도 통과했다. 신규 변경은 미커밋이며 원래 사용자 변경을 보존했다.

재현은 `grammar-debug`에서 위 R 디렉터리를 작업 경로로 하여 같은 바이너리·입력을 사용한다. 출력 파일은 존재하면 거부하며 원래 결과를 덮어쓰지 않는다. 아래 명령의 새 출력명은 실행 전에 미존재를 확인한다.

```bash
timeout 3600 env -i PATH=/usr/bin:/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LUMINA_TOPION_LEVELS_FILE=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/topion_levels.csv \
  LUMINA_IONIZATION_REFERENCE_FILE=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/ionization_reference.csv \
  ./bench_cmfgen_owner_a6 input/atomic_view input/owner_input.bin owner_response_reproduced.csv
```

실제 생성 명령은 같은 인수에서 절대 경로와 `owner_response_a6.csv`를 사용했고, 외부 메모리 상한은 `ulimit -v 16777216`이었다. 입력 준비는 `scripts/prepare_cmfgen_owner_response.py`, collision 준비는 `scripts/build_cmfgen_coldata_all.py --source-manifest .../lumina_model_spec_428222/atomic_vintage_manifest.csv --only 14:1,14:2,14:3,14:4,16:1,16:2,16:3,16:4 --write`, 요약은 `scripts/summarize_cmfgen_owner_response.py --root R`다. 각 도구의 출력 root는 새 디렉터리/파일이어야 한다. 기존 R과 원본 덱에서 writer를 다시 실행하지 않는다. 정상 완료·전90깊이 coverage·a5/a6 공통 수치 일치는 운전자가 실제 실행/원자료에서 재현했으며, 최종 분포 동질성은 재현·판정하지 못했다.

## 후속 묶음 실행 준비 — 원래 수용기준 유지

- Astra가 정한 다음 묶음의 owner 확인은 Si1·61, S1·19·20의5개 풀이로 제한했다. `bench_cmfgen_owner.c`에 대상점 선택과 기존 dump root 전달만 추가했으며 full90 기본 실행의 물리식은 변경하지 않았다. 선택하지 않은 중간 깊이를 건너뛸 때도 복사장 publication generation은 연속되게 유지한다. 첫 선택 실행의 중단은 이 adapter 세대 관리 결함이었으며 기존 a6 결과는 영향받지 않는다. 실패 출력도 보존했다.
- 완결된 선택 실행은 `/gpfs/kjhan/cmfgen_runs/lumina_owner_points_20261019-WR2jSs`이며 `CMF_OWNER_SELECTED_RESPONSE_COMPLETE input_depths=90 solves=5`와 exit0이다. 5개 소형 행렬/기존 진단만 보존하고 다른85깊이 행렬을 생성하지 않았다. S1의 실패는 permutation 분율차 `4.849609686945e-9>1e-10`, S19는 `9.388210663184e-11`로 통과, S20은 `1.497212320452e-10`으로 실패였다. 세 점의 pivot growth는 모두1이며 κ는 기존 한도 이내다. 즉 S의 실제 거부 원인은 순서 민감성이고 조건수나 pivot growth가 아니다.
- 소스 확인에서 `DO_LAM_IT`는 `cmfgen_sub.f:558`의 `RD_LAMBDA`로 읽히고, 최초 물리 반복 전608행에서 `LAMBDA_ITERATION=RD_LAMBDA`가 설정된다. `solve_for_pops.f:285`는 RD_LAMBDA가 참이면 자동 BA/LAMBDA 교대부를 건너뛴다. `FLUX_CAL_ONLY=F`, `USE_FIXED_J=F`, `DO_LAM_AUTO=F`, `DO_T_AUTO=F`를 유지하므로 의도한 첫60회부터 고정T LAMBDA 경로를 사용할 수 있다. 과거 StageA 구성기의 “19회 모두 full BA”라는 잘못된 주석은 역사 제어값을 바꾸지 않고 정정했다.
- `run_cmfgen_lambda_probe.py`는 StageA59 포인터·SCRTEMP/RVTJ 사후 상태 및 부모가 사용한 동일 바이너리를 확인하고 별도 job-ID root에 입력과 restart를 복제한다. `VADAT`는 바이트 동일하게 유지하며 `IN_ITS`에서 `NUM_ITS:19→4`, `DO_LAM_IT:F→T`만 바꾼다. 원자 링크/원본·CMFGEN 소스는 수정하지 않는다. 새 계산은 비수렴 seed의 안정화 시험이지 oracle 인증이 아니다.
- 실행 자원은 CPU16·200GiB·GPU0·최대4시간/4회이며 노드를 고정하지 않는다. staging과 후처리 여유를 위해 자식 CMFGEN의 내부 한도는13,800초다. Slurm 내부 실행 관리자가 분당 최대1회 현재 자식과 새 로그 바이트만 확인하고 moment/grey/nonfinite 오류 또는 예상치 못한 BA 재진입 때 해당 자식만 중단한다. 별도 상주/백그라운드 감시·Slurm 반복조회·자동 재제출은 없다. 입력 복제와 예상 출력의 저장공간 한도는8GiB이며, 실행 전 GPFS 여유255TiB와 사용자 hard quota 미설정을 확인했다.
- 신규 제어·중단 관리자 직접시험11건과 제출 스크립트 shell syntax 검사가 통과했다. 실행할 runner와 두 의존 모듈은 제출 전에 별도 디렉터리에 봉인해 큐 대기 중 저장소 변경이 실행 코드를 바꾸지 않게 한다. 이 실행은 사용자의 기존 자율 진행·관련 계산 사전승인 범위에서 CPU-only로 진행하며 GPU 예산은 소비하지 않는다. 앞 절의 지속 후보/조기중단 기준을 그대로 적용하고 원래 최종 물리 기준을 완화하지 않는다.

## 2026-10-19 후속 실측 — 수치 채택과 물리 원인 분리

- 보존한5개 raw generator는 양의 율을 임의 절단하지 않으면 모두 strongly connected1·closed class1이었다. Si의 의도한 정확 rank는270이며 기존269 판정은 수치적 스케일 문제다. 그렇다고 기존 rank/κ 수용 기준을 바꾸지는 않았다. 실제 생산 코드의 positive-generator GTH로 Si/S 후보와 permutation 시험을 계산하고 dense LU는 원래 조건 진단에 유지했다. 선택점 GTH permutation 분율차0, componentwise 잔차 `~1e-16`이었다.
- 전체90점 GTH 실행은 `/gpfs/kjhan/cmfgen_runs/lumina_owner_gth_20261019-RBF82k`에서 exit0·900행을 남겼다. **S 채택69→90/90, Si0/90 유지**다. 전후 인구·율·입력·rank/κ 기준은 변경하지 않았고, 후보는 이전 dense 대비 최대 Si `8.60e-8`, S `1.24e-8`만 달랐다. 이는 수치 순서 민감성의 수리이지 물리 불일치의 해소가 아니다. CSV SHA256 `a0474deb866b60d946b73f5eb12fe28b534c3e2e0bbfb764b22e2377ff03e6f4`, `gth_summary.json` `86eac1d12d387f8201b0de4dc2309b1583e63dfccef01004f047f9b023eb2a2a`, 바이너리 `428eab52cee9afe43955936898256c1a673c449e21bc21497a74659c8e6ae247`이다. 이 소스의 population C selftest 및 전체 CPU 링크도 성공했다(`lumina_final` SHA256 `1d8ae82bbf8879ccea41a8050ab0c8351bdf809beea16663367822ac6aa2cc48`).
- 단일 CMFGEN 안정화 시도는 job428468·syn04에서 실행됐다. 첫60회에 이미 `MOM_J_REL_V9` excessive iteration 오류가2번 발생해 guard가660.127초에 자식을 중단했다. `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_lambda4_428468/seq_logs/lambda-probe-runtime.json`의 return137은 이 중단 결과이며 OOM 증거가 아니다. `mom_jrel_v9.f:995`는 내부100회 뒤 `MAX_ER`가 `1e-8`보다 작은 조건을 못 채우면 오류를 쓰고 ACCURATE를 참으로 강제한다. 해당 출력 오차는1과102였으므로 정상 수렴으로 인정하지 않는다. 원본·실패 출력은 보존했고 자동 재시도·온도 해제·추가 CMFGEN 제출은 하지 않았다.

### 직접 확인·수리한 Einstein B 규약 오류

- 실제 C 소비자는 `R_abs=B_lu*Jbar`, `R_stim=B_ul*Jbar`, `R_spont=A_ul`이며 J는 각도 평균 강도다. 그런데 최근 사용한 덱428222의 `line_list.csv` **2,588,798선 전부**가 `B_ul=A_ul*c²/(8πhν³)`였다. 올바른 이 소비자 규약은 `B_ul=A_ul*c²/(2hν³)`다. 흡수·유도방출이 `4π`배 과소였으며, B끼리의 `g_l B_lu=g_u B_ul` 검사만으로는 검출할 수 없다.
- `expand_atomic_data_cmfgen.py`의 선 CSV와 macro seed 생산식을 함께 고쳤다. `finalize_cmfgen_ref_npy.py` 자체는 이미 올바른 식이었지만, 이번 R1 driver는 그것을 호출하지 않고 `finalized:true`를 기록했다. 이 driver의 새 기록은 `finalized:false` 및 `line_einstein_convention:J_nu`다. 과거 덱의 잘못된 기록을 소급 덮어쓰지 않는다. 전체 GPU 덱의 NPY·macro·K-SHAPE finalization 완료와 선 원자 core의 수리는 별개다.
- 원본은 불변이다. `/gpfs/kjhan/cmfgen_runs/lumina_excitation_20261019-7ylCcB/atomic_view_J`에서 B두 열만 A·ν·g로 재계산했다. 나머지 CSV 열은 문자열 그대로 유지했고 공통 해시 `78385817156c755a8b2cc2899c3ca8cc876008ccf5546ddda1b3930a3eedb01e`로 확인했다. 보정계수 범위 `12.5663450–12.5663957`은 원래7자리 유효숫자 반올림 범위다. 원래 CSV SHA256 `c302f54282381fabfdd5517d28762af6c499bb5bf8e4a07657e9648a18fa9db9`, 새 CSV `90f75474b31811bb49443049a4f049f9a4ec9a710e4e70f86ee76b8984bb74a9`다. 이 뷰는 C 고정장 시험용이며 완성된 GPU 생산 덱으로 표시하지 않는다.
- 같은 C 바이너리·같은 봉인 `owner_input.bin`으로 Si61·S27·S53의 전후 행렬을 생성했다. 실제 변화는 `rad_bb`에만 있으며 `coll_bb`, `rad_bf`, `coll_bf`, 기존 외부 fit의 `autoion_DR`는 전부 bit 동일했다. 같은 C BF계수에 인구만 가중한 Si III61의 meanΓ는 원래 CMFGEN 인구 `0.3406600`, 잘못된 B의 후보 `0.002257749`, B 수리 후보 `0.3355940 s^-1`였다. 작은 들뜬 인구의 손실이 광이온화를 바꾸는 경로를 정량 확인한 것이다.
- B만 고친 전체90점도 exit0·900행이며 S90/90 채택, Si0/90이다. Si raw rank269·κ `9.26e11–1.37e18`, S raw rank246·κ `1.04e7–1.43e11`로 판정 기준을 유지했다. `owner_response_90.csv` SHA256 `d36ce8e8b712df86ac63dec2b0cfa8deb945a2bedf54b616d03841ad14349c4f`, `owner90_summary.json` `336ecd1af9e7dfd0513de7253cc808b5f9b993f959f123284857f1b067e45263`이다. 미량 원소 구간도 포함한 동일 가중90점에서 III 분율 절대차 p95는 Si `.70460→.19492`, S `.59789→.25942`였다. 이는 후보 차이의 감소이며 물리 합격 문턱이나 스펙트럼 유사도 점수가 아니다.

### 외부 DR fit의 선택물리 불일치 — 생산 수리는 다음 내부 작업

- 생산 조립 `lumina_plasma.c:18826–18861`은 외부 Badnell 등 DR fit을 upper ground→lower ground에 추가한다. 반면 봉인 MODEL의 `DIE_Sk2/SkIII/SkIV/SkV`와 `DIE_S2/SIII/SIV/SV`는 모두 `F,F`다. `DIE_AS_LINE=F`만 보고 추정한 것이 아니다. native `cmfgen_sub.f:764–779`는 이 각 이온의 두 flag로 별도 DIE 파일의 선 또는 PHOT 추가를 선택한다. C에 들어간 외부 fit을 동일한 native 선택이라고 취급할 근거는 없다.
- 실제 같은 참조 인구의 Si III61에서 C rad-BF 역유량은5455.7278인데 외부 DR fit은9182.7353 cm^-3 s^-1를 추가한다. S III27은 각각30.2606과26.0449다. 따라서 B를 고쳐도 별도 재결합 추가가 큰 차이를 남길 수 있다. 이것은 PHOT가 모든 실제 DR을 포괄한다거나 DR 자체가 불필요하다는 주장이 아니다.
- 생산 코드는 아직 변경하지 않았다. 보존된 B 수리 raw generator에서 현재 외부 fit만 들어 있는 `autoion_DR` 평면을 빼고, 기존 C GTH로 독립 진단했다. 다른 원인 자동이온화까지 생산에서 삭제하라는 의미가 아니다. native의 `STEQ_AUTO_V2`도 별도로 존재하므로 독립 과정의 자료·조립과 구별해야 한다.

  | 깊이·이온 | 비수렴 CMFGEN 원래 인구 | B 오류 C 후보 | B 수리 실제 C 후보 | 외부 fit 제외 오프라인 진단 |
  |---|---:|---:|---:|---:|
  | Si III61 | .04983306 | .95464097 | .12378080 | .04936077 |
  | S III27 | .33111757 | .91796997 | .46819327 | .32116926 |
  | S III53 | .96378729 | .83468363 | .91992569 | .96396486 |

- 마지막 열은 **수리된 생산 실행이나 채택 결과가 아니다**. 진단 잔차 `~8e-17`은 해당 행렬 풀이의 일관성만 증명한다. 진단 `external_DR_diagnostic.json` SHA256 `aeabb3c2fcaa4b7f7ef6f50de7320c45b30c074e36d5f3413da50c69919c3ce4`; B 전후 과정 동일성·여기 예산 `einstein_response_summary.json` `519d6d2c30bf84ba3c048df75b0aeb7bfe241f8926486e2a847e77323b53b41f`다.

### Astra 평가·다음 내부 구현 계약과 재현

- 의미 있는 새 결정 두 번만 Astra에 평가/계획을 의뢰했다(`01a152c8-5f68-7a92-9b6d-fcfbba4065a2`, `01a152d7-4b1d-7a80-93e4-179384bd6fa1`). 두 agent는 read-only로 평가했고 종료했다. 새 정례 감사·Q-GOAL/Q-LEAN·다른 모델 호출은 없다.
- 다음 구현은 별도 정책 CSV를 추가하지 않고 **기존 BF 명세의 버전 있는 정책**에 MODEL 경로·해시, 각 lower 이온 경계의 정확한 `DIE_*` 두 flag, PHOT 자료 정체성, PHOT/Milne 및 외부 DR fit 허용 여부를 함께 기록한다. 생성기·생산 로더·조립이 실제로 소비해야 한다. flag가 켜졌는데 해당 독립 DIE 선택을 지원하지 못하면 생성/로딩을 거부하고, 새 정책 누락을 구형 동작으로 조용히 처리하지 않는다. 정책 없는 진짜 구형 덱과 비대상 원소의 기존 동작은 보존한다.
- 제외 범위는 이 정책이 금지한 **외부 fit의 실제 조립 기여만**이다. v4 단면적이 있다는 이유만으로 전역 DR을 끄거나 `autoion_DR` 전체를 무조건 비우지 않는다. PHOT/Milne·전자충돌·BB·다른 native 자동이온화는 이 변경에 섞지 않는다. 정확한 source→lower/upper 경계 대응을 직접 확인하고 세 점에서 오프라인 진단과 생산 구현 등가성을 재현한 뒤 같은90점으로 확장한다. Si rank/κ 문턱은 별개로 유지한다.
- 현재 수정의 직접 회귀는 grammar-debug CPU1·16GiB에서 **90 passed, skip0**다(`regression-final.log`). 실제 CSV 생산기의 Planck detailed balance, 일부러 `4π` 틀린 계수의 실패, macro seed 규약, 원본/비B 열 보존, 덮어쓰기·규약 손상·깊이 누락 거부를 포함한다. 첫 회귀의2개 실패는 필수 `SI_ROUTE_TEST_TMP` 미설정이었고 봉인 PHOT/원본 smoothing library와 임시경로를 지정한 최종 실행에서 해소했다. 첫 선택 덤프의 미생성 디렉터리 오류도 보존하고 별도 baseline/corrected 디렉터리를 준비해 정상 재현했다. 기준을 완화한 것이 아니다.
- 원본 덱·인구·기존 결과는 보존했고 GPU 사용0, 신규 CMFGEN 제출0이다. 수렴한 CMFGEN anchor, Lumina 전하·RE·수송 공동수렴, Si 채택, 최종90깊이 동질성과 종료감사는 계속 미완료다. 비수렴 참조와 세 점 진단의 접근도를 새 합격 기준으로 만들지 않는다. 후속 native collision 미등재 pair·log–log 보간 및 선 profile 차이도 아직 수리하지 않았다.

이 절의 원자료 루트를 `E=/gpfs/kjhan/cmfgen_runs/lumina_excitation_20261019-7ylCcB`, 기존 봉인 입력 루트를 `R=/gpfs/kjhan/cmfgen_runs/lumina_owner_response_20261019-oibhpY`, 실행 바이너리를 `G=/gpfs/kjhan/cmfgen_runs/lumina_owner_gth_20261019-RBF82k/bench_cmfgen_owner`로 둔다. 계산은 grammar-debug에서 CPU1·16GiB·각 timeout3600으로 실행한다. 기존 출력은 덮어쓰지 않으며 아래 새 출력명은 실제 재현 전에 미존재를 확인한다.

```bash
timeout 3600 env -i PATH=/usr/bin:/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LUMINA_TOPION_LEVELS_FILE=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/topion_levels.csv \
  LUMINA_IONIZATION_REFERENCE_FILE=/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/atomic/ionization_reference.csv \
  "$G" "$E/atomic_view_J" "$R/input/owner_input.bin" "$E/owner_response_reproduced.csv"
```

생성기는 `python3 scripts/repair_cmfgen_einstein_view.py --deck "$R/input/atomic_view" --out NEW_VIEW`이며 반드시 새 뷰를 사용한다. 세 점 실행에는 새로 만든 덤프 디렉터리를 준비하고 C 인수 뒤에 `14:61,16:27,16:53 NEW_DUMP_DIR`를 붙인다. 요약은 `scripts/summarize_cmfgen_einstein_response.py --root E --input R/input/owner_input.bin --library /gpfs/kjhan/cmfgen_runs/lumina_owner_points_20261019-WR2jSs/population_generator.so --output NEW_JSON`이다. 그 입력 root 안의 `baseline/selected_response.csv`·`corrected/selected_response.csv`와 각 matrices가 정확한 전후 증거다. 운전자가 이 전후 실행과 평면 차이·입력 동일성을 실제 원자료로 재현했으며, 최종 목표 달성을 재현했다고 표시하지 않는다.

## 2026-10-19 재결합 선택물리의 생산 반영·90점 재현

이 절은 위 「생산 수리는 다음 내부 작업」을 수행한 최신 기록이다. 전체 동질성 목표나 사전 수용 기준은 그대로이며, 국소 원인 분리나 수리 완료를 묶음 최종 종료로 처리하지 않는다.

- BF v5를 추가했다. 기존 BF 이진 배열 뒤에 버전 있는 선택물리 블록을 넣고, 같은 내용을 기존 `cmfgen_bf_channels_manifest.json`에 기록한다. 별도 CSV·전역 DR OFF 스위치는 추가하지 않았다. 실제 MODEL 경로/해시, MODEL_SPEC·원자 링크에서 확인한 lower/upper 이온 경계, 해당 `DIE_*`의 두 flag, PHOT 원자료 경로·해시, 외부 fit 허용 여부를 연결한다. 채널 CSR이 없는 IV도 정책에서 빠뜨리지 않는다.
- 생성기와 이관 도구 `cmfgen_recombination_policy.py`는 독립 DIE가 활성인데 지원되지 않으면 거부한다. C 로더는 v5의 정책 누락·중복·잘못된 경계·미지원 flag·원자료 정체성 누락을 거부한다. `LUMINA_CMFGEN_BF_CHANNELS=0`으로 새 정책을 버리고 구형 외부 fit을 복구하는 우회도 차단했다. 실제 구형 v1–4의 읽기와 명시 OFF, 비대상 원소는 기존 동작을 유지한다.
- 생산 pair 조립과 선택적인 frozen-in DR 소비자는 공통 `nlte_external_dr_fit_enabled`를 사용한다. 정책에 해당하는 외부 fit의 실제 추가와 경험적 floor만 제외한다. PHOT/Milne·BB·충돌·다른 원인의 자동이온화는 삭제하지 않는다. EW 원자 신원에도 새 정책을 포함했다. 새 덱 생성용 Slurm 스크립트는 v5와 `EXPECT_RECOMB_POLICY` 소스 해시를 요구한다. 이번에 그 큰 덱 작업이나 새 Slurm 계산을 제출한 것은 아니다.
- 결과 루트 `P=/gpfs/kjhan/cmfgen_runs/lumina_dr_policy_20261019-wYEzAc`의 `atomic_view`는 원래 B 수리 뷰를 보존한다. BF의 v4 수치 payload SHA256 `b230b582c385ad43ea357583999027703502ed0c9c75abdcf7b9707bb6616a5d`를 그대로 유지하고 버전·정책만 추가했다. 새 BF 바이너리 SHA256 `625d60c2873eeec47e313d0242edf30139ec8755ce0d1c29fa0dee083a35eae0`, 명세 `fa3e61254b670f5f8caf6265ee442d1765b8173a2800c5cc3ff5530e029bfb17`이다. 이 뷰 역시 아직 GPU 생산 배열 finalization 완료 덱으로 승격하지 않는다.
- 같은 새 바이너리의 `baseline`(B 수리 v4)·`policy`(v5)에서 Si61·S27·S53을 각각 실행했다. 구형 결과는 앞선 B 수리 결과와 분율 문자열까지 동일했다. 실제 C의 `rad_bb/coll_bb/rad_bf/coll_bf` 평면은 bit 동일하고 외부 fit 기여만0이다. v5 생산 후보와 앞선 독립 진단의 최대 분율차는 각각 `2.22e-16`, `2.78e-16`, `1.11e-16`이었다. `policy_response_summary.json` SHA256 `e6ab5e9b09476508906fba94e6ed46d62087efeeecf9abb9714047f097913a3a`다. 이는 생산 구현 등가성이지 외부 모델 독립 감사나 최종 수렴 증거가 아니다.
- 전체90점·900행은 exit0이다. 최종 바이너리로 `owner_response_final.csv`를 다시 생성해 첫 `owner_response_90.csv`와 `cmp` exit0을 확인했다. 두 CSV SHA256은 `71031de7beb8a282e1c344de8d6712ea9d29b1ffc9ab60f4268e670dd8714425`다. BF/BB의 stale·unsampled·OOG·profile block은0이며 기존 domain 제외 범위는 유지했다. S90/90 채택, Si0/90이다. Si raw rank269·κ `9.26e11–6.14e17`, S raw rank246·κ `1.68e7–4.16e11`; 기준을 완화하지 않았다.

  | III 분율 절대차 | B만 수리한 p95 / 최대 | 선택물리까지 수리한 p95 / 최대 | 새 최대 위치 |
  |---|---:|---:|---|
  | Si III | .194915 / .270578 | .054711 / .162795 | 깊이28, nSi=1.382e6 cm^-3 |
  | S III | .259419 / .269082 | .123014 / .139320 | 깊이35, nS=3.240e6 cm^-3 |

  통계는 미량 구간까지 포함한 동일 가중90점이며 활성 shell의 최종 허용치를 대신하지 않는다. II의 비율 차이도 남아 있다: Si II `|log10 비율|` median/p95/max `.03957/.19412/.36244 dex`, S II `.02483/.77616/.88951 dex`. `owner90_summary.json` SHA256 `4f4dd76fe13b14720afd750e56c5da04d3812d5d9d4185bdbb63da0ade480245`다. 일부 세 점이 가깝다는 이유로 전체 동질성 통과를 주장하지 않는다.

### 최대 잔여 차이의 참조 자체 예산 재확인

- 최종 코드에서 Si28·S35 두 점만 추가 추출했다(`P/residual`). 같은 참조 인구에서 실제 C와 native PRRR의 photo/RR 총량은 아래처럼 근접하지만 native 자체의 역/정 부분 BF 비는1이 아니다.

  | 위치 | native photo / RR (cm^-3 s^-1) | C photo/native, RR/native | native RR/photo |
  |---|---:|---:|---:|
  | Si III28 | 13.64877 / 31.25345 | .99038 / .99803 | 2.28984 |
  | S III35 | 342.90467 / 530.43091 | .99405 / .99644 | 1.54688 |

- 부분 BF의 두 단계 추정 `III/(III+IV)`는 Si `.77100`, S `.61578`이며 실제 C 후보는 `.75800`, `.64818`, 참조 인구는 `.59520`, `.50886`이다. 이 결과는 참조 비수렴의 영향과 양립하지만, 부분 BF만으로 native 전체 SE 또는 연산자 동등성을 입증하지 않는다. `residual_balance.json` SHA256 `3fcfb22f86c2d278497890798d9214c383b4e7eacaa9901340dd5014eb225f2f`다.
- native source·유효 입력에서 `DO_DDT=F`, `TRT_NON_TE=F`, `INC_XRAYS=F`, `INC_CHG=F`를 확인했다. 독립 `DIE_*`도 대상8개 경계 모두 F,F다. 실제 run에서 8개 `*_AUTO_DATA` 파일은 모두 없고, `STEQ_AUTO_V2`는 그 파일이 없으면 경고 후 또는 바로 RETURN하므로 해당 추가 항을 조용히 존재한다고 가정하지 않는다. 그렇다고 독립 자동이온화 구현을 전역 삭제하지 않는다.
- **기존 `STEQ_VALS`에 전체 native SE 배열이 저장돼 있었다.** NETRATE/TOTRATE 부재를 전체 SE 부재로 확대하면 안 된다. `solve_for_pops.f:123`은 SOLVEBA 전 매 반복에 `WR_ASCI_STEQ`를 호출한다. 마지막24MiB만 읽어 마지막 `STEQ ARRAY`의 27이온×9깊이 블록을 판독했다. 마지막 native 반복40, pre-solve SCRTEMP39의 phase다. 대상 이온의 첫 N_SL bound 행만 합산하고 추가 전자/온도 등 행은 제외했다.
- 출력의 `ES12.4` 반올림 오차를 합산하면 Si III28의 native bound SE 합은 `28.0234 ± 105.7302`여서 0 포함·정확한 합 미확정이다. S III35는 `241.6482 ± 170.0721`로 0을 포함하지 않는다. 이 오차는 출력 자릿수의 합산 상한이지 물리 오차 허용치가 아니다. 큰 내부 상쇄 때문에 Si 결과를 수렴 또는 비수렴의 확정 근거로 쓰지 않는다. S 결과는 해당 참조 상태의 bound SE가 남았다는 직접 증거다.
- `native_steq_snapshot.json` SHA256 `2b32bb4483ffd1597c7618e8f54ae48f8998852d2029540485b4941da1334780`에 입력 tail의 정확한 offset/길이/해시, POINT1 해시와 행 범위를 기록했다. 더 정밀한 native 전체 잔차·과정 분해는 아직 확보하지 않았다. 다음에는 원래 binary/보존 배열에 full-precision RHS가 있는지 `store_ba_data_v3.f`·`read_ba_data_v3.f`와 GATHER 경로부터 확인하며, ASCII의 부정확한 합을 정답으로 고정하지 않는다.

### 검증 상태·다음 내부 작업·보존

- Python **104 passed, skip0**(`regression-final.log`), 새 v5의 실제 C 로더 정상/손상/미지원/우회 거부와 구형 OFF, Si/S v4·v5 생산 owner 정상·손상 transaction 시험, 전체 CPU 링크가 통과했다. SE reader는 마지막 epoch·bound 행 선택·추가 방정식 제외·누락 block 거부·인쇄 반올림을 합성 입력으로 검증했다. `git diff --check`와 Slurm shell syntax도 통과했다. 개별 확인을 별도 감사나 완료 단계로 만들지 않았다.
- 최종 `bench_cmfgen_owner_final` SHA256 `92fbbce1a6f8a1ac81eb492cc76e29dc20c81e41966f207efbf3fdfe5954ee10`, 전체 CPU `lumina_final` `b6ae4ae1f6c6551e9c298b08a84f4a8c762f90c8bc1d548337f820e1d6be87a1`이다. 소스 snapshot은 `P/source_snapshot`; 마지막 native 예산 reader는 같은 디렉터리의 `summarize_cmfgen_recombination_response_with_native.py`에도 보존했다. 최종 C source SHA256: atomic `336f57328ad6a57b7cb8d69d9a6954549bb782d7e86f2ce29753ff89a4f2b455`, plasma `aff0e39b5d1e22732552a2549c40862044153c5f2149942ac738b4649f61718a`, EW `9f57675341e0bf67c4ac3edb476e65744a8080429ebdc61780135bc4b6c8cb61`.
- Astra `01a152ef-68cd-7743-8f4b-2421942c4a50`은 제공 증거에 대한 read-only 평가를 완료했고 종료했다. 외부 fit 전달/제외는 지지하되 잔여 차이를 즉시 누락 물리로 판정하지 말고 native 전체 예산·anchor 실패·Si 수치 인증을 분리하라고 권고했다. 그 원인 분리 제안은 **현재 물리 동질성 묶음의 내부 작업으로만 채택**한다. 작은 「다음 대상 결정」 목표로 원래 묶음을 바꾸거나 종료하지 않는다. 이후 운전자가 직접 AUTO 부재와 native 전체 SE 출력을 확인했으며 이를 Astra의 독립 재현이라고 표기하지 않는다.
- 다음은 보존 자료에서 native 전체 예산의 정밀도/phase를 확인하고, 같은 물리에서 실제 누락 항과 비수렴 참조 효과를 분리하는 것이다. job428468의 MOM 내부 실패는 source/유효 입력·초기화·경계 조건의 구체적 원인이 아직 미확정이다. 고정 T의 큰 광도 차이만으로 수송 오류 또는 LSTAR 입력 오류라고 단정하지 않는다. 명시된 LSTAR는19.48일 내부 decay power를 뜻한다. 임의 T 해제, 설정 탐색, 같은 continuation 자동 재시도, 참조 인구에 맞춘 율 조절, Si rank/κ 기준 완화는 하지 않았다.
- 모든 수치 실행은 grammar-debug CPU1·16GiB·각 timeout3600 안에서 수행했고 GPU 사용0, 새 CMFGEN·Slurm 제출0이다. 원본·앞선 B 수리 뷰·첫 결과도 보존했다. 신규 변경은 미커밋이며 기존 사용자 변경을 유지했다. 수렴 CMFGEN 기준해, Lumina 전하·RE·수송 공동수렴, Si의 실제 채택, 최종 활성 shell 분율·스펙트럼 동질성과 자체 종료감사는 계속 미완료다.

재현은 위 기존 `R/input/owner_input.bin`과 `P/atomic_view`, `P/bench_cmfgen_owner_final`을 사용하며 기존 출력은 거부한다. 동일한 clean env/카탈로그/CPU·RAM 상한에서 새 CSV 이름으로 실행한다. 정책 이관은 `python3 scripts/cmfgen_recombination_policy.py --deck E/atomic_view_J --run /gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern --out NEW_POLICY_VIEW`다. 이관은 원래 MODEL/링크/PHOT 신원을 확인하고 숫자 배열을 재적합하지 않는다. 세 점 검증은 `python3 scripts/summarize_cmfgen_recombination_response.py --root P --previous E`를 새 root의 전후 증거에 적용한다. 기존 요약은 덮어쓰지 않는다. native 잔차 reader는 같은 모듈의 `native_se_snapshot(run, [(14,3,28),(16,3,35)])`이고, 부분 BF 예산은 `reference_balance`다. 두 결과의 의미와 정밀도 한계를 혼동하지 않는다.

## 2026-10-19 추가 원인 분리 — 충돌과 제외 파장 민감도

사용자에게 최신90점 차이를 보고한 뒤, 목표를 축소하지 않고 같은 묶음의 내부 원인 조사를 계속했다. 이 절의 계산은 보존한 실제 생산 C 행렬 두 점을 대상으로 하며, 생산 소스·덱·복사장·허용 기준을 바꾸지 않았다. 새 CMFGEN/GPU/Slurm 실행도 없다.

### 보존 자료의 한계와 실제 활성 항

- `store_ba_data_v3.f`는 `SE%BA`, `BA_ED`, `BA_T`, `BA_T_EHB`만 쓴다. `read_bcd_mat.f`의 `WRITE_BCD_MAT`는 C/BC/D 행렬과 row/column scaling, POPS, pivot, replace/zero flag만 쓴다. `BAMAT`·`CSCRATCH`를 원래 고정밀 SE RHS의 저장물로 취급할 수 없다. 두 소스 SHA256은 각각 `55b1e071b9db38724078833a67347efbc0e90cfce0c89f94960743481949eff5`, `7ec9f7c4a6ba91a1d4eeed714065ab202c25ba14f813e9ec54d87460d6535d51`다. 큰 바이너리를 불필요하게 읽지 않았다.
- `DO_DDT=F`만으로 모든 population advection이 꺼졌다고 추정하지 않았다. `cmfgen_sub.f`의 ELSE는 별도 `STEQ_ADVEC_V4`를 호출한다. 실제 modern `VADAT:120`에 `INC_ADV=F`가 있고, 해당 함수는 그 flag가 거짓이면 즉시 RETURN한다. 이 참조에서는 population advection도 꺼져 있다. 복사 수송의 `INCL_ADV_TRANS`는 별도 제어이며 이를 같은 flag로 혼동하지 않는다.
- 실패한 job428468의 `MOM_J_ERRORS`에는 최종 경고 주파수 `6.40196372e15 Hz`보다 앞선 `1.01964257e16 Hz`부터 외곽 깊이1·2의 음수 XM이 기록돼 있다. moment 수렴 실패의 원인을 아직 특정한 것은 아니며, 원래 로그·실행을 보존했다. 기본 CHK_J의 FORM_VAL 대체와 CHK_H의 MAX_VAL 제한이 존재한다는 점도 정상 수렴 증거로 쓰지 않는다.

### 두 최대 차이 위치의 작은 동일입력 시험

운전자가 `scripts/analyze_cmfgen_collision_gaps.py`를 작성했다. 원래 생산 `rad_bb/coll_bb/rad_bf/coll_bf` 등 모든 평면은 보존하고, 다음 효과를 분리한 복제 행렬만 같은 실제 GTH로 풀었다. baseline의 원래 C 후보 재현을 먼저 확인했다.

1. **복사선·실제 충돌표 모두에 없는 fine-level 쌍:** 각 원래 `col_data`에서 확인한 `OMEGA_SET=.1`을 사용했다. 같은 SL 내부 쌍은 SL 방정식에서 상쇄하고, SL 사이 쌍만 실제 기여한다. Si28은32,198쌍, S35는93,440쌍이다. 이는 임의 floor 추가가 아니라 native `OMEGA_GEN_V3`의 미등재 쌍 처리를 진단한 것이다. 생산 코드에는 추가하지 않았다.
2. **표 보간:** 생산의 linear-in-T Upsilon과 native의 log–log T/Upsilon 보간을 같은 fine-level/SL mapping·열분포·계수에서 대조했다. SAME_N Gbar, 8.629/8.630 상수 차이와 다른 fallback은 변경하지 않았다.
3. **등록 BB 파장 범위 밖의 흡수·유도방출:** 원래 domain `100–20000 Å` 밖 선 중 보존 native J가 전체 profile을 덮는 선에 같은10km/s·±4 Doppler Gaussian을 적용했다. spontaneous A는 원래 행렬에 남아 있으므로 중복 추가하지 않았다. **native J가 덮지 않는 Si140선·S118선은 미확정으로 집계하고 제외했다.** 따라서 전 파장 또는 native 선 profile의 등가성을 입증하지 않는다.

| 시험 | Si III28 분율 | S III35 분율 |
|---|---:|---:|
| CMFGEN 참조 인구 | .595200289147 | .508857020566 |
| 원래 C 후보 재현 | .757995211881 | .648177058460 |
| 누락 충돌쌍만 | .757995212040 | .648176949720 |
| log–log 표 보간만 | .757995267798 | .648175430758 |
| 두 충돌 차이 함께 | .757995267957 | .648175322019 |
| 가용 J 범위의 제외 BB pumping만 | .758002236688 | .648167564597 |
| 위 세 효과 함께 | .758002292762 | .648165828126 |

- 두 충돌 차이를 합친 III 분율 변화는 Si `+0.00000561%p`, S `−0.00017364%p`다. 가용 J 범위의 제외 BB pumping만의 변화는 Si `+0.00070248%p`, S `−0.00094939%p`다. 기존 최대 차이 `16.28%p/13.93%p`를 설명하지 못한다. **이 두 위치의 큰 잔여 오차를 고치려 충돌망 전체를 우선 개조할 근거는 얻지 못했다.** 다른 깊이·미포괄 파장·native profile까지 원인 배제한 것은 아니다.
- 결과 `P/collision_and_domain_response.json` SHA256은 `da74aebc4c3857a4d2acfc2e5575acd849260d51e7b92b5d6340ecc1650efe7e`다. 입력·모든 실제 표·원래 col source·native omega source·실제 C 덤프·GTH library 해시는 JSON에 있다. 앞선 충돌 두 효과만의 `collision_gap_response.json`도 보존했다. 최초 실행에서 Si V manifest의 빈 `n_trans_source`를 만났으며, 이를 임의0으로 바꾸지 않고 실제 Si V col_data의 선언이0임을 확인하도록 수정했다.
- Gaussian의 상수/선형장 적분, native 입력 depth stride, 흡수·유도방출만의 추가, 미포괄 선 분리, 열평형 보존·SL 내부 상쇄, 표 정체성·중복·절단 거부 등 직접시험5건이 통과했다. 이전104건과 합친 회귀는 **109 passed, skip0** (`P/collision-gap-regression.log`, SHA256 `250b56871cefa50c8b705061149a810f24106033bff1381a65dd522f2b7decd5`)다. `git diff --check`도 통과했다.
- 분석기/시험 SHA256은 각각 `3cd49a9e05e76cd44bc0027809af1f75a4d89bd3ff78232dd142f1ec5f9a049d`, `b34269a9dc65a5be6596d167919b5c019dfb0c430b84c3eb330b68589ffc7346`이며 `P/source_snapshot`에 byte 동일한 사본을 보존했다. atomic/plasma/EW/header의 해시는 앞 절의 최종 생산 소스와 동일하다. 이번 민감도를 생산 수리·실제 채택·최종 수렴으로 표시하지 않는다.

재현은 grammar-debug CPU1·16GiB·timeout3600, OMP/OPENBLAS/MKL 각1에서 다음과 같이 한다. 기존 출력은 거부하므로 `NEW_OUTPUT.json`을 새 이름으로 정한다.

```bash
python3 scripts/analyze_cmfgen_collision_gaps.py \
  --root /gpfs/kjhan/cmfgen_runs/lumina_dr_policy_20261019-wYEzAc/residual \
  --deck /gpfs/kjhan/cmfgen_runs/lumina_dr_policy_20261019-wYEzAc/atomic_view \
  --input /gpfs/kjhan/cmfgen_runs/lumina_owner_response_20261019-oibhpY/input/owner_input.bin \
  --library /gpfs/kjhan/cmfgen_runs/lumina_owner_points_20261019-WR2jSs/population_generator.so \
  --native-source /gpfs/kjhan/cmfgen_src/cur_cmf \
  --output NEW_OUTPUT.json
```

Astra의 기존 평가 agent `01a152ef-68cd-7743-8f4b-2421942c4a50`를 한 번 재개해 이 실증을 묶어 다음 수리 우선순위를 평가받았고 종료했다. 충돌/미세율 추가 조정보다 **실제 실패한 MOM 호출의 재현과 원인에 따른 수송 안정화**를 우선하라는 판단이다. native40/pre-solve39의 SE 잔차와 job428468 첫60회의 MOM 실패는 다른 phase이므로 직접 인과관계로 합치지 않는다. Si 고정밀·전방오차 인증은 별도로 진행할 수 있지만 기존 raw rank/κ 실패를 자동 PASS로 바꾸지 않는다. 원래 물리 동질성 목표는 계속 미완료이고 다른 모델 의뢰·별도 작은 종료감사·기준 완화는 없다.

### 다음 수송 수리의 실행 범위와 부모 빌드 복원

- 단일 호출 재현에 필요한 상태를 실제 `MOM_JREL_V9`와 `MOD_RAY_MOM_STORE`에서 읽었다. 함수는 이전 청색측 주파수의 `SAVE/PREV` 배열·경계값과 같은 주파수의 재호출 이력을 사용한다. `INIT=T`, 이전 J/H=0 또는 첫 번째 같은 주파수 호출로 대체하면 기존 실패 재현이 아니다. `comp_j_blank.f`는 실제 frequency index와 `J_IT_COUNTER`, `FIRST_FREQ/NEW_FREQ`를 보유한다. 이 phase와 원시 주파수, 실제 인수·배열 shape/kind·formal moments·이전 상태를 함께 보존해야 한다.
- Astra의 제안은 복제 run에서 실패 주파수 `6.40196372e15 Hz`에 해당하는 실제 호출 직전 상태를 보존하고 최초 내부100회 실패 때 단 한 번 저장 후 종료하는 것이다. 포획 CPU16/OMP16·200GiB·1200초·1회, 추가 상태자료256MiB/전체복제·출력8GiB 이내; 단일 호출 replay는 CPU1·16GiB·300초다. 이는 대규모 continuation이나 설정 탐색이 아니다. 인구·온도 갱신이나 다음 외부 반복 전에 끝내고, `MAX_ER>=1e-8`에서100회 도달을 그대로 실패로 재현한다. 첫 음수 XM의 원래 삼중대각계/RHS·대체 전후 값과 미분 반복 이력을 확인해야 하며, 이후 수리는 실제 원인에만 한정한다. 포획/driver는 아직 미구현·미실행이다.
- 준비 중 **현재 native 소스·라이브러리와 실제 사용 실행파일의 불일치**를 발견했다. 현 `libsubs.a`로 링크한 `cmfgen_baseline.exe`는 SHA256 `824340523c17e43eee0e84f8cd2a2422ff5645556cf57713e10bc3df5683055d`이며 기존 `f2b9afcc...`와 달랐다. ELF 심볼 비교에서 크기가 다른 함수는 `comp_opac_`(6655→6777 bytes)였고, 해당 소스의 git diff에는 `CHI_NOSCAT >= 1e-3*ESEC` 하한 및 NaN 대체가 추가돼 있었다. **그 추가 처리를 기존 실행파일에 들어 있던 물리로 해석하면 안 된다.** 실패 원인 또는 수리 효과로도 아직 주장하지 않는다.
- 별도 `/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST`에서만 현재 `comp_opac.f`를 복사한 뒤 그 추가 부분을 되돌렸다. native HEAD `dd1660af039e10f8f8c61d892c4e226e1cd6438c`의 파일과 `cmp` exit0이었다. 원본 source/library는 변경하지 않았다. GFortran13.2.0과 원래 Makefile의 fixed-form 옵션으로 해당 객체만 만들고 **개인 복사본** `libsubs.a`의 같은 member만 교체했다. 이 사본을 사용해 같은 main 객체·나머지 library로 링크한 `cmfgen_parent_rebuilt.exe`는 기존 실행파일과 `cmp` exit0, SHA256도 **`f2b9afcc064037a413bf206047a6b7c813882dc40f894436e5de704a32b232f1`로 완전히 동일**했다. 다음 포획 빌드는 현재 전역 library가 아니라 이 일치한 부모를 기준으로 해야 한다.
- 개인 `comp_opac.f`/객체/archive SHA256은 각각 `46aa68419d3bd6ef359d43ac95e8de6ee19ab0e33f52f1a61dca17431d21ea41`, `43872288d211048f3e38f467a35dab6f88dcfe0e8757829a8c58a39fa5537c0f`, `811b34c1a114e3f8f7f0906297e9571cae9b184d93b271219a77b6d4bcb133eb`다. source/object/library 및 MOM/호출부 신원은 `parent-build-identity.json`에 봉인했다(문서 SHA256 `c0517d0428643d71c13569a0e0d35059fa6476ad71333693f5b69656aa611efd`). 이 재구성은 native 계산을 실행한 것이 아니라 컴파일·링크만 한 것이다.
- **메타데이터 변경 기록:** 초기 ELF section 추출에서 output ELF를 별도로 지정하지 않은 `objcopy --dump-section`이 원본 `cmfgen_dev.exe`의 mtime을 갱신했다. 원본 바이트/SHA256은 위 기존 값 그대로임을 즉시 확인했다. 이전 시각의 초 단위 원자료가 없어 임의 timestamp 복원은 하지 않았고, 이후 추출은 복사본에서만 하도록 변경했다. 원본 실행파일의 mtime을 과거 실행 시각의 독립 증거로 재사용하지 않는다. 사용자에게도 이 사실을 보고했다.
- 현재까지 신규 native 실행/Slurm/GPU는0, Lumina 생산 코드·입력·판정도 불변이다. 빌드 정체성의 공백은 해소했지만 실패 MOM 포획·수리, 수렴한 CMFGEN 기준해, Si 실제 채택, 새 공동수렴 계산과 최종 스펙트럼/분율 동질성은 계속 미완료다.

## 2026-10-19 MOM 단일 호출 포획·독립 재현의 구현 및 실행 계획

앞 절 Astra 계획을 같은 물리 동질성 묶음 안에서 구현한다. 목표는 기존 실패 호출을 원래 인수·phase·이전 주파수 상태에서 재현하여 수송 수리 대상을 정하는 것이다. 수렴 reference나 최종 분율/스펙트럼 일치의 대체 목표로 종료하지 않는다.

- 입력은 기존 StageA59와 실패 job428468의 `lambda-probe-inputs.json`에 기록된 봉인 파일이다. 같은 StageA59에서만 복제하며, 실패 run이 덮어쓴 EDDFACTOR를 새 출발점으로 쓰지 않는다. 새 runner는 실제 seed hash 집합을 기존 실패 기록과 대조한다. 원본·이전 결과는 보존한다.
- 구현은 `scripts/prepare_cmfgen_mom_capture.py`, `run_cmfgen_mom_capture.py`, `submit_cmfgen_mom_capture.slurm`이다. 생성/컴파일 출력은 개인 parent root의 `capture_v3`에 한정했다. 먼저 **포획 대상 네 native source를 수정 없이 전부 재컴파일·링크**해 기존 부모 실행파일과 SHA256 `f2b9afcc064037a413bf206047a6b7c813882dc40f894436e5de704a32b232f1`가 다시 일치함을 확인했다. 새 `GET_J_CHI.INC` 의존성도 포함했다. 단순히 옛 객체를 다시 링크한 확인보다 강한 source/binary 연결이다.
- 원래 MOM·경계·정확도·음수해 대체의 산술은 바꾸지 않는다. 선택한 주파수의 호출 직전에 `MOD_JREL_V9`66개와 `MOD_RAY_MOM_STORE`63개 field 및 실제 인수를 메모리 사본으로 보존한다. 미할당 배열, lower/upper bounds, kind, 문자열 길이, 원래 INIT와 SAVE/PREV도 그대로 보존한다. native CMFGEN/호출부는 실제 outer counter·frequency index·J_IT_COUNTER·FIRST_FREQ/NEW_FREQ만 전달한다.
- 기존 `IN_ITS NUM_ITS=4`를 유지한다. 이를1로 줄이면 첫60회에 LST_ITERATION이 켜져 다른 경로를 타므로 하지 않는다. 포획에 성공하면 내부100회 및 `MAX_ER>=1e-8`일 때 상태를 한 번 쓰고 STOP93으로 종료한다. 포획이 없더라도 첫 SOLVE_FOR_POPS 직전 STOP94로 종료해 인구·온도 갱신과 다음 외부 반복을 막는다. 부모의 물리 제어·허용치·DIM은 바꾸지 않는다.
- trace는 그 주파수의 호출·각 내부 반복의 원래 삼중대각 행렬/RHS, FORM_VAL 적용 전 XM, 이후 J와 로그미분·오차 이력을 저장한다. 포획 상태224MiB와 trace32MiB의 합256MiB 상한을 별도로 강제한다. native EDDFACTOR/BAMAT가256MiB보다 크므로 전체 native 파일을256MiB로 제한해 원래 경로를 바꾸지 않는다.
- 실행은 기존 job428468과 같은 **syntax Slurm 클러스터**임을 `scontrol`/`sacct`로 확인했다. 새 run은 `toy06_19p48d_mom_capture_<jobid>`다. 노드 고정 없이 a10/a40/a100/a100_pcie/h100/h200, CPU16·OMP16·200GiB·GPU0·벽시계20분·단 한 번이다. 자식 실행 상한1080초, 복제·정리 여유를 두고 자동 연장/재제출은 없다. Python runner와 의존 파일은 제출 전에 별도 runner root에 봉인한다. 직접 replay는 별도 새 출력 root에서 CPU1·16GiB·300초 안에 같은 native 한 호출만 수행한다.
- 관측상 성공은 상태 파일이 실제 실패의 phase60을 담고, 독립 replay가 원래100회 실패 및 trace를 재현하는 것이다. STOP93이나 Slurm COMPLETED 자체는 수렴/물리 PASS가 아니다. 실패가 재현되지 않으면 phase/입력 완전성부터 확인하며 다른 설정을 탐색하거나 기존 수치 기준을 완화하지 않는다. 실제 데이터가 나오면 삼중대각 풀이 오차와 음수해 대체/비선형 미분 반복 중 어디가 실패하는지 분리한다.
- 생성된 실제 native 모듈/인수의 바이너리 round-trip이 byte 동일했다. 미할당·음수 lower bound도 복원했고 손상 kind/절단/후행 바이트를 거부했다. CAPTURE 모드의 인구 갱신 전 STOP94와 비활성 모드의 정상 복귀도 직접 시험한다. 초기 Fortran 선언 문법 및 include 경로 오류는 수리했고 실패 빌드 `capture_v1`도 보존했다. `capture_v2`는 추가 출력 한도 보강 전 사본으로 보존하며 실행하지 않는다.

## 2026-10-19 실제 포획·동일 replay·원식 조사 — 수렴 표시와 방정식 만족의 불일치

### 실제 실행과 독립 재현

- 위 계획으로 **syntax job428531**을 한 번 실행했다. 노드는 자동 배정 `syn03`, CPU16·200GiB·GPU0·벽시계20분 상한이다. 실제 전체264.540초, native 자식251.887초에 첫 실패를 포획했다. Slurm 부모는 COMPLETED, 자식 step은 **의도한 STOP93**이다. 이것을 수송 또는 물리 PASS로 해석하지 않는다. 첫 인구/온도 갱신 전에 종료했으며 자동 재시도는0이다.
- run `R60=/gpfs/kjhan/cmfgen_runs/toy06_19p48d_mom_capture_428531`, build `B=/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/capture_v3`다. 기존 StageA59 출발 hash 집합이 job428468의 기록과 일치했다. 실제 포획 phase는 outer60, frequency index34540, J_IT_COUNTER0, FIRST_FREQ=F, NEW_FREQ=T, INIT=F, ND90이다. 주파수는 native 단위 `6.4019637160951506`(×1e15 Hz), 이전 주파수 `6.4022522629731631`이다. 상대론·이류·coherent scattering을 유지했고 DIFFUSION/HONJ·LOGMON·N_ON_J·FORM_VAL/MAX_VAL/NONE도 그대로다.
- 포획 상태는76,178bytes(SHA256 `797022a958ac245a557c7349d76d841afc3949673db5921cfc09b1f6f12a5ebe`), trace580,444bytes(`5b43b7bb48c361c0589f084a878ab6337b2c2a0ab22ec65a0d8884bccc82d83b`)다. 추가 출력 합계656,622bytes이며256MiB 상한 안이다.
- 독립 `D=/gpfs/kjhan/cmfgen_runs/lumina_mom_replay_20261019-FCL4il`에서 CPU1·16GiB·timeout300으로 native MOM 한 호출만 실행했다. **입력 snapshot 전체와100회 반복 trace 전체가 포획본과 바이트 동일**했고 동일 STOP93을 재현했다. `D/mom-analysis.json` SHA256은 `77d824a12cac6edd689ffc8e95a3c0f9408a1735e0b9eb987e4cb9aea85656b5`다. 이 한 호출에서 필요한 상태 보존은 실행으로 재현됐으나, 임의의 다른 CMFGEN 상태/버전까지 포괄하는 일반성은 주장하지 않는다.

### 무엇이 실제로 실패하는가

- 최초 반복의 선형계 잔차는 `1.50e-16`, 마지막100회도 `1.05e-16`이었다. 마지막에는 깊이6의 XM `−0.0012610609`를 FORM_VAL로 대체한 뒤 **같은 선형계 잔차가 `.9948602194`**로 커졌다. 음수해가 없는2–15회부터 반복 변화가 증가하므로 FORM_VAL을 최초 원인이라고 단정하지 않는다. 비선형 반복 악화 이후 음수와 대체가 추가된다는 증거다.
- Astra가 제안한 최초 음수16회·마지막100회의 같은 행렬/RHS를 독립 pivoting 및80자리 mp LU로 대조했다. 두 행렬의 κ2는 약 `5.656e8`, native 대 mp 해 상대차는 각각 `1.19e-12/1.44e-12`이고 **깊이6의 음수 부호가 유지**됐다. 단순 THOMAS 풀이 오류를 주수리 대상으로 삼을 근거가 없다.
- Astra는 이 증거를 read-only로 평가하고 같은 물리·경계의 원래 비선형 이산식에서 양의 근을 먼저 확인하도록 권고했다. 계획·평가 의뢰는 이 의미 있는 수리 결정에서 한 번만 했고 agent를 종료했다. 초기값/기준 탐색, 이류 OFF, CHK 변경, 완화계수/반복수 조정은 하지 않는다. 운전자 재현을 Astra의 독립 계산으로 표시하지 않는다.

### 같은 원식의 양의 근 조사와 직전 주파수의 상태

- 새 CMFGEN 실행 없이, 동일 replay의 별도 `D/kernel_v1` 사본에서 반복 직전의 정적 계수만 추가로 저장했다. 원래 capture/trace는 다시 바이트 동일했다. `MOM_KERNEL.bin`은 **반복 직전 계수 판독용**이며 최초 호출 입력 `MOM_CAPTURE.bin`과 용도가 다르다.
- `scripts/solve_cmfgen_mom_kernel.py`가 native `subs/nordtau.f`의 LINMON 분기·산술을 그대로 재구성했다.100회 trace와의 대각 재조립 상대차 `1.95e-16`, 실제 기울기 갱신 차이 `2.61e-16`, 우변 차이 `2.08e-16`, 두 경계 우변 차이0이었다. 나머지 삼중대각 계수와 RHS는 모든 반복에서 bit 불변이었다. 따라서 이전 기울기를 고정한 선형계만이 아니라 후보 J에서 미분을 다시 계산한 원식을 조사했다.
- 자연 초기값 두 개(native 첫 iterate, 보존 formal장)에서 독립 root 풀이가 모두 **깊이6 음수 해 `−.04801694543`**에 도달했다. 원식 잔차는 `9.09e-17/1.18e-16`이나 물리적으로 채택할 수 없다. 양수 제약 풀이가 success를 반환했지만 원식 잔차는 `.0007816183`으로 남아, 이것도 성공으로 인정하지 않았다. **현재 고정 입력에서 양의 근은 미확인**이다. 엄밀한 근 부재 또는 전역 유일성은 증명하지 않았다.
- 깊이6의 음수 우변은 `local emission +.0000130`, `previous H −6.7817147`, `previous J +1.4261183`, 두 N 항 `−.0358916/+.2239378`의 합이다. 현재 한 호출만 안정화하면 해결된다고 단정할 수 없어, 호출 전에 남아 있는 **직전 주파수 `6.4022522629731631e15 Hz`의 정적 연산자와 SAVE 반환값**도 원래 식으로 재평가했다. 현재 formal raw STORE 대신 이전 SAVE 경계값을 사용했다.
- 직전 반환 J의 원래 선형식 잔차는 `.997063387270`, 그 J에서 LINMON을 새로 계산한 비선형 잔차도 `.997063387268`이다. 깊이1–4에서 잔차가1e-8보다 크고 최대 위치는4다. H 반환값도 원래 관계식의 값과 달라진 edge4·5에서 native MAX_VAL 제한이 필요했던 상태다. 실제 `R60/MOM_J_ERRORS`에도 이 이전 주파수의 깊이1–3 음수 XM이 기록돼 있다. **직전 입력 field를 방정식을 만족하는 정상해로 가정할 수 없다.** 적어도 값 대체 후 반환/변화량 수렴을 원식 만족으로 읽어서는 안 된다.
- 이전 입력이 오염돼 있으므로 현재 주파수의 양수값을 강제로 맞추는 생산 수리는 하지 않았다. 다음은 최초 J/H 값 대체 또는 원식 불만족이 발생하는 더 앞선 주파수의 입력·이산화를 추적하는 것이다. `R60/MOM_J_ERRORS` 첫 음수 J 기록은 `1.01964257e16 Hz`지만, 기록되지 않은 H 제한이 더 앞서 발생했을 수도 있어 이를 최초 모든 오류 지점으로 단정하지 않는다. 추가 큰 continuation/동일 재시도는 하지 않았다.
- `D/kernel_v1/positive_root_v2.json` SHA256 `7b63747e32a12c7a937aebbf85018680172ed69fde17db1212e8a94dce5eb703`에 재조립, 두 독립 선형 확인, 음수/양수 제약 결과와 직전 주파수 재평가를 함께 보존했다. 이번 원식 계산 실제1.46초이며 CPU1·16GiB·600초 상한 안이다. 최초 결과와 그때 분석기 `solve_cmfgen_mom_kernel_v1.py`도 보존했다.
- 이 수송 증거는 **native60 phase**다. 앞선 분율 비교의 modern pre-solve39/iteration40과 직접 같은 상태가 아니므로, 잔여 `16.28%p/13.93%p`를 이 실패 하나의 인과 효과로 단정하지 않는다. 다만 modern의 기존 `MOM_J_ERRORS`에도 `1.00052714e16 Hz`부터 음수 XM 기록이 있으며, 수렴 참조로 이미 부적격인 사실은 바뀌지 않는다. modern에 대해 이60회 호출의 정량 잔차를 복사해 주장하지 않는다.

### Si 고정밀 대조·검증·보존

- 운전자는 별도로 보존한 Si28·61 두271차원 행렬을 독립120/160자리 mp LU로 풀었다. GTH와 같은 **양의 off-diagonal 율**을 사용하고, 대각은 그 유출률 합의 음수로 고정밀 조립했다. 저장 double 대각의 반올림 오차를 새 물리로 삼지 않는다. 두 정밀도 사이 최대 확률차는 `6.12e-116/2.84e-117`이었다.
- C GTH의 Si II/III와 독립 해 상대차는 깊이28 `5.62e-18/2.48e-16`, 깊이61 `5.86e-16/2.73e-16`이다. 즉 이 두 고정 입력 지점의 계산 후보는 독립 풀이와 매우 가깝다. **기존 raw rank/κ 기준은 바꾸지 않았고 Si0/90 채택도 유지**한다. 전90점 또는 실제 공동수렴의 전방오차 인증으로 확대하지 않는다. `B/si_multiprecision.json` SHA256은 `89a9fd57c6410c572e2bb2c3042bf324f3d9e33bec29b95b146e16e81293d31c`다.
- 최신 직접 회귀는 **116 passed, skip0**, `B/regression-final.log` SHA256 `0c6d77865d154ee859dd0d437519c4b32f1a693f683e4adef47514313ae997ce`다. source/분기/제약·입출력 형식 시험과 실제 보존 데이터 재현을 구분한다. 실행 당시 봉인 runner/test와 이후 분석 source/test는 버전이 다르므로 각각 `B/runner`, `B/analysis_snapshot`에 보존했다. Lumina 생산 C 소스와 판정 기준은 이번 작업에서 변경하지 않았다.
- 통합 신원 정본은 `D/progress-identity-v2.json`(SHA256 `4cd9f17a72e6a982c1f36b91d293141e1a63691f014c097a3cb20bdcdb49a87b`)이다. 초기 `progress-identity.json`의 sacct는 grammar-debug의 다른 Slurm cluster에서 동일 숫자 jobid를 읽은 것이므로 무효다. 수정본은 **syntax 호스트/cluster의 syn03·CPU16·200G·4분25초·자식93**을 명시적으로 연결하고 잘못 읽은 값도 invalid로 보존했다. GPFS 공유가 scheduler 공유를 뜻하지 않는다. 실제 syntax 자식 MaxRSS는5,702,472K였으며, 향후 같은 pre-pop 포획의 메모리 요청은 이 실측을 근거로 축소 검토할 수 있다.
- 새 실행은 이 포획 Slurm1회뿐이다. GPU0, 자동 재제출0, 모든 실제 계산·replay는 종료됐다. 원본·기존 실패·새 포획·양의 근 미확인 결과를 보존했다. 현재는 최초 비물리적 spectral 입력을 추적해야 하며, CMFGEN 수렴 앵커·Si 채택·새 자기일관 복사장과 분포/스펙트럼 동질성·자체 종료감사는 계속 미완료다.

핵심 재현은 `B/replay_mom.exe R60/MOM_CAPTURE.bin`을 **새 cwd**에서 `LUMINA_MOM_CAPTURE_MODE=REPLAY`, OMP/OPENBLAS/MKL1·CPU1/16GiB·timeout300으로 실행하는 것이다. 원래 실패를 재현하므로93이 예상된다. `scripts/analyze_cmfgen_mom_capture.py --run R60 --replay NEW_REPLAY_ROOT --build B --output NEW_JSON`이 원래/재현 trace와 snapshot을 대조한다. 원식 재조립은 `scripts/solve_cmfgen_mom_kernel.py --kernel D/kernel_v1 --original R60 --build B --native /gpfs/kjhan/cmfgen_src/cur_cmf --output NEW_JSON`이다. 기존 파일은 거부하며, 이 두 결과를 수렴한 스펙트럼 비교로 해석하지 않는다.

## 2026-10-19 최초 수송 대체 지점 추적 — 실행 전 계약과 직접 재현

이 작업도 최종 Si II/III·S 분포/스펙트럼 동질성 목표의 내부 원인 조사다. 현재 불만족 입력의 특정 주파수만 수리하지 않고, 같은 StageA59에서 시작한 첫60회 수송 중 최초 J/H 대체 직전의 입력·경계·이산식을 확보해 수리 대상을 결정한다. 최초 지점 발견이나 직접시험 통과를 최종 목표 달성으로 선언하지 않는다.

- **범위와 제외:** 최초 MOM 호출부터 모든 주파수·동일 주파수 재호출·내부 반복을 관찰한다. 원래 FORM_VAL/ABS_VAL의 음수 J 대체, MAX_VAL/AV_VAL의 양·음 방향 H 제한 또는100회 실패 반환 직전에 STOP95한다. 원래 물리·경계·미분 분기·CHK·반복 한도와 NUM_ITS=4는 유지하고 인구/온도 갱신 전에 종료한다. 이 실행에서 앞선 MOM J/H 대체가 없다는 것만 보장하며, seed59나 formal solver의 다른 모든 처리까지 정상이라고 주장하지 않는다.
- **입력과 빌드:** 기존 StageA59 및 job428468의 출발 hash 집합을 다시 대조한다. private build `/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/capture_first_v1`의 무계측 네 translation unit 재빌드는 기존 부모 SHA256 `f2b9afcc064037a413bf206047a6b7c813882dc40f894436e5de704a32b232f1`와 바이트 동일하다. 원본 소스/library/exe와 이전 실행은 변경하지 않는다.
- **기획·평가:** Astra의 기존 묶음 평가에서 최초 대체 포획을 후속 최소 실행으로 채택했다. 첫 INIT의 미할당/미정의 module 값을 읽지 않도록 실제 formal raw 입력만 보존하고 나머지 diagnostic placeholder의 무효 상태를 명시한다. 직전 서로 다른 주파수의 마지막 완료 호출도 저장한다. 개별 checkpoint마다 추가 평가를 의뢰하지 않는다.
- **소유자와 자원:** 운전자 구현·직접시험·syntax cluster 제출·재현. 새 Slurm 실행은1회, 노드 미고정 CPU16/OMP16·RAM16GiB·GPU0·벽시계1200초, native 자식1080초, 자동 재시도0이다. 이전 실측 MaxRSS 약5.44GiB에 근거했다. 복제·기존 출력 예산8GiB, 추가 포획 합계256MiB, 후처리 CPU1·16GiB·총600초다.
- **관측 산출물과 판단:** `MOM_CAPTURE.bin`은 진입 상태, `MOM_KERNEL.bin`은 내부 반복 직전 계수, `MOM_FAULT.bin`은 대체 전 행렬/우변·J/H·사용 기울기·사건 metadata, 선택적 `MOM_PREVIOUS.bin`은 직전 서로 다른 주파수의 완료 상태다. 실제 phase·입력/빌드 hash·원식/경계 잔차·부호를 조사한다. 동일 원식에 물리적인 해가 있으면 수치 수리를, 그렇지 않거나 미확정이면 입력·경계·closure·이산화를 조사한다. root 미발견을 근 부재 증명으로 표시하지 않는다.
- **직접 증거:** 새 포획 모드의 Fortran IO·미할당 초기 상태·이전 주파수 metadata·원래 대체보다 앞선 hook 등을 포함한 직접시험9건이 통과했다. `/gpfs/kjhan/cmfgen_runs/lumina_mom_first_shadow_20261019-1Nmb6I/shadow-comparison.json`에서 기존6.40196371609515e15Hz 실패 입력의 첫 음수 J를 iteration16·depth6에서 STOP95로 재현했고, 기존 trace의 해당 행렬·보정 전 J·진입 snapshot·반복 전 kernel이 모두 바이트 동일했다. 이것은 새 실제 첫 사건의 포획 결과가 아니다.
- **보존·재현:** sealed runner와 hash 목록을 새 build 아래에 두며 기존 출력은 덮어쓰지 않는다. syntax에서 `sbatch --parsable --mem=16G --job-name=cmf-mom-first BUILD/runner/submit_cmfgen_mom_capture.slurm BUILD/runner BUILD first`로1회 제출한다. 첫 사건의 독립 재현은 새 cwd에서 `LUMINA_MOM_CAPTURE_MODE=REPLAY_FIRST BUILD/replay_mom.exe RUN/MOM_CAPTURE.bin`이며 예상 종료95다. 현재 Si0/90 채택과 기존 물리 수용 기준은 변경하지 않는다.

### 실제 최초 사건·원식·경계 처리의 분리

- **실행:** syntax job428588, syn03, CPU16/16GiB/GPU0, 전체219.42초(회계3분40초), 자식213.33초에 의도한 STOP95다. 자식 MaxRSS5,707,196K이다. run은 `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_mom_first_428588`이며 인구/온도 갱신·자동 재시도는0이다. `seq_logs/mom-capture-runtime.json`이 실제 사건과 입력 신원을 연결한다.
- **사건:** outer60, frequency index23732, J_IT_COUNTER0, 첫 내부 반복1의 depth1 음수 J다. 주파수는 `1.0196425671617638e16 Hz`다. 그 이전 이 실행의 MOM J/H 대체는 없지만 formal/경계의 다른 대체까지 없었다는 뜻은 아니다. 과거 특정6.4e15Hz 실패보다 앞선 실제 최초 사건을 확보했다.
- **동일 재현:** `/gpfs/kjhan/cmfgen_runs/lumina_mom_first_replay_20261019-yZp824`에서 CPU1·16GiB로 STOP95를 재현했다. entry/kernel/event가 모두 바이트 동일하다(`replay-comparison.json`). 각각76,178/76,178/5,136bytes이며 SHA256은 `a432e6be22c24d5f33f1297754ca3392198e59b7df984b7bdfecc4108db0d813`, `e0accf100e156c8cabc37aba4563cfe4ab8feb83048baf40d605e9e356f72cd4`, `9b52890499a362db355823f451c3c7bce693e9e8161c5fda4b399161f85bffbb`다. 직전 완료 호출 snapshot도 보존했다(`6d794968b65ac4a2919649461aec9334efd91d95d9335b135a7eb9effd3417db`).
- **직전 입력과 현재 원식:** 직전 서로 다른 주파수 `1.0196885241044287e16 Hz`의 마지막 완료 호출은 index23731/JIT1이다. 저장식 잔차2.81e-16, 기울기를 새로 계산한 비선형 잔차7.33e-10, H 대체0이다. 현재 첫 선형계도 잔차1.39e-16, 독립80자리 해와 상대차1.51e-15다. 현재의 원래 비선형식은 두 자연 초기값에서 depth1·2 음수 해로 갔고 잔차는1e-15 수준이다. 양수 제약 풀이의 success는 잔차.6412 때문에 기각했다. 이는 엄밀한 양의 근 부재 증명이 아니다. `first-kernel.json`을 읽는다.
- **항 예산:** depth2 우변은 emission `+4.08e-8`, previous H `−4.5585e-5`, previous J `+2.3142e-5`, 두 N 항의 합 약`+5.01e-7`로 전체`−2.1901e-5`다. depth3도 음수 우변이며 χ·η·formal J는 양수다. 단순 선형 솔버 오류나 반복수 부족만을 수리 대상으로 삼지 않는다. 첫8깊이의 실제 계수·formal/이전 closure는 `first-budget.json`에 있다.
- **새 경계 부정합:** `plane/cmf_formal_rel_v4.f:621–624`는 raw H/J와 N/J를 계산한 다음 음수이면 각각0으로 바꾼다. 사건의 raw 비는 `−.196480/−.114002`, 직전 비는 `−.192661/−.111402`지만 전달 HBC/NBC는 모두0이다. 실제 THK_CONT/THK_LINE=T이며 외부 연장 영역이 있으므로 inward flux 자체를 곧바로 비물리로 판정하지 않는다. outward/inward half J/H는 모두 양수다. 별도 half NPLUS_OB/NMIN_OB는 해당 루틴의 초기화가 확인되지 않아 판단에 쓰지 않았다; 실제 NNU_STORE는 매 호출0으로 초기화된다.
- **반증 시험:** 같은 고정 입력에서 현재/이전 경계 scalar의 raw H 부호만, 또는 H/N 모두를 복원한 오프라인 반응에서도 depth1·2 음수가 유지됐다. 이전 주파수의 해 전체를 다시 계산한 것은 아니므로 이 결과를 경계 수리의 실패 또는 부호 clamp의 인과 배제로 확대하지 않는다. `boundary-response.json`과 private `capture_first_v1/analysis_v1/boundary_response.py`가 정확한 재현 자료다. 이 민감도를 생산 코드에 반영하지 않았다.
- **다음 판단:** 정상식의 입력과 formal 경계 처리, 원래 공간/주파수 이산식의 양성 보존을 구분해야 한다. 이 새로운 수리 결정에서만 Astra의 read-only 계획·평가1회를 요청했다(agent `01a15364-92b5-72d2-a5d2-5188b5ec2402`). 현재 변경은 분석기와 직접시험뿐이며 원본 native나 Lumina 생산 물리는 그대로다. 직접시험은12 passed다. 새 자기일관 분포·수렴 기준해·스펙트럼 유사도와 최종 목표는 계속 미완료다.

### 두 주파수 연결과 실제 각모멘트의 비물리성 — 후속 오프라인 결과

Astra의 위 read-only 평가는 완료·종료됐다. 권고는 전체90깊이를 유지한23731→23732 연결에서 먼저 THICK 확장과 실제 유입 경계를 일치시키고, 이전 J/H까지 다시 전달하는 것이다. 그래도 문제가 남을 때만 보존적·양성보존 결합 이산화를 검토한다. 다음 오프라인 한도는 연산 노드 CPU1/4GiB/GPU0/300초이고, 실제 광선 상태가 꼭 필요하면 두 주파수에 한정한 native 포획1회(CPU16/16GiB/GPU0/600초·인구 갱신 전 종료)를 제안했다. 아직 그 추가 포획은 제출하지 않았다.

운전자는 먼저 보존된 상태로 다음 작은 시험을 완료했다. 소스는 `capture_first_v1/analysis_v2/connected_boundary_response.py`, 실제 native MON_INTERP wrapper `analysis_v2/mon_mid.f90`·`.so`, 후속 판독 `analysis_v3/angular_support.py`다. 전체90깊이를 그대로 사용하고 새 native 실행은 하지 않았다. CPU1/4GiB, 연결 root·고정밀 계산 실제4.43초다.

1. **이전 해도 다시 푼 signed 경계 후보:** 23731의 원식을 signed H/N 경계로 다시 풀면 J는 모두 양수고80자리와1e-14 수준으로 일치한다. 그러나 H의 edge1이 기존 MAX_VAL 상한을 넘었다. 따라서 이 해를23732에 정상 상류 입력으로 전달하거나 생산 수리로 채택하지 않았다. 이 단계는 경계 전달의 완전한 물리 계약을 검증한 것이 아니며, 이전23730의 이력도 원래 상태다.
2. **같은 비음수 각강도로 표현 가능한 상류 모멘트:** 실제23731의 formal J/H/K/N을 사용했다. native MON_INTERP로 H·K·N의 중간점 모멘트를 재구성하고 기존 KMID_ON_J·NMID_ON_J와 **상대차0**임을 확인했다. node/mid의 정규화된 0~3차 모멘트에 대해 `∫(1±μ)[1,μ]^T[1,μ] I(μ)dμ`의 최소 고유값은 각각`.0178731/.00877942`로 양수다. 이는 `[-1,1]` 위 비음수 각분포가 존재하는 모멘트라는 증거이지 보존하지 않은 실제 광선 강도 전체의 독립 재현은 아니다.
3. **같은 현재 원식에 위 상류 모멘트만 전달:** 원래0 경계와 signed 경계 모두에서 두 자연 초기값이 전90점 양의 J 해로 갔다. 원식 잔차는 각각 약`2.74e-16/1.63e-16`,80자리 상대차`~8e-15`, 같은 미분 분기, H MAX 제한0이다. 원래 χ·η·격자·주파수 간격·이류·상대론은 유지했다. 실제 전체 스펙트럼의 역사를 새로 계산한 결과가 아니라 고정 입력의 조건부 원인 분리다. 결과는 `connected-boundary-response.json`이다.

**핵심 정정:** 「직전 MOM 해의 원식 잔차가 작다」는 「직전 해가 물리적으로 가능한 복사장이다」와 다르다. 같은23731 저장 J/H와 실제 전달 K/N closure를 함께 조사하자, 원래 MON_INTERP의 중간점 규약에서 edge2·3·4·8이 비음수 각강도 모멘트가 될 수 없었다. 특히 edge2·3은 `H²>K·max(J_left,J_right)`여서 **어떤 monotone J 중간값을 택해도 Cauchy 조건을 위반**한다. 단순 `|H|<max(J_left,J_right)`는 통과했지만 충분한 물리 조건이 아니었다. 최악 edge3의 정규화 `(J,H,K,N)=(1,−.851632,.272599,−.125253)`이고 모멘트 행렬 최소 고유값은`−.431176`이다. 반면 formal 상류로 푼 현재 두 후보는 같은 검사를 전 구간 통과했으며 최소 고유값`.00834374`였다(`angular-support.json`).

이로써 현재 실패 지점의 원래 이산식이 **물리적으로 일관된 모든 상류 입력에서도** 필연적으로 음수라고 주장할 근거는 없어졌다. 우선 수리 대상은 이미 물리적으로 불가능한 moment/closure 상태가 주파수 간에 전달되는 경로다. `comp_j_blank.f:783–798`은 K/J(FEDD)의 변화만 보고 주파수별 반복을 종료하며 J/H 및 다른 closure의 정합을 함께 확인하지 않는다. 이것이 단독 원인이라고 아직 확정하지 않으며, 단순 반복수 증대·새 gate 추가만으로 수리하지 않는다.

다음 내부 작업은 필요한 두 주파수의 actual formal 광선·입사 전달을 재현해 THICK 경계와 moment/formal 수지·모멘트 정합을 함께 수리하는 것이다. 필요한 경우에만 위600초 포획1회를 사용한다. 임의로 formal 값을 production fallback으로 복사하거나 외부 연장/이류를 끄지 않는다. 그 수리가 실제 전주파수·공동수렴·분포 개선으로 이어져야 원래 목표를 달성한 것이다.

첫 보간 판독은 MOM의 근사 광속 상수로 계산한 GAM_REL을 formal 보간에 사용해1e-10 수준 mismatch로 거부됐다. 이 오류 판본을 `connected_boundary_response_v1.py`로 보존했고, 원래 GET_MOMS_REL이 실제로 쓰는 GAM_REL_STORE로 수정하자 위 상대차0을 재현했다. 두 gamma 자체 상대차4.17e-9를 큰 음수해 원인으로 해석하거나 원본 상수를 수정하지 않았다. 새 분석 source와 실행 당시 runner는 별도 판본으로 보존한다. 변경 후 직접시험12건이 다시 통과했고, 기존 물리 기준·Si0/90·S90/90·최대 III 차이16.28/13.93%p는 그대로다.

### 실제 광선 재현의 최소 포획 — 실행 전 범위

위 Astra 계획이 조건부로 제안한 native 포획1회를 사용한다. 아직 보존되지 않은 실제 광선의 입사 강도와 신선한 N half 적분이 경계 전달 수리에 필요하기 때문이다. 수렴 기준해나 새 Si/S 분포 계산을 이 포획으로 대체하지 않는다.

- **동일 입력·범위:** 같은 StageA59와 job428468 출발 hash, NUM_ITS=4/고정 T/모든 물리·기하·경계 설정을 유지한다. outer60의 index23731 모든 실제 재호출과 index23732 첫 formal 호출만 저장하고,23732 MOM 및 모든 인구 갱신 전에 STOP96한다. 예상 호출은23731/JIT0·1과23732/JIT0의3개이며, 실제 phase로 검증한다. 실패 자동 재시도0이다.
- **최소 입력:** 호출 인자와 formal에 필요한 raw 모듈 필드, 각 광선의 실제 유효 이전 `I_P/I_M`, 출력 강도와 기하·quadrature의 실제 사용 영역만 보존한다. DIFFUSION/비초기화 호출에 한정한다. 미초기화 output 인자·미사용 quadrature tail·누적 N half를 입력으로 읽지 않는다. 원래 native 변수는 바꾸지 않고 diagnostic 사본의 비소비 필드는0으로 표시한다.
- **재현과 관측 결과:** 새 cwd의 standalone replay는 원래 routine의 초기화로 같은 기하를 만든 뒤 그 일치를 확인하고, 저장한 실제 이전 광선 강도와 raw 상태를 복원해 본 호출을 수행한다. 광선별 출력의 바이트 정체성과 모멘트의 적분·병렬 reduction 오차를 구분한다. 부호가 가능한 실제 입사 강도로 원래 표면의 유입과 외부 연장 계산을 대조한다. 단순 signed 비율 복원과 완전한 입사 경계를 구분한다.
- **빌드·직접시험:** private `/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/formal_v1`에서 원래 formal translation unit까지 재컴파일한 부모가 기존 SHA256 `f2b9afcc064037a413bf206047a6b7c813882dc40f894436e5de704a32b232f1`와 일치했다. 기존 MOM 도구를 재사용했고 원본 source/library/exe는 불변이다. 직접시험19건은 실제 Fortran IO 왕복, 유효 PREV/SAVE 선택, output/미사용 필드 분리, 손상·절단·후행 거부, population 전 종료, 실제 광선 half 적분을 검증했다.
- **자원·보존:** syntax 노드 미고정 CPU16/OMP16·16GiB·GPU0·600초, native 자식500초,1회다. 개별 포획64MiB·합계256MiB, native 기존 파일 한도8GiB다. 이후 오프라인은 연산 노드 CPU1/4GiB/300초 안에서 실행한다. 새 run root와 sealed runner를 사용하고 모든 기존 입력/출력/오류 판본을 보존한다. 제출은 `sbatch --parsable BUILD/runner/submit_cmfgen_formal_capture.slurm BUILD/runner BUILD`다.

별도 소스 근거로 `pp_form_cmf_v2.f:1460` 및 `pp_mom_cmf_v1.f:580–581`는 `H=HBC*J−HBC_INCID` 형태의 입사 상수항을 구현한다. 현재 spherical `MOM_JREL_V9`의 HONJ에는 그 별도 항이 없다. 실제 THICK 표면 유입은0이 아니므로, 다음 오프라인 수리 후보는 단순 H/J 부호 변경이 아니라 outgoing 관계와 incoming 모멘트를 분리한 경계 전달이다. 이것은 아직 구현·검증 또는 생산 채택한 수리가 아니다. 두 형상의 물리식을 그대로 같다고 가정하지 않고 상대론/이류 경계항을 원식에서 따로 재조립해야 한다.

## 2026-10-19 실제 광선과 입사 경계 수리 — native 국소 검증까지

### 실제 포획과 외부 확장→표면 전달의 재현

- **syntax job428619**, syn03, CPU16/16GiB/GPU0, 전체213.77초(회계3분35초), 자식209.18초의 의도한 STOP96으로 종료했다. MaxRSS5,702,556K, 추가 자료3,364,116bytes다. 실제 세 phase는 outer60/index23731/JIT0·1과23732/JIT0이며23732 MOM과 인구 갱신 전에 멈췄다. 자동 재시도0이다. run `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_formal_capture_428619`, 실행 build `formal_v1`을 사용했다.
- **독립 재현 정본:** `/gpfs/kjhan/cmfgen_runs/lumina_formal_replay_20261019-j8PO9m`. 각 세 호출의 기하와 모든 I+/I− 출력은 standalone replay와 바이트 동일했다. 실제 광선 강도 음수는0이다. 모멘트의 OMP16/CPU1 합산 순서 차이는 실제 가중 강도로 유도한 IEEE 상한의 최대4.03% 안이었다. 물리 허용치를 완화한 것이 아니라 바이트 재현과 합산 반올림을 분리했다(`replay-comparison.json`).
- **표면 전달 시험:** 원래 확장층 적분이 만든 I−(R1)을 경계값으로 놓고 R1 안쪽 적분을 재시작해도, 세 호출 모두 모델 내부 I+/I−가 바이트 동일했다(`surface-transfer-comparison.json`). 이 split 시험의 외부 inward 상태는 비교하지 않았으며 다음 주파수 전체 이력에 재사용하지 않는다. 실제 확장층의 유입과 formal 적분 자체가 불일치한다는 가설은 이 구간에서 지지되지 않았다.
- replay용 `formal_v2`는 짧은 후행 바이트까지 거부하도록 IO를 보완한 판본이며, `formal_split_v1`은 위 표면 전달 시험만 위한 별도 판본이다. 두 판본 모두 수정할 native translation unit을 원형으로 재컴파일한 부모가 기존 f2b9afcc…와 일치했다. original native source/exe/library는 변경하지 않았다. v1 실제 실행과 이후 판독/시험 판본의 작성 시점·해시는 구분한다.

### 실제 입사 모멘트의 affine 경계와 반증 대조

fresh half 모멘트에서 `h=H+/J+`, `n=N+/J+`, `Hinc=h*J−−H−`, `Ninc=n*J−−N−`를 직접 계산했다. H−/N−는 signed 적분값이다. 현재23732의 값은 h`.1653440032`, n`.0669090212`, Hinc`4.6772770330e-15`, Ninc`2.3386295298e-15`다. 누적 NPLUS/NMIN 값을 사용하지 않는다.

고정된 원래 Q/VEF에서 경계의 H/N을 `H=hJ−Hinc`, `N=nJ−Ninc`로 치환하면, native 외곽 우변에 `γR²{P_H Hinc+Δ[Hinc−Hinc_prev+(γ²(σ+1)−1)(Ninc−Ninc_prev)]}`가 추가된다. γ²R²가 아니라 **γR²**다. 독립 원시식과 소거식의 부호·이전 주파수 항을 시험했고, 실제 두 입력에서1e-16 수준으로 일치했다. 반환 H도 `hJ−Hinc`로 고쳤다.

- 잘못된 old MOM 상류 이력을 그대로 두면 affine 경계만 국소 적용해도 물리적으로 불가능한 해가 남았다. 이 후보는 채택하지 않았다.
- 실제23730의 비음수 광선에서 얻은 상류 모멘트를 주고23731→23732를 연결하면 legacy도 양수/모멘트 실현 가능성을 만족한다. 그러나 바깥쪽 J/formal은`.996278→.990808`이었다. affine는`1.000383→1.000640`으로 가까워졌다. 두 포획의 raw 상태 차이는 최대1.66e-13 수준이었다. 이 결과는 `affine-boundary-response-v2.json`에 있으며 affine를 전주파수 실패의 유일 원인으로 확정하지 않는다.
- 이 clean 상류 주입은 **국소 원인 분리용**이다. 향후 whole outer60에는 중간 formal 모멘트를 주입하지 않고 실제 첫 주파수부터 native 광선/MOM 이력을 연결해야 한다.

### native 구현과 공간기울기·첫 내부 flux의 추가 수리

`scripts/prepare_cmfgen_affine_repair.py`는 별도 private library/실행파일만 만든다. `LUMINA_AFFINE_BOUNDARY`가 최초 주파수와 동일 주파수 재호출, 서로 다른 이전 주파수를 구분하고 fresh h/n/입사상수를 전달한다. 실제 native 단독 MOM도 clean 상류에서 두 호출을 값 대체 없이 반환했다. 다만 최초 `affine_v1`은 공간기울기에 raw H/J를 남겨 둔 후보였다.

Astra의 묶음 평가(agent `01a1538b-dcaf-7001-8160-f5c9acdfcf66`, read-only 완료·종료)는 affine 주파수 경계항의 부호와 γR² 스케일을 지지하면서, `VdHdR_TERM=β(H/J)_formal`과 Q의 lag를 별도로 지적했다. 운전자가 실제 native에서 확인한 미수리 기울기 결함은 두 호출에서 행 scaling 대비`3.63e-5/6.11e-5`였다. **양의 해와 작은 frozen 원식 잔차만으로 이 첫 후보를 완료 처리하지 않았다.**

후속 `affine_spatial_v1`은 Q를 원래 lagged VEF로 고정하되, 모델 표면의 실제 `K+βH` 공간 flux를 affine으로 치환했다. 경계의 계수/상수만 바꾸지 않고 첫 내부 flux에도 같은 상수를 넣고, 그 flux가 들어가는 첫 내부 보존행에는 반대 부호로 반영했다. 구현은 Q·광학깊이를 만든 뒤 경계의 `VdHdR_TERM(1)=βh`를 사용하고, 외곽 우변의 `βγR²Hinc Q/Δτ_H`, 첫 내부 flux의 그 값`/(P_H+W)`, 첫 내부 보존행의 음의 상수항을 함께 추가한다. 이류 비활성일 때 추가 공간항은0이며, 다른 물리·격자·미분·반복 한도는 그대로다.

실제 native 두 호출의 결과(`native-affine-spatial-check.json`):

| 항목 |23731|23732|
|---|---:|---:|
| 내부 반복·종료 |13회·정상|10회·정상|
| 독립 조립 대비 행렬/우변 상대차 |≤1.99e-16|≤2.20e-16|
| 비선형 원식 잔차 |8.29e-10|3.67e-10|
| 독립80자리 해 상대차 |1.96e-9|7.54e-10|
| 실제 affine 공간 경계 원시식 잔차 |2.09e-17|6.42e-17|
| signed H 반환 |−2.50136e-15|−2.53865e-15|
| J/formal, 바깥쪽 |1.00033284|1.00057943|
| edge 모멘트 최소 고유값 |.0070353|.0066450|
| 표면 모멘트 최소 고유값 |.133576|.131278|

두 해 모두 H 반환이 affine 식과 정확히 같고, 첫 내부 flux는 실제 연산의 IEEE 상한 안에서 재현됐다. 표면 `J−Jminus≥0`도 만족했다. native 기존 반복 종료치1e-8을 변경하지 않았으므로 그 결과를80자리의1e-16 해와 동일 정확도라고 주장하지 않는다. Q의 raw/affine 경계 integrand 차이는 약`−7.22e-12/−1.13e-11`이고 첫 간격과의 곱은`~1e-11`이었다. **이는 두 입력의 점 진단이지 전체 Q lag 오차의 상한 또는 전주파수 수렴 증거가 아니다.**

### 검증·보존·다음 실행의 조건

- 최종 직접시험은 **28 passed**다. 실제 Fortran에서 최초/동일/다음 주파수의 입사상수 전달, 초기화·진공 극한·bad history/지지 거부, 누적 N 값 미소비, 원시식/소거식과 첫 내부 flux 보존을 확인했다. 첫 공간 단위시험의1 ULP 차이는 절대1e-14 비교가 너무 작아 발생했다. 실패 로그를 보존하고 실제 항 합에 따른 IEEE 반올림 상한을 사용한 최종 시험을 별도 로그로 남겼으며, 물리·수렴 기준을 완화하지 않았다.
- 전체 source/compiled source/바이너리·입력과 이전 오류 판본은 새 private 경로에 보존했다. 초기 replay 및 공간항 전 Python 판독 소스도 기록된 SHA256과 정확히 같은 판본으로 `affine_spatial_v1/source_snapshot`에 보존했다. 예전 보고서의 source hash를 현재 수정 파일의 hash로 바꾸지 않는다. 변경은 미커밋이며 새 push는 없다.
- Astra는 단독 검증 후 CPU16/OMP16·16GiB·GPU0·600초·1회·retry0의 whole outer60 pre-pop 진단을 조건부 제안했다. 필요한 증거는 첫 주파수부터의 실제 이력, native 보정/비유한값/실현 불가 최초 종료, 원식·formal/MOM 차이·표면과 edge 모멘트·Q lag의 같은 위치/좌표계 판독이다. K/J 변화만 작다는 이유로 통과시키지 않는다. 그 진단은 아직 구현·제출하지 않았다.
- 자원 문구의 전제도 정확히 해야 한다. 기존 OUTGEN은 수송 전에 약5.99e8 words의 BA pool 할당을 기록하며 실제 MaxRSS는약5.44GiB다. 따라서 「어떤 population 배열 할당보다 먼저 종료」라고 표현하면 사실과 다르다. 보존 가능한 범위는 이 startup pool을 포함한16GiB 안에서, **SOLVE_FOR_POPS 및 인구/온도 갱신 전** 종료하는 것이다. 93GiB급 full-linearization 계산으로 자동 전환하지 않는다. 향후 전체 수렴 예산128GiB/14시간은 Astra의 별도 제안일 뿐 실행 승격/제출 기록이 아니다.
- 현재 완료된 것은 두 실제 주파수의 입사 전달 수리 후보와 native 국소 검증이다. 실제 전주파수 이력·공동수렴 기준해와 새 Lumina 분포/스펙트럼은 남아 있다. 기존 최대 Si III/S III 차이16.28/13.93%p, Si0/90·S90/90 채택, 원래 최종 허용치와 전체 동질성 목표는 **변경되지 않았으며 계속 미완료**다.

### 실제 첫 주파수부터 연결하는 확인 — 실행 범위 개정

실제 OUTGEN의 전체 주파수는 **197,291개**다. 위600초 제안은 앞선23,732개 포획213.77초 및34,540개264.54초와 맞지 않아 폐기한다. Astra(agent `01a153a5-3de0-72f1-8aa4-1dfd0b21fb57`)의 묶음 계획 평가는 **CPU16/OMP16·16GiB·GPU0·부모45분/자식40분·노드 미고정·1회/retry0**를 타당하다고 판단했다. 예상25–35분은 추정이지 완주 보장이 아니다. 시간초과·OOM·출력상한 도달을 성공 또는 자동 연장 사유로 바꾸지 않는다. Astra는 read-only 계획·평가만 했고 구현·시험·실행은 운전자가 맡는다.

- **목적·결정:** 같은 StageA59에서 수리한 affine 입사 경계와 공간/첫 내부 flux가 처음부터 실제 광선·MOM 주파수 이력으로 연결되는지 확인한다. 첫 물리적 실패의 정확한 상태를 확보하거나 전주파수 수송을 마친 뒤 인구 갱신 전에 끝낸다. 결과는 fixed-T 안정화로 이어갈 수 있는지의 근거이며 최종 이온분포·스펙트럼 합격을 대신하지 않는다.
- **보존 입력·제외:** seed `toy06_19p48d_stageA_fixedT_427702`, 기존lambda4 출발 해시, 모든90깊이와 전체 주파수·물리/기하 설정을 유지한다. NUM_ITS=4/DO_LAM_IT=T지만 첫 스윕의 SOLVE_FOR_POPS 전에 멈춘다. 중간 formal 모멘트 주입, split 외부 이력 재사용, native J/H 대체, floor, 기준 완화, full-Jacobian 승격은 없다. startup BA pool은 이미 할당되며16GiB 안에 포함한다.
- **관측·판독:** 모든 실제 MOM 호출의 원식·affine 경계·첫 내부 flux, 같은 위치의 J/H/K/N 실현 가능성, 표면 J−Jminus, formal/MOM 차이와 Q lag를 계산한다. 모든 실제 formal 호출의 사용 강도를 검사한다. 비유한·음수·실현 불가·원식/flux 실패는 첫 호출에서 보존 후 종료한다. 주파수별 마지막 재호출은 별도 CSV로 구분하며 중간 호출의 극값·실패 phase도 남긴다. 기존1e-8 반복 종료치를 바꾸지 않고 K/J 변화만으로 joint closure를 선언하지 않는다. PSD와 flux 반올림 허용은 IEEE 연산 상한뿐이다.
- **출력·재현:** 추가 요약≤64MiB, 최초 snapshot≤256MiB, 기존 파일당8GiB 상한이다. private `whole_v5`와 새 run root에만 쓴다. 실패 MOM은 원래 인자/모듈과 입사상수7개를 함께 보존하며 `LUMINA_MOM_CAPTURE_MODE=REPLAY_FIRST LUMINA_WHOLE_MODE=1 BUILD/sweep_mom_replay.exe RUN/MOM_CAPTURE.bin RUN/AFFINE_INPUT.bin`으로 주입 없이 재현한다. 입력·runner·소스·실행파일 해시를 봉인한다. 원본 native source/library/exe와 기존 모든 출력은 그대로 보존한다.
- **완주 조건:** 197,291개 연속 coverage, 첫 실패0, 정상 pre-pop STOP97이 있어야 연결 진단 완료다. 마지막 formal/MOM 차이·실제 Q/gradient 폐합은 따로 판독하며 frozen-Q 원식 통과를 실제 closure 수렴으로 승격하지 않는다. 수렴 released-T 기준해, 새 자기일관 Lumina 및 Si II/III·S 분포/스펙트럼 동질성 목표는 여전히 미완료다.

구현은 `scripts/prepare_cmfgen_affine_sweep.py`, 실행은 `scripts/run_cmfgen_affine_sweep.py`와 `scripts/submit_cmfgen_affine_sweep.slurm`이다. 제출 명령은 syntax의 `sbatch --parsable BUILD/runner/submit_cmfgen_affine_sweep.slurm BUILD/runner BUILD`이며, 직접시험과 기존 두 주파수 결과 불변을 확인한 뒤1회만 제출한다. 초기1–3 삽입점 판본은 중복 native 구간을 거부하고 실행파일 생성 전에 중단했으며 빈 private 디렉터리로 보존한다.

**실행 착수:** private `whole_v5`가 빌드됐으며 직접시험 **36 passed in8.85s**다. 기존 실제 두 주파수의 해와 MOM 반복 trace가 관측 코드 추가 후에도 전체 바이트 동일했다. 실현 불가능한 모멘트 음성 대조, 누락 coverage 거부, 정상/비정상 pre-pop 종료와 F 반복 상한 판독도 직접시험했다. `whole_v5/direct-tests.log`와 해시로 봉인한 `whole_v5/runner`의16개 파일을 보존했다. syntax Slurm **428641**에 위45분/16CPU/16GiB/GPU0 범위로1회 제출했다. 새 run root는 `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_affine_sweep_428641`이다. 제출 사실은 전주파수 완주나 물리 합격 증거가 아니다.

**최신 역할 변경:** 사용자가 Fable을 **검증 2**로 지정했다(2026-10-19). 운전자의 구현·직접시험과 Astra의 기획·계획·평가는 유지한다. 과거 Fable 감사 절차 및 Q-GOAL/Q-LEAN을 복구하지 않는다. 정본은 AGENTS.md와 CLAUDE.md 맨 위다. 이 변경만으로 봉인한428641의 물리 입력·기준·자원 한도를 바꾸지 않았다.

제출 직후428641은 syn01/a10에서 RUNNING이었다. scheduler 기본 Requeue=1을 발견해 `scontrol update JobId=428641 Requeue=0`을 성공 적용했다. 이후 제출 소스에는 `--no-requeue`를 명시했다. 이미 제출한 sealed runner의 해시는 바꾸지 않았으며, 이 제어 변경은 controller override로 구분한다. native 물리·입력·시간/메모리 한도 변경이나 재제출은 없다.

**Fable 의뢰 취소(최신 사용자 지시):** 검증 2의 묶음 요청은 `docs/VERIFY2_AFFINE_BOUNDARY_REQUEST_2026-10-19.md`, 기록은 `whole_v5/verify2_fable`에 보존했다. `--bare` 호출은 인증 부재, 별도 안전 모드 호출은 인증 후 HTTP429 `credits_required/model_requires_usage_credits`로 각각 종료됐다. 두 호출 모두 API 입력/출력 토큰0·모델 검증 미수행이다. 이후 사용자의 「fable의뢰는 취소」에 따라 해당 의뢰를 취소했으며 진행 중인 Fable 호출은 없다. 실패를 검증 판정으로 꾸미거나 다른 모델에 대체 의뢰하지 않는다. 재지시 전 추가 Fable 호출은 하지 않는다. 운전자 구현·직접시험과 Astra의 기존 계획·평가 배정은 바꾸지 않는다.

**428641의 첫 정지 재분류:** 실제 실행은156초/자식138초·MaxRSS5,704,200K에 outer60/index1/JIT0의MOM20회 반환을 보존하고 STOP95로 끝났다. 추가 snapshot102,628bytes, 인구/T 갱신0, 완주0/197291이다. 최초 관측의 PSD 최솟값−.161420은 **native의 초기화 시 비소비 KMID/NMID 계수를0으로 지운 값을 물리 모멘트로 읽은 관측 오류**였다. `scripts/recover_cmfgen_initial_closure.f90`로 같은 raw 상태에서 원래 GET_MOMS_REL을 다시 실행하자 실제 closure의 최솟값은+.126809였다. 독립80자리 active-branch 해의 잔차1.49e-16·native 상대차1.80e-9·동일 미분 분기 및 PSD+.126809도 확인했다. 이를 native 물리 실패나 경계 수리 실패로 표시하지 않는다. 재현 root `/gpfs/kjhan/cmfgen_runs/lumina_sweep_first_replay_20261019-62k8onh5/initial-closure-analysis.json`이 정본이다. entry/event는 바이트 동일했고 kernel의 차이는 P_J·LU_MOM·RMID_STORE tail·N_ON_J_SAVE로 구분됐다. 이 비소비/초기화 차이를 모두 동일 상태라고 숨기지 않는다. 다음 내부 작업은 관측이 INIT 이전 실제 closure를 사용하도록 고치는 것이다. 아직 수정 후 전주파수 계산을 제출하지 않았으며, 첫 JIT의 formal/MOM 약50% 차이와 Q lag를 수렴 해의 오차라고 해석하지 않는다.

**관측 수정 — whole_v6:** GET_MOMS_REL 직후의 실제 KMID/NMID를 관측 사본에 보존해 INIT의 비소비 영계수와 구분했다. 원래 MOM 계수·반복·스펙트럴/공간 경계 수리는 바꾸지 않는다. private `whole_v6`와 직접시험 **38 passed in8.43s**를 보존했다. 새 음성 대조는 계수를 보존한 뒤 원본을0으로 지워도 물리 사본이 유지되는지 확인한다. 실제428641의 같은 첫 entry/입사상수를 새 standalone native에서 재현하자 정상 반환했고 J/H가 이전 포획과 바이트 동일하며 PSD>+.12였다. 기존 두 주파수의 해/trace도 바이트 동일했다. 이 실제 반례를 고친 후에만 새 연결 실행1회를 진행한다. 무변경 실패 작업의 자동 재시도가 아니라 확인된 관측 버그의 수정본이며, 자원과 물리 입력·수용기준은 위45분/16CPU/16GiB/GPU0·retry0 그대로다. 원래 f2b9afcc… 실행파일과 모든 이전 포획/판본은 보존한다. 이후 runner는 첫 정지를 원인 분류 전 `first_observed_failure`로 부르며, 관측 실패를 곧바로 물리 실패라고 확정하지 않는다.

수정본은 **syntax Slurm428645**로1회 제출했다. run root는 `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_affine_sweep_428645`, build는 `whole_v6`, 봉인 runner는18파일이다. 이번 제출에는 `--no-requeue`를 처음부터 포함했다. 계측 오류의 실제 반례 수정이 유일한 재실행 사유이며 중간 formal 주입·기준 완화·새 물리 변경은 없다. 실행이 끝나면 마지막 JIT의 실제 closure와 전주파수 coverage부터 판독한다. 현재 최종 이온분포·스펙트럼 동질성이나 기준해 수렴은 미완료다.

### 428645 결과 — 극미량 꼬리의 상대차와 최종 물리 판정을 구분

- 실제 연결은 **2088/197291주파수, MOM3752호출**까지 끝난 뒤2089/JIT0/MOM7회의 관측PSD−.00812478로STOP95했다. 총156초/자식149.50초, MaxRSS5,687,188K, 추가snapshot233,782bytes, 인구/T갱신0이다. native J/H 대체나 반복 상한 실패가 아니라, 원래 F-loop 종료 전의 관측 중단이다. 직전2088/lastJIT1은F변화9.44e-8·원식2.70e-9·PSD+.00342848이지만 formal/MOM J차이최대.602578이었다. 작은K/J변화와 같은 조립식의 잔차를 공동수렴 증거로 쓰지 않는다.
- **절대량이 결정적이다:** 2089의worst depth71은J_MOM1.38788e-279/J_formal3.47760e-279, 전깊이J최대도7.79372e-247이다. 실제 부모comp_opac의ETA/ETA_NOSCAT≥1e-280 및외부확장ETA_EXT≥1e-280은원래존재하며최근제거한1e-3ESEC opacityfloor와별개다. ETA와J는다른양이므로비슷한규모만으로floor가PSD직접원인이라고확정하지않는다. 이60%를Si/S차이나전복사장가중오차60%로확대하지않는다.
- `/gpfs/kjhan/cmfgen_runs/lumina_sweep_2089_replay_20261019-rgb53nq2/event-analysis.json`에 entry/kernel/event바이트동일, 독립80자리원식잔차1.07e-16·native차1.76e-9·동일미분분기, 실제PSD와절대J를보존했다. formal의내부Iplus=B+mu*dBdtau와MOM의H=DBB/(3CHI)는같은상태에서실제로다르다. half_J합은rawJ와동일/half_H합은약2e-15차이로신원을확인했다. 그러나row-only affine국소대조는innerJ비율1.07837→1.03015만바꾸고depth71의.399091비율/PSD위반을전혀해소하지못했다. 그진단을일관된경계수리로채택하지않는다.
- **Astra의묶음평가 보완:** agent `01a153cc-6afc-76a1-9f9b-a0df7fb34611`는새절대량·대조원자료를읽고이사건에대한내부경계우선수리제안을철회했다. 극미량중간JIT의PSD를무조건적인실행차단/최종물리실패기준으로계속쓰는것은v2보다강한추가조건이다. 최초사건기록은보존하되격리진단에서원래F-loop종료까지보고raw위반·절대량·lastJIT를함께분류한다. v2의J비교Δm·Δlnν가중치는변경하지않고, 양쪽활성합집합의기여>1e-4를개별νbin삭제문턱으로오용하지않는다. 실제σJ/hν와열가중차이는기존율예산에병기하되상쇄없는절대차예산도구분한다. 아직전체기여가없으므로비활성확정/전파무해를주장하지않는다.

### 다음 내부 작업 — 같은 주파수의 원래 종료 상태 확보

현재MOM모멘트만으로광선별PREV를유일하게복원할수없다. 따라서원래물리/분기를바꾸지않고현재최초사건에서만실제formal인자·현재/PREV광선·기하/가중치와FOLD/JIT/ACC_EDD_FAC/COMPUTE_EDDFAC/산란분기/소스배열을붙여한번포획한다. rawformal입력은호출자가TA를MOM소스로덮기전에저장한다. 사건시SAVE는이미현재광선이므로입력writer에는NEW_FREQ=false를명시해실제PREV를선택한다. incoherent일때만RJ_ES를읽으며coherent의미소비배열을입력으로꾸미지않는다. 누적NPLUS/NMIN은사용하지않는다.

범위는 **CPU16/OMP16·16GiB·GPU0·부모600초/자식500초·1회/retry0**, 포획후offline CPU1/4GiB/300초다. snapshot256MiB/요약64MiB상한과기존입력·소스·exe보존을유지한다. private `event_v1`은이최초상태에붙이는진단자료만늘리며전체재시도·내부경계교체·floor변경·반복수증가·새인구주입을하지않는다. 이후격리재현에서기존STOP95지점의MOM반환다음원래문장부터NEW_FREQ/counter/FOLD/종료판정을진행하고,원래조건이요구할때만후속formal→MOM을수행한다. 중간PSD가회복되면중간상태위반,끝에도남으면종료꼬리결함,복원/자원/원래상한실패이면lastJIT미확정으로기록한다. 어느경우도전체수렴·Si/S동질성PASS가아니다. Fable취소는유지하고Astra평가를검증2대체로부르지않는다.

실행 판본은 **event_v2**다. `event_v1`의 첫시험은 과거formal모듈에없는새split심볼을시험replay가참조한fixture오류1건/나머지38통과였으며그로그를보존했다. v2에서는실행당시formal replay소스를재사용하고,입력raw모듈을사건시점이아닌실제formal진입시점에기억하도록명확히했다. 최종 **39 passed in8.47s, skip0**이며원래입력packet·PREV광선·현재광선·기하가모두기존캡처와바이트동일한음성/왕복시험을포함한다. INITIALIZE의미할당raw입력을만들어내지않고비초기사건만포획한다. 새진단은21파일runner로봉인했고원래한주파수F-loop만진행하는 `replay_cmfgen_event_floop.f90`와별도링크신원 `event_v2/floop-replay-build.json`을보존했다. 이replayer는격리cwd에서기존PSD관측중단만비활성화하되nativeJ/H대체·100회실패중단은유지하며,새인구나추가강제반복을넣지않는다. 실제packet연결재현은포획뒤확인해야한다.

**syntax job428691**로600초상한을명시해1회제출했다. syn02/a10의RUNNING, CPU16/16GiB/GPU0·Requeue0·TimeLimit00:10:00를controller에서확인했다. run `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_affine_sweep_428691`, build `event_v2`다. 다음재현명령은새격리cwd에서 `LUMINA_MOM_CAPTURE_MODE=CAPTURE_FIRST LUMINA_WHOLE_MODE='' BUILD/replay_event_floop.exe RUN`이며,최초formal광선/기하와MOM의실제캡처정체성을먼저대조한다. CPU1/OMP16의모멘트합산오차와후속원래반복의lastJIT를구분한다. 이작업은전체연결다시시도나최종수렴실행이아니다.

### 실제 원래 반복 종료까지의 결과 — 극미량 꼬리 결함은 남는다

428691은같은2089/JIT0를총약168초/자식150.02초에포획하고의도한STOP95로종료했다(회계2분50초, MaxRSS5,701,248K). `event_context_complete=true`, 추가snapshot1,358,110bytes다. 원래formal입력/출력·실제PREV/현재광선·기하·F-loop상태가모두있으며인구/T갱신0이다.

`/gpfs/kjhan/cmfgen_runs/lumina_event_floop_replay_20261019-tf7_hekf`의격리CPU1재현은종료0, 기하·최초광선바이트동일, 첫MOM의J/H도포획값과바이트동일이었다. 원래제어는COMPUTE_EDDFAC=false·coherent=true·ACC_EDD_FAC=1e-4다. 첫F변화4.4718938657e-4로원래조건이요구한두번째호출만진행했고, JIT1의변화8.6833315094e-8에원래순서대로종료했다. 강제세번째호출·반복상한확대·인구주입·native대체는없었다.

| 항목 | 최초JIT0 | 실제종료JIT1 |
|---|---:|---:|
| 새원식비선형잔차 |4.85e-10|2.51e-9|
| edge70 정규화PSD최소 |−.008124780877|−.008124780861|
| formal/MOM J최대상대차 |.60090930716|.60090930666|
| worst depth71 J_MOM |1.38787709144e-279|1.38787709145e-279|
| 같은위치J_formal |3.47759824106e-279|3.47759823669e-279|

따라서이번PSD는단순히다음JIT에서회복되는중간상태위반이아니라 **native 종료에도남는극미량꼬리의결함**이다. 반면Si/S의활성물리원인또는전주파수영향으로확대하지않는다. 기존율·열예산과연결한정량기여및더적색인구간의실제연결은아직미확정이다. 이자료를근거로안쪽경계수리/floor제거/격자전체재작성에착수하지않으며원래v2수용기준을완화하거나해당ν를J통계에서삭제하지않는다.

정본은 `floop-analysis-v2.json`이다. 최초FORMAL의CPU1출력J와복원한원본OMP16 rawJ에는byte차이가있지만, 실제광선/가중치의보상합대비IEEE상한분율은각각.042365/.021182로검증됐다. 이는물리오차4.24/2.12%라는뜻이아니다. 첫분석v1과정확한producer소스를보존했고후속v2에서이합산차이를명시적으로분리했다. 마지막JIT의FORMAL인자J와rawJ는byte동일하다. 회귀는원래두번호출·세번째호출없음·최초J/H동일재현까지추가해 **40 passed in8.59s, skip0**다. 코드/재현의통과는Si/S동질성합격이아니다.

현재본건native작업428641/428645/428691과Astra평가agent는모두종료했다. Fable의뢰는취소상태다. 다음필수작업은이극미량raw진단과최종수용판정을혼동하지않는연결계획안에서, 실제활성율/열과더적색구간에관련된복사장·인구불일치를계속해결하는것이다. 새CMFGEN수렴앵커/자기일관Lumina분포·스펙트럼은여전히없고Si0/90·S90/90·최대III차이16.28/13.93%p는그대로다. 원본native source/library/exe와모든기존출력은보존했으며새commit/push는없다.

### 후속 묶음 — 같은 상태의 전주파수 J60과 Si/S 율 반응

Astra의후속계획평가1회(agent `01a153cc-6afc-76a1-9f9b-a0df7fb34611` 재개)는같은StageA59물질상태에서 **전주파수J60을확보하고그장이실제Si/S광율·가열에주는변화를판독**하는묶음을타당하다고판단했다. 기획·평가만수행했으며Fable검증2대체가아니다. 꼬리재포획/내부경계교체/legacyfloor재작성/반복수증대는추가하지않는다.

- 같은seed/NUM_ITS4/DO_LAM_IT=T·전체90깊이/197291주파수·private affine공간수리와모든물리/격자/해석기준을보존한다. 첫주파수부터실제이력으로이어SOLVE_FOR_POPS전에종료한다. 새인구/T갱신은범위밖이다. 원본source/library/exe/부모출력은불변이며새runroot만쓴다.
- 관측PSD(code3/4)·추론한outgoing J−Jminus의support위반·formal/MOM상대차·Qlag는원값/절대량/위상/마지막JIT를남기며연결진단을계속한다. 강도cutoff·ν삭제·PSD허용치확대는없다. 실제J/실제광선의음수·비유한값, nativeJ/H대체시도·100회실패, 원래식/경계식/flux허용실패, 원래F상한미수렴, coverage/입력정체성/자원오류는보존후중단한다. signed H/N의음수자체는실패가아니다. PSD가먼저발견됐어도원식/flux/F실패를가리지않게우선처리한다.
- 출력은lastJIT CSV와중간호출의집계/단일최악위치·절대량으로제한한다. 상세행/PSD별snapshot을증가시키지않는다. 최초실제중단사건에만기존completeCTX를붙인다. 모든연속coverage+prepop종료는diagnostic_complete일뿐closure/물리PASS가아니다.
- 예산은CPU16/OMP16·16GiB·GPU0·부모45분/자식40분·노드미고정·1회/retry0·Requeue0, 최초사건256MiB/요약64MiB/기존파일당8GiB상한이다. 완주예상은보장이아니며자동연장/재실행은없다. 과거128GiB/14시간released-T제안은실행으로승격하지않는다. rollback은private새출력보존과실패원인분류이며기존run/소스에되쓰지않는다.
- 완료시새EDDFACTOR의J60과보존된부모J59를 **같은SCRTEMP59의인구/T/ne** 에서읽고,기존PHOT/BF평가기함수로SiII/III·S의광율/광가열변화와상쇄없는절대차를판독한다. 기존최대위치Si28/S35부터해석하되최종v2 J의Δm·Δlnν가중통계/활성합집합은바꾸지않는다. 단순최대상대차로정확한σ가중formal−MOM차이를계산했다고주장하지않는다.

**출력정체성 주의:** outer60은LST_ITERATION=false이므로WRITE_RATES=T여도새PRRR/GENCOOL은없다. SCRTEMP/RVTJ는복제한59의물질상태이지새인구출력이아니다. prepop의STEQ_VALS에는새RE/전자에너지식은있지만전체Si/S SE배열은SOLVE_FOR_POPS안에있어아직없다. EDDFACTOR는매주파수새MOM J를기록하고R/V/좌표를flush하나COMPUTE_EDDFAC=false이면finish flag는부모값을물려받을수있다. 따라서새field신원은이번fullcoverage/정상prepop종료/주파수대응을함께사용하며부모finish flag만믿지않는다. 미완주EDDFACTOR의나머지부모record를새J60으로소비하지않는다. PRRR을요구하는기존owner준비CLI를그대로실행하거나마지막iterationflag를조작해출력을만들지않는다.

구현판본은private `whole_observe_v2`다. 기존code4에같이있던추론한outgoing sign을rawcode8로구분하고PSD3/4와같이관측계속대상으로분리했다. 실제광선/J의음수와혼동하지않는다. 원식5·flux6·원래F미수렴7이rawflag에가려지지않게했고, COMPUTE_EDDFAC=false에서20회상한종료시남은F오차도구분한다. 중간호출은고정크기집계와최악위치/물리절대량만저장하고lastJIT CSV는3개절대J/2개위치열을추가해전체197291행도64MiB안에든다. 치명적첫사건의phase와직전완료검사의phase를별도로남긴다. 초기미산출값의관측사본은표시를초기화해이전호출의값을오인하지않는다.

직접시험 **44 passed in8.77s, skip0**다. 실제2089의PSD를유지한채같은MOM해/반복기록을바꾸지않고, 주입한원식/flux오류와F상한이여전히중단대상이되는음성대조를포함한다. rawflag가있는coverage완료를diagnostic완료로기록하되명시적으로NOT_CLOSURE_OR_PHYSICS_PASS를유지하는경로와CSV출력예산을검증했다. 빌드/21파일runner를봉인했으며원래f2b9afcc…exe와부모입력은보존한다. 첫구현whole_observe_v1은실행하지않았고, 미산출관측값초기화보완판v2만새실행에사용한다. Astra평가agent는계획기록후종료했다.

### 428713 결과 — 관측 PSD가 아닌 실제 100회 반복 상한

- **syntax job428713**, syn01/a10, CPU16/OMP16·16GiB·GPU0·Requeue0, 회계2분53초/자식165.45초·MaxRSS5,704,656K에 종료했다. run은 `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_affine_sweep_428713`, build는 `whole_observe_v2`다. 최초부터5690/197291주파수(Last JIT의PSD rawflag3494개 포함)를 완료한 뒤5691/JIT0의 MOM100회 미수렴(kind3)을 대체 없이 보존하고STOP95했다. 최초사건 snapshot1,358,110bytes·인구/T갱신0·자동재시도0이다. 부분 EDDFACTOR의 부모 나머지를 새 J60으로 소비하지 않는다.
- **재현 정본:** `/gpfs/kjhan/cmfgen_runs/lumina_affine_5691_replay_20261019-lSVfbZ/native-event-analysis.json`과 `.csv`. 같은 private standalone 실행에서 entry/kernel/event가 모두 바이트 동일했다. `native_trace/MOM_TRACE.bin`에는 원래100회 전체가 있으며 마지막행의 실제 matrix/RAW가 원사건과 바이트 동일하다. 생산 물리/입력/원본 native와 이전 자료는 바꾸지 않았다.
- 주파수는 `3.512818708720037e16 Hz`, 실제 광선과 모든 J는 비음수다. J최대 `9.54213e-30`이다. 마지막 native 변화는91회5.14725e-8→100회2.05228e-8로 감소하지만 기존 종료치1e-8에는 미달했다. 같은100회 frozen 선형식 잔차1.04e-16·80자리 상대차1.65e-15이고, fresh기울기의 비선형식 잔차는5.42e-10이다. affine 입사/첫 내부 flux를 포함한 독립 우변 재조립 상대차2.05e-16이다. 따라서 선형계 오풀이·J음수·PSD관측 차단으로 분류하지 않는다.
- **같은 원식의 독립 근:** native/formal 두 자연 초기값에서 전90점 양수, 비선형 잔차7.20e-17·80자리와≤4.03e-14·같은 limiter 분기·native H MAX위반0을 확인했다. SciPy의 `success=false`도 그대로 기록하며, 판정은 status가 아니라 잔차/독립 해/분기에 근거한다. 같은 분기의 log-gradient Picard 국소 수축률은 damping.8일 때.90288, damping.1일 때.98786이다. 이는 느린 원래 반복이100회 상한에 걸린 증거이지 물리적 양의 근 부재가 아니다.
- 이 고에너지·작은 절대량의 반복 문제를 최대 Si III/S III 차이16.28/13.93%p의 원인으로 확대하지 않는다. 전체 J60→실제 Si/S 율·열의 반응이라는 묶음 목표는 아직 달성하지 못했다. 새 복사장 비교기 `scripts/compare_cmfgen_field_response.py`는 작성 중이며 실제 결과/검증 완료 도구로 인계하지 않는다. 현재 추가 전주파수 작업은 없고, 같은 원식을 더 잘 푸는 최소 구현 방안을 Astra에 한 번 기획·평가 요청했다. Fable 취소는 유지한다.

**Astra 계획·평가 완료:** 같은 agent의 읽기 전용 평가1회는 위 증거를 느린 Picard 수축으로 분류했다. 100회cap은 정당한 중단이며 기준을 낮추지 않는다. 후속 구현은 native90회까지 원래 연산을 유지하고 미수렴일 때 같은 LINMON 분기 원식의 Newton 후보를1회 푼다. 실제 DERIVCHI 분기·양수/유한성·원식 잔차·H 검사를 만족한 후보의 **기울기만** 다음 조립에 전달한다. 후보 자체를 해로 반환하거나 JOLD를 덮지 않는다. 후보풀이1회도100회예산에서 차감하고, 원래THOMAS/변화1e-8/J/H/원식/flux/F-loop종료까지 실제로 확인한다. 후보 거부 시 남은 원래 반복으로 진행하고 최종cap실패는 그대로 보존한다.

반증시험은 실제5691의 발동 전 native trace불변·후보 fresh분기와독립근·100내 원래종료 및같은주파수원래F-loop, 기존첫주파수/2089/두정상호출의미발동byte불변, 잘못된분기/음수/비유한/H위반/작은변화만있는wrongroot거부를 한묶음으로 수행한다. 실행환경CPU1/4GiB/300초를유지한다. 미발동 정상경로의byte불변만주장하며,90회이후개입한모든미래정상경로까지byte동일하다고확대하지않는다. 직접시험통과후에만 같은전주파수연결1회(CPU16/OMP16·16GiB·GPU0·45분/자식40분·노드미고정·retry0)를진행한다. 전체목표와전주파수증거/부분EDD금지는변하지않는다.

**가속 수리 구현·격리 검증:** `scripts/cmfgen_mom_acceleration.f90`와 `scripts/prepare_cmfgen_mom_acceleration.py`를 추가했고 sweep builder의명시적 `--accelerate-late`에서만사용한다. 실제LINMON의현재분기는내부3점삼중대각이므로dense해석기/외부root반복을생산에넣지않고원래THOMAS1회로후보를구한다. 같은반복의affine/Q/PREV/closure/우변은고정하며단위보정/floor가없다. proposal풀이도ICOUNT에서1회를차감한다. 기본미발동빌드의원식은불변이다.

private `whole_accel_v1`의실제5691재현은첫90회trace전체byte동일,후보1회(예산91),원래native92·93회후변화<1e-8로정상반환했다. 92회JOLD는90회실제J,93회JOLD는92회실제J와동일하므로후보로변화량을위조하지않았다. 원식잔차1.76e-16/경계식4.54e-17/flux IEEE상한분율.04659이며독립원식근과차이<1e-8이다. 남은rawPSD−.34589/극소J의formal차이~.997은숨기거나수리완료라고부르지않는다.

정본replay는 `/gpfs/kjhan/cmfgen_runs/lumina_affine_5691_replay_20261019-lSVfbZ/accelerated`다. `floop/`에서는actualformal기하/첫광선byte동일,원래F변화.0496953→5.72763e-8로요구된JIT0·1만수행하고정상종료했다. 원래F상한을늘리거나강제추가반복을하지않았다. 최종직접시험 **53 passed in9.06s, skip0**다. 최초시험의1건실패는formal IO fixture가보존된구판formal_v1을기본선택해짧은후행byte를허용한것이며,그로그를보존하고현재strictformal_v2를명시한회귀로그를별도로남겼다. 새solver의실제53시험에는원식/분기/비유한/음수/H위반후보거부와기존실패우선순위가포함된다.

이구현은전주파수계산이막힌내부반복의수리후보이며Si/S목표완료가아니다. 다음한번의연결실행은수정된실제solver를검증한뒤의실행으로,실패상태무변경재시도와구분한다. Astra는이한정계획을평가한뒤종료했으며Fable취소는유지한다.

**새 연결 실행:** syntax Slurm **428716**을 `whole_accel_v1/runner`의25개해시봉인파일로1회제출했다. run은 `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_affine_sweep_428716`, build는 `/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/whole_accel_v1`이다. CPU16/OMP16·16GiB·GPU0·45분/자식40분·노드미고정·Requeue0·retry0과같은StageA59를유지한다. 이번실행의전체coverage/정상prepop종료를얻기전까지Si/S새장반응에소비하지않는다. 가속후보집계도종료증거에포함한다.

21:11:38KST 조회에서같은428716은syn02에서RUNNING(6분19초),실제연속lastJIT64217/197291까지진행했다. 기존5691과23732의중단지점을넘었다. 마지막ν1.775e15의depth49 J는1.16e-4/formal2.12e-4로차이가약45.3%였으므로전주파수차이를모두수치꼬리로축소하지않는다. 아직정확한σ가중영향/공동수렴/최종동질성판정은없다.

`compare_cmfgen_field_response.py`는같은59인구의850full준위/90깊이를대상으로raw PHOT의선형J광율·광가열함수를구한다. 전체native격자를쓰며기존1234bin영역으로자르지않는다. 문턱별열가중을분리하고양수nodal가중으로상쇄없는절대차상한을계산하되참해오차상한으로부르지않는다. 출력에인구·율·가열과입력/생산소스신원을남기며,250초예산으로미완료면부분합/미완료표시를유지한다. CPU1/4GiB/300초상한내에서만실행한다.

비교기사전시험은 **8 passed in1.42s**로analytic광율/열,기존raw적분기일치,부분EDD읽기금지,실제59인구/모델/PHOT/smoothing신원과4이온의실제PHOT채널을확인했다. 임의상수J를쓴마지막시험은평가기시험일뿐새물리결과가아니다. 실제링크입력은일반쉘redirect가있는setup_links.sh가아닌atomic_links.txt를사용하고같은인구복원의전하는stage−1을쓴다. source snapshot은위5691replay의`field_response_source`에보존했다. 실제새J60→율반응결과는아직없다.

### 428716 종료 — 실제 광장 구간에서 J/H 이력 불일치 확인

- 428716은 **66068/197291** 주파수 완료 뒤66069/JIT0/첫 MOM 반복의edge36 H MAX위반(kind2)을보존하고STOP95했다. syn02/a10·16CPU/16GiB/GPU0, 전체6분27초/자식382.45초·MaxRSS5,706,092K, snapshot1,358,110bytes, 인구/T갱신0이다. late기울기후보는4회모두허용됐고최후발동은27372/JIT0이므로이번실패위치에서가속을발동하지않았다. 현재이묶음의live Slurm은없다. 부분EDD소비/무변경재시도는하지않는다.
- 새원인분리 정본은 `/gpfs/kjhan/cmfgen_runs/lumina_affine_66069_replay_20261019-Pyi0PH/native-event-blue-response.json`이다. 원래실행과독립native replay의entry/kernel/event는모두byte동일이다. 실제주파수1.6378328249644118e15Hz, J최대.00218018·최소2.46787e-5로모두양수이며입력/출력실제광선도비음수다. 이전극미량수치꼬리와혼동하지않는다.
- 첫선형식잔차1.38e-16/mp80차이1.79e-15이고, 같은비선형원식의두초기값해는양수/잔차≤2.52e-14/mp80≤5.02e-14/같은분기다. 그정확한근도edge36에서H/max인접J=1.003151, H/Jmid=1.187513·K/Jmid=.215752·N/Jmid=.117033, PSD최소−1.02664로실현불가다. 따라서반복상한수리나수렴조건완화로고칠수가없다. 이관측H중단은유지한다.
- 같은위치의직전주파수값은MOM J=.000147099/실제광선 J=.000300358로약2배갈린다. 현재formal의J를현재MOM원식에그대로평가한잔차도.33461이다. **현재기하/물질/Q/closure/affine원식은고정한** 한주파수반응에서직전J만또는H만실제blue광선값으로대응시키면H MAX는통과하나PSD는각각−.32772/−.31139로여전히실현불가다. 직전J/H를함께대응시키면전90점양수/원식7.62e-15/mp80차이1.53e-14/같은분기/PSD+.00321063/H MAX위반0을얻었다. 이는실제blue광선과MOM history의불일치가이번국소실패에인과적으로관여한증거다.
- 위반응은기존출력을다음history에주입한생산수리가아니다. formal값을원하는답으로강제하거나native H대체를켜서계속하지않는다. 서로다른이산수송에서이력이갈리는기작과K/J-only 종료/Q·J·H 공동정합문제를구분할후속묶음계획을Astra에1회요청했다. Fable취소를유지하고다른모델로검증2를대체하지않는다.
- `scripts/analyze_cmfgen_affine_native_event.py`는같은원식분석을H사건에도적용하도록확장했다. 최초분석v1과blue반응v2의producer를별도보존했고v1은보고서source SHA256 `67d8fec28f85e5739ad207cc4f83e2d0ef5ea0df19048079ba5de751a77839d6`와일치한다. 단일H MAX통과만으로실현가능성통과를대체하지않는반증회귀를추가한다.
- 전체J60/Si/S율·가열반응묶음은미완료다. 수렴released-T 기준해와새자기일관Lumina분포/스펙트럼·최종동질성도미완료다. 기존Si0/90·S90/90·최대III차이16.28/13.93%p와원본/이전자료/미커밋변경보존·새commit/push없음을유지한다.

### 후속 원인 분리 — 실제 두 blue 이력 고정의 공동 반응

Astra의 읽기 전용 기획·평가1회가 완료됐다. H/J·N/J·Q가 전혀 갱신되지 않는다는 해석은 수정한다. 각 formal 재호출 뒤 GET_MOMS_REL/MOM 진입에서 갱신되지만, MOM 내부는 J 기울기만 반복하고 바깥 종료는 K/J 변화만 본다. **공동 반응 부족인지, 공동 반응 뒤에도 남는 이산 수송 불일치인지** 먼저 구분한다. 평가 agent는 이 계획 뒤 종료했으며 추가 모델 호출은 없다.

- 같은66069의 실제 ray blue와 실제 MOM blue를 각각 고정한다. trial마다 관련 PREV/SAVE·주파수·affine 이전 상수를 복원하고 현재 source J만 별도로 전달한다. 양의 trial J에서 formal η=ηtrue+χscat·J를 풀고 새 K/J·H/J·N/J·중간점 closure·affine 입사로 Q·광학깊이·행렬·flux를 다시 조립한다. MOM 방출항에는 ηtrue만 넣어 coherent 산란을 이중 계상하지 않는다.
- 생산의 H 제한은 유지한다. 격리 fixture는 같은 원식의 비채택 raw trial을 독립 풀이하며 H/PSD 위반도 그대로 기록한다. MOM 풀이·가속 후보는100예산, 바깥 trial은 원래 F-loop 상한 안에서만 한다. K/J가 먼저 작아져도 J·H·소스·Q가 계속 변하는지 함께 관측하며 상한에 걸리면 미수렴이다. partial EDDFACTOR는 사용하지 않는다.
- 실제 blue를 유지한 공동 고정점이 H/PSD/원식/경계/flux를 회복하면 같은주파수의 공동 갱신·조기 종료 수리를 검토한다. 공동 고정점에도 위반이나 큰formal/MOM 차이가 남으면 F-loop 변경으로 해결했다고 하지 않고, edge36 주변의 ray blue-source 각적분과 MOM PSIPREV·Jprev/HS·Hprev/EPS_PREV를 같은좌표·γR²·주파수 차분에서 대조한다. K+βH 공간 flux/Q 항과 구별한다.
- 오프라인 CPU1/4GiB/300초 안에서 원래66069·blue상태복원·소스지연 분리·J-only/H-only 반증·변경항 독립조립과 기존fixture를 한 묶음으로 검증한다. 기존출력·입력은 보존하고 새 private 경로만 사용한다. 물리 수리 근거와 검증 전에는 새whole을 제출하지 않는다. 수리를 채택할 때만 처음부터16CPU/16GiB/GPU0·45분/자식40분·1회/retry0을 고려하며 현재는 미제출이다.

현재 원식 분석 회귀는 **2 passed in7.77s**다. 실제 H 사건은 고정밀 근에도 H/PSD 위반이 남고, J-only/H-only의 MAX 통과가 실현가능성 합격이 아님을 재현한다. 이전5691 cap의 같은 원식 근은 H MAX를 넘지 않는다는 구별도 확인했다.

### 공동 반응 결과 — 반복은 안정되지만 물리 위반과 약50% 차이가 남음

격리 fixture `scripts/prepare_cmfgen_coupled_packet.py`/`replay_cmfgen_coupled_packet.f90`는 첫 MOM 행렬을 **풀이 전에** 포획하고STOP96한다. 생산 H 검사를 비활성화하거나MOM 정상반환으로꾸미지않았다. `run_cmfgen_coupled_packet.py`가동일LINMON원식의비채택trial을독립풀이해다음산란source J에만전달한다. 실제ray blue와MOM blue는매trial모두byte동일한고정입력이다. 새주파수이력으로쓰지않는다.

- 실행build는 `coupled_v3`다. v1은경계상수추가전compile-only, v2의첫실행은17byte문자literal을16byteheader로읽는fixture IO오류로중단했다. 둘다보존했고, 물리변경없이writer를정확한16byte로고친v3만유효원인분리에사용했다.
- 정본은 `/gpfs/kjhan/cmfgen_runs/lumina_affine_66069_replay_20261019-Pyi0PH/coupled_actual_blue_v2/coupled-response.json`이다. CPU1/4GiB, **3trial·0.226초·총3,788,784bytes**, 각trial 독립선형풀이2회로100예산안이다. 원래F-loop20회상한을늘리지않았다. 첫trial의실제광선/기하는원본과byte동일, 첫MOM행렬의최대항별상대차<1e-10을직접시험했다. 이항별시험허용치를IEEE오차상한의별도증명이라고부르지않는다.
- 마지막J변화6.70e-11/H변화÷J3.12e-12/Q변화1.14e-10/K/J변화5.95e-11, 산란소스지연8.50e-12로공동반응이안정됐다. 원식잔차2.13e-16이다. **그런데H MAX edge36·PSD−1.026644·formal/MOM J차이.501577은그대로다.** 산란소스지연을제거한formal원식잔차도.334618이다. 이고정점을물리PASS로판정하지않았다. 같은입력에서반복만늘리거나K/J종료만바꾸는수리는현재실패를해결하지못한다.

### 실제 주파수 source와 공간 flux의 항별 대조

`analyze_cmfgen_coupled_terms.py`는같은공동종료packet에서실제REL_VARIABLES가소비한각광선의B·I를0·1차각적분하고native PSI/PSIPREV·HS/EPS와비교한다. 현재/직전의차분에서큰항이상쇄되므로, 양쪽큰source의작은상대차와순물리수지차이를구별한다. primitive0차식과실제소거행렬의독립재조립차이는8.72e-15로확인했다.

정본 `coupled_actual_blue_v2/primitive-terms-v2.json`의depth36에서, 이전0차source는native MOM7.74807e8 대실제ray1.58334e9, 이전1차source는native1.38279e11 대실제ray6.85457e10이다. 두차이는공간flux변경없이도같은정확한식에들어가는이력이크게갈렸음을보인다. 현재0차source의native/ray차이는내부88점최대.08534%이고1차의89개edge최대상대차4.1839%는signed값/상쇄위치도포함하므로그대로전체율오차로해석하지않는다. 큰현재/이전항을각각비교하는것과,실제쌍의차분을비교하는것은다르다.

양쪽source를동일ray각적분으로맞춘순수지진단에서depth35–37의0차차이는22.7%/11.2%/26.1%,1차차이는31.9%/35.8%/32.8%였다. 이는이packet의공간/주파수이산연산자정합진단으로서원래v2의율·복사장허용치나전주파수오차를뜻하지않는다. 현재근거로단일물리항누락·특정각계수오타·전역불안정성을확정하지않는다. 생산수리는source를맞춰답을강제하는것이아니라일반적인장의계수/각투영·보간/차분정합을고쳐야한다.

통합직접시험 **16 passed in9.62s**는실제packet광선·기하/blue불변,첫행렬대응,엄격한binary길이,가짜공동고정점의물리기각,이전H/cap분류,광율비교기입력/적분을함께검증했다. producer와원자료를새경로에보존했다. primitive v1/v2도각report source hash와일치하는별도소스를보존한다. 현재새Slurm/외부모델은없고원본native/이전출력/미커밋변경은유지했다. 신규commit/push는없다.

다음우선순위는반복수확대나동일whole재제출이아니라 **두수송경로의일반적인이산연산자정합수리**다. 완료한것은실제광장에서의이력불일치와가짜공동수렴반증까지다. 전주파수J60/실제Si/S율·가열반응/수렴released-T CMFGEN기준해/새Lumina자기일관분포·스펙트럼/최종동질성은계속미완료다.

### 공통장 수리 계획 — uniform formal angular transport

후속 설계결정에서 Astra는 **모든 주파수에 같은 formal angular RTE를 적용하는 B안**을 선택했다. 각투영 계수만 바꾸는 A안은 확인된 공간 이산 차이를 해결하지 못하며, 완전한 discrete moment 재투영 C안보다 기존 native angular solver 재사용 B안이 현재 범위의 작은 일관된 구현이다. 생산 채택이나 새 whole 승인은 아직 아니다. Fable 취소·운전자 구현 분장은 유지한다.

추가 packet 진단에서 실제 B 계수와MOM β/σ 다항식의 최대차이는3.2849e-7,σ는모두0이었다. 큰속도계수누락증거는없다. source를formal J로갱신한독립반응 `coupled_formal_source_v1/coupled-response.json`은4trial에J변화4.80e-9/소스지연1.49e-10/실제광선비음수·nodePSD+.00321057로안정됐다. 같은MOM blue를남긴hypothetical MOM은여전히약50%차이/H위반이므로그자료를정답강제나생산해로채택하지않았다.

- **목표·입력:** 같은StageA59의물리/원자료/기하/경계에서 하나의 angular ray 이력과 그 모멘트가 J/H/K/N·율·에너지소비자에 공급되도록 한다. saved formal J를 외부정답으로주입하거나실패주파수에서만fallback하지않는다. 전체J60→같은59의Si/S율·열반응은계속묶음출력이며수렴released-T앵커/새자기일관Lumina분포·스펙트럼을대체하지않는다.
- **물리보존:** USE_J_REL/USE_FORMAL_REL/INCL_REL/이류/경계/산란을그대로두고private COMP_J_BLANK 수치dispatch만분리한다. USE_J_REL=F는메인의상대론MECH_LUM과RLUMST변환까지바꾸므로금지한다. GET_RSQH_REL의interface는γ없는r²H이며메인γ변환을이중적용하지않는다. K와내/외부flux도같은수렴광선에서얻는다. 실제inner Iplus와순flux를DBB/(3CHI)와자동동일시하지않는다.
- **풀이:** 매주파수blue rays를고정한 ηtrue+χscat·J 산란고정점을풀고수렴current ray만다음blue로확정한다. K/J뿐아니라실제source/J잔차1e-8을확인하고원래반복상한을늘리지않는다. 필요하면같은고정점을보존하는diagonal ALI를이묶음에포함하며최종잔차는실제formal호출로확인한다. MOM반환/SAVE는공통장으로소비하지않는다.
- **사전증거:** CPU1/4GiB/300초안에서실제66069/이전packet,blue복원과확정,비등방/급격한공간변화/광학적으로두꺼운일반장,v→0흡수·열평형·coherent산란·팽창/redshift를검증한다. 같은ray RTE의방출/흡수/경계/팽창일을에너지장부로연결하고작은manufactured문제의공간·각·주파수h/h2/h4수렴을보인다. source누락/blue잘못전진의반증도포함한다. 실제packet재보간을독립정답으로쓰지않는다.
- **판정·자원:** 알고리즘변경을명시하고v2의M/E/S·에너지·격자문턱을유지한다. old MOM잔차.334618을0으로표시하거나E동등PASS로꾸미지않는다. 새로운실제angular RTE/산란잔차·보존·연속극한·격자증거로판독한다. fixture의보존/두꺼운장비용을확인한뒤에만16CPU/OMP16·16GiB·GPU0·45분/자식40분·노드미고정·1회/retry0의전주파수실행을검토한다.14시간/128GiB계산은여전히미승격이다.
- **보존·재현:** 새private build와runroot를쓰며기존부모소스/바이너리/물리입력/모든실패자료를보존한다. 직접시험·producer신원·정확한재현명령을함께남긴다. fixed-T 결과는staging일뿐released-T matched-physics앵커PASS가아니다. 이후인구/T풀이의radiative response/Jacobian도같은producer와정합해야한다.

구현의첫실제packet시험4건은같은formal광선/기하/blue불변,source/J잔차,실제η입력대응,γ없는H interface,wrong-source/wrong-blue중단을통과했다. 새수치경로는private환경선택만추가하고기존물리flag는불변이다. 사용하지않는MOM경계비를계산하지않으며영장에서는0/0대신사용되지않는K/J표현만0으로두고광장자체는그대로0이다.

일반장fixture의초기Pgrid는접선광선을core에포함해잘못된geometry였으며자료를보존했다. native규약NP=NC+ND와core P<Rinner/별도limb P=Rinner로고친fixture에서순흡수/영장/열평형/급한공간변화는정상종료했다. 그러나τ=10·albedo.9와τ=100·albedo.99의순산란source반복은20회상한에미달(마지막J잔차각각.00252/.01551)했다. 이를완주가능성으로보고전체계산을제출하지않는다.

같은고정점가속의구현내부범위는nativeSC의실제광학깊이/양의local선형source반응에서얻은근사대각ALI와최대4개의source잔차이력extrapolation이다. 작은QR로source후보만제안하며음수/비유한후보는기각하고ALI또는실제Picard값으로되돌린다. 광장/J/H를floor/clamp하지않고모든trial을원래20회에포함한다. 실제angular source/J잔차1e-8과H·K안정성을매번직접판독하므로가속후보가최종해를대신하지않는다. 이수치가속은물리/최종수용기준변경이아니다.

### 공통장 후보의 직접 물리 시험 — 수렴과 보존을 구분

`formal_primary_v4`의 대각 ALI/4개 이력은 산란 시험에서 20회 상한에 실패했다. 이력을 예산 안에서 전부 보존한 v5는 albedo .9를 17회에 풀었지만 albedo .99에서는 끝까지 미수렴했다. 이를 통과나 전체 실행 근거로 사용하지 않는다. v6는 작은 선형 풀이의 라이브러리 심볼 부재로 링크에 실패했고 기존 디렉터리를 보존했다.

현재 성공한 private 후보는 `lumina_mom_capture_build_20261019-RbUeST/formal_primary_v7`이다. 가속용으로만 양의 **선형 source 비국소 angular Lambda**를 각 광선에서 전파·합산하고, `(I-Lambda) deltaJ = Jformal-Jsource`를 pivoting으로 푼다. 실제 formal의 포물선/단조 source 보간, 물리 플래그, 광선 이력, 최종 수용 잔차는 변경하지 않는다. 최대 19개 잔차 차이는 원래 20회 안에서 얻은 자료이며 추가 formal 호출이 아니다. 음수/비유한 후보는 전체 기각한다. Lambda는 주파수 첫 호출에서 만드는 근사 전처리자일 뿐이며 실제 source/J/H의 `1e-8` 판정과 K/J 종료 확인을 대신하지 않는다.

운전자의 직접 시험은 **7 passed in 5.28s**다. 원래 실제 66069 packet의 첫 광선·기하·blue 입력 신원, source 누락/blue 잘못 전진의 음성 대조, gamma 없는 H interface를 유지한다. 별도 시험은 pivoting/특이행렬 기각, 독립 구형 순흡수 연속해, 비균일 적분, 작은 실제 native 광선 제조해 24건을 함께 확인한다. 이 시험을 외부 모델 감사로 표시하지 않는다.

- **원본 격자의 실패도 보존:** uniform r, 65 shell의 albedo .99 시험은 source 반복을 11회에 끝냈지만 에너지 수지 오차는 **24.98%**였다. 수렴을 물리 보존 PASS로 바꾸지 않는다. core angular grid만 개선해도 이 오차는 남았다.
- **해상도 원인 분리:** 경계층을 해상하는 cosine r mapping의 65/129/257 shell에서 albedo .99 에너지 상대 결함은 **4.186% / 1.910% / 0.8774%**, 실제 formal 호출은 **9/7/6회**였다. albedo .9는 **0.8476% / 0.3165% / 0.1156%**, 호출은 **6/5/5회**다. 순흡수·급한 source 변화·영장도 정상이며 열평형 깊은 곳의 J=B와 전역 수지를 구분했다. 열평형 129 shell의 전역 에너지 결함은 **0.0654%**다. 거친 fixture의 실패를 지우지 않았고 실제 생산 격자를 바꾸거나 257 shell 결과를 실제 90 shell의 검증으로 간주하지 않는다.
- **각도·주파수 반응:** 고정된 65 shell에서 core angular 구간 32/64/128의 독립 순흡수 J 최대 절대오차는 **1.38248e-3 / 6.91170e-4 / 3.45561e-4**다. 고정된 65 ray 점에서 log-frequency 구간 512/1024/2048의 팽창 제조해 `I=nu^3` 최대오차는 **2.20399e-3 / 1.10031e-3 / 5.49736e-4**로 감소한다. 마지막 주파수의 경계·팽창 희석·주파수 유입 장부의 결함은 약 **5.99e-11**다. 이 시험은 native 단일 characteristic kernel의 redshift 극한이지 전체 상대론 구형 수송의 독립 정답은 아니다. 고정 주파수 간격에서 공간만 세분한 결과에는 주파수 오차 바닥이 남으며 이를 공간 정확해 수렴이라고 꾸미지 않는다.

최종 자료는 `/gpfs/kjhan/cmfgen_runs/lumina_affine_66069_replay_20261019-Pyi0PH/formal_manufactured_resolved_v1/manufactured-report.json`이다. 실행파일/실제 빌드 소스/transport manifest/producer SHA와 source 사본을 남겼다. 초기 실패와 uniform/cosine의 거친 결과도 각 별도 root에 보존했다. 계산은 grammar-debug CPU1/4GiB에서 각각 수 초 안에 수행했으며 syntax 로그인에서는 수치 계산을 하지 않았다.

재현은 `CMFGEN_FORMAL_PRIMARY_TEST_BUILD=/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/formal_primary_v7`을 설정하고 연산 노드에서 `python3 -m pytest -q tests/test_cmfgen_formal_primary.py`를 실행한다. 보존된 fixture exe `formal_primary_v7/manufactured_mu_v1.exe`와 `scripts/run_cmfgen_formal_manufactured.py --grids 65 129 257 --radial-mapping cosine --exe <그 exe> --out <존재하지 않는 새 root>`로 같은 제조해를 재현할 수 있다. CPU1/4GiB/300초 상한을 유지한다.

**다음 연결 범위:** 실제 전주파수 J60과 같은 물질 상태59의 Si/S 율·열 반응은 아직 없다. 전체 실행용 새 coverage/field producer와 pre-pop 종료를 구현·직접시험해야 하며, 기존 MOM 잔차 항목을 0으로 꾸미거나 기존 whole monitor의 MOM schema를 그대로 성공 처리하지 않는다. private v7의 `whole_executable_released=false`를 유지한다. 다음 meaningful 실행 판단에서만 Astra에 묶음 평가를 요청하고 Fable 취소는 유지한다. 실제 90 shell의 보존·후속 released-T anchor·새 자기일관 Lumina 분포와 스펙트럼은 미검증이며, 기존 Si 0/90·S 90/90 및 III 최대 차이16.28/13.93%p는 바뀌지 않았다.

Astra의 이 묶음 실행 평가는 완료됐고 agent를 종료했다. **같은59·197291개 주파수·실제90깊이의 전주파수 diagnostic 1회는 타당**하며, 앞서 명시한 CPU16/OMP16·16GiB·GPU0·45분/자식40분·retry0·노드 미고정·pre-pop 종료 범위다. 단, 제조해를 실제90격자의 보존으로 간주하지 않고 실제 J/H·격자·주파수 가중치와 깊이별 방출/흡수/팽창일, 내외 경계 순flux, 주파수 끝항을 함께 보존해야 한다. signed 결함과 절대 규모를 같이 판독한다. 완주는 diagnostic_complete일 뿐 보존/앵커 PASS가 아니며, 실제 수지 결함이 크면 최소 후속 수리는 경계층 공간해상 보강이다. 부분 EDD 소비·문턱 완화·Fable 검증2 대체는 없다.

whole 준비 중 별도 확인한 소비자 연결은 `GET_RSQH_REL` 뒤의 native `REGRID_H`가 H를 다시 node로 보간한다는 점이다. uniform formal 경로에서는 실제 node H의 `r²H`를 SOB에 직접 공급하고 H_MOM/K_MOM도 같은 광선의 H/K를 소비하도록 분리한다. gamma 없는 RSQH interface와 main의 이후 상대론 변환은 유지한다. 이 변경의 구현·직접시험은 운전자가 수행하며 새 소스·실행파일 신원을 별도 private build로 보존한다.

### 새 공통장의 실제 전주파수 연결 — Slurm 428766

운전자가 새 producer/소비자 신원·coverage·최초 실패 보존·pre-pop 종료를 구현하고 **13 passed in 5.44s**를 확인했다. `FORMAL_PRIMARY_FIELDS.bin`은 모든 주파수의 J/H/K/N·실제 chi·true eta·chi_scat·source J 및 원래 r/v/sigma/gamma·주파수 가중치를 남긴다. 이를 통해 실제90격자의 방출/흡수/팽창일·경계 순flux·주파수 끝항을 사후 판독한다. 기존 MOM equation residual을 0으로 채우지 않으며 새 schema는 `NOT_APPLICABLE_NOT_ZERO`를 명시한다. 실제 J/H/K/광도 소비자가 같은 광선 모멘트와 다르면 중단한다. 부분 field 또는 EDD와 다른 J는 완료 증거가 아니다.

- 제출 작업: **428766**, run root `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_formal_primary_428766`.
- 자원·범위: CPU16/OMP16·16GiB·GPU0, 부모45분/자식40분, 노드 미고정, `a10,a40,a100,a100_pcie,h100,h200`, 1회/retry0. 원래 StageA59에서 주파수197291개/90깊이를 연결하고 `SOLVE_FOR_POPS` 직전에 종료한다. 부모 입력·원본 native·기존 실패자료는 그대로다. 전체 field 2GiB, 주파수 요약64MiB, 최초 실패 snapshot256MiB 상한이며 전체 파일별 출력 상한8GiB다.
- build: `/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/formal_primary_v10`; exe SHA256 `d35ee84f408671b985f837b0755141a34d36c7a44f85330d60fb99e2eea2d2af`; manifest SHA256 `2650d5cafd149d15f9aac93ef4b3236c16e82188c88f4911c34596dff4a2f4aa`.
- 봉인 runner: 같은 build 부모의 `formal_primary_runner_v1`; `runner.sha256` SHA256 `06bada924f25e0af281fe63b6b777d185a068f8a13f065ca1060b3b328b24b27`. 실행 시 5개 파일의 해시를 재확인한다. `formal_primary_v10/direct-test-src`와 `direct-tests-full.log`에 실제 시험 소스/결과를 보존했다.
- 재현: 위 build를 `CMFGEN_FORMAL_PRIMARY_TEST_BUILD`로 설정해 연산 노드 CPU1/4GiB/150초에서 `python3 -m pytest -q tests/test_cmfgen_formal_primary.py tests/test_cmfgen_formal_primary_whole.py`. 실제 계산 재현은 봉인된 `submit_cmfgen_formal_primary.slurm <runner> <build>`를 통해서만 새 root에 제출한다. 상태 관측 timeout은 재제출 사유가 아니다.

이 기록 시점에는 **제출만 확인**했으며 완주/실제 에너지 보존/새 Si·S 반응·앵커는 미판정이다. 정상97 종료와 모든 coverage·소비자 신원을 만족해도 diagnostic_complete일 뿐 최종 물리 동질성 PASS가 아니다. Fable은 취소 상태이며 Astra는 위 실행 판단 뒤 종료했다. 새 commit/push는 하지 않았다.

**22:50:04 KST 상태 갱신:** 428766은 syn02에서 RUNNING 7분40초이며 새 stream은 **43,904/197,291**까지다. 마지막 주파수 `FL=4.2730438334`(1e15 Hz 단위)는 formal 3회, J/source/H 잔차 약 `7.73e-10/9.72e-11/9.33e-10`으로 실제 소비자 검사를 통과했다. 아직 전체 완료나 새 Si/S 분포 결과가 아니다.

`analyze_cmfgen_formal_energy.py`의 부호·상쇄·주파수 끝항 단위 직접시험은 **2 passed in 0.12s**다. native main의 상대론 mechanical coefficient를 사용하되 실제 장부는 미산출이며 물리 PASS를 판정하지 않는다. 특히 stream의 FL은1e15 Hz 단위지만 **실제 FQW는 Hz**이므로 주파수 끝항에1e15를 명시적으로 곱한다. γ 없는 H와 이후 main 변환을 혼동하지 않는다. 초기 구문 오류는 직접시험 수집 시 발견·수정했으며 실행 중인 봉인 runner/build에는 영향이 없었다. 전체 field가 완성될 때만 실제90격자의 signed/절대 항 장부를 만든다.

### 428766 종료 — 50번 shell의 불연속 SC source 분기

**428766은 종료했다.** 64,657/197,291개 뒤 64,658번째 주파수 `FL=1.7413450430056086`의 20회 source 상한에서 STOP95했다. 전체10분16초, 자식609.17초, MaxRSS5,691,312KiB, 최초 실패 자료1,106,934bytes다. 시간초과나 음수/비유한 광장으로 중단된 것이 아니다. 마지막 source/J 잔차는1.613e-4/1.615e-4인 반면 H·K/J 변화는약1e-12였다. 이 부분 출력을 새 전체J60·Si/S 분포·에너지 장부로 소비하지 않는다.

운전자의 단일 packet 재현은 `/gpfs/kjhan/cmfgen_runs/lumina_formal_source_64658_replay_20261019-v1`이다. **실제 마지막 formal 호출의 광선은 원본과 byte 동일**하다. 50번 shell만 source guess와 반환 J가 약1.6e-4 상대차로 갈리고 다른 깊이는 반올림 수준이다. Picard/ALI는 `2.61948512e-4 ↔ 2.62027346e-4`의 2-cycle을 반복한다. 근사 Lambda의 최대 행합.0021266/분광반경.0021138이므로 강한 산란의 느린 수축으로 해석할 근거가 없다. Anderson도 이 불연속 사이에서 종료하지 못한다.

추가 관측 전용 kernel은 산술을 바꾸지 않고 실제 분기값만 기록했다(`branch_probe/ACTUAL_SOURCE_BRANCHES.bin`,770880bytes). **ray51·iz74·outgoing**에서 원래 포물선 source 적분이 `-2.03736e-11 → +2.35488e-11`로 0을 지날 때 native는 음수 쪽에서만 선형값 **2.87779e-6**으로 바꾼다. 따라서 실제 적분값은 약2.88e-6에서약2.35e-11로 불연속 도약한다. 이 구현의 비연속 source map이 2-cycle의 직접 원인 후보이며, 반복수나 상한을 늘리는 수리로 채택하지 않는다. 관측 전용 수정 뒤에도 실제 마지막 광선의 byte 동일성을 유지했다.

**자료 한계도 명시한다:** 현재 실패 packet에는 실제 total eta와 source J는 있지만 true eta가 별도 보존되지 않았다. causal 반복의 true eta는 두 값을 차감해 복원한 것이며 정확한 원본이라고 부르지 않는다. 50번 shell의 복원 eta는8.23341214e-13, 차감 반올림 규모는약2.07e-25다. 실제 마지막 호출은 원래 total eta로 byte 재현했으나 이후 반복은 `DERIVED_SOURCE_CAUSAL_REPLAY_NOT_EXACT_SOURCE_OR_ANCHOR`로 구분했다. 다음 실제 source 기록은 true eta·chi_scat·source J를 직접 보존하도록 보완해야 한다. 기존 자료를 소급해 exact source로 바꾸지 않는다.

다음 의미 있는 설계 판단은 **positive·continuous SC source 보간**이다. endpoint를 보존하는 연속 quadratic Bezier control-point limiter와 일괄 linear SC를 대조하고, 물리 플래그/blue 이력/실제잔차1e-8/20회 상한을 유지한다. 원하는 J에 RHS를 맞추거나 intensity/source 고정점을 floor로 바꾸지 않는다. 현재 source 반복을 통과한 것만으로 목표를 종료하지 않으며 실제90격자의 전체J→Si/S 반응이라는 원래 묶음 목표는 미완료다. Fable 취소를 유지하고 이 수리 결정에서만 Astra의 기획·평가를 받는다.

Astra의 위 설계 평가는 완료됐고 agent를 종료했다. 선택은 **A: 양방향의 positive·continuous quadratic Bezier source**다. endpoint 보존, 매끈한 비제한 구간의 2차 정확도, 양의 Bernstein 가중치와 상수/선형해·얇음/두꺼움 극한, 실제 source 0 교차의 jump 제거, 실제 packet 2-cycle 해소 및 기존 제조해 에너지 해상도 반응을 직접 확인한다. control-point 형상 제한과 J/source floor를 구분한다. 복원 true eta의 자료 한계는 유지하고 다음 packet에 true eta·chi_scat·source J를 직접 남긴다. CPU1/4GiB/300초의 직접 확인 뒤 수정본으로 기존 CPU16/OMP16·16GiB·GPU0·45분/자식40분·노드 미고정·retry0의 **처음부터 whole 1회**가 타당하다는 판단이다. 부분 재시작/부모 EDD 잔여값 소비·물리/상한/기준 변경을 허용하는 판정이 아니다.

### 연속 SC 수정본 직접 증거

`formal_primary_v11`은 모든 주파수의 양방향 내부 SC 구간에서 같은 positive·continuous quadratic Bezier source를 적용한다. 기존 경계의 선형 적분과 상대론/이류/산란/blue 이력은 그대로다. endpoint 사이 control point만 연속적으로 제한하며 intensity나 반환 J/H를 clipping하지 않는다. 매우 작은/큰 optical depth의 Bernstein 가중치는 양수이고 합은 `1-exp(-tau)`이며, 상수·선형 source와 비제한 quadratic source를 독립 적분으로 확인했다. 기존 source 0 교차의 약2.88e-6 jump는 같은 두 입력에서1e-14 미만으로 줄었다.

직접시험은 **21 passed in6.06s**다. 새 경로를 끄면 실제 마지막 native 광선이 byte 동일하고, source를 켠 원래66069 재현은 geometry/blue/H interface 신원을 유지하면서 새 source 고정점을 푼다. 포물선 분기를 바꿨으므로 새 경로의 광선값이 원래 산술과 byte 같아야 한다고 요구하거나 그렇게 기록하지 않는다.

실패64658의 같은 파생 source 입력에서 **Anderson/ALI/Picard 모두3회에 종료**했다. 마지막 J/source/H 잔차는 Anderson 약5.13e-14/2.41e-14/9.38e-14, plain Picard1.54e-9/5.21e-10/2.17e-11이다. 이는 상한/완화계수 조정의 효과가 아니다. 자료는 `lumina_formal_source_64658_replay_20261019-v1/positive_sc_v1/source-response.json`에 있으며 source 신원은 여전히 `derived_eta_not_original_exact`다. 새 실제 실패에는 true eta·chi_scat·source J를 `FORMAL_PRIMARY_SOURCE.bin`으로 직접 기록하며 잘못된 소비자/이력/미수렴 거부 시험에서 byte 배열을 확인했다.

같은 root의 `manufactured_positive_sc_v1/manufactured-report.json`은24건 모두정상,4.77초다. cosine65/129/257의 albedo.9 에너지 결함은 **.9417/.3346/.1190%**, albedo.99는 **5.146/2.143/.9322%**이며 후자의 formal 호출은9/7/6회다. 거친 격자는 여전히 보존 실패이며 가장 촘촘한 제조해의1% 이내 결과를 실제90격자나 최종앵커의 PASS로 확대하지 않는다. 다음 실제whole은 새 원본계산에서 exact source를 사용해 이 파생재현의 한계까지 확인한다.

### 연속 SC 전주파수 연결 — Slurm 428782 제출

21건 직접시험·24건 제조해와 실제 실패 packet의 3회 종료를 확인한 뒤 **428782**를 제출했다. run root는 `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_formal_primary_428782`, build는 `/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/formal_primary_v11`, 봉인 runner는 같은 부모의 `formal_primary_runner_v2`다. 원래StageA59에서 처음부터 실행하며428766의 부분 field/EDD를 restart로 쓰지 않는다. 물리 플래그·20회/1e-8·CPU16/OMP16·16GiB·GPU0·45분/자식40분·노드 미고정·1회/retry0는 동일하다. 과거 작업/소스/산출물은 보존했다.

- exe SHA256: `f911749a956055077fe69f28f2d64182b37c60cb387303b7f160e7c5e43acfdd`.
- build manifest SHA256: `33ef247a89bf195a9ce6a803a75d3904354f20b022392e9f1a18115712052551`.
- `runner.sha256` SHA256: `71aeeea6707c6078fb3b621e16567fd42768a63e531693a7d2fbe68d109c194b`.
- `direct-tests-full.log` SHA256: `90501f2140dc8ae0da5c4adfdbd956537be1e521352d04cbfa2472834d0e4298`; 실제 시험 소스4개는 `formal_primary_v11/direct-test-src`에 있다.

제출 직후 상태·완주·실제90격자의 에너지 수지와 Si/S 반응은 미판정이다. Fable 취소와 Astra의 묶음 결정만 호출하는 배정은 유지하며 새commit/push는 없다.

**23:21:38 KST 확인:** 428782는 syn02에서 실행 중이며 새 stream은10,482/197,291까지다. 초기에는 BA/restart 로딩 뒤 첫 stream 기록이 늦었으나 이후 진행을 확인했다. 마지막 formal3회, J/source/H 잔차약4.89e-16/0/1.19e-16이다. 첫 source 기록 부재를 시간초과·종료로 해석하거나 같은 작업을 재제출하지 않았다.

후처리 helper는 동일한 Gamma/heat 선형 함수를 묶어 DRAM 중복 읽기를 줄였다. 정확히0인 가중치의 외부 구간만 생략하며1e-280/1e-290 등의 비영값은 그대로 계산한다. 기존 적분기·실제 PHOT/상태59 입력·부분 출력 거부·수학 장부를 포함한 직접시험은 **11 passed in1.46s**다. 실제 whole 완료 전이므로 이 최적화의 새 Si/S 반응 결과는 아직 없다.

### 428782 전주파수 완주와 실제 장의 첫 물리 판독

**428782는 COMPLETED, 전체26분00초**다. 자식 step의 FAILED/97은 `SOLVE_FOR_POPS` 전 의도된 diagnostic STOP97이며 부모 runner는 정상0이다. 전197291개/90깊이, 최대 formal6회, 최대 source/J/H 잔차 `9.933e-9/9.994e-9/9.999e-9`다. geometry·J/H/K/광도 소비자·EDD J·부모SCRTEMP불변을 확인했고 인구/T갱신은0이다. field SHA256은 `8dea1f46cf2e85d23dc141a755aa2e97e17a6c5596b402d68e0bea985befecd2`다. 원래64658 source jump와66069 MOM H실패 주파수를 새 전체 이력에서 넘어섰다. 이것은 아직 수렴앵커/SiS동질성PASS가 아니다.

판독 root는 `/gpfs/kjhan/cmfgen_runs/lumina_formal_primary_428782_analysis_v1`이다. 에너지와 광율/가열을 grammar-debug CPU1/4GiB에서 독립 병렬 판독했다. **850준위 모두/76500행**을151초에 계산했으며, 행 key의 누락/중복과90깊이×모든준위를 확인하고 CSV를 다시 합산해 JSON의 모든 이온 예산과 대조했다. 시간상한이 이온 사이에서 걸렸을 때 거짓 전체완료가 되지 않도록 완료 수를 실제 이온별 준위수로도 확인한다. 후처리/장부/팽창 구형 제조해 회귀는 **16 passed in9.73s**다.

**비교 phase 주의:** 원래427702는 exit0/마지막 반복59/EDD완료flag1이다. native는 복사장 기록 뒤 인구를 갱신하므로 부모J59는 물질58에서 생성됐고 새J60은 물질59에서 생성됐다. 두 장의 율은 같은 물질59에서 재평가했다. 물질58→59만으로도 SiIII분율 최대7.2921%p(깊이30), SIII최대2.0018%p(깊이31), ne최대1.8710%가 변했고 T는고정이다. 따라서 전체 J59→J60 차이를 source 수리 단독 효과로 해석하지 않는다. 과거39/40 단계의16.28/13.93%p 차이와도 직접 같은 phase의 전후라고 비교하지 않는다.

아래는 **이온분율이 아니라**, 같은59 인구로 가중한 이온당 평균 광이온화율의 `Gamma60/Gamma59`다. III는III→IV 전이를 뜻하며 역재결합/전체SE/새분포를 대신하지 않는다. 모든90깊이 자료는 `ion_rate_readout.csv`에 있다.

| 깊이 | SiIII 광율 비 | SIII 광율 비 |
|---:|---:|---:|
|10|3.7430|1.7037|
|28|1.9660|2.4726|
|35|1.9979|1.4847|
|50|1.2546|1.0442|
|61|.99770|.99943|
|90|1.01030|1.01439|

**실제90격자 에너지 장부는 미통과/원인 분리 중이다.** 연속식의 point-trapezoid 판독에서 경계순flux `1.47291e21`, 부피RHS `7.87364e20`, signed 결함 `+6.85541e20`으로 순량 대비46.54%다. 반면 총 방출/흡수는 각각약1.36249e27로 커서 절대항 합으로 정규화하면2.52e-7이다. 후자를작다는이유로보존PASS라부르지않는다. 이 장부는 quadrature 근사이므로46.5% 전체를 수송 오차로 단정하지도 않는다.

가장 큰 기여는 마지막89–90cell의 `+8.58995e20`이고, 그밖의 signed합은약`-1.73454e20`다. 실제 내부경계의 양의 냉각 기여를 가중한 흡수 optical-depth median/p95는 **7.18e4/1.11e6**이며 강한 선에서 산란비는약1e-8이다. 폭81.61(radius 단위)의 cell에 극히 얇은 경계층이 존재할 수 있어, endpoint의 큰 냉각을 cell 전체의 선형 평균으로 취급한 오차가 유력하다. 실제 flux/독립 cell 적분과 구분해 확인해야 하며 boundary node를 삭제하거나 flux에 맞춰 RHS를 정답 강제하지 않는다. cell32–52에도 잔차가 있어 core 한 점의 문제로 모두닫지않는다.

### 에너지 판독 보완용 독립 팽창 구형 제조해

단일 characteristic redshift 외에 `beta=.025*r`, `chi=eta/nu^3=1`, 진공 외부/확정 내부경계의 구형 제조해를 실제 native solver로 계산했다. 독립 연속해는 `I_nu/nu^3` 불변량과 `ds=gamma*(1-beta*mu_lab) dz`의 해석 적분으로 구했고, v→0에서 기존 순흡수 구의 연속해와도 확인했다. 유한 주파수 경계항을 빼면 약16%의 수지 결함이 생기는 사례이므로 그 항을 실제로 검증한다.

자료는 `lumina_formal_source_64658_replay_20261019-v1/expanding_manufactured_v1`과 `expanding_manufactured_v2`다. 공동33/65/129r 격자·512/1024/2048주파수·64/128/256core에서 J 최대절대오차 `.0038918/.0021358/.0011375`, 장부결함 **1.1994/.6433/.3408%**로 감소했다. 순수 공간/각/주파수 변화도 분리했다. 고정65공간에서 주파수만512→4096으로 세분하면 analytic J오차가 `.001466→.002708`로 커지지만 Cauchy 차이는 감소한다. 이는 공간오차와 주파수오차의 상쇄/고정공간 오차 바닥이며, 단일축 세분을 곧바로 최종정확도PASS로 주장하지 않는다. 제조해의 결과를 실제90격자의 보존으로 대체하지 않는다.

### 2026-10-20 같은 물질59의 실제 C 인구 응답과 내부경계 적분 분리

Astra의 직전 묶음 평가는 독립 에너지 적분과 같은59의 외부장 C 응답을 병행하고 추가 whole/grid를 보류하라는 것이었다. 그 두 작업을 운전자가 실제 수행했다. 이 절의 루트는 `A=/gpfs/kjhan/cmfgen_runs/lumina_formal_primary_428782_analysis_v1`이다. 원래 CMFGEN/원자 덱/생산 C 바이너리는 변경하지 않았으며, Fable 의뢰 취소와 운전자 구현·의미 있는 결정만 Astra 평가 배정을 유지한다.

**새 입력과 실제 인구 풀이:** 기존 PRRR/SCRTEMP39 adapter를 수정하거나 물질39 입력을 재사용하지 않았다. `prepare_cmfgen_field_owner_pair.py`는 실제59의518개 Si/S SL·terminal 인구, T/ne/밀도/속도를 보존하고 양쪽 외부장만 바꾼다. 원래 덱은 기존 엄격 검증으로 확인하고, 격리 seed에서 경로가 바뀐 SIII/CoII PHOT 복사본을 포함한 모든 이온/역할/PHOT 분기의 원자료 byte 동등성을 따로 확인했다. 링크 텍스트가 다르다는 최초 거부는 출력 생성 전에 발생했으며, 원자료 내용 동등성 확인 없이 검사를 우회하지 않았다. v5 선택물리와 smoothing도 유지한다. 새 PRRR 매칭이나 수렴 참조를 주장하지 않는다.

- 입력 `A/owner_pair_inputs_v1/J59_owner_input.bin` SHA256 `e2ffed897510f9b4f716e58148e2a967b1bf379be75477fc5fc6182fec7afa17`, J60 입력 `66e289fe3bb8db676a18a30c0b8c16659544441d1ef60a35a9443c0dd26b0d02`다. 상세 신원은 `pair-inputs.json`에 있다.
- `P=/gpfs/kjhan/cmfgen_runs/lumina_dr_policy_20261019-wYEzAc`의 기존 `bench_cmfgen_owner_final`/`atomic_view`를 그대로 사용했다. 바이너리 SHA256 `92fbbce1a6f8a1ac81eb492cc76e29dc20c81e41966f207efbf3fdfe5954ee10`를 실행 전에 직접 확인했다. 새 격리 cwd의 clean env에서 CPU1/4GiB/timeout300초로 수행했다. 두 점 예비 응답은 각각8.55/8.62초, **전90깊이 양쪽 실행은77.73/77.54초에 exit0**다. GPU/새 Slurm 사용은0이다.
- 전90점 결과는 `A/owner_pair_full_J59_v1`과 `A/owner_pair_full_J60_v1`에 있다. 각900행의 유일 key/전체 coverage·원소합·실제 완료 footer·같은 T/ne/초기 인구를 확인했다. 두 CSV SHA256은 각각 `c516f474f496a3902aff678e478c703f85f0f63b85f9e9b343fb64cbf1b2a0ea`, `3f8b5538df448712385f8142562e22e62a72da3402674a217c7b1765f59a678b`다.
- 요약 정본은 `A/owner_pair_summary_v1/owner-pair.json`, 모든 II–VI 분율은 `fractions.csv`다. 후자 SHA256 `986c90d6acbfe2e2d9742c980703d6e96a072c15da2d051547e3cf73bf9864bf`다. Si는 양쪽 **0/90 채택**, S는 **90/90 채택**이다. Si의 raw rank269/목표270·κ 기준을 완화하지 않았다. 후보 SE/원소보존 잔차가 작아도 수치 채택 실패와 실제 채택을 구별한다.

아래는 **III 분율(%)**이다. CMFGEN59는 아직 fixed-T staging 인구이며, J59/J60 열은 같은59 T/ne의 조건부 C 후보다. 서로 다른 물질에서 생성된 장을 썼으므로 전체 차이를 수리 단독 효과나 새 자기일관 분포로 부르지 않는다.

| 깊이 | CMFGEN59 SiIII | C·J59 SiIII | C·J60 SiIII | CMFGEN59 SIII | C·J59 SIII | C·J60 SIII |
|---:|---:|---:|---:|---:|---:|---:|
|10|.187694|.188830|.000562|.151504|.151026|.045898|
|20|1.623949|1.626795|.059394|1.849237|1.846838|.916396|
|28|37.005489|37.092799|7.452144|40.881915|40.862446|17.114834|
|29|30.316725|30.375431|5.740157|51.702465|51.687577|27.310659|
|31|58.130184|58.140676|24.844334|71.489906|71.523717|50.254316|
|35|66.559356|66.507853|49.793381|97.494213|97.510147|95.640787|
|50|99.693835|99.692438|99.583693|99.592657|99.593702|99.657742|
|61|7.230130|7.164846|7.238118|41.404372|41.339224|41.408708|

**해석:** 이전J59에서 C 후보와 고정59 인구의 III 차이는 Si 최대.192684%p, S 최대.0651475%p다. 따라서 과거의 큰 차이를 단순히 원자자료의 미수리로 해석할 근거는 없다. 새J60에서는 크게 변하므로 복사장·물질의 공동 반응이 주 조사 대상이다. 이전39/40 단계의16.28/13.93%p를 현재 오차로 재사용하지 않는다.

**내부경계 독립 적분:** `analyze_cmfgen_core_layer.py`는 실제 local source와 native 상수/DTDR에서 계산한 `I+=B+mu*dB/dtau`를 쓰며 실제 flux로 RHS를 맞추지 않는다. 고정 chi/source·정적 평면층의 경계 모드 `deltaI=(B-S+mu*dB/dtau)*exp(-chi*x/mu)`를 공간 방향으로 해석 적분하고 r²를 정확히 포함했다. 실제 팽창/blue·source 변화까지 재현한 수송해가 아니라 quadrature 원인 분리다.

- `A/core_layer_v2/core-layer.json`/`core-layer.npz`가 최신 정본이며 v1도 보존했다. 모든197291주파수를 포함했다. 순수지 원래 결함 `+6.85541e20`에 대해 이 독립 모드의 point-trapezoid는 `-8.87525e20`, 해석 적분은 `-1.98542e18`이다. 근사 적분 차이 `-8.85539e20`만 적용해 보아도 나머지 signed 결함은 `-1.99998e20`, 원래 순경계량 대비 **-13.58%**로 남는다. 이를 보존 수리/PASS로 처리하지 않는다.
- 해당 모드의 core 실제/예측 source 잔차 L1 차이는3.11%이며, angular32→64의 해석 적분 차이는3.20e11이다. core의 큰 양은 매우 얇은 흡수 경계층의 nodal cooling을 폭81.61의 셸 평균으로 취급한 오차와 잘 맞는다. 엄밀한 실제 수송 오차 상한이나 core 외부의 해소를 뜻하지 않는다.
- **core 근사를 내부32–52로 확대할 수 없다.** 실제 source 기여로 가중한 depth32/35/50/51/52의 인접 tau median은20.8/35.5/11.0/3.48/7.39지만, 주파수 이류 alpha/chi의 상동팽창 scale median은1.68/1.02/3.49/6.01/3.36이다. core90은 tau median7.18e4·alpha/chi5.01e-6인 흡수 지배층이다. 내부에는 blue 이력과 이류를 포함한 공간/주파수 적분이 필요하다. 현재 전체 장 파일에는 각 주파수의 전체 angular blue 이력이 없으므로 nodal J/H/K만으로 정확한 cell 보존을 역으로 재구성하지 않는다.

**수리 단독 효과의 최소 확인:** `compare_cmfgen_completed_blue_band.py`는 같은59에서 실패했던 legacy428716의 실제 완료66068개 direct-access record만 읽고, 그 이후 red/부모 record를 수치에 쓰지 않는다. 네 ground 광율은 전체 비영 적분 지지가 완료 구간 안에 있음을 먼저 확인했다. `A/same59_blue_band_v1/blue-band.json`에서 formal/legacy의 min/median/max는 SiII `.97923/.99974/1.02784`, SiIII `.94493/1.00000/1.05385`, SII `.95580/.99999/1.21440`, SIII `.96535/1.00001/1.09810`이다. 이것은 네 ground 준위의 같은59 방법 비교이며, 전체 이온율/SE/미완성 legacy 전체장을 완성한 증거는 아니다. 전850준위의 같은 blue 예산과 완성 장의 전체 예산을 비교해 phase 기여를 추가 판독 중이다.

현재 새 helper/기존 입력 adapter/장부 직접 회귀22건과 완료-prefix 시험3건이 통과했다. 음성 대조는 물질39 재사용, 비영인 미완료 지지, 손상된 red tail의 비소비, 행 누락/중복·원소합·미완료 owner 거부를 포함한다. 이는 직접시험이며 별도 모델 감사가 아니다. 신규 코드/문서는 미커밋, 새push 없음이다. **최종 수렴 released-T 앵커, 실제 채택 Si, 전하·RE·수송 공동수렴, 새 스펙트럼 및 사전 v2 동질성은 계속 미완료**다.

### 같은59의 전준위 완료 구간 대조 — 큰 광율 변화의 phase 분리

위 「전850준위 추가 판독 중」은 이제 완료됐다. `A/same59_blue_all_levels_v1/blue-all-levels.json`과 `levels.csv`가 정본이다. **850준위/76500행을142.46초에 완주**했다. 완료된 blue 구간은 `nu>=1.6379066e15 Hz`인66068개이며 미완료 red record를 읽어 수치에 쓰지 않았다. 물질59의 같은 fine-level 인구를 가중해 다섯 예산(legacy59 blue/formal59 blue/parent58 blue/formal 전체/parent 전체)을 따로 남겼다. legacy 전체율·전체 SE를 새로 얻은 것은 아니다.

| 이온 | 같은59 formal/legacy blue min/median/max | legacy59/parent58 blue min/median/max | 새 전체광율 중 blue 비중 min/median/max |
|---|---|---|---|
|SiII|.93675/.99832/1.06389|.99976/1.19235/2.51724|.87745/.92181/.99466|
|SiIII|.97332/.99986/1.06770|.99956/1.31066/3.95899|.84036/.96788/.99867|
|SII|.95282/.99951/1.05147|.99986/1.19712/4.46731|.85338/.94752/.99497|
|SIII|.95275/.99965/1.06744|.95939/1.00352/2.44546|.94308/.98315/.99915|

**운전자 해석:** 완료 blue 구간에서 방법 차이는 약7% 이내인 반면, 물질58→59에 대응한 legacy 장 변화만으로 최대2.5–4.5배 광율 반응이 생긴다. 이 구간은 새 전체광율의84–99.9%를 담당한다. 따라서 앞선 J59→J60의2–4배 이온당 광율 증가를 angular 수리 단독 효과로 해석하는 것은 틀리다. 물질–장 반복의 강한 비선형 반응이 주요 원인 후보로 좁혀졌다. 다만 미완료 red의 legacy 값과 BB pumping·전체 SE를 모르는 채 모든 분율 차이를 물질 phase 하나로 확정하지 않는다. 새 J60에 맞춘 실제 물질 갱신/전하·열 공동수렴과 내부 cell 보존을 함께 해결해야 한다.

운전자는76500행의 key/준위별90깊이를 다시 확인하고 다섯 예산을 CSV에서 독립 재합산했다. 이전 `field_response/field-response.json`의 전체 photo59/photo60 예산과도 `rtol=3e-13` 안에서 일치했다. 두 owner 입력을 binary로 따로 해독해 header/주파수/SL키 및90개 모든 물질·인구 block의 byte 동일성도 확인했다. 새 후보 Si/S를 모두 채택한다고 가정하면 전자수 증분은 고정ne 대비 최대11.43%다. 실제 Si 미채택 상태이며 전체 원소 전하 풀이도 아니므로 전하 잔차 통과로 표시하지 않는다. 최신 관련 통합회귀는 **58 passed in16.00s**, 로그 `A/regression_20261020.log`다.

### 다음 실행 계획 — Astra 묶음 평가와 운전자 적용

Astra는 이 전90점/phase/에너지 결과를 묶어 평가했고 agent를 종료했다. 별도 감사나 Fable 호출은 없다. 다음 whole의 선행 작업은 **실제 SC·연속 Bezier·angular blue가 살아 있을 때의 독립 cell 적분**이다. 작은 제조해와 기존 packet에서 적분 정확도/보존 수렴·계산비용·원래 J/H 불변을 확인한 뒤에만, 기존 CPU16/16GiB/GPU0·45분/자식40분·retry0 범위의 한 번 실행을 고려한다. 정확한 깊이별 signed/L1 예산·경계·팽창일·주파수 끝항만 누적 보존하고 전체 ray 로그를 정례 저장하지 않는다. 현재 새 whole은 제출하지 않았다.

Si 수치 채택의 후속 수리는 이 적분 작업과 병행할 수 있다. 다만 **추가 영공간인지 척도 문제인지의 이전 진단은 이미 끝났다**: 본 문서 「2026-10-19 후속 실측」과 「Si 고정밀 대조」에 graph rank270와120/160자리 GTH 비교가 있다. 같은 진단을 처음부터 반복하지 않고 그 증거를 인수해 새 물질59/장60 상태의 수치 인증·채택을 해결하는 데 필요한 부분만 진행한다. 미채택 후보를 감쇠한 것만으로 채택했다고 하지 않으며 기존 rank/κ 기준은 유지한다.

공동 물질 갱신은 에너지 판독과 인증된 인구 풀이에 연결한다. 새 장에서 강한 반응이 확인됐으므로 보존·양수성을 지키는 수치 감쇠와 전하 갱신이 필요하지만, 이를 근거 없이 적용하거나 T만 무작정 해제하지 않는다. 최종 목표는 여전히 수렴 released-T matched-physics 앵커와 새 Lumina의 자기일관 Si/S 분포·스펙트럼 동질성이다. 새 적분기나 국소 인증 성공을 종료 조건으로 바꾸지 않는다.

### 다음 적분 구현의 첫 착지 — SC 셀 내부의 강도·정확한 공간 적분

`scripts/cmfgen_sc_cell_integral.f90`를 별도 passive 모듈로 구현했다. 기존 SC의 같은 quadratic Bezier control point를 받아 De Casteljau로 셀 내부 강도를 계산하고, 끝점 flux 차이를 답으로 쓰지 않는 **양의 Bernstein 공간 적분 가중치**로 `integral I dtau`를 구한다. 얇은 셀에서는 `tau/3-endpoint_weight`의 상쇄를 피하는 직접 급수로 leading source 면적 `O(tau^2)`를 보존한다. 음수/비유한 입력을 거부하며 intensity floor·clamp는 없다.

독립적인 연속 이중 Gauss 적분, 비균일/영 source, `tau=1e-280...1e280` 극한 및 `Iup=0`의 얇은 셀을 확인한 직접시험은 **5 passed in1.34s**다. 기존 `cmfgen_positive_sc.f90`·whole build v11·생산 바이너리는 바꾸지 않았고 새 함수는 아직 solver에 연결하지 않았다. 따라서 실제 whole의 공간·주파수 에너지 적분기 완성이나 J/H packet 불변의 새 연결 시험까지 끝났다고 하지 않는다. 다음에는 실제 chi/eta·blue·팽창항과 구형 적분 측도를 이 기본 함수에 연결해야 한다.

소스/시험 사본과 통합 재현 root는 `A/sc_cell_integral_v1`이다. 작은 함수 착지는 원래 물리 묶음의 내부 진전일 뿐 종료 판정이 아니다.

최종 통합 재현은 **63 passed in17.56s**, `A/sc_cell_integral_v1/regression.log` SHA256 `3f4009e5f68ea1a08b58cebe86a2fb2f653863c5677e90fdb0d483dc1be6fca8`이다. 새 기본 함수 SHA256 `4ee62ce3f3ed7a1b6888832c57e6fbb9e76447e40f25f97a8b9b5433e31e2341`, 직접시험 소스 `1264b8aef55ce160dad9ba8fd95858f9a448a1b6be9bda586657cb317e73c3ae`를 같은 root에 사본 보존했다. 기존 packet/whole 소비자/Bezier/팽창 제조해/PHOT 입력·율/부분 장 거부/새59 인구 짝비교·적분 시험을 함께 실행했다. 모든 수치/외부 모델 프로세스는 종료됐으며 새 whole은 제출하지 않았다.

### 실제 angular blue의 독립 공간·주파수 적분 연결과 다음 whole 계약

위 「새 whole 미제출」은 당시 상태다. 후속 작업은 Astra가 정한 A(실제 cell 적분)의 구현이다. 같은 물리/원자료/StageA59와 v11 RTE·source 고정점·20회/1e-8 기준을 보존하며, 새 물리/인구/T 갱신을 추가하지 않는다. 목표는 **기존 node 장부의46.5% 차이가 공간 적분·이산 source·기하·주파수 중 어디에 생기는지 전90깊이에서 분해**해, 다음 수송 수리 또는 물질–장 공동수렴 방식을 결정하는 것이다. 독립 장부의 생성 자체가 최종 물리 수용이나 묶음 종료는 아니다.

**적분식·분리 원칙:** `D=gamma*(1-beta*mu_lab)`, 실제 ray의 native `s` 좌표에서 `r² dr dmu/2 = p dp ds/(2D³)`를 사용한다. 각 광선의 실제 chi/eta/b/blue를 따로 복사하고, ray의 물질 계수와 D를 s에서 선형 보간한 정의된 cell 방정식을 적분한다. 이는 node 값만으로 원래 연속 물질장을 정확 복구했다는 주장이 아니다. SC의 실제 quadratic Bezier 내부 강도를 적분하며 얇은 지수 경계층은 `y=1-exp(-tau)` 변환으로 전체 지지를 포함한다. 임의 optical-depth 절단·intensity floor·끝점 flux에 맞춘 source 역산은 없다.

다섯 항은 물리 `eta-chi*I`, 팽창일 `b*I`, 실제 upwind 주파수 교환 `alpha*b*(Iblue-I)`, D의 선형 보간에서 생긴 기하 결함, SC source와 원래 물질/blue 보간의 차이다. **마지막 두 항을 물리 source나 합격 보정에 섞지 않는다.** 마지막 두 항까지 더한 수치 항등식의 오차는 적분기 정확도만 뜻한다. 실제 물리 이산 수지는 source+주파수 항과 경계를 비교하고, 연속 주파수 수지는 팽창일과 양끝항까지 따로 비교한다. p² 방향 적분의 경계광도와 원래 native angular J/H의 경계광도 차이도 별도다.

- `scripts/cmfgen_ray_cell_budget.f90`는 accepted formal의 실제 자료를 읽고 원래 물리 R(1) 밖의 thick-outer 확장 부피는 세지 않는다. 확장부가 준 실제 incoming 강도와 source/blue 이력은 보존한다. `cb_compute`는 J/H/blue·source 고정점 배열을 쓰지 않는다.
- 최신 소형 제조해 정본은 `A/ray_manufactured_v4/ray-manufactured.json`이다. 순흡수/두꺼운 열적 경계층/급한 source/산란/팽창/실제 범위의 양의 ESEC 확장경계를 함께 실행했다. 모든 사례가 정상종료했고 수치 항등식과 실제 물리 오차는 분리해 남겼다. 고정 source·tau1e6 시험에서 node 장부는17점 기준 약10054를 주지만 독립 ray source는경계약.993과1e-15 수준에서 맞았다. node 끝점의 냉각을 셸 평균으로 쓰는 오차를 직접 반증했다.
- 급한 source의 물리 결함은 작은 격자에서 비단조이고, 산란65/129/257의 물리 차이는 **47.56/14.34/4.44%**로 아직 크다. 이것을 적분기 실패와 동일시하거나 수렴/에너지 PASS로 바꾸지 않는다. node 물질 보간과 실제 SC source의 이산 차이를 별도 표시한 결과이며, 실제90격자에서도 이 항을 확인해야 한다.
- 추가 팽창129r/256core/512주파수 시험은 `A/ray_expanding_129_v1/refinement.json`이며108.37초에 끝났다. 공동33/65/129의 실제 연속 에너지 차이는 **.4364/.2761/.1746%**, native/p² 경계 차이도 축소됐다. 공간·주파수 오차 상쇄 때문에17→33은 비단조였으며 그 결과도 보존했다. 이 제조해를 실제90격자 합격으로 대체하지 않는다.
- 초판 제조해의 첫 주파수 반복에서 INITIALIZE를 첫 trial에만 켰던 fixture 오류를 고쳤다. 실제 `comp_j_blank.f`는 FIRST_FREQ를 모든 trial에 넘기므로 생산 변경은 아니다. v1/v2의 해당 팽창 시험은 이후 정본으로 쓰지 않는다. 처음의 zero-ESEC thick-outer 시험은 native의 `log(ESECmax/ESECmin)=log(0/0)` 외삽으로 formal 계산 자체가 실패했다. 이 edge case를 해결했다고 하지 않으며 기존 실패를 보존했다. 새 양의 ESEC 사례는 실제 모델 범위의 다른 시험으로 명시한다.

**실제 packet 재현:** `A/ray_actual_packet_v1/packet-cell.json` 및 최신 직접시험은 v11 archive 기준과 새 passive 관측기 사이에서 geometry/첫·마지막 rays/blue/최종 formal 상태/H interface/source 반복 CSV의 byte 동일성을 확인했다. 적분 후에도 ray/blue bytes가 같다. 실제66069 packet의 endpoint 재구성 차이는3.99e-16,8점 적분 수치 항등식 L1÷최대 cell 항은7.03e-10이다. 기하/SC source 차이는 별도 nonzero 값으로 남으며 물리 PASS가 아니다. 8점 적분 비용은 CPU1에서주파수당 약.00421초였다. 직전1551.6초에 단순 환산 추가51.9초, 약1603초 추정이지만 실제 병렬 효율/모든 opacity의 상한은 아니다.

**실행 전 봉인:** 새 build는 `/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/ray_budget_v4`, runner는 같은 부모의 `ray_budget_runner_v1`이다. 해당 build의 직접시험 **19 passed in2.64s**는 passive packet·공간/주파수 적분·전체 field 소비/누락 거부를 함께 확인했다. 원본 CMFGEN/기존v11/원자 뷰/이전 출력은 바꾸지 않았다.

- executable SHA256 `fc5c7256438ce3c9a9034256303c00c4ce3527cbe3ae44deead5809a46487c8e`.
- `formal-primary-build.json` SHA256 `3d3e42a35472d6383296ed15f0037870f16a3e58db7ce757e4f4f2859b10cea1`.
- `runner.sha256` SHA256 `a9566e337f378350cccba534c5e0d644f9151248bb6a0139e01f098316e20ef1`.
- `direct-tests.log` SHA256 `ea87edf3534f6496f2dfe443adf041c6b408934fef9f1ae2fa5ea065458c6199`.

**사전 실행 범위:** CPU16/OMP16·16GiB·GPU0·45분/자식40분·노드 미고정·1회/retry0, 기존 StageA59에서 처음부터197291개 주파수를 계산하고 pre-pop97로 끝낸다. 기존 field1.1GiB와 EDD/재시작 자료에 약9KiB cell 합계만 추가하며, 주파수별 전체 ray 로그를 만들지 않는다. 공용 GPFS 여유252TiB, 사용자 block/file quota hard limit0(미설정)을 확인했다. 새 별도 job-ID root를 쓰고 기존 출력/소스는 보존한다.

진단 완료 기준은 full coverage·부모SCRTEMP불변·공통장의 J/H/K/EDD 신원 및 cell 파일의90깊이/197291누적·native 소비자 광도 대응이다. 이것은 **실행/자료 신원 기준**이며 실제 에너지1%·최종 v2 동질성 문턱을 낮추지 않는다. 물질 source 보간/기하 결함을 에너지 RHS에 넣어 PASS를 만들지 않는다. 실패하면 같은 job의 최초 자료를 읽고 무변경 재시도하지 않는다. 실제 residual의 분해가 다음 수송/공동수렴 수리를 결정한다. 현재 제출 준비를 마쳤으며 job-ID는 제출 후 아래 기록한다.

**제출/현재 상태:** CPU-only Slurm **428832**, `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_formal_primary_428832`를2026-10-20 01:02경 제출했다. 같은 syntax cluster의 **01:20:29 KST** 조회에서 syn02/RUNNING17분58초·**111986/197291**이다. 같은 job을 계속 사용하며 관측 timeout으로 재제출하지 않는다. 당시 마지막 source/J/H 잔차는 약5.16e-11/5.37e-11/4.76e-11이다. 전체 에너지 파일은 마지막에만 완성되므로 부분 장/누적을 전체 판정에 쓰지 않는다. 후처리 `scripts/analyze_cmfgen_ray_energy.py`는 완료97·coverage·cell 원본 해시를 확인하고 물리 수지/보간/수치 항등식을 분리하며, 모든 bin의 J/H/K/N과 입력 계수를v11 기준 장과 대조하도록 구현했다.

### 병행 Si 수치 전방오차의 구간 인증 — 채택 기준은 아직 불변

원인 rank/κ 진단을 반복한 것이 아니라, 고정 입력의 정확한 nominal 해를 명시적 경계로 감싸는 후속 인증을 구현했다. `scripts/certify_generator_interval.c`/`certify_cmfgen_owner_interval.py`는 생산 코드에 링크하지 않은 오프라인 도구다. 생산 GTH와 같은 binary64 양의 off-diagonal 율을 사용하고 대각은 그 정확한 유출률 합으로 정의한다. 매 양의 합/곱/몫을 long double로 저장한 뒤 `nextafterl`로 외향 확장한다. 작은 비영율도 버리지 않는다. nearest 반올림과 x87 extended 정밀도를 확인하며 잘못된 환경/음의 율/모순된 대각/비연결 계를 거부한다. 각 입력율이 정확하다고 놓은 **수치 오차 인증**이며 원자율 불확실성이나 coupled 물리 오차 인증이 아니다.

검증은 두 상태의 정확한 유리수 해,1e-280 연결,1e280/1e-280 혼합, 독립 spanning-tree 열거로 얻은5상태 정확 유리수 해와 순열, rounding/x87 정밀도 결함 주입을 포함한다. 현재 구간 인증과 에너지 판독 직접시험은 **12 passed in.50s**다. 그 이전 전체 관련 회귀는 `A/validation_ray_energy_v1/regression.log`의 **80 passed in19.02s**이며, 이후 추가한 반올림 환경/독립 spanning-tree 시험은 이12건 기록과 구별한다.

같은 새 물질59/J60 입력에서 기존 봉인 생산 C 바이너리로 Si28·61 두 점만 새로 보존했다. 루트는 `A/si_interval_current_v1`, 최신 인증은 `certificate_v2/interval-certification.json`이다. 두271차원 계 모두0.22초에 해의 전방 경계를 얻었다.

| 깊이 | 기존 raw rank/목표 | 기존 κ | 최대 level 확률 구간 폭(근사 표시) | C nominal 최대 오차 경계(근사 표시) | SiIII 확률 구간 폭 |
|---:|---:|---:|---:|---:|---:|
|28|269/270|3.65e16|3.27e-16|2.06e-16|1.89e-17|
|61|269/270|3.29e16|3.40e-16|2.55e-16|1.92e-17|

권위 경계는 각 `z14_depth*_intervals.npz`의 longdouble 배열이며 JSON 십진/폭은 표시용 근사다. 과거 자료의120/160자리 비교를 다시 계산하지 않고 새59/60 입력에서 직접 전방 경계를 얻었다. **이 두 점의 nominal 인구가 크게 잘못 풀렸다는 가설은 지지되지 않는다.** 반면 전90점·다른 물질에서의 인증, 원자율 섭동 민감성, 물리 동질성은 주장하지 않는다. 생산 rank/κ 판정과 Si0/90 채택도 변경하지 않았다. 엄격한 전방 인증을 실제 채택 경로에 연결할지는 명시적 계획 평가 뒤 결정하며, 기존 실패를 몰래 PASS로 바꾸지 않는다.

### Si 실제 채택의 명시적 수치 계약 개정 — CERTIFIED_NUMERICAL

위 미변경 상태에서 후속 `certificate_v3`는 실제 binary64 반환 density와 정확한 원소총밀도를 직접 비교하는 외향 전방오차를 추가했다. 깊이28/61의 모든 준위 최대 확률오차 상한은 각각2.062e-16/2.546e-16, 최대 상대오차 상한은4.271e-16/4.053e-16이다. 미량 준위도 포함하며 후보를 재정규화하거나 구간 중점으로 바꾸지 않는다.

Astra는 요청 `01a154f8-1322-7610-a272-64fefc804c0f`에서 별도 `CERTIFIED_NUMERICAL` 경로를 타당하다고 평가했다. 이전 rank/κ 유지 권고는 직접 전방 인증이 없던 상태의 판단이며, **고정율의 수치 풀이 오차와 입력율 민감도를 분리하는 명시적 수치 계약 변경**이다. 원래 rank/κ 실패 수치와 legacy 판정은 따로 남긴다. 최종 v2 물리 기준·원자료·M/E/S·수렴 및 자원 한도는 바꾸지 않는다. Fable 대체 의뢰가 아니라 현 담당 Astra의 한 묶음 설계 평가이며, 평가 후 agent를 종료했다.

구현 범위는 Si/S의 기존 양의 generator GTH 반환 후보에 직접 인증을 연결하는 것이다. 정확한 binary64 offdiag와 정확한 유출률 합 대각이 인증 대상이며, 저장 대각과의1e-12 점검은 동일행렬 증명으로 해석하지 않는다. **원래 생산 Araw의 SE 잔차·원소보존·positivity·연결성·원자료/coverage·boundary·순열·pivot 검사도 모두 유지**한다. 기존 raw/normalized rank 및 κ 차단만 엄격한 인증 경로로 대체할 수 있다. 모든 준위의 정확해 density 구간 하한이 양수이고 외향 계산한 `e_i/N<=1e-12`, `e_i/lower_density_i<=1e-10`이 동시에 성립해야 한다. 인증 실패는 채택 거부다. 이온 합도 외향 합산하며 출력 십진 근사를 판정에 쓰지 않는다.

사전 실증 범위는 같은59의 J59/J60 전90점 Si/S 실제 채택, 기존 S 후보 불변, 반환 후보 손상·반올림 환경·단절·음율/대각 손상의 거부다. 생산/오프라인은 동일한 구간 함수를 소비한다. 계산은 각 CPU1/4GiB/300초 이하 grammar-debug, 기존 입력/출력/바이너리 보존·새 root만 사용한다. 전체 행렬90점 dump는 만들지 않고 채택 응답과 인증 최대오차/이온 경계만 남긴다. 이 실증은 고정장 nominal 채택의 증거일 뿐 전하·열·수송 공동수렴이나 최종 목표 완료가 아니다.

### 2026-10-20 전90점 실제 채택 및 전주파수 독립 적분 결과

**Si 채택 실증은 완료됐다.** `C=/gpfs/kjhan/cmfgen_runs/lumina_certified_owner_20261020-MNr0rt`에서 새 생산 owner를 빌드했다(SHA256 `9636e47918f12d296a1c7f8093a383f7bc7fa820d645b2b72cee2da9b609d040`). 생산과 오프라인은 동일한 `src/population_generator_interval.h`를 쓰며, 원래 반환 후보를 바꾸지 않고 C 채택 지점에서 strict longdouble 경계를 판단한다. J59/J60 전90깊이 Si/S의 **360개 모든 계가 인증되고 각각90/90 실제 채택**됐다. 실행은109.46/117.27초, CPU1/4GiB/GPU0이다. 이는 격리된 C owner 인구의 실제 commit이며 native59의 인구/T를 갱신한 것은 아니다.

`C/summary_v1/owner-pair.json`/`fractions.csv`에서 이전 두 응답900행과 대조했다. **모든 후보 분율·T/ne·원소총밀도·원래 SE/원소보존·rank/κ/graph 수치는 십진 byte 동일**, legacy topology/numerical 실패 표기도 그대로다. S의 기존 채택은 유지되고 Si의 미채택만 새로운 명시적 수치 인증 경로로 해소됐다. 새 장에서의 III 재분배33.30%p(Si)/24.38%p(S)는 그대로 남아, 채택 수리가 복사장·물질 공동수렴 문제까지 해결했다고 하지 않는다. 고정장 인구 풀이가 잘못돼 이 큰 차이가 났다는 가설은 지지되지 않는다.

**실제 ray 에너지 장부는 물리 합격이 아니다.** Slurm428832는 전197291주파수/90깊이·최대6회·source/J/H 모두1e-8 이하로 완주했다. 부모1728.66초, 자식1722.45초; pre-pop97을 정상 처리했고 SCRTEMP는 불변이다. 실제 cell 파일8632bytes의 SHA256은 `ead0682f50ef5b48960790f1beadf7fa8f120a87133361c29fdbd3f7101b1cad`이다. 판독은 `A/ray_whole_428832_v1/ray-energy.json`/`cells.csv`이며 기존428782와 모든17756190개 주파수×깊이를 대조했다. 관측기 전후 J/H/K/N 최대차÷local J는 각각5.67e-13/3.84e-13/2.86e-13/2.27e-13으로 합산 반올림 수준이다.

같은 공통 단위에서 ray 순경계1.47183e21(native1.47291e21), 선형 물질 재구성의 적분 Q=-1.69345e25, 팽창일3.62096e21, 실제 이산 blue 교환-4.16530e21이다. SC source 보간과 그 선형 물질 재구성의 차이가 **+1.69402e25**, 기하 보간 차이는-3.16325e17이다. 수치 항등식의 누적 L1은2.083e14, endpoint 최대차4.24e-15로 작지만 **이를 물리 RHS에 더해 보존 PASS로 만들지 않는다.** 연속 주파수와 이산 주파수 차이도-5.44339e20으로 별도 남는다.

가장 큰 SC–선형 물질 차이는58–59/59–60/57–58/60–61 셀에서 각각+5.813e24/+4.629e24/+3.010e24/+2.452e24다. 이는 core 하나의 node 냉각 적분 오차를 고치는 것으로 전체 수지를 설명할 수 없음을 보여준다. 다만 **현재 진단은 native s에서 chi/eta/b/blue/D를 선형으로 재구성한 계약**이므로 이 큰 차이를 곧바로 원래 연속 물질장의 실제 에너지 위반 수치로 확정하지 않는다. 실제 SC source의 재구성과 물리 eta/chi 재구성이 양립하는지, 어느 재구성이 native 입력/방정식을 대표하는지 독립 확인이 다음 핵심이다. 임의 보간 항을 물리 방출로 간주하거나 기준을 완화하지 않는다.

확장 회귀 첫 실행은 `C/regression_v1/regression.log`의115통과/1skip/1실패였다. 실패는 현재 producer가 아니라 과거 봉인 `formal_v1/lumina_formal_diag.o`를 링크한 fixture의 한byte trailing 거부다. 현재 producer는 이미 size/position 검사를 갖고 있었다. 시험이 현재 모듈을 격리 재빌드하도록 수정했으며 원본/과거 object/실패 로그는 보존했다. 새 통합회귀 결과는 후속 기록으로 구별한다. 코드/수치 인증·새 관측기 완료를 최종 v2 물리 동질성이나 묶음 종료로 표시하지 않는다.

현재 모듈을 격리 빌드한 후속 **통합 회귀는116 passed/1 skipped in27.97s**다. `C/regression_v2/regression.log` SHA256 `3b251e87f8b748c1b104ef25dbc6750fa3b43fa14a41584a4f9d0fd545736cc1`. skip은 scoped CMFGEN Fortran smoothing witness가 필요한 기존 선택 시험이며 통과로 세지 않는다. 새 인증의 음성 대조는 tiny level의 상대오차만 틀린 후보·0 후보·음수/NaN/잘못된 총밀도·단절/대각 손상·rounding/x87 환경을 포함하며, 기존 후보/진단/legacy 판정 또는 boundary 실패를 숨긴 요약도 거부했다. 생산 source/모든 header/시험과 compiler command는 `C/source_snapshot` 및 `C/build.log`에 보존했다.

### 같은 물질의 독립 RTE 반증 — 대규모 반복 없이 재구성 의미 분리

Astra의 묶음 결과 평가 요청 `01a15504-4f3b-7d62-a1c2-b5c31dbbb35d`는 다음 whole을 보류하고 같은 물질의 독립 RTE·h/h2/h4 비교를 먼저 하라고 권고했다. 이전 「독립 적분」 권고에 물질 재구성 전제가 부족했음을 명시했으며, 1e25를 실제 연속물질의 위반으로 확정하거나 단순 장부 문제로 무시하는 것 모두를 배제했다. native에 C 인구를 주입하지 않고 원래 갱신 경로를 유지하며, 고정T 물질/공유ne 준비는 병행할 수 있으나 released-T는 보류한다. agent를 종료했고 Fable/대체 검증 의뢰는 없다.

**원래 보간 정의를 소스에서 확인했다.** `cmf_formal_rel_v4.f`는 ray point를 추가할 때만 logR의 native MON_INT_FUNS_V2로 logchi/logeta를 각각 보간한다. point가 없으면 원래 radial chi/eta 값을 사용한다. `solve_cmf_formal_v2.f`의 REL_VARIABLES는 끝점 `S'=(eta+alpha*b*Iblue)/(chi+(alpha+3)*b)`를 구성하고, OPTDEPTH_V4의 ZERO는 native s의 effective chi를 trapezoid 적분한다. 이후 source는 광학깊이에서 보간되며 현 private v11은 양의 제한 Bezier다. 따라서 native가 모든 셀 안에서 eta를 s-선형으로 정의한다고 읽는 것은 틀리다. **현재 관측기의 s-선형 eta는 명시된 별도 재구성**이다.

**해석적 반증:** `chi=K(1+u), eta=K, b=0, 0<=u<=1`를 두 방법에 똑같이 주었다. tau=`K(u+u²/2)`에서 끝점 source1/0.5를 선형으로 잇는 SC는 `S_SC=1-tau/(3K)`다. 끝점은 K가 커질수록0.5에 접근하지만 `integral(chi*S_SC-eta)du=K/8`이다. K=1e6에서도 실제 적분기와 이 정확식이 맞았다. 반대로 constant chi와 독립적으로 주어진 linear eta에서는 같은 SC source와 물질 정의가 일치해 이 항이0이다. 이 반증은 큰 cell 적분 차이를 곧바로 큰 끝점/이온화 오차로 읽을 수 없음을 보여주며, 실제 재구성의 비보존을 없던 것으로 만드는 논리는 아니다.

**같은 해석적 물질의 RTE 대조:** 정적·상대론/고정blue 두 양의 모델에서 독립 DOP853 해와 별도 양의 formal 적분 끝점이2e-12 상대 이내에서 일치했다. reference의 독립 공간 에너지 적분 잔차는1.32e-13/3.42e-14였다. 기존 SC의2→4→8→16→32→64 cell 분할에서 다음 오차가 줄었고, Gauss48→96 정밀도 변화는 약1e-14 이하였다. 따라서 이 차이는 단순 적분기 정밀도 부족이 아니라 고정 물질과 SC source 보간의 이산 차이다.

| 모델 | SC2cell 끝점오차 | SC64cell 끝점오차 | SC2cell 물리 수지 오차 | SC64cell 물리 수지 오차 |
|---|---:|---:|---:|---:|
| 정적 | .00460209 | 9.07130e-6 | 2.09896 | 1.45625e-4 |
| 상대론/고정blue | .0115082 | 2.21418e-6 | .230442 | 1.27640e-5 |

정본 `A/same_material_cell_v1/direct-tests.log`는10시험 통과(1.87초)와 해석적 물질의 직접 실측을 포함한다. 후속 oracle 함수의 비교까지 추가한 직접시험도10통과(1.86초)다. 이 작은 제조해가 실제90격자 물리 수용이나 전체 ray의 수렴을 대신하지 않는다.

**기존 실제66069 packet의 조건부 실증:** `scripts/compare_cmfgen_packet_cell_material.py`는 기존 관측기와 같은 s-선형 chi/eta/b/blue를 사전 고정하고, 정확한 양의 formal 적분을 기존 SC와 비교했다. 새 수송/whole/원자료 변경 없이 기존 `ray_actual_packet_v1/observed`의 final args·실제 blue·경계·ray를 읽었다. 해당 packet은 point 추가가 없으며 central ray114nodes/원래90깊이/확장24다. 깊이33/58/59/60/61의 양방향10셀에서 **기존 SC 끝점을 직접 재현**했고1/2/4/8/16/32/64 국소분할을 했다. 정본은 `A/same_material_packet_v1/same-material-packet.json`이다.

33 inward는 원래 끝점과 같은 선형 물질의 독립 해 차이가2.17675%이며64분할에서는4.04569e-5(상대값)다. 58–61 셀은 원래 차이가 작은데도 source 재구성 적분 차이가 남고 분할로 줄어든다. **이것은 한 주파수·중앙ray·고정blue/입사강도의 국소 검증**이다. native 연속 물질의 유일한 복원·전체 angular J·전주파수 58–61 지배성·전체 이온화 오차로 확대하지 않는다. 원래 소스/장/실패·출력은 모두 보존했다.

최신 통합회귀는 **119 passed/1 skipped in28.14s**, `C/regression_v3/regression.log` SHA256 `80fd8fedb4ca88cd29ed51fd3134c06b28f0f7233777f8cbc2f9dadbb741ff22`다. 같은 목적의 중복 외부 검토는 추가하지 않았고 Astra agent도 종료했다. 현재 새 Slurm/수치 프로세스는 없다. 다음 whole은 자동 재시도하지 않는다. 다음 작업은 동일 물질 재구성의 양의 cell 수송을 실제 packet 범위에서 검증하고, 원래 엔진 경로의 고정T 전체 SE/공유ne 갱신을 준비하는 것이다. 새 cell 해나 국소 끝점 통과를 수렴앵커·최종 에너지1%·v2 Si/S 동질성으로 바꾸지 않는다.

### 후속 구현 범위 — 물질과 양립하는 양의 연속 cell 수송

이전 턴은 실제 Si/S 채택과 독립 RTE/실제 packet 증거를 얻었으므로 진행 턴이다. Astra가 제시한 수리 분기를 이어간다. 먼저 같은 s-선형 chi/eta/b/blue 모델에서 양의 formal 적분으로 endpoint를 구하는 private cell 함수를 구현한다. 기존 SC source를 eta로 역산하거나 인구를 native에 주입하지 않는다. 광학깊이 적분의 모든 지지를 포함하되 먼 구간은 양의 적분 가중치와 source 상한에 의한 명시적 오차 경계로 처리한다. Gauss 차이는 추정 오차이며 전체 엄격한 수학 인증이라고 표시하지 않는다. 약한 비영 방출·얇은 셀·두꺼운 지수 경계층·opacity 대비·실제 packet에서 독립 적분과 비교하고 비용을 실측한다. 이 결과와 실제 원래 SE/전자밀도 갱신 경로의 확인을 함께 다음 물질–장 계산 준비에 사용한다. 현재 생산 RTE/원자료/완주장·앵커 기준은 그대로이며 새 whole 제출은 없다. 각 offline CPU1/4GiB/300초·새 root만 사용하고 모든 기존 산출물을 보존한다.

### 병렬 물질 갱신 실행 계약 — native 고정 J60/고정T SE·ne

목표는 완성 J60을 외부장으로 유지했을 때 native 전체 원소 SE/전자밀도가 어떻게 바뀌며 같은 완료 상태의 C owner와 맞는지 확인하는 것이다. 원래 CMFGEN `USE_FIXED_J`/`RD_LAMBDA`/`SOLVE_FOR_POPS`/`POPTOION` 경로를 쓰며 C 인구 주입은 없다. Astra 평가 요청 `01a1551e-a2bb-7cd3-8592-547dd8f384e1`는 이 병렬 실행과 아래 자원/제외 범위를 지지했고 agent를 종료했다. 이는 새 field whole이나 released-T 실행 승격이 아니다.

- 입력: StageA59의 전체 원래 인구/T/ne·POINT/SCRTEMP·물리/원자자료와 native 부모 바이너리. 고정 외부장은428782의 완성 EDDFACTOR(143638040bytes)이며 전197291주파수/90깊이 J가 angular producer와 byte 일치한 파일이다. 초기 물질도59임을 별도 확인한다.
- 유일한 제어 변경: `USE_FIXED_J F->T`, `DO_LAM_IT F->T`, `NUM_ITS 19->4`. `FIX_T=T`, `FIX_NE=F`, `DO_DDT=F`, `DO_NG=F`, `DO_LAM_AUTO=F`, `DO_T_AUTO=F`, `INC_GRID=F` 및 나머지 입력을 유지한다. 물질이 바뀌어도 J를 동결하는 의도된 조건부 문제이지 장–물질 공동수렴이 아니다.
- 관측/판단: 각 완료 상태의 전체 SE/전하잔차·ne·Si/S 분율·양수성·원소보존을 확인한다. 목표 수렴 문턱은 기존 전체 SE/전하1e-8을 유지한다. 네 번 완주나 작은 step만으로 수렴이라 하지 않는다. C 비교는 그 완료 상태의 T/ne/원소밀도와 같은 J60으로 행렬/인증을 새로 구성한다. 이전59 행렬을 다른 물질 상태에 재사용하지 않는다.
- 제외: native RD_CONT_J의 H/K/flux 대용값과 fixed-J heat/luminosity는 물리 출력으로 쓰지 않는다. OBSFLUX·에너지/스펙트럼 PASS·최종 v2 동질성 및 released-T 앵커 판정은 금지한다.
- 자원: **CPU16/OMP16/128GiB/GPU0, 부모45분/자식40분, retry0, 노드 미고정**, partition `a10,a40,a100,a100_pcie,h100,h200`에서1회. RT pre-pop의16GiB를 관성으로 쓰지 않는다. StageA427702의 과거 step RSS97819832KiB(약93.3GiB)/요청200GiB가 근거이며 Lambda 절감은 아직 추정이다. 파일8GiB·초기 복제본 포함 root16GiB를 제한한다. 복제 BAMAT2.40GB/SCRTEMP77.3MB/새장143.6MB와 추가 수GB 예상, 공용 여유252TiB를 확인했다.
- 보존/중단: 새 job-ID root만 사용하고 기존 run/field/native/원자자료를 보존한다. 고정 J 파일의 실행 후 hash와 seed 불변을 확인한다. 오류·비유한·OOM·시간·출력 상한이면 최초 자료를 보존하고 자동 재시도하지 않는다. 부분 결과는 POINT/SCRTEMP가 일관된 마지막 완료 상태만 구별해 해석한다.
- 재현/증거: `run_cmfgen_fixed_field.py --seed SEED --field RUN428782 --run NEW_ROOT --native /gpfs/kjhan/cmfgen_src/cur_cmf`; `seq_logs/fixed-field-inputs.json`에 실제 입력 경로/제어/장·바이너리·runner 신원, `fixed-field-runtime.json`에 완료/변화/보존 상태를 남긴다. 작은 직접시험 후 runner를 봉인해 Slurm에 제출한다.

**제출:** 위 계약의 Slurm **428864**를2026-10-20 02:11:20 KST에 제출했다. 새 root는 `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_fixed_field_428864`, 봉인 runner는 `/gpfs/kjhan/cmfgen_runs/lumina_fixed_field_runner_20261020-hr18tF`다. `runner.sha256` 자체 SHA256은 `adc8e555d5cdd197d2f9532b17aeaa7bb3119d82f2927df29299844ceeffd526`, 실행 전 직접시험은21 passed in.29s다. 한 번만 제출했고 같은 job을 계속 확인한다. 관측 timeout을 재제출 이유로 삼지 않는다.

**첫 실행의 운전자 판독기 결함:** 428864는1분17초에 종료됐으며 native 단계는1분00초 후 운전자에 의해 CANCEL됐다. 일반 `error` 단어 검색이 원래부터 있던 `Warning: Possible error in reading SN_HYDRO_DATA with RD_SN_DATA.`를 fatal로 오인했다. native가 물리 실패했거나 OOM인 것이 아니다. 실제 step RSS5516924KiB, POINT/SCRTEMP는59, T/ne 불변, EDD/seed hash 불변, root3042253631bytes다. 최초 자료와 실패판정은 보존했다.

이 실패를 재현한 음성 대조를 추가하고, 행 머리 Error/Fatal/Fortran runtime error/Program received signal 및 비유한 값만 오류로 잡도록 판독기를 수정했다. isotope 경고는 OUTGEN에 그대로 보존하며 물리 수용으로 승인한 것이 아니다. **동일 바이너리·입력의 무변경 자동 retry는 하지 않는다.** 수정 판독기를 새 runner에 봉인하고 직접시험 뒤, 같은 과학적 조건/자원 상한에서 고정-J 실행을 새 root로 한 번 진행한다. 이는 구현 결함 수정 후의 명시적 재실행이며 이전 실패를 없던 것으로 하지 않는다.

**수정 후 실행:** Slurm **428866**을02:16:54 KST에 제출했다. 새 root `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_fixed_field_428866`, 새 봉인 runner `/gpfs/kjhan/cmfgen_runs/lumina_fixed_field_runner_20261020-RNCMpF`, `runner.sha256` 자체 SHA256 `f163f0655b657a9935c02ed03657d3674ea3b79ba2b0d4584628ac3546dc9cf4`다. 경고/진짜 오류 구별을 포함한 직접시험22통과(.30초) 후 제출했다. 02:17:43 KST 조회에서 syn01/RUNNING48초였고 기존59에서 첫60회차로 진입했다. 두 root와 최초 실패 기록은 모두 보존한다.

### 물질 기반 RTE 수리의 실제 angular packet 착지

새 `scripts/cmfgen_material_cell.f90`는 정해진 양의 s-선형 chi/eta/b/blue를 source 비율 보간 없이 직접 formal 적분한다. 얇은 셀은 원래 공간좌표의 양의 적분, 두꺼운 셀은 optical-depth 적분을 쓴다. 큰 optical depth는 유한 구간 밖 source를 상수로 확장한 양의 Laguerre8/16 적분에서 그 알려진 tail을 정확히 빼고 상한도 확인한다. 다른 경우에는 양의 exponential mass를 보존하는 Gauss4/8 분할을 사용한다. 반올림으로 미량 방출을 지우지 않도록 convex 합을 사용하고, 큰 입사강도와 매우 작은 감쇠의 곱은 log에서 계산한다. intensity floor·eta 역산·flux 맞춤은 없다. 오차 추정과 tail 상한을 구별하며 전체 엄격한 구간 인증이라고 표시하지 않는다.

직접26시험에는 tau1e-280…1e280, target eta=0에서도1e-200인 비영 방출, 큰 Iup와 exp(-1000)의 유한 곱,24개 대비/실제blue 범위 및 잘못된 입력 거부가 포함된다. `A/material_cell_validation_v1/direct-tests.log`가 정본이다. 개발 중 opacity 대비 약4780/얇은 tau.0679의 이전 y좌표 참조 적분이 SciPy 수렴 경고를 내며 잘못된 값을 준 사례도 확인했다. 공간좌표의 독립 formal 적분과 DOP853는 새 함수 값 `.0002545318967136676`에 일치했고, 이전 참조 `.0002545318867758586`는 틀렸다. 참조 적분을 좌표별로 안정화하고 미수렴 경고를 명시적으로 거부했다. 물리 수용 기준을 완화해서 통과시킨 것이 아니다.

`A/same_material_packet_v2/same-material-packet.json`에서 실제 packet10셀의 새 함수와 독립 해 차이는 최대1.75e-14(local I scale)이며 대부분24 source 평가로 계산됐다. 이는 중앙ray의 조건부 검증이다.

이후 **전체 angular ray/공통 source 고정점까지 실제로 연결**했다. `scripts/prepare_cmfgen_material_transport.py`는 원본/기존v11을 수정하지 않고 사설 archive의 양방향 interior·경계4진행 지점을 새 cell 해로 바꿨다. 원래 경계·공간기하·blue·20회/1e-8 source 기준은 그대로다. 정본 `A/material_transport_packet_v1/material-transport.json`에서 기준과 새 함수 모두3회에 수렴했고, geometry/blue byte 동일·모든 실제 ray 비음수·독립 angular 합을 확인했다. **전90깊이 최대 J/H/K/N 차이÷local J는 각각 .00492580/.00661434/.00245521/.00392767**, 최대 J 차이는깊이50이다. 이 한 실제 주파수의 장 변화는≤0.7% 수준이며, 큰 셀 재구성 차이를 곧바로 그만큼 큰 J/이온화 차이로 읽어서는 안 된다.

CPU1 packet 비용은 기준 .06481초, 새 수리 .46526초로 증가했다. 이것을 전주파수 상한으로 간주하지 않으며 **기존40분 child 한도로 새 whole을 제출할 근거가 아직 없다**. 다음 구현은 동일한 선형 물질 응답의 양의 계수를 재사용하는 비용 절감과 cell 수지 검증이다. 정확도를 낮추거나 기존whole로 물리 수리의 효과를 대체하지 않는다. 이번 수리는 원래 SC의 암묵적 연속 eta와 명시적 s-선형 물질을 구별한 유한격자 수송 변화이며, 최종 연속물질의 유일한 정답/스펙트럼·에너지1%/released-T 앵커/SiS 목표 완료로 승격하지 않는다.

### 수용 기준 재대조와 실제 우선순위

위 비용은 CPU1에서 계산과 I/O를 포함한 wall 시간이다. 최신 통합 회귀는 **151 passed/1 skipped in28.38s**, `A/material_cell_validation_v2/regression.log` SHA256 `c09f56212db5c122d88f3c8f1ebf3431a490e4c2207f615dc183e2a52a046f27`이다. skip은 기존 scoped Fortran smoothing witness 시험이며 통과로 세지 않는다. 최신 producer/시험 사본은 같은 root의 scripts/tests, 실제 private RTE 바이너리/소스/archive 신원은 `A/material_transport_packet_v1/transport-identities.sha256`에 보존했다.

운전자가 `CODEX_CMFGEN_EQUIVALENCE_PLAN_v2.md`의 전체 범위와 Stage0/3/6을 다시 대조했다. **선택한 s-선형 eta와 SC source 사이의 cell 장부 닫힘은 Stage0 고정T 안정화에 명시된 별도 수용 문턱이 아니다.** 이를 final formal 에너지1%와 혼동해 새 필수 게이트로 강제하지 않는다. 1e25 차이는 버리지 않고 재구성에 조건부인 이산화 진단으로 보존한다. 전주파수/앵커 조건을 충족하기 전 어떤 장/율/인구에도 parity PASS는 금지한다.

따라서 새 cell 함수는 검증된 private 수송 대조 후보로 유지하되, 약7배의 packet 비용을 지불하는 승격·최적화를 native 물질 갱신의 선행조건으로 만들지 않는다. **현재 우선순위는428866의 완료 상태에서 전체 SE/ne와 Si/S 분포를 추출하고, 같은 새 물질/J60 조건에서 C owner를 다시 계산하는 것**이다. 이후 실제 장–물질 안정화에 필요한 수송 변화와 비용을 결과 단위로 평가한다. 기존 source 고정점 완주 v11/J60을 staging 외부장으로 사용하는 것은 그 장을 최종 oracle로 인증하는 것과 다르다.

최종 수용은 그대로다: 수렴 released-T matched-physics 앵커, 실제 SE/전하/RE·마지막3회 안정성·광도비1±1%, 새 Lumina 권위 상태 및 관측 formal/MC 스펙트럼의 공통 문턱·h/h2/h4·재현이 필요하다. native 고정-J의 대용 flux/heat나151개 시험을 그 증거로 대신하지 않는다. 이번 scope 재대조는 원래 기준으로 돌아가는 것이며 최종 문턱 완화가 아니다.

### 고정 J60의 실제 첫 물질 갱신 — 새 ne에서 전90점 재계산

이번 턴의 직전 작업은 실제 RTE 수리/고정-J native 실행을 착지했으므로 진행 턴이다. 428866에서60회차가 완료되고61,이어62회차로 진행된 것을 실제 Slurm/OUTGEN/POINT로 확인했다. 원시 MAXCH는60회차약1e7%,61회차에도 컸지만, 이것만으로 전체 물질 발산 또는 수렴을 판정하지 않는다.

`scripts/prepare_cmfgen_fixed_field_owner.py`는 POINT1/2가 모두 완료한 record의 고정 byte 구간만 두 번 읽고 일치할 때 스냅샷을 보존한다. 뒤 record가 쓰이는 동안의 SCRTEMP 전체 길이/마지막 상태를 완료했다고 가정하지 않는다. 원래59와 header/T·준위 대응·물리 제어·원자 view 해시를 확인하고, 이미 봉인한 owner J60 입력의 장 block을 byte 그대로 유지한 채 새 인구/T/ne만 교체한다. 기존59 행렬이나 인증은 재사용하지 않는다. 직접시험은 부분 후속 record·미완료 포인터·서로 다른 원래 물질·기존 출력·질량가중·다른ne/총밀도/가짜commit/부분 응답 거부를 포함한다.

**완료 물질60 정본:** `A/fixed_field_owner60_v1`의 `completed-state.npz`, `owner_input.bin`, `reference.csv`, `fixed-field-owner-inputs.json`이다.90깊이/1800변수를 보존했고 저장 record SHA256 `1b07e393b7e2d8cfa7a6ec0dc2ce86173f78ec59c226ad878449606c45dfaa9a`다.59 대비 ne 최대변화는 **10.9582%**, 전 원소 SL/terminal charge를 재합산한 최대 전하잔차는 **1.51724e-8**이다. 아직1e-8 문턱을 넘으며 T는 byte 불변이다. 원소총량 상대변화 최대는 Si1.28e-11/S3.63e-8/Fe3.25e-8/Ca6.48e-9/Co5.75e-5/Ni2.58e-4로, 원소보존이 완벽하다고 표시하지 않는다.

같은 완료60 물질/ne와 **동일한 J60 bytes**에서 봉인 C owner를 전90점 실행했다. `C_response`에 실제900행/완료 로그가 있고, 기존 바이너리 SHA256 `9636e47918f12d296a1c7f8093a383f7bc7fa820d645b2b72cee2da9b609d040`를 실행 전 확인했다. **Si90/90, S90/90 새 인증·실제 채택**, C SE잔차최대약3.8e-15/원소보존1.4e-15 이하다. 원래 Si rank/κ 실패는90점 모두 별도로 보존했다.

비교 정본은 `A/fixed_field_owner60_v1/comparison/{fixed-field-owner.json,fractions.csv}`다. 모든900행의 실제 native60와 C 후보/commit·새T/ne·59기준값을 남겼으며, 깊이 가중치는 같은 입력의 `rho*(v_outer³-v_inner³)`다.

| III 분율 | native60 vs C·물질60/J60 질량가중 median/p95/전체max 절대차(%p) | 전체 max 깊이 | 분율>1e-8 subset 최대 log 차이(dex) |
|---|---:|---:|---:|
|SiIII|.0018415/.130226/1.116594|27|1.66774|
|SIII|.0006098/.028442/1.402547|31|.09781|

원래 물질59/J60의 III 큰 격차(33.29%p/24.39%p)는 첫 native 물질 갱신 후 위 조건부 격차로 좁혀졌다. 비교마다 **같은 그때의 ne와 물질**에서 C를 새로 풀었으므로 단순히59 후보를60 상태로 재명명한 것이 아니다. 반면 SiIII의 저분율 구간 log 차이는 아직 크고, 전하/원소잔차·native 전체 SE 잔차도 해결/추출 전이다. fraction threshold subset은 열률/opacity를 포함한 최종 active union 전체가 아니므로 이를최종동질성PASS로 사용하지 않는다. 고정J는 여전히 공동수렴이 아니며 새released-T 앵커/스펙트럼은 없다.

물질61도 별도 `A/fixed_field_owner61_v1`에 완료 record만 보존해 같은 방식으로 C 재계산 중이다. 진행 중인428866과 최초 실패428864는 그대로 두며 현재 새 Slurm 제출이나 RT whole 재실행은 없다.

### 완료63의 실제 조건부 분포와 다음 물질–장 피드백

**428866은 COMPLETED/부모0/native0,36분16초**로 종료했다. 마지막63까지 네 번 갱신했고 J60/원본59/T 불변·인구 비음수·ne양수를 확인했다. native step36분05초, peak RSS5784328KiB(약5.52GiB), root8557750983bytes다. 정본은 `RUN428866/seq_logs/fixed-field-runtime.json`이다. 관측 timeout으로 재시작한 것은 없으며 종료된 이 job을 재제출하지 않는다.

완료61/62/63을 각 `A/fixed_field_owner{61,62,63}_v1`에 따로 보존하고 그때의 실제 ne/pop 및 동일 J60 bytes로 C owner를 새로 풀었다. 각900행/90점, Si/S 각각90/90 인증·실제채택이다. 최종63 실행108.174초(CPU1), record 저장 SHA256 `3228d4eefaaab4bdba63f648905bea9424889cb9a9ec81de5d63d33c675f5cce`다. 기존59의 행렬/인증 재사용은 없다.

| 완료 물질 | SiIII 최대 절대차(%p) | SIII 최대 절대차(%p) | SiIII 최대 log차이(dex) | SIII 최대 log차이(dex) |
|---:|---:|---:|---:|---:|
|60|1.116594|1.402547|1.667739|.097810|
|61|.1302255|.3693662|.670406|.007393|
|62|.1302255|.3678792|.0075035|.0032576|
|63|.1302255|.3678793|.0075035|.0032576|

log열은 분율>1e-8 부분집합이며 v2의 열률/opacity까지 포함한 전체 active union은 아니다. 최종63의 III 질량가중 median/p95 절대차는 Si .0018144/.1056680%p, S .0006098/.0284410%p다. 최종 비교 `A/fixed_field_owner63_v1/comparison/fractions.csv`에는 전90깊이 II–VI와 native59/native63/C후보·실제commit·T/ne가 모두 있다.

다음 표는 III 분율(%)이다. native63은 고정T/J staging이며 수렴 CMFGEN 앵커가 아니다. 작은 조건부 차이를 새 자기일관 해의 동질성으로 표시하지 않는다.

| 깊이 | native63 SiIII | C SiIII | native63 SIII | C SIII |
|---:|---:|---:|---:|---:|
|20|.0598610|.0596482|.917058|.920276|
|28|8.137427|8.107873|18.324213|18.453651|
|31|26.834463|26.777591|52.418048|52.785927|
|35|50.941901|50.885657|95.753033|95.820021|
|50|99.585065|99.583877|99.657443|99.657649|
|59|50.693501|50.563275|90.349715|90.365687|
|61|7.288630|7.238367|41.386740|41.409609|
|90|.00002920|.00002878|.00111390|.00111177|

native63의 전 원소 전하 재합산 잔차3.46582e-15, 마지막ne 최대|Δln|4.54083e-8이다. 그러나 기존 `ptf_gated_metrics.py --from-it 60 --to-it 63 --threshold 0 --format json`의 `A/fixed_field_native_v1/native-corrections.json`에서 마지막 fraction-active 부분집합 p95는3.516e-5%로 작아져도 최대는878.86%다. **Ni V(내부명NkV) SL12/깊이14, 해당 이온분율1.43702e-4**이므로 비활성 제외로 통과시킬 수 없다. native 전체 SE 분모는 미확보이며 correction은 residual이 아니다. 59 대비 원소총량 변화(Si2.57e-12/S1.99e-11/Co3.62e-6 등)도 구성된 원소보존행의 잔차와 구별한다.

**Ni V 해석 정정/직접 재현:** 위878.86%는 입력물질62에서 산출해63에 적용한 보정이지 완료63의 잔차가 아니다. 해당 변수1693의 실제 준위는 `3d6(1S)_1Se[0]`다. 완료 SCRTEMP59–63과 이번 append의 네 SOLUTION ARRAY를 대조했다. 재현은 `prepare_cmfgen_fixed_field_owner.record_bytes(..., record,90,1800)`의 `[1692,13]` 및 `ptf_gated_metrics`와 같은 printed-SOLUTION 행1693/깊이14 판독이다.

| 갱신 | 갱신 전 인구(cm^-3) | printed fractional correction | 실제 갱신 후 인구(cm^-3) | 실제 증가배수 |
|---:|---:|---:|---:|---:|
|60|2.21467014e-21|-9835.8|2.21467014e-20|10|
|61|2.21467014e-20|-977.9|2.21467014e-19|10|
|62|2.21467014e-19|-96.886|2.21467014e-18|10|
|63|2.21467014e-18|-8.7886|2.16785620e-17|9.7886188934|

앞선 세 번의 큰 보정은 MAX_LAM10의 실제 증가 제한에 묶였고63에서는 제안된 전체 step이 적용됐다. 마지막 printed5자리로 복원한 선형 후보와 실제63 차이1.93e-6은 직렬화 정밀도 영향이며 **물리 SE 잔차가 아니다**. 정밀 OUTGEN MAXCH878.86188934115239%와 실제 증가배수도 일치한다. 따라서 계속 남은879%의 오차나 Ni V 물리 기작 결함으로 단정했던 해석은 철회한다. 완료63의 정규화 SE/이후 장–물질 안정성이 미입증이라는 결론과 T-release No-Go는 유지한다. 진행 중428886의 입력·제어는 이 재해석만으로 바꾸지 않는다.

#### 후속 묶음 계약 — 물질63 재생장에 대한 전체 물질 피드백 연결

Astra 기획·평가 요청 `01a15546-7f28-7931-bb31-7c0c2e7b3e15`를 이번 결과 전체에 대해 한 번 수행했고 agent를 종료했다. 권고는 **완료 물질63→새 전주파수 J64/H64/K64→그 J64를 고정한 native 전체 SE/ne 한 번→완료 물질64**다. 의미 있는 최종 목표는 원래 released-T 앵커·새 Lumina 자기일관 분포/스펙트럼이며, 이번 실행은 그 과정의 교대 피드백 연결이다. 새 진단 통과를 목표 완료로 바꾸지 않는다.

- 범위/입력: 완료63의 정확한 POINT/SCRTEMP/RVTJ·T/ne/full model 상태와 원래59의 물리/원자료, 기존 검증v11의 동일 수송 커널/경계/주파수/producer를 사용한다. C 인구 주입, T해제, rate/원자료 변경, material-cell prototype 승격, full coupled Jacobian 주장은 제외한다. 작은 단계별 외부 감사는 추가하지 않는다.
- 구현: `prepare_cmfgen_feedback_build.py`가 기존v11의 outer60 검사·기록만 명시 세대64로 바꾸고 **STOP97은 유지**한다. 기존B/archive/native는 보존한다. 새 build는 `/gpfs/kjhan/cmfgen_runs/lumina_feedback_build_20261020-Gw516t/build`다. `run_cmfgen_field_feedback.py`는 기존59 전용 stage 검사를 약화하지 않고 새 복제본에 검증한 native63 restart를 이식한 뒤 두 계산을 별도 cwd에서 실행한다. 수송완주 전에 물질 계산을 시작하지 않는다.
- 상태 복제의 수리: 첫 offline preflight는 `CUR_MODEL_DATA`가59와 다르다는 이유로 거부했다. 실제 `WRITE_SEQ_TIME_FILE_V1`은 이 파일에 T/ED/원소 및 full-pop을 기록한다. 불변 물리 카드가 아니므로 완료63의 다른 restart와 함께 복제하도록 수정했다. 모델/원자 대응의 불변 검사는 유지한다. 새 Slurm을 실패시킨 뒤 고친 것이 아니라 제출 전에 실제 입력으로 확인했다.
- 제어: 양쪽 `FIX_T=T`, `FIX_NE=F`, `DO_DDT=F`, `DO_LAM_IT=T`, `DO_NG=F`, 자동전환F. 수송만 `USE_FIXED_J=F`, 물질은T. `LAM_SCALE_OPT=NONE`으로 native의 조건부 component correction 0.999 치환을 끄고 `SCALE_OPT=LOCAL`, `MAX_LAM=2`의 깊이별 공통 step 제한을 명시한다. J 혼합·사후 원소 재정규화·미량준위 삭제는 없다. `WRITE_RATES=F`로 이번 판단에 사용하지 않을 per-line/EW 대량 출력만 줄이며 실제 제어 차이에 기록한다. 원래 SE 조립/율은 바꾸지 않는다.
- 자원: CPU16/OMP16/**32GiB/GPU0**, 부모45분/두 자식 **공유40분**, outer1/물질1, retry0/no node pin, 파일8GiB/전체16GiB. Astra의128GiB 초기 제안은 실제native5.52GiB와 기존RT16GiB 완주로32GiB까지 줄였다. RT26분+물질약8분은 추정이며 보장치가 아니다. 물질용9분이 남지 않으면 완주장만 보존하고 묶음 미완료로 종료한다. 각 자식에40분씩 주거나 자동 재시도하지 않는다.
- 판단/증거: 새 J가 정확히 물질63에서 생산되고 angular producer와 실제 EDD 소비자가 일치하며197291×90/호출상한20/source·J·H1e-8을 만족해야 한다. 그 장을 소비한 native64의 실제 완료/양수성/T보존·전하·단계별 변화와 제한 영향을 함께 본다. J64는물질63의 장이므로 물질64/J64를 곧바로 공동수렴 쌍이라고 하지 않는다. 고정J 물질 단계의 dummy H/K·flux/heat/OBSFLUX는 제외한다.
- 잔차 공백: v2 원래 `max(inflow,outflow,n_i/t_ref)`를 유지한다. 기존 WRITE_RATES가 모든 활성 SL의 방향별 gross rate를 복원하는지 조사하고 부족하면 실제조립의 O(Nrow×Ndepth) 누적만 고려한다. BA norm/DGEEQU/correction/ion총율을 분모로 대체하지 않는다. v2에 없는 t_ref 수치를19.48일로 임의 지정하지 않는다. 미확보는 잔차 인증·T-release를 막되 이번fixedT피드백의 새 필수 게이트로 만들지 않는다.
- 보존/재현: 원래 run/필드/원자자료/실패 및 사용자 미커밋 변경을 유지하고 새 root만 사용한다. 실행 입력은 `seq_logs/feedback-inputs.json`, 각 clone은 `feedback-stage.json`, 결과는 `feedback-runtime.json`에 생산자/세대/해시/실제 단계종료를 남긴다. 재현 명령은 `run_cmfgen_field_feedback.py --original SEED59 --source RUN428866 --run NEW_ROOT --build NEW_BUILD --native /gpfs/kjhan/cmfgen_src/cur_cmf`이며 Slurm 내부 OMP16에서만 실행한다. 연결 완료≠안정화/물리 목표 완료이며 마지막 v2 전체 종료감사는 아직 열려 있다.

**실행 전 직접 확인:** 새 수송 세대 변경/STOP 보존·잘못된 입력 거부, 물리 제어·공유기한·기존 출력 보존 등을 포함해 최신 통합회귀는 **183 passed/1 skipped in28.89s**다. `A/feedback_validation_v1/regression.log` SHA256 `123e7b04e21086e3055fb9fec7fef25382e39ee9b406d360b250555464e81de4`이며 기존 scoped Fortran smoothing witness skip은 통과로 세지 않는다. 이전162통과/1skip 로그 `A/fixed_field_validation_v1/regression.log` SHA256 `602f29dec15657184239c947867cdd7bb5e3fc126681f30b184c8183be79d94a`도 보존했다.

실제 완료63 입력 검증과 격리 clone 작성은 `/gpfs/kjhan/cmfgen_runs/lumina_feedback_preflight_20261020-B8ckR2/{preflight.log,clone}`에서 통과했다. 복제 SCRTEMP 전체 SHA256 `2fa821ab6336335c71841da8a71393079233e20f3009af65c0221b60f0dfb66f`이며 record63 구간 SHA와 전체 파일 SHA를 혼동하지 않는다. 새 장 실행파일 SHA256 `413b516a024b35141c4b48a226106814ca880bb75b5d7cf6d20401585468ecaf`, build manifest SHA256 `276826fb64ebd0241eaf07920c8d0a42aa8aa60d2bf2541c4ed20f5e2f55be5b`다. 봉인 runner는 `/gpfs/kjhan/cmfgen_runs/lumina_feedback_runner_20261020-gpcCY6`, 8개 직접 의존 파일을 확인한 `runner.sha256` 자체 SHA256 `6808aa91ab9af31a45b6e682d9afbb13982770487341065afc4be7dc46995098`다. 원래 종료63 OUTGEN에는 LIMIT 치환 경고가 없어 그것이 실제 발동했다고 주장하지 않는다.

**실제 제출·시작:** Slurm **428886**을2026-10-20 03:03:30 KST에 위 봉인 runner/build로 한 번 제출했다. root `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_428886`, stdout/stderr는 `/gpfs/kjhan/cmfgen_runs/slurm_logs/field-feedback-428886.{out,err}`다. 03:04:05 조회에서 syn02/partition a10/RUNNING34초·CPU16/32GiB/GPU0를 확인했고 runner8개 해시는 모두OK, stderr는 비어 있었다. 노드를 지정한 것이 아니며 Requeue0/Restarts0다. 이 job의 실제 입력과 각 단계 결과를 같은 root에서 이어 확인한다. 제출 또는 실행 중이라는 사실을 새 장/물질64가 이미 완성됐다는 증거로 쓰지 않는다.

### 실행 중 병렬 원인 분리 — 실제 full-level 분배

미완료 묶음을 닫거나 새 게이트를 추가하지 않고, 현재의 작은 C/native 분포 차이가 SL 내부 분배의 큰 누락인지 조사했다. 원래 native `SUP_TO_FULL_V3b`는 occupation probability와 선택된 interpolation sequence를 포함할 수 있으므로 소스의 “Boltzmann” 주석만으로 같다고 가정하지 않았다. 완료63의 실제 `CUR_MODEL_DATA`를 Fortran 순차 record 그대로 판독하고 T/ne가 SCRTEMP63과 byte 같고 각 SL의 full-pop 합이 같은 SL 인구임을 확인했다.

비교 대상 C 식은 봉인 생산자 `C/source_snapshot/src/population_contract.c:374`의 `g_i exp[-(E_i-E_anchor)/(kT)]/Σ`다. 실제 C binary를 한 번 더 실행한 것이 아니라 **그 봉인 식의 독립 재구성**과 native 실제 full-pop의 대조다. 원자 view의 같은 energy/g/SL mapping을 사용했다. `CUR_MODEL_DATA`의 ZXzV는 이 파일에서 SiII=2이며 CSV ion_number=1과 구분했다. 초기 판독의 stage/charge 오인을 수정했고 원자자료 결함으로 분류하지 않았다.

| 이온 | full 준위 수 | SL 수 | 전90점 최대 재구성 차이(dex) |
|---|---:|---:|---:|
|SiII|125|69|3.160e-9|
|SiIII|147|99|3.905e-11|
|SiIV|61|50|1.869e-11|
|SiV|203|52|2.004e-10|
|SII|322|55|3.178e-8|
|SIII|256|79|1.789e-8|
|SIV|176|65|1.791e-8|
|SV|184|47|1.201e-8|

총 **1474준위×90=132660쌍**이며 양수 support 불일치0건, 실제 full→SL 합 최대상대차6.66e-16이다. 이 상태에서 SL 분배의 큰 불일치가 남은 이온분율 차이의 주원인이라는 가설은 지지되지 않는다. 이것으로 모든 rate·전하·열·수송의 E-동등이나 향후 다른 상태의 분배를 통과시킨 것은 아니다.

정본/실행은 `A/within_sl63_v1/{analyze.py,analysis.stdout}`이며 grammar-debug CPU1/4GiB/300초 안에서 재현했다. source SHA256 `ed8db40ec69fc1a38a88f3fc2bcbc946ab2298b30657fec533bf5f0c7fa859bc`, 결과 SHA256 `ab688794689d5ef27d376b71319e12fcab8bc4c5b4756e5c50c64013bdda1115`, native CUR SHA256 `d2b2aad8a4ed43fb982846afae6c2274e30380b054fb1a5544fe9f568bc4c496`, 봉인 C 식 source SHA256 `69bdec0e1475941cde7306f202841f080240c82aa0499f2abd124cdbedff120b`다. 재현 명령은 `python3 A/within_sl63_v1/analyze.py`다.

새 비교 입력 producer `prepare_cmfgen_feedback_owner.py`는 실제 새 J64를 직접 읽고 완료63 또는 실제 갱신64의 T/ne/pop을 전달한다. 예전J60·행렬을 재사용하지 않는다. 기본 완료장/소비자·원자료/SL 대응·세대 검사를 유지하며 native64는 실제 완료 runtime 뒤에만 허용한다. `summarize_cmfgen_fixed_field_owner.py`는 새 장의 생산물질63, 비교물질63/64, 원래59와 직전63의 변화량을 별도로 기록한다. 새 합성 직접시험은 구J/구ne 혼용·미완료64·원자료 변경·기존 출력 덮어쓰기를 거부하고 기존J60/물질59 기본값도 보존한다. 아직 이번 실제 J64 입력/응답이 완성됐다고 주장하지 않는다.

### 새 J64 완주와 고정 물질63의 실제 반응

위 준비 뒤 **새 J64가24분50초에197291×90 전주파수 완주**했다. 원래 pre-pop STOP97을 유지했으므로428886.0은Slurm에서FAILED/97로 표시되지만 계획된 수송 종료이며 부모는 native 물질64를 별도428886.1로 실행한다. 완료 marker64/197291/197291과 기존 angular/source/EDD 소비자 검사를 확인했다. 현재 이 수송 결과를 released-T 앵커나 열·스펙트럼 물리 합격으로 표시하지 않는다.

`A/feedback_owner63_J64_v1`의 실제 새 입력으로 C owner 전90점을108.393초에 실행했고 Si/S 각90/90 인증·채택,900행을 완료했다. 이번 입력은 J64를 직접 읽었으며 구J60를 쓰지 않았다. 이전 `fixed_field_owner63_v1/C_response`와 전900행의 **T/ne/원소총밀도/pre_fraction 십진 byte가 모두 같음**을 확인했다. 따라서 다음은 물질63을 같게 둔 채 복사장만 바꾼 조건부 반응이다.

| 깊이 | C SiIII·J60 (%) | C SiIII·J64 (%) | C SIII·J60 (%) | C SIII·J64 (%) |
|---:|---:|---:|---:|---:|
|20|.059648|.007897|.920276|.558791|
|28|8.107873|1.651646|18.453651|8.122991|
|31|26.777591|6.567104|52.785927|31.560049|
|35|50.885657|21.721979|95.820021|91.026426|
|50|99.583877|99.106721|99.657649|99.756512|
|59|50.563275|50.624689|90.365687|90.380710|
|61|7.238367|7.244474|41.409609|41.429967|
|90|.00002878|.00002836|.00111177|.00109705|

장만 교체한 C III 분율 최대 변화는 **Si31.88091%p(깊이34), S21.22588%p(깊이31)**다. J64 조건부 C 후보와 아직 갱신 전인 native63의 최대차도 Si31.86053%p/S20.85800%p다. Si/S 후보만 반영한 전자수 증분은 고정ne 대비최대10.6353%이나 전체 원소 charge solve가 아니다. 이전의 .1302/.3679%p는 J60 아래의 조건부 일치였고 새 자기일관 물리 해의 오프셋 해소가 아니었다. 이번 비교의 큰 변화는 동일 geometry/물질에서 얻었으므로 이 실험 안에서는 shell 평균과 중심의 차이로 설명할 수 없다. 이를 수렴 사상의 비수축성 또는 다른 물리 기작 결함의 최종 증명으로 확대하지 않는다.

정본은 `A/feedback_owner63_J64_v1/comparison/{fixed-field-owner.json,fractions.csv}`, 장만 비교한 직접 재현은 `A/feedback_field_only63_v1/{analyze.py,analysis.stdout}`이다. 다음은 실제 native64와 그 새 ne/J64의 C 응답이며, 완료63 비교를64로 재명명하지 않는다. 최신 통합회귀는 **190 passed/1 skipped in28.67s**, `A/feedback_validation_v2/regression.log` SHA256 `2c4e48294240e81ee3e3e46ac1415f201217ff9bf6e9c5de41400071d897391a`다. 동작·계측 통과를 최종 목표 달성으로 대체하지 않는다.

### 실제64A 종료 — trace 상대증가 제한이 bulk 갱신을 막음

428886은 **34분22초/부모0**으로 종료했다. 실제수송1490.08초, native물질545.20초(9분06초)/0, peak RSS는각5701524/5645416KiB다. root7743058078bytes·source/T 불변이며 `seq_logs/feedback-runtime.json`이정본이다. source/J/H최대잔차9.4251e-9/9.9174e-9/9.9991e-9·max5calls를확인했다. 연결 완주는 맞지만 이온화 분포 목표는 **REWORK/미완료**다.

`A/feedback_owner64_J64_v1`에서 완료64의 새ne와같은J64로C를108.240초에다시풀었다. 각Si/S90/90인증·실제채택,900행이다. native64A/C III최대차는 Si31.83751%p/S20.84906%p로여전히크다. 원래Si rank/κ실패표기도별도로유지한다. nativecharge잔차는3.78087e-15지만 charge정상만으로물리수렴을판정하지않는다.

직접원인추적은 `A/feedback_native64_v1/{native-corrections.json,scaling.py,scaling.stdout}`이다. native LOCAL은 모든 원소와 ne에 같은깊이배율을준다. 가장 큰raw증가를요구한것은 **CaV SL60/변수703/깊이26**, p63=2.81535e-46→p64A=5.63071e-46이었다. CaV이온분율1.19048e-20, 해당준위/Ca분율2.41927e-51이다. 이 극미량의2배상한이 α=4.4147e-5를만들어전체변수/ne를축소했다. 총 raw최대증가2.2652e6%, fraction-active부분집합최대는SiV SL45/d34의55001%(이온분율1.45e-8)다. 비활성 제외로 최종 판정을 통과시키지 않는다.

| 깊이 | 적용 공통α(직렬화 정밀도) | SiIII 이전→비감쇠 제안→실제64A (%) | SIII 이전→비감쇠 제안→실제64A (%) |
|---:|---:|---:|---:|
|26|4.4147e-5|2.09273→.46376→2.09266|8.25360→3.87394→8.25341|
|28|6.9401e-5|8.13743→1.70125→8.13698|18.32421→8.25881→18.32351|
|31|4.1381e-4|26.83446→6.94636→26.82623|52.41805→32.36269→52.40975|
|34|6.8614e-4|83.82694→52.83812→83.80568|88.69848→77.85112→88.69104|

비감쇠ne최대변화제안10.305%에실제최대는.0775112%뿐이며, d31은7.1148%제안에.00294418%만적용됐다. printed자료의모든비감쇠인구후보는양수(최소인구비율.00779,ne최소비율.99988744)다. 원래 방정식의 최종 SE 잔차 인증은 아니다. 운전자가 적용한 수치제약이 이번 bulk 응답을 막은 것이며 새 원자자료 결손이나 SL 분배 결함으로 재분류하지 않는다.

#### 같은63/J64의64B 통제 실증 계약

Astra 평가 요청 `01a15579-8a8e-7803-a9e4-21b38888bb6c`는 앞선 LOCAL/MAX_LAM2가부적절했다고평가하고 **MAX_LAM만2→1e6**인 native한번을권고했다. agent는종료했다. 이번물질·장피드백묶음의수정이며작은확인별새감사를추가하지않는다.

- 목표: 제한에막힌실제전체물질/ne응답을적용하고같은조건의C비교로다음새장의출발상태를선택한다. 기존64A는대조군으로보존한다. 시험군도같은63에서시작하므로native카운터64이며새root의64B로구분한다. J64의생산물질은계속63이다.
- 유일한물리/수치입력차이는 `MAX_LAM=1.0D6`이다. LOCAL/NONE·원래native·T고정/free-ne/steady/LAMBDA/NG·자동전환F·NUM_ITS1을유지한다. 수송재계산0,새root만사용한다. 충분히큰값은알려진positive후보에서상한을비활성화하기위한값이지새물리기준이나무제한반복허용이아니다.
- 실행glue는 `run_cmfgen_trace_cap_ab.py`/`submit_cmfgen_trace_cap_ab.slurm`이다. 기존clone·공유기한·중단제어를재사용하고원래native/RTE/SE식은수정하지않는다. J64는완료장에읽기전용연결하고물질계산의독립복제본에서소비한다. 원본63/64A/필드는전후해시보존한다.
- 예정step: 같은갱신전상태/J/raw후보를확인하고모든깊이α=1 및실제density의비감쇠후보일치를검사한다. 허용폭은ES12.4출력의반올림과density연산IEEE오차로만계산하며인구floor는없다. 여전히작은배율/사후수정이면실증실패다.
- 상태: finite·양수support·ne양수·T불변·solver오류없음·전하보존과봉인63원소총량보존을확인한다. 총량변화를원래NC행잔차나전체SE잔차로바꿔부르지않는다. 음수/보존위반이면clipping·재정규화없이종료한다.
- 선택: B의새ne/J64로C전90점을새로푼다. 각Si/S의II–VI에서 단계별질량가중p95절대차중최댓값과전90점전체최대차가63/J64와64A보다모두감소하고support/sign문제가없을때만B에서다음새장을권고한다. 지표가악화/엇갈리면tracecap원인실증과B의후속주행선택을분리해보고한다. 그때같은J에서상한증가/여러update반복은하지않는다.
- 자원/보존: CPU16/32GiB/GPU0·부모45분/공유40분·native1/retry0/노드미고정·파일8GiB/전체16GiB다. C비교는후속CPU1/4GiB/300초이내이며기존90점/행렬을재사용하지않는다. 이번C비교는staging선택이지v2잔차인증이아니고T-release와최종목표는계속미완료다.

**실행 전 직접 확인:** 단일 상한 변경/원본 보존·극소 인구의 전체 step·부분 step 거부·직렬화/IEEE 정밀도와 실제 기존64A 제어의 일치를 확인했다. 최신 통합회귀는 **195 passed/1 skipped in28.58s**, `A/trace_ab_validation_v1/regression.log` SHA256 `65f862cf49b86cb1435d03ec44eef1d67b0927feb1cf06f43d08e3f221dcbeb4`다. 봉인 runner `/gpfs/kjhan/cmfgen_runs/lumina_trace_ab_runner_20261020-FxwbSc`의9개 직접 의존 파일을 검사했고 `runner.sha256` 자체 SHA256은 `221cade11693ed3af7f95fb8f4dea63bb3f52fb138fbbee60d64126c92f4b648`다. 원래 native/source/수송 build/대조64A/J64는 바꾸지 않았다.

**64B 실제 제출:** Slurm **428934**,2026-10-20 03:56:47 KST 제출/시작이다. 새 root `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_trace_ab_428934`, stdout/stderr `/gpfs/kjhan/cmfgen_runs/slurm_logs/trace-cap-ab-428934.{out,err}`다. 03:57:38에syn02/a10/RUNNING52초·CPU16/32GiB/GPU0·ReqNodeList null·Requeue0/Restarts0,봉인9파일OK·stderr비어있음을확인했다. 아직예측한fullstep이나B의물리진전을달성했다고하지않는다. 같은job의종료/실제예측재현·상태/보존·B새ne/J64 C응답으로다음새장출발점선택을이어간다. 완료한64A는재시작하거나덮지않는다.

### 64B 종료와 조건부 출발점 선택 — 다음 실제 새 장으로 진행

428934는 **COMPLETED/부모0·9분39초**, native 자식9분12초/0·peak RSS5646580KiB다. `seq_logs/feedback-runtime.json` SHA256 `f3184fd7436eda5499cea9910d9fb12713bcf823c3b284f04fea9f6d2dd24766`에서 다음을 확인했다.

- 같은63/J64의 raw correction은64A와 출력 byte가 같고 모든 실제 density가 비감쇠 후보와 출력/IEEE 허용폭 안에서 일치한다. 전 깊이 최소 scale은1이다. 알려진 극미량 CaV 증가 제한이 bulk 응답을 억제했다는 원인은 실증됐다.
- 최대 ne step10.3050337%, 전하 잔차5.347458e-15, 봉인63 대비 원소총량 변화 최대1.9984e-15다. finite·양수 support·고정T·source/field 보존을 확인했다. 원래 full SE나 NC행 잔차를 인증한 것이 아니다.
- SCRTEMP 전체 SHA256 `41fd16b1d00e3237d46394b9c98e41484075aa2d795511ca43f788a0ff1ad026`, 완료64 저장구간 SHA256 `274baaeaf6e36f06b44779372b60c7f531176266bfbd3e2c4b86c5b8a4a5799a`이며 구간/전체 해시를 구분한다.

`A/trace_ab64B_J64_v1`의 실제 새 입력으로 C를108.503초에 실행했고 각Si/S90/90 인증·실제채택, II–VI900행을 완료했다. **B의 실제 ne·인구와 같은J64**를 사용했으며63/64A 응답이나 행렬을 재사용하지 않았다. 비교 정본은 `comparison/{fixed-field-owner.json,fractions.csv}`다. `compare_cmfgen_feedback_candidates.py`는 세 입력과 동일 C binary/질량 geometry, B의 전체 step/보존을 확인해 `A/trace_ab_selection_v1/candidate-comparison.json`에 **NEXT_OUTER_SUPPORTED/64B**를 기록했다.

| 원소·II–VI 전체 |63/J64 |64A/J64 |64B/J64 |
|---|---:|---:|---:|
|Si 단계별 질량가중 p95 중 최대(%p)|5.350994|5.342028|.129712|
|Si 전90점 최대(%p)|31.860534|31.837506|1.571491|
|S 단계별 질량가중 p95 중 최대(%p)|.302131|.301679|.0773055|
|S 전90점 최대(%p)|20.858302|20.849363|1.137511|

각 원소의 두 지표가63/64A 대비 모두 감소했고 support 불일치0이다. 이 선택은 고정T의 다음 출발점 결정일 뿐 v2 잔차/상태/스펙트럼 PASS가 아니다. **온도 해제·고정J 추가 polish·상한 추가 증가·native 물리 소스 변경은 하지 않는다.** Fable 호출/대체 검증2는 취소 상태이며 작은 확인마다 Astra를 다시 호출하지 않는다.

#### 이미 정한 조건부 후속의 집행 — 64B에서 실제 J65와 물질65

목표는 구 장 조건부 일치가 새 장에서도 유지되는지 보고 장–물질 안정화 방향을 확인하는 것이다. 원래 최종 목표와 문턱은 유지한다. 완료64B를 보존한 새 root에서 **실제 전197291주파수 J65/H65/K65 생성→그 J65를 소비하는 native65 한 번**만 수행한다. C 인구 주입이나 released-T 앵커 주장은 없다.

- `run_cmfgen_field_feedback.py`는 명시한 완료 부모 record와 상한을 받도록 확장했다. 기본63/MAX_LAM2 동작과 이미 봉인된 runner는 변경하지 않는다. 부모64에서는 위 선택 근거와 실제 종료/해시/물리 카드/원자료/POINT/SCRTEMP/RVTJ 연결을 확인하고 **선택된64B**만 재시작한다. 현재는64/MAX_LAM1e6·LOCAL/NONE·고정T/free-ne/steady/LAMBDA1·NG/자동전환F다. 다른 수치 상한을 임의로 받지 않는다.
- `prepare_cmfgen_feedback_build.py`는 기존 검증v11의 세대 검사·완료 기록만65로 바꾸고 STOP97을 유지한다. 새 build `/gpfs/kjhan/cmfgen_runs/lumina_feedback65_build_20261020-7X42lp/build`의 exe SHA256 `513bc17344a5f4f768a69adc9104c4dcc9b651426ad600942a6a55e618a18077`, manifest SHA256 `01f281949a87543d40573ed9352da2d3540559d43ed748aabd6c3dad41da66d1`다. 커널/archive/native 원본은 불변이다.
- 후속 비교 producer는 `--reference-parent`와 `previous_parent`로 실제 부모를 기록한다. 기존 `--reference63`와 과거 NPZ 판독은 유지한다.65 자료를 이전63으로 재명명하거나 구J64/구ne를 전달하지 않는다. 실제 새 완료장과 갱신65의 새 ne로 C를 다시 실행해야 한다.
- 자원은 CPU16/OMP16/32GiB/GPU0, 부모45분/자식 공유40분·outer1/물질1/retry0/노드미고정·파일8GiB/root16GiB다. 수송에 시간이 걸려 native 예산이 부족하면 완주장을 보존하고 미완료로 종료한다. 관측 timeout을 재실행 근거로 쓰지 않는다. 이후 C 비교는 grammar-debug CPU1/4GiB/300초 이하다.
- 원본/63/64A/64B/J64/원자자료/실패를 유지한다. 중단이나 음수/보존위반 때 floor·clip·renormalization·자동retry를 적용하지 않는다. 고정J native의 dummy H/K 기반 OBSFLUX/heat/L은 물리 판정에서 제외한다. 전체 SE 분모/최종 열·광도·수렴앵커 공백은 그대로 미완료로 남긴다.

실제64B의 검증/격리 clone은 `/gpfs/kjhan/cmfgen_runs/lumina_feedback65_preflight_20261020-rUUhA5/preflight.log`에서 성공했다. clone은POINT1/2=64, 다음세대65, native binary SHA256 `f2b9afcc064037a413bf206047a6b7c813882dc40f894436e5de704a32b232f1`를 보존했다. 비교 선택 직접17시험은 `/gpfs/kjhan/cmfgen_runs/lumina_feedback_selection_tests_20261020-hXp2SD/tests.log`에서 통과했다.206통과/1skip 통합회귀는 `/gpfs/kjhan/cmfgen_runs/lumina_feedback65_validation_20261020-IlYGic/regression.log` SHA256 `70c6a492b20f9d89dca0aeb8b6227505f8074ec42c9266a128958dc27a1eb7ac`다. 이후 보존 검사 추가분까지 최종 봉인 전에 확인한다.

**최종 실행 코드 시험/봉인:** 전하·봉인 부모 총량·양수 support 및 기존 root 한도의 위반이 완료로 남지 않도록 한 코드까지 **206 passed/1 skipped in28.78s**다. `/gpfs/kjhan/cmfgen_runs/lumina_feedback65_final_tests_20261020-3DvDGk/regression.log` SHA256 `241806bee21426d2607d54c5620c4d5a089ebc801a9b4c50a628cf0b022c8b62`이며 scoped smoothing witness skip은 통과로 세지 않는다. 봉인 runner `/gpfs/kjhan/cmfgen_runs/lumina_feedback65_runner_20261020-b1QGl4`의8개 직접 의존 파일과 `runner.sha256` 자체 SHA256 `599f58122f89b08313d96e6e05828d970da7345c62236ac8cd3d6f66396dad08`을 확인했다. 원래 봉인 runner와 사용자 변경은 유지했고 commit/push는 하지 않았다.

**실제 새 장 제출/시작:** Slurm **428943**,2026-10-20 **04:15:41 KST** 제출·시작이다. `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_428943`와 `/gpfs/kjhan/cmfgen_runs/slurm_logs/field-feedback-428943.{out,err}`에서 잇는다.04:15:52에syn02/a10/RUNNING11초·CPU16/32GiB/GPU0·노드미고정·Requeue0/Restarts0, 봉인8파일OK·stderr 비어 있음을 확인했다. 아직 새 장/물질65 완료를 주장하지 않는다.

재현 명령은 Slurm 내부에서 `python3 RUNNER/run_cmfgen_field_feedback.py --original SEED59 --source /gpfs/kjhan/cmfgen_runs/toy06_19p48d_trace_ab_428934/material --source-record 64 --max-lam 1.0D6 --selection A/trace_ab_selection_v1/candidate-comparison.json --run NEW_ROOT --build /gpfs/kjhan/cmfgen_runs/lumina_feedback65_build_20261020-7X42lp/build --native /gpfs/kjhan/cmfgen_src/cur_cmf`다. 기존 root에서 다시 실행하지 않는다. 완료 후에는 `prepare_cmfgen_feedback_owner.py --root RUN428943 --reference-parent A/trace_ab64B_J64_v1 --record 65 --out NEW_COMPARISON`로 새 ne/J65 C 입력을 만들며,64를 지정하면 같은64B의 장만 바꾼 대조가 된다. 이후 같은 C binary로 전90점 응답과 summary를 새로 만들고 실제 J/물질 안정성을 함께 평가한다.

**현재 자체 확인:** trace cap 원인 재현과 B의 조건부 C 개선은 직접 증거로 확인했다. 원래 전체 물리 종료감사는 여전히 미완료이며 새 장의 실제 영향·native 전체 SE·마지막3회 안정성·released-T 앵커·최종 Lumina 스펙트럼을 코드/시험 성공으로 대신하지 않는다. goal은 active다.

### 428943과 병렬 준비 — 전체 SE 분모 공백의 보수적 충분조건

**현재 실행/기준은 바꾸지 않는다.** 아래는 기존 Stage0의 SE 잔차 공백을 작은 관찰로 확인하기 위한 private 구현 준비다. native 원본·428943의 봉인 runner/수송/물질 입력에는 적용하지 않았고 추가 native 계산도 제출하지 않았다. 이 준비의 통과를 원래 물리 목표나 SE 잔차 측정 완료로 바꾸지 않는다.

직접 소스 조사에서 `solve_for_pops.f`의 `WR_ASCI_STEQ`는 SOLVEBA 전에 실제 local SE를 출력하지만5유효자리이며, gross 유입/유출은 저장하지 않는다. `cmf_blkband_v3.f`의 `STEQ_ARRAY`는 이미 NC/ground/고정변수 교체를 거친 RHS다. 기존411115200byte `64B/material/STEQ_VALS`의 이 둘과 마지막 보정량을 완료64B의 전체 잔차로 재명명하지 않는다. 실제 충돌 조립은 `steq_multi_v10.f`에서 CNM 계산에 ne=1을 전달한 뒤 실제 ED를 곱한다.

Astra 기존 agent `01a15546-7f28-7931-bb31-7c0c2e7b3e15`에 이 **단일 최종 잔차 증명 설계**만 검토 요청했다(제출`01a1559d-e4dd-74e1-939f-a0fd76995193`). 평가는 **조건부 타당/가역적 소형 준비 진행**이며 agent를 종료했다. 작은 실행별 새 감사나 Fable/검증2 호출은 없다. 실제 native 적용은428943 결과 후 의미 있는 후속 묶음에서 상태·장·phase를 봉인해 결정한다.

원래 `D=max(I_full,O_full,n/t_ref)`는 그대로 둔다. native 실제 양의 CNM과 상태로 충돌 subset의 하측 경계 `Icol−,Ocol−`를 만들면 `D≥L=max(Icol−,Ocol−)`다. 따라서 유효한 `L>0`에 대해 상향 계산한 `U=R_plus/L≤1e-8`은 원래 SE 문턱의 **충분조건**이다. 미정인 `t_ref` 수치를 새로 만들지 않는다. `U>문턱`,0분모,음수/비유한 항,누락·중복호출·잘못된 매핑은 **UNPROVEN**이며 원래 SE의 실패/통과로 치환하지 않는다. 미량 cutoff/floor로 제외하지 않고 전 인구행×전 깊이를 기록한다.

SL j의 충돌 gross는 `I_j=ED*(Σ[l≠j]HN_l CNM_lj+HNST_j CNM_jj)`, `O_j=ED*HN_j*(Σ[l≠j]CNM_jl+CNM_jj)`다. upper-ground에는 ionization 유입 `ED*ΣHN_j CNM_jj`와 recombination 유출 `ED*ΣHNST_j CNM_jj`를 반대로 더한다. 실제 `EQ_IN_BA`로 여러 local 기여를 합치며 마지막 NC행은 SE 합에서 제외한다. 이는 실제 full→SL/dissolution 투영 CNM을 사용하므로 다시 근사 원자율을 계산한 것이 아니다.

구현은 한 관찰 파일을 만들기 위한 세 호출뿐이다.

- `cmfgen_se_collision_bound.f90`: 각 양의 곱·합의 하향 경계, 평가된 local 잔차의 global 합산 구간, 나눗셈의 상향 경계. 계산의 underflow mode가 증명 조건에 맞지 않으면 native 설정을 고치지 않고 미증명으로 남긴다.
- `cmfgen_native_se_witness.f90`: 실제 매핑/호출 coverage, 입력 POPS·교체 전 raw R·구간·두 충돌 하한, 별도 NC 원시 합/원소밀도와 charge 값을 한 파일에 보존한다. `LUMINA_NATIVE_SE_WITNESS=1`일 때만 활성이다. 관찰 함수는 원래 입력/SE를 바꾸지 않고 원래 IEEE 예외 flags를 복원한다. 저장은 `O(NT×ND)`이며 이번1800×90에서는 수MB 규모다.
- `prepare_cmfgen_se_witness.py`: 원래 source/라이브러리를 읽고 private 사본의 zero 직후 reset, CNM 직후 collect, SOLVEBA 전 write 호출만 추가한다. 관찰 코드를 제외하면 원래 source byte가 그대로 돌아오는 직접시험을 포함한다. 원래 식·조건·율·원자자료를 변경하지 않는다.

**증명 범위의 제한:** 여기서 `R_plus`는 native가 **이미 평가한 local binary64 SE 값들**의 global 합산 enclosure다. native 내부의 모든 주파수/과정 산술을 실수연산으로 재평가한 오차상한이 아니며17자리 저장만으로 그 자격을 얻지 않는다. NC·charge는 각 원래 분모로 별도 판단하고 RE/열·광도/스펙트럼은 이 증명으로 인증하지 않는다. 갱신65 직전 관찰은 **입력64의 잔차**이지 완료65의 잔차가 아니다. 실제 native 적용 때도 정확한 입력/장/phase와 이 제한을 유지해야 한다.

원래 물리 결과를 기다리는 동안의 소형 시험은 grammar-debug 각CPU1/4GiB/300초 이내다. 양의 경계는 정확 유리수, raw 합산은 큰 상쇄·subnormal·정확0, native 구조 모형은 bb/ion 방향·upper-ground·여러 local/global 기여·음수/누락/중복/기본 비활성·입력/SE/IEEE flags 보존을 검증한다. 첫 native 연결 시험/빌드는 관찰 glue의 공백 삽입점과 `GET_J_CHI.INC` 경로 때문에 실패했다. 이를 수정한24직접시험은 `.69초`에 통과했고, 원래 물리 계산 실패로 분류하지 않는다. 실패 `/gpfs/kjhan/cmfgen_runs/lumina_se_native_tests_20261020-gMeUDz`와 `/gpfs/kjhan/cmfgen_runs/lumina_se_witness_build_20261020-YN064H`는 보존했다. 이후 FTZ 거부 시험을 추가한 최종 코드의 실제 결과/신원은 이어 기록한다.

**최종 준비 결과:** gradual-underflow 조건을 확인하고 FTZ에서는 상향 경계를 주장하지 않는 코드까지 직접 **25통과/.68초**, 기존 경로를 포함한 통합회귀는 **231통과/1skip·29.26초**다. `/gpfs/kjhan/cmfgen_runs/lumina_se_witness_final_tests_20261020-PsOUOS/{direct.log,regression.log}` SHA256은 각각 `87d32174fc32b10424bd555031a9dba321387886a221cbc419f3407af7e1a13a`, `872d502b0510b66e97140697c3a5c2de65b1383f8fb12a55266affddf63d9bc0`다. skip은 기존 scoped Fortran smoothing witness로서 통과로 세지 않는다. 이전230통과 로그 `lumina_se_witness_regression_20261020-1NquiZ/regression.log`도 보존했다.

private build는 `/gpfs/kjhan/cmfgen_runs/lumina_se_witness_build_v3_20261020-44LxrD/build`다. exe SHA256 `f545980e7613faae7b0a0670121caf7104fa580c5737a1555de80b6b4647efa2`, `se-witness-build.json` SHA256 `ea5ba018ed6c520f99f44535e7517574cdef2e48c96a4e7af9148e16d70e6fbf`다. 빌드 전후 원래 source/실행파일/라이브러리와 참조 입력의 해시 불변을 확인했고, 빌드 기록에는 **actual_native_execution=false/noninterference=NOT_YET_REPRODUCED**로 남겼다. 구조 모형의 관찰 비간섭 시험을 실제 native 전체 계산의 bitwise 재현으로 승격하지 않는다.

재현은 허용된 소형 offline 환경에서 `python3 -m pytest -q tests/test_cmfgen_se_collision_bound.py tests/test_cmfgen_native_se_witness.py --basetemp NEW_TMP -o cache_dir=NEW_CACHE` 및 `python3 scripts/prepare_cmfgen_se_witness.py --native /gpfs/kjhan/cmfgen_src/cur_cmf --reference /gpfs/kjhan/cmfgen_runs/toy06_19p48d_trace_ab_428934/material/seq_logs/feedback-stage.json --out NEW_BUILD`다. 이들은 소형 시험/빌드이지 새로운 native 물리 실행 명령이 아니다. 현재428943은 원래봉인 코드를 계속 사용하며, 그 결과 없이 관찰 binary로 교체·재시작하지 않는다.04:37:19 KST에는 기존428943.0이 RUNNING21분23초, 실제 새 장163903/197291주파수를 처리했고 stderr는 비어 있었다.

### J65·물질65 종료와 실제 새 ne의 전90점 비교

428943은 **COMPLETED/0·34분23초**로 종료했다. RT는1489.886초/계획된97, native는545.158초/0, 각 peak RSS5696260/5643832KiB다. root7758680912bytes이며 source/T/양수 support를 보존했다. 실제 J65는197291×90/max6호출/source·J·H잔차7.99625e-9/9.99607e-9/9.99931e-9, angular/EDD 소비자 검사 완료다. J65의 생산물질은64B다. native65의 전하 잔차2.904308e-15, 봉인64B 대비 원소총량 변화 최대2.88658e-15다. 원래 NC행 잔차로 대체하지 않는다. ne step은 **10.6566204%**로 여전히 크다.

정본 `RUN428943/seq_logs/feedback-runtime.json` SHA256 `e4ad06efa18c809974e54db8a1f18f42a070626589ffd215069144ec595467a0`, 새 field SHA256 `ce81c4318742dd7efb49c00d718281113daa96d59fd20babb26cf5912ee8d551`, EDD SHA256 `0849bd2b018c7d09f2ac05a10f31bf9e97ba718d8e725f5cf43eaf415bc503cb`다. 완료65 저장구간 SHA256 `873f36e4afac05d40941b97a2ceafb4ed7b8485d459fd43bccece82161fe4a97`와 SCRTEMP 전체 SHA256 `81946a3046037e2c558403ea86580035876935f928cb7b020d3aff0472126b2f`는 서로 다른 신원이다.

수송 완료 직후 native 물질 단계와 병렬로 `A/feedback_owner64B_J65_v1`의 C를108.469초에 실행했다. `compare_cmfgen_fixed_material_fields.py`는 기존64B/J64와 새64B/J65의 state/profiles/columns/indices 및 전900행의T/ne/원소밀도/초기분율 decimal byte가 모두 같음을 검증했다. 각Si/S90/90 인증·실제채택이다. **장만 바꾼 C III 최대 변화는 Si33.48767%p(d34), S25.05956%p(d33)**, 질량p95는15.88119/1.44906%p다. II–VI 전체 최대는34.67674/25.06060%p다. 전17756190주파수·깊이 셀의 J/H/K/N 변화÷localJ 최대는.999972/1.142404/.883067/.906028이다. 미약장 극값이 포함되므로 이 최대값 하나를 최종 공통J 계약의 실패 판정으로 쓰지 않는다.

정본 `A/feedback_field_only64B_J65_v1/{field-only.json,fractions.csv}`, JSON SHA256 `8ea46c69e65fe7032795e31c8baa3390495da69c3e5d6afa5d1da14a5042a634`다. 재현은 grammar-debug에서 `compare_cmfgen_fixed_material_fields.py --old A/trace_ab64B_J64_v1 --new A/feedback_owner64B_J65_v1 --out NEW_OUTPUT`이다. 이 source는 과거63의 일회성 판독을 재사용 가능한 동일물질 비교로 만든 것이며, 다른ne/물질을 장 단독 효과로 받아들이지 않는다.

native 종료 후 `A/feedback_owner65_J65_v1`에서 완료65의 **실제 새ne/J65**로 C를108.495초에 다시 풀었다. 원래 구ne/행렬/인증은 재사용하지 않았다. 각Si/S90/90인증·실제채택/II–VI900행/support불일치0, CscaledSE 최대2.78e-15/3.74e-15이고 원래Si legacy rank/κ 실패90은 그대로 보존했다. native 전체 SE 인증과는 별개다.

| 원소 | II 최대 차이(%p) | III 최대 차이(%p) | II–VI 전체 최대(%p) | 단계별 질량p95 중 최대(%p) |
|---|---:|---:|---:|---:|
|Si|.00007841|1.013509|1.087272|.252386|
|S|.00166960|1.533010|1.533010|.0793062|

III의 fraction-only subset 최대log차이는Si .0423995dex/S .0114495dex다. full heat/opacity active union이 아니므로 원래 최종 분포 계약을 이 숫자만으로 통과시키지 않는다. 실제 native64B→65의 III 최대 이동은Si30.84028%p/S23.89773%p다. conditional C 일치는 상당히 가까우나 ne step10.305→10.657%와 큰 장 유발 변화가 남아 공동수렴·T-release는 **No-Go/미완료**다. 서로 다른J/ne의 conditional 최대값이 혼합 추세라는 사실만으로 일반적 비수축성이나 원자자료 구멍을 확정하지 않는다.

완료65 비교 JSON SHA256 `e388e449385accc324a8e502ff50360d4a8fdc3f0cd164903121bbaf65970740`, 실제 적용량 재현은 `A/feedback_native65_v1/{analyze.py,analysis.stdout}`다. source/output SHA256은 `d0c826eed22185b4868d7c19db436f7d9a242a11f372f633d0d9c68b4ae71f4a`/`a6cee707cd29f767077a79e2da990d703842a5f923166ad482e3748702b880ba`다. 실제 full-step은 ES12.4/IEEE 폭 안에서 일치했고 최소 scale1이다. 적용한 보정의 fraction-only subset p95/최대1091.58/135880%는 **완료65의 남은 오차가 아니다**. 재현 명령은 `python3 A/feedback_native65_v1/analyze.py`다.

#### 같은 새 장의 입력 재bin 중복만 제거

`prepare_cmfgen_feedback_owner.py --same-field-input A/feedback_owner64B_J65_v1`는 **동일root·동일부모64B·동일새J65/그field해시**만 허용한다. 기존 `replace_material`로 실제65의 T/ne/pop을 넣고 현재J65의 재bin/native-frequency block bytes를 그대로 복사한다. 원래 epoch·원자열·질량 geometry와 부모 인구를 확인하며 **율/행렬은 재사용하지 않는다**. 새 C 프로세스가 이 입력으로 다시 조립·인증·채택했다. 실제 준비시간6.769초, J65 block SHA256 `6b1c653b7245f775f04cb82ce3d11eeae96470fea4d6859cb6d6fffca0b0b31b`다. 전체재bin 입력과 byte동일을 시험했고 oldJ/oldne/다른epoch/변경된해시를 거부했다.

관련직접32시험 `.97초`, `/gpfs/kjhan/cmfgen_runs/lumina_same_field_cache65_tests_20261020-ankmLQ/tests.log` SHA256 `073f4b2a579f873893553406b79a105b96fdd94a3920ff01bcb44428640b37cb`, 관련통합 **243통과/1skip·30.01초** `/gpfs/kjhan/cmfgen_runs/lumina_feedback65_analysis_regression_20261020-AL9ZIh/regression.log` SHA256 `2231358261c690a7a5f1f3d9ae12b896e6989cdea16450d1b1da2abbac08adff`다. 준비 코드가 작동함을 실제 물리 목표의 완료로 대신하지 않는다.

### 다음 집행 계약 — 완료65에서 최대 두 번의 실제 장–물질 피드백

Astra의 전체 결과 평가 `01a155bd-d6b4-7250-a58b-5d78c55c9856`는 위 source/C/장변화/65종료를 묶어 **(a)동일 제어·최대2outer**를 권고했다. agent는 종료했다. 현재 큰 이동은 고정장 inner와 강한 outer 비선형 피드백으로 가장 잘 설명된다. 원자자료 완전성/전체nativeSE가 인증된 것은 아니지만 이를 새 결손으로 우선 설명할 추가 근거는 없으며, 지금 물리식/결합법을 무작정 바꾸거나 수렴을 보장하지 않는다.

- 목표/범위: 원래 Si II/III·S의 수렴 released-T matched-steady 분포/스펙트럼 동질성으로 가는 실제 공동 안정화 방향을 판단한다. **물질65→J66[생산65]→native66**, 유효하고 아래 조기중단이 아니면 **66→J67[생산66]→native67**까지다. 두 번 완주를 목표 완료로 바꾸지 않는다.
- 봉인 입력/변경: 완료65/원래59·검증v11 수송·native원본/원자자료·T고정/free-ne/steady/LAMBDA1·MAX_LAM1e6/LOCAL/NONE·NG/자동전환F를 유지한다. 수송 build는 기존세대검사/완료기록만66/67로 바꾸고STOP97은 유지한다. C 인구 주입,물리식/율/원자료/경계/T변경,추가cap증가,고정J polish,새witness 실제native 적용은 제외한다.
- 실행/비용/보존: 운전자가 각각CPU16/OMP16/32GiB/GPU0·부모45분/공유40분·물질1회·retry0·노드미고정·파일8GiB/root16GiB로 새root만 만든다. 최대2outer이며 각남은 실행이 중단되면 자동 연장/무변경재시도가 없다. 이전63/64A/64B/65/각실제장·실패·원본·사용자변경은 보존한다. 오프라인 C 전후 비교는 각CPU1/4GiB/300초이며 큰 계산의 공유40분을 늘리지 않는다.
- 관측/소비: 기존 세 묶음만 소비한다. `F_Z`=같은입력물질의old/newJ에 대한 C II–VI 단계별질량p95 중최댓값/전체최대, `P_Z`=실제부모대비 native II–VI 이동의동일두지표, `Q_Z`=새ne/현재J에서 C와native조건부 차이의동일두지표, `N`=최대ne step이다. 전bin J/H/K/N 최대는보조이며 단독 실패기준으로삼지않는다. 모든CSV는전90점/II–VI900행을유지한다.
- 66후67의선택규칙: 기존수송/신원/양수성/보존 또는Ccoverage·채택·support가실패하거나relativecap등으로full-step전제가깨지면남은67을중단한다. 수치적으로유효한경우, **Si/S각각에서 F_Z·P_Z의네성분모두65대비감소하지않고최소하나는증가**, 동시에 **N과양원소Q_Z의모든성분도개선되지않으면**67을생략한다. 그외혼합/감소는67까지수행한다. 실제수치로그대로판단하며새임의tol/단조감소최종게이트를추가하지않는다.
- 결과판단/한도:67뒤같은shell·stage의부호반전과두-step상태차이대비각한-step차이를함께기록해2주기후보/포화/지속이동을구분한다. 감소하더라도공동수렴PASS를선언하지않는다. 두outer뒤추가자동연장은없고실제결과를묶어다음방향을평가한다. 원래전체SE·RE·마지막3회·releasedT·에너지/스펙트럼/격자문턱은그대로미완료다.
- 실행glue:현재frozen65runner는그대로보존한다. 다음runner의종료기록에는기존A/B와같은ES12.4/IEEE full-step확인과최소scale를남긴다. 완료상태는보존하되전제미확보일때`next_outer_eligible=false`로남겨67자동사용을막는다. 이는남은SE잔차/최종수렴판정이아니며추가계측체계가아니다.
- SE witness:현재큰이동구간에추가native9분을쓰거나선행게이트를만들지않는다. **처음안정화가관측된뒤** 원래native의동일입력/J를private관찰build로한번재현해비간섭/실제하한coverage를확인하는후속묶음을계획한다. RT재계산은없고phase/UNPROVEN정책은유지한다. T-release는계속No-Go다.

실행 재현은 기존 `submit_cmfgen_field_feedback.slurm`의명시부모record/source/MAX_LAM 인수를사용한다.66은source `RUN428943/material`,record65,MAX_LAM1e6이며기존65전체step재현/종료/원자료 신원으로 시작한다. 실제build/직접시험/봉인runner/제출ID는 이어 기록한다. 소형확인마다추가Astra/Fable감사를하지않고역할/기준변경을이유로65를재시작하지않는다.

**66 실행 전 확인:** 현재 code의통합회귀 **245통과/1skip·30.15초**, `/gpfs/kjhan/cmfgen_runs/lumina_feedback66_validation_20261020-WAsrJv/regression.log` SHA256 `8b2a648d318280c20199521b0acc3abfd7229c7a46f6d2f7b06dc863c0cd0979`다. 전체step 정밀도는 기존A/B 함수와 일치하고, 완료66이라도 full-step 전제가 미확인/변경됐으면67의입력으로 사용하지 않는 것을 시험했다. skip은 기존 scoped smoothing witness이며 물리통과가 아니다.

실제65검증/격리clone은 `/gpfs/kjhan/cmfgen_runs/lumina_feedback66_preflight_20261020-LLveIm/preflight.log`에서 성공했다. source record 저장해시873f36…·POINT1/2=65·다음66·native원본f2b9af…·LOCAL/NONE/MAX_LAM1e6을 확인했다. 새build `/gpfs/kjhan/cmfgen_runs/lumina_feedback66_build_20261020-DaEVKo/build`는 기존v11의 세대검사/완료기록만66으로 바꿨다. exe SHA256 `bcd9117af0fc20a6b44e6dea619250f98460a450861f57b10e71e4d5494ece87`, manifest SHA256 `25d4b9c9d8e5d78910180cf3ddbad24e598ccdc8453e289d9390c58d3a1217ca`다.

봉인 runner `/gpfs/kjhan/cmfgen_runs/lumina_feedback66_runner_20261020-zfwnYc`의9개 직접 의존파일이 모두OK이며 `runner.sha256` 자체 SHA256 `803feccb3024aeb520e78e3311fd9fbb695ddfb64e027ac8afd1f9240f7c4f6d`다. 기존65runner/원래native/SE관찰build는 변경하지 않았고 commit/push도 하지 않았다. 두outer중 첫66을 이 정확한 입력으로 제출하며, 이 확인을 실제66/67의 완주나 최종동질성 달성으로 해석하지 않는다.

**66 실제 제출/시작:** Slurm **428954**,2026-10-20 **05:09:27 KST 제출/05:09:28 시작**이다. root `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_428954`, 로그 `/gpfs/kjhan/cmfgen_runs/slurm_logs/field-feedback-428954.{out,err}`다.05:10:27에syn02/a10/RUNNING59초·CPU16/32GiB/GPU0·ReqNodeList null·Requeue0/Restarts0·봉인9파일모두OK·stderr비어 있음을 확인했다. 이 source65에서 새 J66과 native66이 아직 완주한 것은 아니다.67은 미제출이며 위 규칙 적용 후에만 시작한다.

다음 C 입력은 완주장 확인 후 `prepare_cmfgen_feedback_owner.py --root RUN428954 --reference-parent A/feedback_owner65_J65_v1 --record 65 --out NEW_PRE66`로 만들고 같은 C binary로 전90점을 새로 푼다. 이후 native66의 실제 종료 뒤에는 `--record 66 --same-field-input NEW_PRE66 --out NEW_POST66`을 사용한다. native65 oldJ65 C와 NEW_PRE66의 장단독 비교, NEW_POST66의 부모대비실제이동/조건부차이를 합쳐67 여부를 결정한다. source/장/oldne/행렬을 혼동하지 않으며 새raw/출력을 기존root 위에 덮지 않는다. 원래전체 물리 목표는 계속 **미완료/active**다.

05:14:44 KST 재조회에서428954/자식.0이 각각RUNNING5분16초/5분02초이고 실제native로그의 counter66 진입을 확인했다. stderr는 비어 있다. 같은job/handle을60초 이상 간격으로 이어 확인하며 아직J66/물질66 완료를 주장하지 않는다.

#### 66/67 기존 증거의 묶음 판독 — 추가 실행/물리 게이트 없음

`evaluate_cmfgen_feedback_66_67.py`는 위에 정한 F/P/Q/N과 최대2outer 규칙을 기존 완료 증거에 적용하는 소비자다. 계산을 제출하지 않으며 전체 목표·공동수렴·T-release를 통과시키지 않는다. 실제65 기준을 `A/feedback65_policy_baseline_v2/feedback-progress.json`에서 재현했다(SHA256 `491ce3ef6c4a7224a218385eb7ab3c3e8bfc321e7631638ed35c3955953443dc`). 단계65/66/67의 연속 부모, 동일 C/질량 격자와 전900행/전90점 증거를 확인한다. 65의 과거 runtime에 없던 full-step flag는 `null`로 보존하며 앞서 별도로 재현한 결과를 꾸며 넣지 않는다.

직접21시험은 `/gpfs/kjhan/cmfgen_runs/lumina_feedback_policy_tests_20261020-cV7WfJ/tests.log`, SHA256 `53915de3a717174034637c80c59720ea9e780e2fca38a0e2f452aeef854166b9`다. 관련 통합회귀 **255통과/1skip·30.06초**는 `/gpfs/kjhan/cmfgen_runs/lumina_feedback_policy_final_20261020-wZVbZE/regression.log`, SHA256 `c681cd376509a2335002b1ebabe9f95ce482b4f8ebd654e062987f2e99d0ccb7`다. 도구/시험 성공을 물리 성공으로 세지 않는다. 출력은 새 디렉터리에만 만들며 봉인 runner/실행 중66은 변경하지 않았다.

#### 대기 중 read-only 물리 조사 — BB 범위와 실제 선폭

실제 C binary의 봉인 source를 추적했다. `bench_cmfgen_owner.c`의 `profile_integral`은 **native 전주파수 J의 구간선형 함수**를 등록 Gaussian으로 적분하며, EW의 `ew_run_impl`은 `nlte_assemble_rate_matrix`→`nlte_assemble_rate_matrix_impl`의 production split을 소비한다. 따라서 현재 BB가 단순 coarse-bin 중심 J만 쓰거나 과거 shadow의 escape factor를 그대로 쓴다는 설명은 맞지 않는다.

반면 다음 두 차이는 source/입력에서 확인했다. 물리 영향량과 native에서 실제 선택된 전이 집합은 아직 입증하지 않았으므로 현재 큰 outer 이동을 이것만으로 설명하지 않는다. 현재 봉인66/67 계획의 물리식/원자료는 변경하지 않는다.

1. C의 일반 BB 경로는100–20,000Å 밖에서 J-driven 흡수/유도방출을 제외하고 spontaneous A/충돌은 유지한다. `JEQB`는 그 범위 밖에도 Planck 값을 직접 대입하는 별도 경로이므로 그 진단의 상세평형 통과를 일반 경로의 범위 밖 흡수/유도방출 증거로 쓸 수 없다. 봉인 C line_list의 **범위 밖·서로 다른 SL·현재 field가 ±4×10km/s를 덮는** IR 전이는 Si II/III/IV/V 각각616/196/75/187개, S II/III/IV/V 각각1659/556/118/122개다. Si V에는100Å보다 짧은 동일 조건15개도 있다. 이 재고는 “전부 같은 SL 내부라 상쇄된다”는 설명은 반증하지만, native 채택 여부나 유효율의 크기/이온분율 변화량은 증명하지 않는다.
2. 실제 `VADAT`에는 `FIX_DOP`가 없다. native `rd_control_variables.f:555–567`의 초기값은true이며 이 경우 `GLOBAL_LINE_PROF=DOP_FIX`를 설정한다. 따라서 덱 끝의 `GLOBAL_PROF=DOPPLER`, `DOP_LIM=3`만 보고 해당 분기를 적용하면 잘못이다. `set_line_opac.f:354–359`에서 실제 Gaussian 폭은 `12.85*sqrt(TDOP/AMASS_DOP+(VTURB/12.85)^2)`이고 현재 TDOP2/AMASS4/VTURB10으로 **13.51152286km/s**다. C 등록 Gaussian은10km/s·±4D다. native는 실제 grid의 trapezoid `LINE_QW_SIM*J`를 누적하므로 단지 폭 상수만 맞추면 모든 Jbar가 같아진다고도 아직 주장하지 않는다.

정본은 `A/bb_contract_inventory65_v1/{analyze.py,inventory.json}`이며 source/output SHA256은 `bd9e2512f8c40c66e2866d8cf8c7094177a18de17e71097e8509982e89176fdb`/`4d82a7b28f1d80d2a0e44aa567366b79ce77561b7407922a28093cf8b8aa9b69`다. 실제 완주 J66의 범위는3.49897e12–1e18Hz다. 재현은 grammar-debug CPU1/4GiB에서 `timeout 60 python3 A/bb_contract_inventory65_v1/analyze.py`이며 exact deck/source/controls의 해시가 출력에 포함된다. 관찰만 수행했고 선택 cell의 대조 실행이나 source/율/프로파일 변경은 하지 않았다. Fable/대체 검증2 호출도 없다.

### 66 실제 종료 — 새 복사장 단독 영향은 혼합 추세

Slurm **428954는2026-10-20 05:43:58 KST, COMPLETED/0·34분30초**다. J66은1489.382초/계획된STOP97, native66은548.011초/0이다. 전197291×90, 최대5formal 호출, source/J/H 잔차는9.96066e-9/9.97190e-9/9.96993e-9다. 장 생산 물질은65이며 **물질66/J66은 공동수렴이 입증된 쌍이 아니다**.

source/T/양수 support는 보존됐다. 부모65 대비 ne 최대변화는 **9.18535288%**, 전하 재합산 잔차3.3572003e-15, 원소총량 변화 최대3.33067e-15다. 원래 NC행 잔차와 구분한다. `printed_minimum_scale=1`, `actual_full_step_within_print_precision=true`, `next_outer_eligible=true`이며 완료 상태는 정밀도 규약 안에서 실제 전체 갱신을 적용했다. 이를 남은 SE 잔차/수렴과 혼동하지 않는다. root는7774192666bytes다.

`RUN428954/seq_logs/feedback-runtime.json` SHA256 `577c68babfba230328e7cffc3f76ec57440b8c67eaa7d974c8ed956d2628c84e`, J66 field SHA256 `92bb9fe5f17d9c2f2a1d6220d18447bf7bd5d3e08b1fbd61a339b91ccf90c2e2`, EDD SHA256 `416430cd74ae41203bb99eff37659437a18129036ec7e25bdd302d7546309255`다. 실제 완료66 저장 구간의 신원은 전체 SCRTEMP 파일과 구분해 보존한다.

물질65를 그대로 둔 J65→J66 대조는 `A/feedback_owner65_J66_v1`의 C 전90점·108.545초로 완료했다. `A/feedback_field_only65_J66_v1/{field-only.json,fractions.csv}`에서900행/T/ne/밀도/초기분율 decimal byte와 NPZ 물질의 동일성을 확인했다. **III 최대 변화는Si45.7906396%p/S8.9511807%p**, 질량p95는18.0904516/.4924174%p다. II–VI 전체 최대는45.7906396/8.9516034%p다. Si는65보다 커지고 S는줄어 공동 단조개선/안정화라는 주장은 불가하다. JSON SHA256 `a490c5b9204fa3799c2bb4f02fe44b521d9126583717581c08ee17dab6a4e092`다.

66 자신의 실제 새ne/J66 C는 `A/feedback_owner66_J66_v1`에서 **108.370초에 완료**됐다. 같은새J66의block만복사한입력준비는6.745초였고,율/행렬/인증은새C프로세스에서다시계산했다. 각Si/S90/90인증·실제채택/II–VI900행/support불일치0, CscaledSE최대2.32800e-15/4.92282e-15다. Si의legacy rank/κ실패90은보존했다. `comparison/fixed-field-owner.json` SHA256 `a0a52c9a7ba7d5be9977fb3b019da4969c773562e3d37c3456bfe76c25e1756a`다.

| II–VI 소비 지표(%p; p95/최대) | Si·65 | Si·66 | S·65 | S·66 |
|---|---:|---:|---:|---:|
|F·동일물질/장만교체|15.88124/34.67674|18.09050/45.79064|1.452798/25.06060|.503639/8.951603|
|P·실제부모대비native이동|15.50892/33.30320|17.48265/41.95969|1.421452/23.89873|.483160/9.725642|
|Q·새ne/현재J의C/native차이|.252386/1.087272|.865353/2.724406|.0793062/1.533010|.0723390/.227015|

III만의조건부최대는Si2.724406%p(d37)/S.1713570%p(d33)이고 II최대는.000078479%p/.00111559%p다. Si의F/P네성분과Q두성분은65보다악화했지만 S는감소했고 N도10.65662→9.18535%로감소했다. 따라서사전명시한공동악화중단조건은성립하지않으며 `A/feedback66_decision_v1/feedback-progress.json`은 **CONTINUE67_WITHIN_LIMIT**다(SHA256 `7567a283625abaec9bf94ac36e729adb4b3bb7baf0d85ea1df95af8126fb807e`). 결과를보고기준을줄이지않고기존최대2outer의남은67만집행한다. 큰실제이동과조건부일치는구분하며T해제/공동수렴PASS는없다.

`A/feedback66_decision_v1/{reproduce.py,reproduction.stdout}`에서actualfull-step·최소scale1·전하·위지표/다중shell분포를독립재현했다. source/output SHA256은 `1b248ac7f3f36bb88a01950984d456df6a4e02a06c2ba7e0fb5a5796228d860a`/`18f3d48256c9aaf377b91f3c54b146e29b683bf2b9ce8e2eb709eac6f8cfab83`다. 재현명령은허용된grammar-debug CPU1/4GiB에서 `timeout 60 python3 A/feedback66_decision_v1/reproduce.py`다. 실제66record저장구간 SHA256 `9c30ad81d412050498d0ecb654bb24ccc20d36e26eccf8690f12afaf2cf0833d`와전체SCRTEMP SHA256 `3c88a9b627cc2ee9e3456167fe0bebe7e338e9854fe2dda14599c72b70a0347a`를구분한다.

#### 남은67의집행 — 승인된두번째outer, 자동연장없음

새build `/gpfs/kjhan/cmfgen_runs/lumina_feedback67_build_20261020-JMeRKR/build`는기존v11의세대검사/완료기록만67로바꾸고STOP97/수송커널을보존했다. exe SHA256 `43b7f47a9f52ab211c3b34d7ee3835407e71ebe56d5b480995909f6bd527c6da`, manifest SHA256 `2efc887373fd364302f46af8b830b483d7ee516de4e3323db34ee1f847d6502e`다. 봉인runner `/gpfs/kjhan/cmfgen_runs/lumina_feedback67_runner_20261020-laKtUJ`는66runner의9파일을그대로복제했고전부해시OK, `runner.sha256` 자체SHA256 `803feccb3024aeb520e78e3311fd9fbb695ddfb64e027ac8afd1f9240f7c4f6d`다. 실행코드/원래native/C/원자료/물리식은변경하지않았다.

실제66검증/격리clone은 `/gpfs/kjhan/cmfgen_runs/lumina_feedback67_preflight_20261020-cM8BGi/preflight.log`에서성공했다. source66/다음67/POINT1·2=66,원래native SHA256 f2b9af…/LOCAL/NONE/MAX_LAM1e6을보존한다. 제출명령은 `sbatch --parsable RUNNER67/submit_cmfgen_field_feedback.slurm RUNNER67 BUILD67 /gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_428954/material 66 1.0D6`이다.

**실제 Slurm428960, 2026-10-20 05:47:41 KST 제출·시작**. root `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_428960`, 로그 `/gpfs/kjhan/cmfgen_runs/slurm_logs/field-feedback-428960.{out,err}`다. 05:48:00에 syn02/a10/RUNNING19초·CPU16/32GiB/GPU0·ReqNodeList null·Requeue0/Restarts0·봉인9파일 모두OK·stderr 비어 있음을 확인했다. 67뒤 자동68은 없고 부모45분/두 자식 공유40분·retry0 한도를 유지한다. 관측 timeout은 재실행 근거가 아니다. 아직 J67/native67 완주나 최종 물리 목표 달성을 주장하지 않는다.

### 67과 병렬 원인 분리 — BB 폭/범위의 단순 치환으로 잔여 차이는 해소되지 않음

새 native/자기일관 실행이나 생산 소스 변경 없이 grammar-debug CPU1/4GiB·각300초 이하에서 아래 두 물리 비교를 수행했다. 운전자 구현·직접시험·재현이며 외부 감사/Fable 검증으로 표기하지 않는다. 최종 v2 동질성 기준과 현재67의 입력/자원/중단 기준은 불변이다.

먼저 `A/bb_field_rates65_J66_v1`에서 실제 **물질65/J66**의 full-pop/T/ne를 같은 phase로 확인하고, 같은 Gaussian 적분에서 폭10→13.51152km/s 효과와 C범위 밖에서 빠진 J-driven 순율을 따로 계산했다. Python 적분은 봉인 `line_jbar.c`의 실제 kernel로 8이온×3선×90점을 대조해 상대차 최대6.66e-16이다. 범위 밖이거나 미약한 선의 Jbar 상대최대만으로 영향을 판단하지 않았다. 주요III 구간의 범위 밖 순율은 BB 총 in+out 흐름에 비해 작았지만, BB 총 흐름은 전체 SE 분모가 아니며 작은 값만으로 population 영향이 작다고 결론내리지 않았다. 실제 native 전이 선택/중심/끝 인덱스가 없는 trapezoid 재생은 **식 대조**로만 표기했다.

율 source/result SHA256은 `b45ac8182febb7ee71bcb58bb84f2d1ff7430b95b2b55bf079ac596e8d780a0b`/`8faffbd7ac889eb2c327d52cdd05c6e975eb31456ba5ed463000dba4f344196f`다. 새 순율을 원래 전체SE/NC/RE 잔차나 새 이온분율로 대신하지 않았다.

그다음 **완료66 자신의 ne/J66**에서 대표6셀(Si깊이34/37/61, S깊이33/34/61)의 실제 C를 새 격리 root `/gpfs/kjhan/cmfgen_runs/lumina_bb_sensitivity66_20261020-FvXPGU`에 실행했다. 원래 봉인 binary·입력·원자료 그대로이며 selected marker는 **input_depths=90/solves=6**, 응답30행이다. 전90점 재실행으로 부르지 않는다. 각 셀의 T/ne/N/초기분율/후보/실제채택은 원래 전90점 `feedback_owner66_J66_v1/C_response/response.csv`와 decimal byte까지 같다.

실제 채널별 행렬과 FL/SL 분율을 dump에서 읽었다. 실제 행렬의 원래 해를 같은 봉인 GTH로 재현하고, BB만 다시 구성한 행렬의 support가 같으며 항별 상대차 최대1.23e-15임을 확인했다. 비BB 채널을 빼기 연산으로 복원하지 않고 원래 채널의 양의 전이율을 보존해 더했다. 그 뒤 **행렬 사본에서만** 다음을 비교했다.

- `width_only`: 같은100–20,000Å 범위에서 정규화Gaussian의 폭만13.51152km/s로 바꿈.
- `outside_only`: 현재 실제 field가 전체프로파일을 덮는 범위 밖 선의10km/s 펌프를 복원함. 미coverage 선의 펌프를 알았다고 꾸미지 않고 원래baseline을 보존·개수명시함.
- `width_and_outside`: 위 두 변경의 결합.
- `native_discrete_inside/covered`: 같은 CSV중심·±6D를 가정해 native의trapezoid Jbar와 Gaussian 면적을 사용하고 spontaneous A에도 그 면적을 곱함. native상수2.998e5km/s를 그대로 사용함. 실제 native 전이 선택/중심/끝 인덱스가 모두 같다는 증명은 아님.

모든 후보는 봉인 GTH와 기존 `population_generator_interval.h`의 **원래 long-double 엄격 오차 규약**을 그대로 적용했다. 원래 전체SL/미량/terminal을 포함한 고정행렬의 수치 인증일 뿐 물리 불확실성·native SE·공동수렴의 인증이 아니다. 전 개체수 상대오차 상한은 표시값 기준 최대4.05e-16이며 floor/clip/미량 cutoff/재정규화는 추가하지 않았다.

아래 단위는 III 분율 **%**다. `구적/면적`은 `native_discrete_covered`인 잠정 식 재생이다.

| 원소/깊이 | native66 | 원래 C | 폭만 변경 | 범위만 변경 | 구적/면적 식 재생 |
|---|---:|---:|---:|---:|---:|
|Si/34|63.95763|66.52895|66.35448|66.53100|66.48586|
|Si/37|67.97607|70.70048|70.80185|70.70598|70.73282|
|Si/61|7.296928|7.246698|7.246232|7.240960|7.240871|
|S/33|36.47508|36.64644|35.90190|36.64645|36.44414|
|S/34|47.66080|47.74884|46.08302|47.74887|47.28853|
|S/61|41.41871|41.44156|41.44473|41.43798|41.43857|

**운전자 판독:** 폭만 바꾸면S/34에서−1.66583%p 움직여 원래+.08805%p 차이보다 크게 악화한다. 범위만 바꾸면Si/61에서−.005738%p 움직이지만 원래−.050229%p 차이를 해소하지 않는다. 구적/면적 식 재생도S/33은 개선하지만Si/37·Si/61은 해소하지 못하고S/34는 악화한다. 따라서 이6셀에서 **BB범위/선폭의 단순 치환만으로 현재 오프셋을 설명하거나 즉시수정하는 가설은 지지되지 않는다**. 작은 BBtraffic 비율도 population 영향의 크기를 그대로 뜻하지 않는다는 점을 실제 분포로 확인했다.

이 판단은 폭/범위 차이를 숨기거나 최종 공통장 계약의 불일치를 허용한다는 뜻이 아니다. 물리 정합은 원래 계약대로 해결해야 하며, 현재 비수렴 staging의 숫자를 맞추기 위해 폭을 선택하지 않는다. 큰 장–물질 이동과 완료66의9.19% ne step이 남아 있으므로 고정J/ne 민감도를 새 자기일관 해로 부르지 않는다. native 자체의 구적/전이와 post-ne 비선형 갱신을 구분할 후속 판단은67까지의 실제 결과와 묶어 Astra가 한 번 평가한다.

정본 `sensitivity.json` SHA256 `a5ac9ad709cc6467674421d75266838945de0e1d6cecbcc583d2a6c46e5605d9`, `sensitivity-discrete.json` SHA256 `3853c14b36db6dc15261da51b6df54b732a1be922ca8137c6c8391bda75e966e`다. 각각 source `analyze.py`/`analyze_discrete.py` SHA256은 `da378d9bb8c8db3e5180d8b1421aa25d3a6d9c2552148e080943fd604a319357`/`02a252218fdfc66cbefa87b8f41012e0588c52972ba476366dcfd2511e7d69fa`다. wrapper source/lib SHA256은 `e188eddc1be12ece94f88c88952c833a4a3b0c6e40639acf9ea29b75454d4b95`/`500df73c1151155e0d0bd91a4208adc06b1d0cf0c417007253baf8af12439a8d`이며 원래 source/header는 결과에 별도로 봉인돼 있다. 기존 출력은 보존했고 새 source/native/runner 수정이나 commit/push는 하지 않았다.

**격리 재현:** 원본의 정확한6셀×solution/identity/matrix_raw와 source/같은lib/response만 새 root `/gpfs/kjhan/cmfgen_runs/lumina_bb_sensitivity66_replay_20261020-u92wGj`에 복사했다(재귀복사/원본출력 덮어쓰기 없음). CPU1/4GiB·300초 이하에서 `python3 NEW_ROOT/analyze_discrete.py`를 수행한 뒤 원본과 재현 JSON의 `results` 전체를 비교했다. **6셀×6변형의 모든 분포·전SL재현오차·엄격인증 수치가 그대로 일치**했다. `compare.stdout` SHA256 `14d6e15c0e7bb491b5f67809df62dde7459d8ffece4b3023748b8bcaf880f905`다. 재현이 최종 물리 통과를 뜻하지 않는다.

같은 J66의 전후 C/실제 ne 문맥은 위 원인분리 root의 `ne-context.json`이다. 깊이37 ne step은−9.18535%지만 깊이61은−6.18078e-6(상대값; −.000618078%)뿐이다. 깊이61 C III 전후 변화는Si−.000041356%p/S−.000149965%p이고 남은native66와의차이는Si−.0502294%p/S+.0228509%p다. 따라서 모든 잔여 차이를 전영역의 큰 ne 갱신 하나로 설명하지 않는다. 다만 입력의 초기 분포도 바뀌므로 이 전후 표만을 ne 단독 효과의 증명으로 부르지 않는다.

### 추가 발견 — 역Milne의 서로 다른 주파수 격자 번호 오용

BB 가설 조사 뒤 실제 BF 소비 경로를 읽다가 **별도의 구체적 배선 결함**을 찾았다. 아직 이를 고친 C/새 분포는 없으며, 물리 영향량과 전체 불일치 해소는 미판정이다.

- 봉인 `src/lumina_plasma.c:16515`는 NLTE/BF를1234bin·`NLTE_NU_MIN`에서 시작하도록 설정한다. 반면 `src/radiation_field.h`의 공통장은K2·J_LO−1398·N3866이고, `radiation_field_read_view`는 실제3866bin과 그 edges를 반환한다. `bench_cmfgen_owner.c:255–277`의 actual 경로는 이3866bin 장을 발행/취득하며 estimator는NULL·JEQB는 꺼져 있다.
- 기본 `nlte_bf_field_source`는0을 반환한다. `lumina_plasma.c:18340–18395`의 역Milne loop는1234bin의`bb`로 주파수/σ/온도중량을 구성하면서 `bf_rate_view_jnu(&nlte->radfield_view,shell,bb,...)`를 호출한다. getter는 그대로canonical `bb`를 읽는다. forward Γ의 주파수 겹침 처리와는 다르다. 두 격자의 시작과 폭이 다른데 같은 번호를 쓰므로 다른 주파수의 J가 들어간다. 소스상의 ordinary 경로와 합성 대조로 확인했으며 실제 전체 분포 영향은 아직 계산하지 않았다.
- `A/inverse_bf_grid_witness66_v1/witness.c`는 **봉인 getter 그대로** 상수장과J(ν)=ν를 읽는다. 상수장은오용을못잡지만 BF bin0/600/1233의읽은주파수는각1.4410e12/7.0627e12/3.7779e13Hz,해당BF구간의J평균은5.8568e13/1.4069e15/4.0256e16Hz다. getter가 잘못된 것이 아니라 BF번호를canonical번호로 넘기는 호출자 문제다. `CALLER_GRID_INDEX_DEFECT_REPRODUCED mismatched=3/3`이며 합성 대조의0종료를 물리성공으로 부르지 않는다.

witness source/stdout SHA256은 `8e56f4ffb00a1b4359b2eaa9958021a087e2b8112786c2ce0fa916cccc978032`/`408ffcd662324c4b00ef55720abc12b156c7bb561d29d0041337bf61ed2d06c8`다. grammar-debug에서 봉인 `bf_rate_jnu.c`와 그 header만 링크해CPU1/4GiB·60초 이하로 수행했다. 원래source/기존C/native67을바꾸지않았다.

**한 번의 의미 있는 Astra 기획/계획/영향 검토**를 기존agent `01a15546-7f28-7931-bb31-7c0c2e7b3e15`에 재개 의뢰했다(submission `01a15604-0474-7e00-b991-36dae7a1693b`). 번호/주파수 관계의 반론·실제 dispatch·동종 inverse 전체범위, 하드코딩 없는 최소 주파수 겹침 수정과 최소 직접시험/같은66/J66 C A/B를 묶어 판단하도록 요청했다. 감사/추가게이트/Fable 호출이 아니고 새outer/T-release 승인이 아니다. 현재는 **계획검토 진행 중**, 생산 패치는 미적용이다.

**위 Astra 검토의 최신 결과:** 실제 owner까지 버그로 확인하고 최소 수정 계획을 제시했다. agent는종료했다. pair(`lumina_plasma.c:18376`)외에 **EW target callback(`lumina_element_wide.c:743`)과Si/S V→VI terminal(:1461)**도같은index오용이며callback은기존accessor반환값까지무시한다. 반면자기BF격자의`nlte->J_nu`를읽는legacy ground-only/TOPSTAGE_IV/Fe boundary및spontaneous-only경로는이index결함으로함께바꾸지않는다. Astra의확인은운전자getter합성재현과독립된소스판독이며물리영향량을확정한결과는아니다.

최소수정은공통주파수구간reader하나다. 활성BF구간 `a=max(nu_lo,nu_threshold), h=nu_hi`에서canonical겹침`[u,v]`의 `J_eff=sum(J_q*log(v/u))/log(h/a)`를사용한다. 실제edges/차원으로겹침을찾고1234/3866/offset을새로하드코딩하거나nearest-bin/두bin단순산술평균을사용하지않는다. missing/stale/OOG/불완전coverage는기존오류경로로전파하고0/legacyJ로대체하지않는다. JEQB source2·spontaneous·현재Milne열적평가점/thresholdσ/Saha/target가중은유지한다. 현재σ/열적구적전체를고치는범위를섞지않는다.

운전자는 `/gpfs/kjhan/cmfgen_runs/lumina_inverse_bf_20261020-9tBN4q/source_snapshot`에봉인14C와compiler가식별한18header를정확히복제했다. **이시점에는아직수정source/새binary/수정후분포는없다.** 다음은사본에서세callsite와reader/오류전파를수정하고상수J·기울기·threshold부분bin·잘못된낮은번호변조불변/올바른겹침변조반응·J0/spont/JEQB/invalid를한묶음직접시험한다. 그뒤같은66/J66의대표6셀과전90점기존C900행을A로재사용해수정C의II–VI/gap질량p95·최대/support/인증·실제채택을비교한다. CPU1/4GiB/각300초private검증이며현재67/기존frozenC/원자료는불변,새RT/native실행은필요없다.

기존interval인증은당시조립한행렬의해인증으로그대로보존한다. 다만기존C분포/native gap/C기반장민감도는이결함의영향대상이므로수정C와함께재평가한다. native의실제큰분포이동·ne변화는그대로유효하다. 이수정으로d61포함전체차이가해소된다고예측하지않으며추가outer/T-release/원래v2전체SE·RE·3grid·spectrum문턱변경은없다.

병렬로67의J는 **197291×90 완주**했고 `.0`은06:12:37 KST에의도된97/24분41초로끝났다. `.1` native67은06:12:49 시작,06:12:54 RUNNING5초를확인했다. 운전자는`A/feedback_owner66_J67_v1`에서같은물질66/새J67 C와장단독비교를진행중이다. 이분석에는기존봉인C를사용해65/66/67 유한묶음의비교계보를보존한다. 수정C가생기면별도A/B로분리하고기존결과를소급덮어쓰지않는다.

**67장단독비교완료:** 같은66/J67의기존C는112.569초·각Si/S90/90인증·실제채택·900행으로끝났다. `A/feedback_field_only66_J67_v1`에서같은물질66/새장만변경한것을확인했다. III최대변화는 **Si15.45884%p/S4.12814%p**,질량p95는2.394260/.174113%p다. II–VI전체최대는15.47610/4.12814%p이고단계별p95중최대는2.394270/.200691%p다. F는앞선66보다감소했지만이값은기존C기준의조건부장민감도다. 위BF결함영향의재평가대상이며공동수렴/최종목표로확대하지않는다.06:20:24조회에서native67은RUNNING7분35초였다.

### 최신 집행 결과 — 67 한도 종료와 역Milne 수정의 실제 영향

428960은 **06:22:06 KST COMPLETED/0·34분25초**다. RT1481.171초/의도된97,native549.360초/0,전197291×90/max4formal/source·J·H잔차6.60904e-9/9.99911e-9/9.99399e-9다. source/T/양수support 보존,전하3.930901e-15,부모대비원소총량변화max4.44089e-15,ne step3.6646669%,root7789362266bytes다. minLOCALscale.98936·fullstepfalse·next_outer_eligiblefalse를그대로보존한다. 원래최대2outer는끝났고자동68은없다. runtime SHA256 `9398b6e955c42d00bfd7ad09e6b4e517aa2c0b4da87992db0ccb9b568016a512`다.

기존C0의새ne67/J67 전90점은108.258초·Si/S각90/90인증·실제채택/900행/support0으로완료했다. `A/feedback67_bundle_evaluation_v1/feedback-progress.json`은 **LIMIT_REACHED_EVALUATE/latest_full_step_premise=false**이며SHA256 `aafa48e410c126f9ab7d600efbf6021ddb79e0664935343791fc492f9e87f633`다. native II–VI실제이동max는Si14.59423%p/S3.56688%p다. 65/66/67의부호반전후보는Si224/S168,두step가두one-step보다작은후보는123/74다. max후보진폭은Si14.59423%p(V,d11)/S.319575%p(IV,d39)다. 실제진동/비수축성의증명이나마지막3회안정성통과로읽지않는다.

**67 감쇠 원인:** CaVI terminal·변수714·d37 raw correction+1.0108,비감쇠p비율−.0108이다. p_old6.527013e-40/원소내5.868917e-46→실제p6.527013e-46,ne로재구성한공통scale.9893658이다. 최악증가는NiV SL39/d14의17.317배다. `solveba_v13.f:297–326`의BIG_LIM/LOCAL식과맞는양수성보호이며이전증가cap정체가아니다. MAX_LAM추가확대는음수제안을양수로만들지않는다. 미량행을지우지않고fullstepfalse를유지한다. source `step-context-v2.stdout`에전수raw후보의최소비율/좌표를남겼다. 초기출력의numpy integer직렬화오류는별도보존했고수치실행실패로해석하지않았다.

#### 역Milne 최소 수정 — 저장소 통합과 검증 완료

`/gpfs/kjhan/cmfgen_runs/lumina_inverse_bf_20261020-9tBN4q/source_snapshot`의사본에서3잘못된callsite와 `bf_rate_view_log_band_jnu`를구현했다. 격자edges를탐색해활성대역의log-weighted J만읽고사용한모든overlap의validity/완전coverage를확인한다. EWcallback실패도target_fail로전파한다. source2의Planck,기존spontaneous/σpref/thermalpoint/Saha/target는그대로이며기존x<700사용범위밖의J를불필요하게읽지않는다. 마지막BF구간은metadata.nu_max로닫아exp/log roundoff의격자밖요구를피한다. 숫자offset/새미량cutoff/fallback은추가하지않았다.

이exact사본으로새binary `bench_cmfgen_owner`를빌드했다(SHA256 `5447bd091db9e47f001a6bb0183d145940fb9ef2de35ff7af7574c6811c4da46`). 기존C0의4대상파일과작업트리가같음을확인한뒤exact차이만 `src/bf_rate_jnu.{c,h}`, `src/lumina_plasma.c`, `src/lumina_element_wide.c`에apply_patch로통합했다. 기존미커밋변경을지우지않았고원래native/C0/원자료/frozenrunner는바꾸지않았다. commit/push는하지않았다. 새직접시험은 `tests/bf_log_band_reader_selftest.c`, `tests/test_bf_log_band_reader.py`다.

상수J/주파수기울기/partial대역/범위밖낮은번호변조불변/실제overlap변조반응/EXACT_ZERO/missing·stale·OOG/잘못된인자를직접시험했다. 관련6pytest모듈은 **78통과/1skip·1.40초**, `/gpfs/kjhan/cmfgen_runs/lumina_inverse_bf_validation_20261020-kFuO9y/regression.log` SHA256 `61a16c115512347ad914d57f025d85ce64a189b72d8ca92f6a335f9eb2fda0a8`다. skip은sealedLINKS/MODEL_SPEC·PHOTfixture env미지정이다. `git diff --check`를변경범위에적용했고문제없다.

`conditions_v2/conditions.json`에서JEQB와영복사장의양원소d61 각2셀을원래/수정실제owner로비교했다. **두조건모두전채널raw행렬byte동일**이다. JEQB양쪽해인증도모두통과했다. J0에서는photo0이고spontaneous역율합이Si.04616218/S.20233955로양수다. 단원래S J0가denseLU실패하여GTH앞에서비인증이었으므로이를새패치의통과/실패로위조하지않았다. 최초test의과도한해인증요구실패/조건v1/실패source를보존하고v2에서는 **율보존만**판정했다. 합성J0의S해인증은여전히미확보다. conditions JSON SHA256 `a5defba7de2d16851b0163d366f72a55529d02c7a309c43398d2bc8c5db115ff`다.

build-manifest SHA256 `879cb1e2e554576f7abc4f609d4d5caf8c662cc1bd59dc374cab74d0b81be1f6`는원래/수정14C·18header·binary·exactpatch·직접시험·조건실패/보존·실제66입력을묶는다. 저장소수정4파일은이사본과byte같다. 새물리결과가불리하다는이유로코드의주파수오용을되돌리거나원래기준을완화하지않는다.

#### 실제66/67·같은새장/전자밀도의C0→수정C1

수정C1은각66/J66 **114.126초**,67/J67 **113.977초**에전90점을새프로세스에서풀었다. 입력은기존완료상태를그대로읽기전용재사용했으며율/행렬/인증은새로계산했다. 각Si/S90/90인증·실제채택/II–VI900행/support불일치0이다. 조건부소형오차/수치인증을released-T/전체SE/새자기일관해로부르지않는다.

| III 최대차(%p) | Si | S |
|---|---:|---:|
|66·C0|2.724406|.171357|
|66·C1|2.742825|.175500|
|67·C0|.183354|.398727|
|67·C1|.212720|.403748|

67 C1의II최대차는Si.000020573%p/S.0100791%p,II–VI전체max는.398525/.403751%p,단계별질량p95중max는.098863/.0842141%p다. fraction-only II/III 최대log차는Si.004683/.007117dex,S.002671/.003959dex지만fullheat/opacity활동집합이아니므로최종기준을이숫자만으로통과시키지않는다. d61 SiIII는native7.297203%/C1 7.312758%,SIII는41.418712/41.502926%다. 실제배선버그수정과잔여물리차이해소는구분한다.

정본은위private root `owner{66,67}_J{66,67}_v1/comparison`이다. 66/67 JSON SHA256은각 `12d735299b55da239e9b160e21a7b45c275842e71344dffa7037a8a49cd7a9f4`/`3227c245451d9ddd5b7019f24060421bb8638f46843e61979cdded370b4f44ba`이며 `impact66_67.stdout` SHA256 `b53ecbb04952117e76bfabbe092360c37411439ddd9f4f61d90b610badf3e319`다. C0 기반gap/장민감도의과거수치인증은그행렬의증명으로보존하고수정영향대상으로표기했다. native의실제P/N/보존은그대로유효하다.

### 다음 목적 묶음 — 완료67/J67의 원시SE로 물질식과 장 결합을 분리

Astra의한번종합평가(submission `01a1561c-91dc-73e1-a1ab-bae1ef14870f`)와67감쇠원인추가실측반영(`01a1561e-f741-7cc1-90ea-9f05e1efbe9b`)은 **추가outer보다원시SE witness**를권고했다. agent는종료했다. 원래목표는released-T 자기일관분포/스펙트럼이며이진단을목표완료로대체하지않는다. 현재추가job은미제출이다.

- 목적/끝결정: 최초의이동감소뒤남은차이에native물질식미수렴이어느정도관여하는지정직하게평가하고다음안정화대상을결정한다. witness비간섭성재현과전행의원시잔차/충돌하한/NC/charge/PROVEN·UNPROVEN사유가관측결과다. 결과가판단에불충분하면목적미달/미증명으로남기며로그생성만으로PASS를꾸미지않는다.
- 입력/기준/제외: 원래물질66·J67·67실행제어·원래native/C0/C1/원자료/모든실패를봉인보존한다. 첫실행은원래66→67그대로,두번째는실제완료67/J67의조건부물질식평가다. 새로운RT/장/outer/T해제/상한확대/미량cutoff/새원자자료나source물리변경은없다.
- 운전자실행1: 관찰build의native1회로원래67을재현한다. 완료인구/ne/T/rawcorrection/실제.98936scale을기존67과비교해관찰비간섭을확인한다. α1이나사후확대한오차를요구/허용하지않는다. 입력신원불일치는실행전중단,재현실패는1회에서종료한다.
- 운전자실행2: 재현성확인후에만완료67/J67을1회평가한다. solve전캡처가물질67임을기록하고CaVI포함전체population행을유지한다. solve로사본에생긴후속record는관찰부산물이며주행checkpoint로채택하지않는다. source67 nexteligiblefalse를자동outer승인으로바꾸지않는다.
- 비용/보존: **RT0/추가outer0/native최대2**,예상합계18–20분. CPU16/OMP16/32GiB/GPU0,부모45분·합산child40분·no-pin·retry0이다. 후처리는grammar-debug CPU1/4GiB/300초다. 실행writer는새root만사용하고원래67/C0/C1/사용자변경을보존한다.
- 해석/미완료: 충돌subset하한으로원래evaluated-SE의충분상한을증명할수있지만native내부전산술의실수잔차/RE인증은아니다. 작은하한으로큰상한이생긴행을SE실패로바꾸지않는다. 전행미증명사유/원시값을함께보고하며반대로양수성/작은C gap을nativeSE통과로대신하지않는다. 전체SE·RE·마지막3회·energy/releasedT·분포/스펙트럼/3grid문턱은그대로미완료며T-release는No-Go다.

구현인수는기존 `scripts/{prepare_cmfgen_se_witness.py,cmfgen_native_se_witness.f90,cmfgen_se_collision_bound.f90}`를먼저읽는것부터다. private v3 build는 `/gpfs/kjhan/cmfgen_runs/lumina_se_witness_build_v3_20261020-44LxrD/build`이며빌드/직접시험만끝났고실제native적용/비간섭은아직미재현이다. 다음runner에서원래입력·binary/출력root정체성과합산시간제한을확인한뒤위두실행을한목적의증거로묶는다. 정확한실행명령/Slurm ID/해시/재현결과는실제준비·제출후여기에추가한다.

### 원시 SE 목적 묶음 구현·제출 — 2026-10-20 06:57 KST

운전자가 위 관찰 소스 세 파일과 기존 staging/완료 물질 검증을 읽고 `scripts/run_cmfgen_native_se_observation.py`를 추가했다. 원래 outer 적격 검증은 보존했다. 별도 관찰 validator는 완료67의 부모66·J67·정상 종료·원자료/제어/완료 저장 bytes·RVTJ를 확인하며, `next_outer_eligible=false`를 새 outer 승인으로 바꾸지 않는다. 첫 native는 원래66/J67→67을 재현하고 **완료 record 저장 bytes/pop/ne/T/출력 correction/원래 출력 scale/제어가 모두 일치할 때만** 두 번째 native로 넘어간다. 사후 오차 확대는 없다. solve 직전 관찰 인구가 완료66 또는67과 같은지도 정확히 확인한다.

전 population 행의 원시 합산 구간과 충돌 subset 하한을 읽어 `population-se-rows.csv`에 PROVEN/UNPROVEN 및 사유를 출력한다. 미량 행/terminal을 제외하지 않는다. nc/charge 원시값은 관찰 binary에 보존하고, 평가값의 NC 상한/인구에서 재구성한 charge도 기록한다. 큰 충분상한을 실제 SE 실패로 해석하지 않으며, native 내부 전산술 또는 RE 인증으로 표기하지 않는다. 두 번째 실행이 만든 record68은 채택하지 않는다.

새 시험과 기존 관찰/충돌 하한 시험 **35통과·0.82초**다. root `/gpfs/kjhan/cmfgen_runs/lumina_se_observation_tests_20261020-gNpTuH`, `regression.log` SHA256 `c4d05229fd309c9b765f61b04828132774c8037ccd7a7a8a4243db511070dccb`. 관찰 전용 코드의 직접시험 결과이지 실제 native 비간섭/물리 동질성 성공이 아니다. 쉘 구문과 해당 변경의 diff 공백 검사도 통과했다. 최초 runner 복사 명령은 상대 경로의 작업 디렉터리 착오로 입력을 찾지 못해 실패했고 아무 native도 실행하지 않았다. 프로젝트 경로에서 다시 준비해 10개 파일을 봉인했다.

정확한 제출은 다음 한 번이다.

```bash
cd /gpfs/kjhan/cmfgen_runs/lumina_se_observation_runner_20261020-ij73Ra
sha256sum -c runner.sha256
sbatch --parsable submit_cmfgen_native_se_observation.slurm \
 /gpfs/kjhan/cmfgen_runs/lumina_se_observation_runner_20261020-ij73Ra \
 /gpfs/kjhan/cmfgen_runs/lumina_se_witness_build_v3_20261020-44LxrD/build
```

**Slurm428973**, 06:57:15 제출/06:57:16 시작. 06:57:24 RUNNING8초·syn02/a10·CPU16/32GiB/GPU0·노드 미고정·Requeue0/Restarts0다. `runner.sha256` SHA256 `4586fe353134e327c89413efa27713c7b062b02aa0ad591ece50fd2ade55b788`; Slurm에서도 10개 모두 일치했다. root `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_native_se_observation_428973`, 로그 `/gpfs/kjhan/cmfgen_runs/slurm_logs/native-se-observe-428973.{out,err}`다. 관찰 exe SHA256 `f545980e7613faae7b0a0670121caf7104fa580c5737a1555de80b6b4647efa2`, 원래 native/source/C0/C1는 보존했다. 새 commit/push·Fable 의뢰·대체 검증2 호출은 없다.

native 최대2/합산40분·부모45분·retry0/RT0/outer0이며, 완료 결과는 root `seq_logs/observation-runtime.json`, 중간 결과는 `after-reproduction.json`, 각 실행의 재현/평가 JSON과 전행 CSV에 남긴다. 아직 actual native 비간섭/67 원시 SE 결과는 기다리는 상태다. 원래 전체 물리 목표와 T-release No-Go를 유지한다.

### 원시 SE 묶음 중간 증거 — 첫 재현 byte 일치, 67 평가로 진행

07:07:24 KST 확인에서 첫 native가 종료했고, 07:08:35에는 `after-reproduction.json`과 완료 관찰/두 번째 입력 manifest를 확인했다. 첫 native **565.6694697476923초/returncode0/stop_reason없음**이다. 완료67의 저장 record/pop/ne/T·출력 correction 배열·원래 .98936 scale·POINT1/2·제어가 모두 일치해 `noninterference=REPRODUCED`다. 운전자가 별도 `cmp`로 원래67과 재현본의 전체 SCRTEMP·POINT1/2도 byte 같음을 확인했다. SCRTEMP SHA256 `01a3860787401869c00abf22db545fad9e729668b25efe65d43d5ec713f112fb`, 재현 JSON SHA256 `6a4c923311c3f7d7135390c38719a10cbf898c9b1d8b25ceb47b2f49cebcea72`다. 기존 규약보다 허용오차를 넓히지 않았다.

첫 `NATIVE_SE_BOUND_000067.bin`(SHA256 `0738431dc6d781830af9d1fcced007b7fd3a25b2e7c81c5ee2c1678c69ec0cc6`)의 실제 phase는 **물질66/J67, solve전**이다. header는 population1798/깊이90/전체변수1800/ion33/species-slot28이다. 평가 인구는 봉인 완료66과 정확히 같다. 전체161820행 중1000행만 충돌 subset 하한의 충분상한으로1e-8 이하를 입증했고160820행은 상한이 충분히 작지 않아 UNPROVEN이다. coverage/유효성 누락이나 새 cutoff는 없다. Si2/24390, S0/22230 PROVEN은 **미수렴의 증명으로 바꾸지 않는다**. 첫 phase는 새 J67로 이미 바뀐 물질66을 평가한 것이므로 완료67의 SE 판정도 아니다.

NC 유효540행은 실제 모형6원소×90이고, 비활성 원소 slot을 포함한2520행과 구분한다. NC 평가상한 최대2.742371848070432e-15, 인구에서 계산한 charge3.3572003084541448e-15다. 첫 `native-se-assessment.json` SHA256 `cf27392be7c864b47f408bc1f8d33a428fd15d95db5e1277d8ae760d04a760ac`, 전행 CSV SHA256 `bcf4bfe1d1d5826b45aa576e5d25c77c69edcd567120d296eb369a19a8746c8c`다. binary/JSON/CSV는 모두 같은 `reproduction` 사본에 있다. native 내부 전산술·RE·원래 전체 물리 목표는 여전히 미인증이다.

같은428973의 `evaluation` 사본에서 **입력67/J67의 두 번째 native가 시작**됐다. 추가 제출/RT/outer는 없고 후속68은 관찰 부산물로 남긴다. source67의next_outer_eligiblefalse는 바뀌지 않는다. 완료 후 두 phase를 묶어 평가하고, 이 첫 관찰 통과를 전체 목적 달성으로 대신하지 않는다. Fable/대체검증2 호출도 없다.

### 원시 SE 묶음 완료 증거와 후속 계획 — 07:24 KST

Slurm428973은 **07:17:00 COMPLETED/0·19분44초**다. child1/2는565.669/568.586초, MaxRSS는각5549580/5549340KiB다. 정확히native2/RT0/outer0, 원본 보존true이며 허용 예산 내에 끝났다. 재현은 전체 SCRTEMP/POINTS/pop/ne/T/출력correction/원래.98936scale까지 일치했다. 두 번째 캡처는 **67/J67 solve전**이며 입력 인구가 봉인 완료67과 byte 같다. 후속68은 관찰 부산물로만 남겼다.

161820행 중 **38164 PROVEN/123656 UNPROVEN**, coverage/유효성 누락0이다. Si4408/24390, S4037/22230행을충분상한으로1e-8이하입증했다. 나머지는 전부충분상한이작지않은것이지실제SE실패가아니다. 미량/terminal 제외는없다. 실제6원소×90 NC540행 평가상한max3.164275209312037e-15, 인구재구성charge3.930901182735637e-15다. 원래fullD/native내부전산술/RE/전체SE수렴·물리동질성은 미인증이다.

- runtime SHA256 `4c7e0432119069527cad173648b0d0d6f7f08a26fcfa17a8353399e60dfe982a`
- evaluation `NATIVE_SE_BOUND_000068.bin` SHA256 `bc918237ec85bf7e0a59d4748b75d780d45553763f18c06945555186b6289445`
- evaluation `native-se-assessment.json` SHA256 `c0775e7a02c7bd876772d45153f6275fd1e1346c36c93562c0ec3b278bb792ef`
- evaluation 전행 CSV SHA256 `5fe26286333ddc1dff18b48dccf1da574a7e2cc26d17ddf89a40c521c13c773b`

운전자 phase 비교는 `/gpfs/kjhan/cmfgen_runs/lumina_se_observation_analysis_20261020-VKtHzb/analyze.py`로 grammar-debug CPU1/4GiB/300초에서 실행했다. `phase-comparison.json` SHA256 `eb6166dc0bf0285a5ad9b870774163ea785fb587c0e33dee744e6cfee37becce`다. 첫 파일 조회는 SSH 분석이 아직 실행 중이라 파일없음으로 실패했으나 session51606을 유지해 정상 완료했으며 분석/native를 재시작하지 않았다. net 분율률은 원래 scaled-SE norm이 아닌 진단이다. 같은J67의 부산물67→68은 minscale1/ne최대step0.000537640555687724, III 최대이동 Si.2026561315601194%p/S.06106015813747145%p다. 전체SL raw correction p95는Si.01338365/S.0036148(비율)이며 원래열/율/불투명도합집합active규약을대신하지않는다.

전자밀도차혼입을피하기위해 `compare_C68.py`로관찰68의실제인구/T/ne와동일J67을사용한수정C1을깊이34/35/61×Si/S의6셀만새로풀었다. 입력field block byte 일치,6/6인증·실제채택이며새native/RT는없다. 결과 `selected-C68/comparison.json` SHA256 `f5a7fefa80103f408f085203d2140e89d56c1226eb8d2dc839d6b6a4e3419324`다. III C1−native gap(%p)은d34 Si+.212720→−.003371/S+.403748→+.352218, d35 Si+.024979→−.062864/S+.330995→+.264219, d61 Si+.015554→+.015554/S+.084214→+.084214다. d61 ne변화는1.12035e-11이다. 이는 대표6셀의 조건부 비교이며 전체90점/공동수렴/앵커 인증으로 확대하지 않는다. Si한위치에서갭이크게줄었다고모든차이를해결했다고하지않고 S의잔여차이도기록한다.

Astra가원시SE관찰전체를한번평가했다(submission `01a1563f-8ccb-7c12-aaa7-ada355777eb0`,agent종료). 판독은 **고정J의물질응답은작아졌으나전체SE수렴은미증명, 다음은실제장결합검증**이다. 새기획목적은 **실제장갱신에서Si/S분포와ne이동이함께감소하는지확인해현재결합법을계속쓸지결정**하는것이다. 원래67→새J68→새물질68, 중단사유가없으면새68→새J69→새물질69의 **최대2outer/RT2/native2**다. 관찰부산물68은선택하지않는다. 대표6셀C후처리는Astra평가뒤완료됐고새모델호출은추가하지않았다.

구체적범위·입력·보존·부분step정책·종료/69선택기준·자원은 `docs/CMFGEN_COUPLING_68_69_2026-10-20.json`에기록했다. 기본fullstep적격검증을유지하면서이번별도계약에서만양수성제한부분step을허용한다. source67의기존nexteligiblefalse를바꾸지않고성장상한정체면중단한다. C1동일입력장유발변화/실제부모자식II–VI질량p95·max/ne/현재장실제ne의C1조건부차이/원시SE전행사유를기존출력으로소비한다. 첫outer에서두원소의장·물질이동과ne가모두악화되고조건부차이도개선되지않으면69생략,혼합이면69까지허용하되자동연장없음이다. 큰반전·정체가지속되면다음수정은결합안정화이며단순반복연장은권고하지않았다.

각outer CPU16/OMP16/32GiB/GPU0·부모45분/공유child40분·no-pin/retry0,예상각38–39분이다. 물질update와동시에비간섭확인witness를사용하며별도관찰native를더하지않는다. 기존67/관찰68/C0/C1/원자료/사용자미커밋변경/실패는보존한다. 다음실행runner/직접시험/정확한재현명령은아직구현전이며 **새Slurm은미제출**이다. 필요한수정C1과거field기준은같은입력으로만재산출해C0지표와혼용하지않는다.

가역적build준비만먼저끝냈다. 기존 `prepare_cmfgen_feedback_build.py --parent /gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/formal_primary_v11 --iteration 68`을grammar-debug CPU1/4GiB/300초/GNU13에서실행해 `/gpfs/kjhan/cmfgen_runs/lumina_coupling68_build_20261020-Mzkj1R/build`를만들었다. exeSHA256 `fc95ff32870139e1ab5ce304825e400439524a8987028d75d175e82d21fdea72`,manifestSHA256 `ce6375078e51663505b5a8e096495bd979f52de07fc4f5debe5cce08ec1445f1`다. 세대기록만68로바꾸며수송커널/STOP97/native원본불변이다. 69build/실행은첫결과뒤결정한다. 최종원래v2목표는미완료/active며T-release는No-Go다.

### 실제 장 결합 68/69 구현·첫 제출 — 07:34 KST

운전자는 `cmfgen_coupling_contract.py`로명시적이번계약의67/68만허용하고기존기본fullstep거부규약은그대로뒀다. 기본검증으로실제67은계속거부되며계약을명시할때만재검증/수락한다. 원래67의nexteligiblefalse는수정하지않았다. nativeLOCAL과같은population+ne범위에서ES12.4반올림폭을포함해최대증가상한17.3175<1e6,감소보정하한1.01075>1−1e−6,실제scale.98936을확인했다. fixed-T보정0을확인하고증가상한/온도보정/원인불명부분step/관찰68/계약밖세대는거부한다. 이는양수성보호입력허용이지잔차감소line-search증명이나전체SE통과가아니다.

프로젝트변경은 `run_cmfgen_field_feedback.py`의명시계약인수/검증witness실행/동시SE판독, `run_cmfgen_native_se_observation.py`의명시적인장생산물질인수, 새 `submit_cmfgen_coupling.slurm`, 계약모듈과직접시험이다. 원래native/sigma/수송커널/C1물리소스는변경하지않았다. 새물질update는비간섭확인witness를동시에활성화해입력67/J68의원시잔차를남기며추가관찰native를요구하지않는다. 실제사용exe와원래native부모신원을 `material/seq_logs/material-execution.json`에구분한다. 각세대배타청구파일로retry0/최대2회를집행하며기존청구를지우거나재시도용으로되돌리지않는다.

기존판독기 `evaluate_cmfgen_feedback_66_67.py`는기본65–67동작을보존하고계약명시시에만67–69를소비하도록확장했다. 실제부모자식연속성/C1신원/SE물질·장phase를확인하고큰공동악화면69생략,혼합이면69까지,세번째새outer는거부한다. 과거fullstepfalse는true로재명명하지않는다. 새선택증거는생산자사본을해시해이후작업트리편집으로이미만든판독의재현성이깨지지않게했다.

제출전관련104통과·1.92초 및 실제source67검증은 `/gpfs/kjhan/cmfgen_runs/lumina_coupling_verification_20261020-jR6oU0/{regression.log,preflight.log}`에있다. SHA256각 `88f493ca56fadf28401cc411b755566794cae1f4ac4ae9e2f7f896ad31c66b76`/`2dd00cf0b8284319f1a47a4925137ec0841c062e956b2c6736b5045d89c530fd`다. 판독기범위/분기시험까지최종 **106통과·2.00초**, `/gpfs/kjhan/cmfgen_runs/lumina_coupling_final_tests_20261020-6nHpLc/regression.log` SHA256 `85dc9b29317f9886f1e9093dd4f3727cb83d5dd07f89fcb5280d14c84fb54bfd`다. 실제검증이원래67 recordSHA `e65db392af2daeefb6bfc5b8d69bd696cff6e3ff770511e82893749e77cf1f98`와pointer67/67을확인했고물리실행0이다. 원시LOCAL범위확인중population뿐아니라ne도포함함을확인해제출전검사를보완했으며104시험에는ne증가상한음성대조가포함됐다. 최종판독기변경은현재실행중인봉인runner를바꾸지않는다.

프로젝트계획JSON의구현상태만갱신했다. 범위/과학기준/비용은기존Astra계획그대로이며새SHA256 `4a67206d8622f4397d03fa5acf7919d26a41d10f882094c16a1e9d0af6928608`이다. runner `/gpfs/kjhan/cmfgen_runs/lumina_coupling_runner_20261020-73UGpi`의12파일을봉인했고 `runner.sha256` SHA256 `57d09375ea2f102f0a663998c99e861712c785e2806437e4921d953c79b26133`이다. source/build/관찰비간섭근거는변경하지않았다.

정확한제출은한번이다.

```bash
cd /gpfs/kjhan/cmfgen_runs/lumina_coupling_runner_20261020-73UGpi
sha256sum -c runner.sha256
sbatch --parsable submit_cmfgen_coupling.slurm \
 /gpfs/kjhan/cmfgen_runs/lumina_coupling_runner_20261020-73UGpi \
 /gpfs/kjhan/cmfgen_runs/lumina_coupling68_build_20261020-Mzkj1R/build \
 /gpfs/kjhan/cmfgen_runs/lumina_coupling_runner_20261020-73UGpi/CMFGEN_COUPLING_68_69_2026-10-20.json \
 4a67206d8622f4397d03fa5acf7919d26a41d10f882094c16a1e9d0af6928608 \
 /gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_428960/material 67
```

**428985**, 07:34:07 제출·시작, 마지막07:45:38 RUNNING11분31초·syn02/a10·CPU16/OMP16/32GiB/GPU0·노드미고정·Requeue0/Restarts0이며 stderr는비어있다. root `/gpfs/kjhan/cmfgen_runs/toy06_19p48d_field_feedback_428985`, 로그 `/gpfs/kjhan/cmfgen_runs/slurm_logs/coupling-428985.{out,err}`다. Slurm 안에서도12파일전부일치했다. 물질67→J68→물질68이며관찰68은사용하지않았다.

병행한C1기준재계산도완료했다. `/gpfs/kjhan/cmfgen_runs/lumina_coupling_c1_20261020-TsyI0J/run_baseline.py`가동일한물질66/J67입력으로전90깊이Si/S C1을새로풀어각90/90인증·실제채택/II–VI900행/support불일치0을확인했다. `owner66_J67_v1/comparison/fixed-field-owner.json` SHA256 `0cf1276fd5128e70ab39f68501cb0618274a3a01dc5db22c38bc8989f03e1094`, `field_only66_J67_v1/field-only.json` SHA256 `913f3979cf40d8d79eaa1c0e951c2bf99f86fa944e7fa599b5708f53fe1e6c30`, `baseline.json` SHA256 `94e8b5a74b46170f97c87a67faaecdef8aba4f758b75407aa49032cba6bf2a7d`다. 물질66고정J66→J67의III최대변화Si15.447255%p/S4.127808%p이며새J68의완료결과와혼동하지않는다. 새RT/native0,기존C0율/해재사용0이다.

현재진행은새RT68이다. 장완주후C1 pre67/J68을native와병행하고, native완료후실제ne68/J68 C1을추가해field-only67/J68·실제물질이동·ne·조건부gap을한번의첫outer판독으로묶는다. 기준67은기존C1 `owner67_J67_v1`과이번수정C1 field_only66_J67이며C0비교와혼용하지않는다. 계약인수/새SHA와두frame을명시한판독의 `CONTINUE69_WITHIN_LIMIT`이실제새68을선택할때만69build/제출을진행한다. 이번턴에Astra/Fable/다른모델재호출,commit/push,원본/실패/관찰68삭제는없다. 전체물리목표는계속미완료/active이며T-release No-Go다.
