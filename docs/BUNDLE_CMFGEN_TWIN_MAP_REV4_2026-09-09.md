# DET-CMFGEN-TWIN-MAP 개정 4 — 두 분모와 실패 보존 수정묶음

2026-09-09 · 사용자 승인: 「수정으로 진행」 · 설계/사전감사: 운전자.
**Q-GOAL=PASS · Q-LEAN=APPROPRIATE · DESIGN=PASS. 구현·실물 실행 PASS 아님.**
계약 정본은 `docs/PREREG_CMFGEN_TWIN_MAP_2026-09-08.md`다. 본 문서는 해당 계약의
개정 4 변경집합과 수용기준이며 개정 3의 상충하는 단일 분모 정의보다 우선한다.
과거 집행 기록을 다시 구현하지 않는다. 이 수정은 기존 묶음의 재작업이며 신규 GPU 레인이 아니다.

## 목적·범위·보존

생산 원자량을 바꾸지 않고 생산 분율과 A=56 비교량을 구분하여, 질량 규약 차이를
population 결손으로 오독하지 않는 관측 지도를 만든다. 비유한 NLTE의 전체 중단과 잘못된 닫힘 표기도 함께 고친다.
구현 소유자는 GPT-5.6-sol, 검수·통합·실행·독립 재현 소유자는 운전자다.

입력은 개정 3과 같은 BASE·대응 소스·바이너리·참조 자료다. 출발 구현 identity는
`docs/REVIEW_CMFGEN_TWIN_MAP_REV3_2026-09-09.md`에 고정돼 있다:
판독기 `73a9042b5b77646a4488319ce2e238678b265609a0ef701cc95b9062ec686ad5`,
stager `ee3f2ca2cf9f2c6c9ee21969609b7e59a2722030fb71f30ffcfa422fe763ec38`.
원 검수·로그·출발 구현은 별도 보존하고 덮어쓰지 않는다.

변경 파일은 기존 `scripts/cmfgen_twin_map.py`, `scripts/stage_cmfgen_twin_map.sh`뿐이다.
생산 C/CUDA·바이너리·원자량표·밀도·조성·BASE·표본·GPU 횟수·자원 상한을 변경하지 않는다.
기존 아홉 산출물에 필드만 반영한다. 새로운 gate·runner·상주 작업·증거 장부를 만들지 않는다.
신규 실물 RUN은 여전히 미승인 자격 상태다. 실패 시 코드·증거를 보존하고 해당 판본을 비채택한다.

## 변경·수용기준

| 항목 | 구현과 실제 소비 경로 시험 |
|---|---|
| 두 원소 분모 | `N_Z_model`은 이제 `N_Z,prod`의 명시적 alias. 생산 mass CSV에서 모든 원소의 `rho*X/(a_sealed*m_u)`를 계산한다. `N_Z_comparison`은 IGE A=56·나머지 sealed, `N_Z_production_to_comparison`은 둘의 양수 총량비. 0/0은 null/정의 불능. 덤프 합은 분모가 아니다. |
| 두 전체 원자수 | `n_atom_production=ΣN_prod`, 기존 `n_atom_model=ΣN_cmp`, 둘의 비를 기록한다. 전체 조성(비교 6원소 밖도 포함)을 각각 합산한다. |
| 지표 명칭 | `q_comparison=ne/n_atom_model`, `z_production=ne/n_atom_production`. 오해 소지가 있는 `zbar_lumina`는 새 schema에서 제거하고 verifier/consumer를 같이 변경한다. `d_Z`·E-Z는 `q_comparison/(ne_C/natom_C)`로 유지한다. metadata에 분율/생산/비교 규약을 구별한다. |
| 생산 분율 | Saha·NLTE 분율과 `N_Z_dump` 대조 모두 `N_prod`를 사용한다. 실제 봉인 Fe/Co/Ni 원자량으로 만든 생산 정규화 fixture가 `ELEMENT_DENOMINATOR_MISMATCH` 없이 소비돼야 한다. 진짜 총량 교란은 여전히 검출한다. |
| 닫힘 | 생산 전 단계와 참조 전 단계의 원값·표현 구간에서 `Σn_q/N_prod-Σf_C`의 잔차·구간·0 포함 여부를 계산한다. 참조 합 검사와 독립된 필드다. 생산 합만 교란하고 참조 합=1을 유지하면 닫힘 불일치여야 한다. 결손·NaN·분모 0·최고통 미정의는 일치 null이며 재정규화하지 않는다. 직접통·누적통 양쪽 시험이 필요하다. |
| 비유한 NLTE | NaN·±Inf·음수 n_pop이 있어도 셀 상태·원토큰을 남기고 나머지 지도/summary를 출력한다. `R_SN`을 계산하려고 같은 실패 토큰을 재판독해 전체 중단하지 않는다. 실패 셀의 수치 R_SN은 공란이며 출력 0은 결손과 구분한다. |
| 기존 대조 | 두 직접 selftest와 실제 참조 50셸 prepare를 실행한다. 두 분모 분리 전후 같은 BASE의 E-Z 피연산자는 변하지 않아야 한다. 공개 n_e나 선정값을 대체하거나 누락 셸을 제외하지 않는다. |
| stager | 새 schema/규약 필드 소비만 갱신한다. 최고통·loader·source 미확인 및 A100-only runner 차단은 정직하게 유지한다. 출처 없는 최고통 의미를 임의 지정하지 않는다. 이번 승인만으로 hardware guard를 삭제하거나 새 runner를 만들지 않는다. |

준위 피복·공개 세대·참조 최고통 의미의 기존 한계는 그대로 남는다.
고정-T 실제 loader 시험은 여전히 필요하나 생산 빌드/입력 변경 없이 가능하다는 근거가 없으면
미시행으로 보고한다. 이 수정에서 이미 아는 runner 미적격을 반복 GPU 제출로 재현하지 않는다.

## 자원·증거·재현

기존 전체 상한을 유지한다: GPU 1회·8 GPU-hour, 런 128 할당 CPU-hour,
오프라인 단일 CPU 누적 1 CPU-hour, 신규 총량 16 GiB. 이번 구현/사설 시험의 GPU 소비는 0이다.
로그인 노드에서는 파일 조회만, 단일 CPU 시험은 grammar-debug에서 한다.
기존 테스트 로그와 개정 3 재현을 되풀이 감사하는 별도 모델 호출은 만들지 않는다.

재현 명령(작업 디렉터리 `/gpfs/kjhan/lumina/cmfgen_map/worktree`):

```bash
python3 -B scripts/cmfgen_twin_map.py --selftest
bash scripts/stage_cmfgen_twin_map.sh --selftest
```

운전자는 개정 3의 `impl_rev3/operator_reproduce.py`를 덮어쓰지 않고 그 실패 입력을
현행 schema에 맞춘 사설 재현으로 확인한다. 생산 소비 경로 시험과 독립 산술을 혼동하지 않는다.
증거는 변경 diff·파일 해시·각 시험의 stdout/stderr/rc·결정적 셀 값·미결 사유다.
코드 검수에서 이상이 없으면 오프라인 준비로 진행하되, runner·loader·비용/보존 자격 없이 GPU를 제출하지 않는다.
묶음 산출 완료 전 Opus5 종료감사를 미리 호출하지 않는다.

## 운전자 설계감사

Q-GOAL — 이 수정은 최종 MC SN Ia 스펙트럼 목표에 부합하는가?
분율과 전자/원자수비의 비교 가능성을 회복하는 오프라인 관측 준비다. 구체적으로 Fe/Co/Ni의
가짜 분모 결손과 NaN으로 소실되는 다른 셀을 없애고, 닫힘 실패의 크기를 사실대로 남긴다.
물리 오차를 줄였다는 주장은 하지 않는다. 생산 원자량을 바꾸는 대안은 T-only 대조를 훼손하고
추가 물리 변경을 섞는다. 이번 수정을 건너뛰면 이미 재현된 약 5% 규약 차이를 population 결함으로
오독하게 된다. 직접 수리 표적 D-C-2/D-D를 대신 완료했다고 주장하지 않는다. **Q-GOAL=PASS**.

Q-LEAN — L1: 분모 필드는 분율과 규약 차이 해석, 잔차 구간은 닫힘 대조에 각각 소비한다.
L2: 참조 합 flag의 재사용을 제거하고 실제 잔차를 계산하며 새 인증 gate는 만들지 않는다.
L3: 서로 연결된 분모·닫힘·비유한값 처리를 한 수정과 한 코드 검수로 묶는다.
L4: `zbar_lumina`의 오해 소지 alias를 제거하고 기존 검증/집계를 교체한다. 생산을 바꾸는 별도
질량 실험·새 runner·GPU 획득런은 배제한다. **Q-LEAN=APPROPRIATE**.

대수 대조: `N_prod/N_cmp=a_cmp/a_sealed`; `Σ(n_q/N_prod)=1`은 생산 전건 닫힘에만 적용한다.
직접통의 고단계 꼬리와 중성항, 누적통의 중성항을 원식에서 소거하면 둘 다
`Σn_q/N_prod-Σf_C`가 남는다. E-Z는 비교 분모를 유지하므로 새 분율 분모 변경과 독립이다.
전제 identity와 기존 실패 재현은 개정 3 검수문에 고정했다. 이 대수 검토는 실행 시험 PASS가 아니다.
**전체 설계 PASS. 구현·직접 시험·코드 검수는 후속 실측 대상으로 남긴다.**
