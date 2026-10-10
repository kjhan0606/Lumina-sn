# 사전등록 — DET-BLUE-ION-MAP: 청색 결손 기준 O의 철족 이온 분율·절대 대역 광도 비교

2026-10-10 · 읽기 전용 기안  
운전자 파일화 대상: `docs/PREREG_DET_BLUE_ION_MAP_2026-10-10.md`  
상태: **기안. 구현·직접시험·기계 게이트·실행·판정·종료감사는 미집행.**

## §0. Q-GOAL·Q-LEAN 자기답과 과학 질문

**Q-GOAL — 최종목적에 직접 기여하는가? 자기답: PASS.**

최종목적은 CMFGEN과 비교 가능한 SN Ia 이온화 상태와 스펙트럼을 얻는 것이다. 폐합된 진단은 O의 2500–4000 Å 출력 약 88.7%, 4000–6000 Å 출력 약 61.5%가 매핑 선 레인에 기록됨을 확인했다. 그러나 그 레인을 담당하는 이온·셸과 결손 원인은 식별하지 않았다.

이번 과학 질문은 **「청색·짧은 광학 출력의 매핑 선 원천을 조사할 후보인 Fe·Co·Ni에서, O의 II·III 분율은 공개 CMFGEN보다 어느 셸에서 얼마나 높거나 낮은가? 이 분율 지도와 함께 제시할 절대 대역 광도 결손은 얼마인가?」**다.

새로 채우는 칸은 **Fe II/III·Co II/III·Ni II/III × 전 50셸의 CMFGEN 대비 dex**다. 기존 청색·광학 결손 칸은 절대 광도로 함께 제시한다. 이 단을 생략하면 철족 개체수 차이가 있는 위치조차 모른 채 매핑 선 원천·여기·수송 중 다음 조사 대상을 골라야 한다.

이번 단은 이온화 상태의 비교 공백을 좁힌다. 분율 차이가 청색 결손을 설명하는지, 해당 이온이 청색 출력을 지배하는지는 판정하지 않는다.

**Q-LEAN — 각 측정은 어디에 소비되는가? 자기답: APPROPRIATE.**

| 유지하는 측정 | 소비하는 칸·결정 |
|---|---|
| Fe·Co·Ni의 NLTE II·III 합과 원소 총량 | 셸별 분율·dex 300행 |
| 셸 속도와 공개 최근접 속도 | 분율 비교의 공간 대응과 그 한계 |
| O·공개의 여덟 대역 및 OPT 적분 | 절대 광도·dex 9행 |
| 입력 신원·세대·시각 | 동일 O 관측 상태의 비교인지 확인 |
| 실제 CPU·벽시계 사용 | 작은 오프라인 범위 준수 |

기존 진단의 성분 분해는 출발 근거로 승계한다. replay 재실행·이온별 수송 분해·준위 역산·J̄ 재구성·순율 재계산은 하지 않는다. 공개 분율만으로 선 원천이나 탈출 광도를 정할 수 없으므로, 이를 흉내 내는 보조 지표도 만들지 않는다.

D5-2의 공개 철족 분율 비교를 채택한다. 이온별 replay 도구는 이번 질문에 필요하지 않아 채택하지 않는다. D5-3에 따라 장파장과 청색을 독립적인 물리계로 나누지 않으며, 동일 O 스펙트럼의 여덟 대역을 함께 보고한다.

## §1. 기준선·입력·범위

| 식별자 | 고정 대상 |
|---|---|
| 프로젝트 P | `/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn` |
| 기준 O | `/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1` |
| 진단 D | `/gpfs/kjhan/lumina/cmfgen_map/diag7_bluedef_20261010` |
| 새 작업 기반 B | `/gpfs/kjhan/lumina/cmfgen_map/blue_ion_map_20261010` |
| 공개 자료 | `data/standart_data1/toy06` |

사용자 결정은 「1번 청색 결손부터, Ω는 기본값으로 채택」이다. O는 N에 `LUMINA_OMEGA_CMFGEN=1`을 추가한 12회 계산이며 COLL_FIX를 유지한다. 이번 단은 그 저장 결과를 읽는다.

고정 입력은 다음과 같다.

- O의 `work/physics_comparison/physics_DET_iter0010.ion.csv`와 대응 manifest·shell CSV.
- O의 `work/lumina_spectrum_det_obs.csv`와 `work/det_obs_capture/manifest.json`.
- 공개 `ionfrac_fe_toy06_cmfgen.txt`, `ionfrac_co_toy06_cmfgen.txt`, `ionfrac_ni_toy06_cmfgen.txt`, `spectra_toy06_cmfgen.txt`.
- O의 STAGE_IDENTITY.txt·RUN_FOOTER.txt, 직전 판정문과 진단 폐합 기록.

**생산 소스·바이너리·입력·환경변수 변경 0, 생산 모델 진입 0, GPU 0**이다. 새로운 물질·복사 상태를 만들지 않는다. Si·S·Ca 비교 확장, 온도·밀도 비교, 충돌 처방 평가도 이번 범위에 넣지 않는다.

## §2. 관측 세대·비교 시각·입력 신원

이온 덤프는 **population_generation=13**, 대응 manifest는 **radiation_generation=12**, iteration=10이다. 이 관계를 오류로 간주하거나 세대 14로 교체하지 않는다.

관측 스펙트럼의 캡처 manifest는 **material=13, radiation=13, iter=11**이다. 분율과 스펙트럼은 같은 관측 물질 세대를 가리키지만, 이온 덤프에 기록된 복사 세대까지 같다고 쓰지 않는다.

공개 시각은 이온 자료의 **19.480 d** 블록과 스펙트럼의 **19.48 d** 열로 고정한다. 최근접 시각 대체·시간 보간은 하지 않는다. 물리 덤프의 epoch_s=1683072와 연결한다.

실행 시 실제 읽는 입력·재사용 모듈·새 코드의 SHA-256과 경로를 evidence.json에 기록한다. O의 저장 스펙트럼은 캡처 manifest의 신원과 대조한다. 캡처 payload 전체, 생산 바이너리의 재현성, 이미 폐합한 replay를 다시 검사하지 않는다.

## §3. 셸·이온 대응과 분율 정의

대상은 **Z=26,27,28, q=1,2, shell_id=0…49**로 고정한다. II는 q=1, III는 q=2다. 기존 다섯 셸에 제한하면 진단 장부가 지목한 안쪽 셸을 놓치므로 전 50셸을 사용한다.

\[
f^O_{Z,q,s}=
\frac{\mathrm{Z}Z\_\mathrm{nlte\_q}q\_\mathrm{cm3}}
     {\mathrm{Z}Z\_\mathrm{element\_total\_cm3}},
\qquad
d^f_{Z,q,s}=\log_{10}\!\left(\frac{f^O_{Z,q,s}}{f^{C}_{Z,q,s}}\right).
\]

SAHA에 해당하는 stage 열, II+III 합을 분모로 한 재정규화, 다른 세대의 원소 총량을 사용하지 않는다.

공개 대응 열은 Fe II/III=`fe1`·`fe2`, Co II/III=`co1`·`co2`, Ni II/III=`ni1`·`ni2`다. 공개 헤더의 0…5를 I…VI로 읽는 명명 가정에 조건부임을 표에 명시한다.

공간 대응은 다음 규칙으로 사전 고정한다.

1. Lumina 셸 속도는 대응 shell CSV의 `(v_inner_cm_s+v_outer_cm_s)/2/1e5`로 읽는다.
2. 등록 격자는 `v_mid(s)=4264+728s` km/s다. shell_id와 속도가 이 관계를 만족하는지 확인한다.
3. 각 원소의 지정 공개 시각 블록에서 속도 차의 절댓값이 가장 작은 행을 선택한다. 동률이면 낮은 속도를 선택한다.
4. Lumina 속도가 공개 중심 속도 범위를 벗어나면 해당 비교는 미정의다. 끝점 대체·외삽하지 않는다.
5. 각 행에 양쪽 속도, 부호 있는 속도 차, 공개 열 이름과 원문 분율 토큰을 남긴다.

최근접 비교는 같은 질량 셸이나 셸 체적 평균의 비교가 아니다. 여러 Lumina 셸이 같은 공개 행에 대응해도 보존하며 독립 표본으로 세지 않는다. 분율값을 보고 대응 방식을 바꾸지 않는다.

## §4. 절대 대역 광도 정의

여덟 대역은 **2500–3000, 3000–3500, 3500–4000, 4000–5000, 5000–6000, 6000–7000, 7000–8000, 8000–10000 Å**다. OPT는 **4000–6000 Å**다.

\[
L_b=\int_{\lambda_{b,0}}^{\lambda_{b,1}}L_\lambda\,d\lambda,
\qquad
d^L_b=\log_{10}\!\left(\frac{L^O_b}{L^C_b}\right).
\]

기존 판독기의 경계 선형 보간·사다리꼴 적분을 재사용한다. 입력 단위는 erg s⁻¹ Å⁻¹, 적분값은 erg s⁻¹다. 거리 환산·추가 광도 정규화·외삽은 하지 않는다.

**주 표의 dex는 절대 광도 비다.** `L_b/L_OPT`의 색 잔차나 R₈를 대신 넣지 않는다. OPT는 여덟 대역과 겹치는 요약 행이며 아홉 번째 독립 대역으로 합산하지 않는다.

## §5. 주 산출과 기대치

주 산출은 **하나의 CMFGEN 대비 표**이며, 분율부와 광도부를 같은 기계 표·표시본에 담는다.

| 행 종류 | 행 수 | 필수 자료 |
|---|---:|---|
| Fe II/III·Co II/III·Ni II/III × 50셸 | 300 | 셸·양쪽 속도·공개 분율·O 분율·부호 있는 dex·수치 구간·상태 |
| 여덟 절대 대역 광도 | 8 | 대역·공개 광도·O 광도·비·부호 있는 dex·수치 구간·상태 |
| OPT 절대 광도 | 1 | 같은 항목, 중복 요약 표시 |

총 **309행을 항상 유지**한다. 영 원소량·공개 범위 밖·미정의 로그도 행을 삭제하지 않고 원량과 사유를 남긴다. 분율부는 Z·q·shell_id, 광도부는 파장 순으로 표시한다.

**기대치 사전등록**

- Fe·Co·Ni II·III 분율 잔차의 부호·크기·방사상 순서는 예측하지 못한다.
- 셸 3의 Co III 국소 순율이 크다는 선행 관측으로 Co III 분율 과잉이나 청색 출력 지배를 예측하지 않는다.
- 기존 O의 여덟 절대 대역 광도는 모두 공개값보다 작았다. 따라서 동일 입력·정의의 광도부는 모두 음의 dex를 재제시할 것으로 기대한다.
- 직전 판정의 대역 비는 순서대로 약 **0.163, 0.106, 0.101, 0.162, 0.294, 0.470, 0.878, 0.739**, OPT 비는 약 **0.2111**이다. 반올림된 이 값을 새로운 수치 통과 문턱으로 삼지 않는다.
- 새 상태를 계산하지 않으므로 스펙트럼 개선·악화 또는 N→O 처방 효과를 이번 단의 새 결과로 등록하지 않는다.
- II와 III가 모두 공개값보다 낮거나 높은 결과도 허용한다. 나머지 이온 단계를 역산해 채우지 않는다.

대역값이 선행 기록과 질적으로 다르면 입력·단위·시각·절대값/색 혼동을 먼저 확인한다. 기대를 결과에 맞춰 바꾸거나 해당 대역을 제외하지 않는다.

## §6. 수치 구간·행 분류·해석 규칙

공개 이온 분율은 원문 토큰의 표시 정밀도 구간을 사용한다. O 덤프는 저장된 binary64 값을 출발점으로 기존 구간 산술을 적용한다. 대역 광도는 기존 적분의 부동소수점 경계를 사용한다.

이 구간은 물리 모형·관측·공간 대응·반복 절단 오차를 포함하지 않는다. 공개 스펙트럼 원문값의 불확실성 전체를 포괄하는 구간으로도 해석하지 않는다.

| 코드 | 의미 |
|---|---|
| 0 | dex 구간 상한 < 0: 공개값보다 낮음 |
| 1 | dex 구간 하한 > 0: 공개값보다 높음 |
| 4 | 유효한 dex 구간이 0을 포함: 부호 미분리 |
| 5 | 원량 또는 비교가 미정의: 사유 병기 |
| X | 잘못된 입력 신원·세대·열·셸 대응 또는 행 누락·중복으로 비교 부적격 |

영 분율·영 분모·비양의 로그 인수는 작은 양수로 바꾸지 않는다. 음수·비유한 원량은 그대로 사유를 남기며 dex를 계산하지 않는다. 정확한 중앙값 일치도 코드 4에 포함하고 물리적 일치 판정으로 확대하지 않는다.

과학 해석은 다음 순서를 고정한다.

1. 여덟 대역과 OPT의 절대 결손을 제시한다. 이는 기존 O의 재표시임을 밝힌다.
2. 여섯 이온 각각에 대해 낮음·높음·부호 미분리·미정의 셸 수를 **분모 50**과 함께 보고한다.
3. 같은 원소·셸에서 II 과잉/III 결손이면 「두 단계 사이에서 낮은 전리 쪽으로 치우친 패턴」, 반대면 「높은 전리 쪽 패턴」이라고만 기술한다.
4. 같은 부호·혼합 부호·미정의가 있으면 그대로 보고한다. 전체 평균 전하나 에너지 부족으로 번역하지 않는다.
5. 큰 분율 어긋남이 있는 이온·셸은 후속 개체수 조사 후보다. 어긋남이 작더라도 여기·J̄·선 원천·수송의 적합성은 확인되지 않는다.

셸별 분율 잔차와 관측 대역 잔차를 곱하거나 상관계수·가중 점수·결손 설명률로 만들지 않는다. CMFGEN의 셸별 관측 대역 기여가 없으므로 **셸×대역 결손 표를 만들어 낼 수 없다.**

## §7. 최소 변경집합·재사용·출력

### 변경집합 — DET-BLUE-ION-MAP (표)

| 파일 | 허용 변경 |
|---|---|
| `scripts/det_blue_ion_map_readout.py` | 신설: 명시적 이온 열 대응, 전 셸 결합, 절대 dex 조립, 행 분류, 음성 대조와 출력 |
| `/gpfs/kjhan/lumina/cmfgen_map/blue_ion_map_20261010/tools/run_det_blue_ion_map_offline.slurm` | 신설: a10 CPU에서 직접시험·판독을 실행하는 얇은 러너 |

### 변경집합 끝

기존 판독기·러너·생산 소스의 수정은 0이다. **새 C 도구도 만들지 않는다.**

det_nrb_judge_readout.py 약 794행의 `det_nrb_judge_public_block`과 약 141행의 `det_nrb_judge_log10i`를 재사용한다. 표시 정밀도와 구간 산술은 같은 판독기가 사용하는 `POP.printed`·`I`를 재사용한다.

기존 `det_nrb_judge_ions`는 Si·S 열 대응을 고정했고 `det_nrb_judge_ion_dump`도 기존 원소 집합에 의존한다. 이를 Fe·Co·Ni로 가장해 호출하거나 전역 상수를 바꾸지 않는다. 새 래퍼에 **고정 덤프 한 개의 필요한 열 읽기와 Z–공개 열 대응만** 둔다.

스펙트럼은 `det_siiii_repair_public`, `det_siiii_repair_spec`, `det_siiii_repair_bands`를 재사용하고, 반환된 절대 광도 구간의 비에 로그를 취한다. 색·R₈를 생성하는 전체 분석 경로는 호출하지 않는다.

출력은 다음 세 파일로 제한한다.

- `cmfgen_comparison.csv`: 309행의 기계 원본. 분자·분모·공개 토큰·대응 속도·구간·사유 포함.
- `cmfgen_comparison.md`: 같은 행의 표시본과 §6 순서의 짧은 해석.
- `evidence.json`: 입력·코드 신원, 직접시험, 게이트, 작업 ID, 비용, 이탈.

별도 결정 파일, 저장 CSV 재독해 검사기, 결정표 재계산 장치는 만들지 않는다. 행 검사는 저장 전 조립 자료에 적용한다.

## §8. 게이트·직접시험·종료감사

| 게이트 | 검사와 막는 실재 실패 |
|---|---|
| G-DOC | 기존 계약·참조·정본·발주·러너 검사를 적용해 다른 계약이나 미등록 변경으로 집행하는 실패를 막는다. |
| G-INPUT | §1–2의 고정 파일·시각·세대·스펙트럼 신원을 확인해 세대 14나 다른 런의 값을 O로 보고하는 실패를 막는다. |
| G-TABLE | §3–6의 열·속도 대응·309행 전수·중복·단위·정의를 확인해 잘못된 이온 결합, 불리한 행 누락, 색을 절대 dex로 보고하는 실패를 막는다. |
| G-COST | 실제 할당·실행 위치·누적 비용을 확인해 로그인 계산, GPU 사용, 등록 상한 밖 실행을 막는다. |

새 직접시험은 작은 정상 입력을 바탕으로 한 다음 **음성 대조**에 한정한다. 정상 입력과 주입 결함의 결과를 함께 기록한다.

1. 세대 13 이온 입력을 세대 14로 바꾸면 G-INPUT이 거부한다.
2. 셸 3을 이웃 셸의 속도로 결합하거나 Fe 공개 열을 Co 열로 지정하면 새 대응 검사가 거부한다.
3. NLTE·stage·원소 총량·II+III 합이 서로 다른 소형 입력에서 잘못된 분자 또는 분모를 사용한 변형이 등록 분율과 맞지 않아 실패함을 보인다.
4. O·공개 대역 광도 비와 OPT 정규화 색 비가 다른 소형 입력에서 색 비를 절대 dex로 넣은 변형이 실패함을 보인다.

승계 적분·구간 함수의 시험은 복제하지 않는다. 저장 산출 재검·쓰기 결함 모사·결정표 재계산 시험을 추가하지 않는다.

**종료감사의 1차 대상은 과학 표의 수치·셸/이온/대역 대응·주장 한정이다.** 필요한 수치 재현은 원입력에 대한 제한된 감사로 수행하며 상시 검사 장치를 추가하지 않는다. 수리에 필요한 범위는 실제로 영향을 받은 계산·행으로 제한한다.

## §9. 환경·비용·재현 명령

| 항목 | 등록 규칙·상한 |
|---|---|
| 계산 환경 | a10 CPU Slurm, 동시 1작업 |
| 작업당 할당 | 1 CPU·8 GiB·GPU 0 |
| 누적 할당 경과 | **600초 이하** |
| 누적 CPU 할당 비용 | **1/6 core·h 이하** |
| 포함 비용 | 문서 게이트·직접시험·판독·실패 실행·필요한 감사 재현 전부 |
| 생산 계산·replay | 모두 0회 |

로그인 노드에서는 파일 조회·편집만 한다. 계산은 Slurm 안에서 실행한다. lageunha로 우회하지 않는다. 작업명은 `LUMINA_` 접두이며 `--exclude=syn102,syn03`를 적용한다.

러너는 `module purge` 후 사용할 모듈을 명시적으로 로드하고 실제 버전을 남긴다. Python은 `-B`로 실행한다. 신규 출력 경로만 사용하고 기존 경로의 덮어쓰기를 거부한다. 일반 파일 입출력만 사용한다.

다음은 구현할 인터페이스다. 이번 기안에서 실행하지 않았다.

```bash
BLUE_BASE=/gpfs/kjhan/lumina/cmfgen_map/blue_ion_map_20261010
sbatch --partition=a10 --ntasks=1 --cpus-per-task=1 \
  --mem=8G --gres=gpu:0 --time=00:04:00 \
  --job-name=LUMINA_BLUE_ION_MAP --exclude=syn102,syn03 \
  "$BLUE_BASE/tools/run_det_blue_ion_map_offline.slurm" analyze a1
```

`analyze a1`은 새 코드의 직접시험을 먼저 수행하고 통과하면 고정 입력으로 표를 만든다. 남은 누적 상한은 문서 게이트와 필요한 감사 재현에 사용한다. 추가 작업의 시간 요청도 잔여 예산 이하여야 한다.

실제 할당 경과를 누계하고 대기시간은 별도 보고한다. 실패 실행도 비용에 포함하며 상한을 자동 연장하지 않는다. 원본 보존을 위한 복구 절차는 원본 무수정과 새 결과 경로 분리로 충분하다.

## §10. 완료 조건·처분·주장 상한

다음이 갖춰지면 분율 잔차의 부호나 크기와 관계없이 **PASS_LIMITED**로 보고할 수 있다.

- 등록된 309행이 빠짐없이 있고 정의 가능한 비교값이 계산됐다.
- 미정의 행은 원량·사유와 함께 남으며 숨겨지지 않았다.
- 입력·공간 대응·정의가 적격이고 비용 상한을 지켰다.
- §6의 해석 순서와 아래 주장 한정을 지켰다.

필수 입력 자체가 없어 등록 표를 만들지 못하거나 예산 안에 실행을 마치지 못하면 `INCOMPLETE`, 잘못된 신원·대응으로 만든 표는 `INADMISSIBLE`이다. 표 틀만 만든 것을 과학 질문의 완료로 표시하지 않는다.

새 정보는 **O의 철족 II·III 분율 어긋남 지도**다. 대역 광도는 기존 O 결손의 비교 좌표다. 개별 미정의 행 때문에 유효한 다른 행을 버리지는 않되, 해당 이온·셸의 질문은 답하지 못했다고 명시한다.

생산 승격·물리적 정답·인과 기여율·수렴·씨앗 독립성을 선취하지 않는다. O는 유한 반복 상태이며 이번 단에서 수렴을 평가하지 않는다. 분율 일치도 선 원천 일치나 청색 결손 해소를 뜻하지 않는다.

후속 보고는 분율 조사가 필요한 위치와 여전히 남은 출력 연결 공백을 구분한다. 새 replay 도구, 원천 변경, 추가 반복 또는 GPU 계산은 이 계약으로 자동 승인되지 않는다.

## §11. 집행 기록·분장

| 업무 | 담당·현재 상태 |
|---|---|
| 기안·개정 | astra 배정, 미파일화 기안 |
| 경량 사전감사 | GPU 0·생산 변경 0에 대한 개정25-ab에 따라 생략 |
| 구현·직접시험·게이트·실행·판독·판정 | 운전자, 미집행 |
| 종료감사 | astra, 과학 표 중심, 미집행 |

기안자의 Q-GOAL·Q-LEAN 자기답을 독립 감사 통과로 표시하지 않는다. 운전자의 구현·판정 겸임과 계약 저자의 종료감사 이해관계를 최종 기록에 남긴다.

기안자는 공유 맥락, 프로젝트 위치·규약, 직전 계약·판정, 최종 진단·폐합·감사 자료, 지정 입력의 헤더·manifest, 재사용 함수의 구현을 읽었다. 파일 변경·계산·Slurm 제출·게이트 실행은 하지 않았다.

운전자는 실제 모델 ID·코드 신원·작업 ID·시험·게이트·비용·이탈·판정·종료감사 결과를 이 절에 이어 기록한다.

## §12. 기계 선언

직전 계약의 changeset·branches·references 형식을 따른다. 분기 코드는 이동·개선 코드가 아니라 §6의 **공개값 대비 부호 분류**다. 참조에는 실재 파일만 넣는다.

```prereg-preflight
{
  "changeset": {
    "table_heading": "### 변경집합 — DET-BLUE-ION-MAP (표)",
    "table_end": "### 변경집합 끝",
    "path_pattern": "(?:scripts/det_blue_ion_map_readout\\.py|/gpfs/kjhan/lumina/cmfgen_map/blue_ion_map_20261010/tools/run_det_blue_ion_map_offline\\.slurm)",
    "symbol": "det_blue_ion_map_",
    "roots": [
      "scripts/det_blue_ion_map_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/blue_ion_map_20261010/tools/run_det_blue_ion_map_offline.slurm"
    ],
    "expected_extra": [
      "scripts/det_blue_ion_map_readout.py",
      "/gpfs/kjhan/lumina/cmfgen_map/blue_ion_map_20261010/tools/run_det_blue_ion_map_offline.slurm"
    ]
  },
  "branches": {
    "regimes": [[0,0],[1,1],[2,2],[4,4],[5,5],[9,9]],
    "metrics": {
      "valid": "len(v) == 1 and all(x in (0,1,4,5) for x in v)",
      "c0": "sum(1 for x in v if x == 0)",
      "c1": "sum(1 for x in v if x == 1)",
      "c4": "sum(1 for x in v if x == 4)",
      "c5": "sum(1 for x in v if x == 5)"
    },
    "rules": [
      {"name":"X","predicate":"not valid"},
      {"name":"BELOW_PUBLIC","predicate":"valid and c0 == 1"},
      {"name":"ABOVE_PUBLIC","predicate":"valid and c1 == 1"},
      {"name":"UNRESOLVED","predicate":"valid and c4 == 1"},
      {"name":"NOT_IDENTIFIABLE","predicate":"valid and c5 == 1"}
    ],
    "adversarial_fixtures": [
      {"name":"below","mix":[[0,1]]},
      {"name":"above","mix":[[1,1]]},
      {"name":"unresolved","mix":[[4,1]]},
      {"name":"undefined","mix":[[5,1]]},
      {"name":"invalid","mix":[[2,1]]},
      {"name":"missing","mix":[]},
      {"name":"multiple","mix":[[0,1],[1,1]]}
    ]
  },
  "references": [
    {"path":"docs/PREREG_DET_COLL_PRESC_2026-10-09.md"},
    {"path":"docs/VERDICT_DET_COLL_PRESC_2026-10-09.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag7_bluedef_20261010/evidence/USER_DECISION_2026-10-10.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/evidence/USER_DECISION_2026-10-09.txt"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag7_bluedef_20261010/LOOP_CLOSED.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag7_bluedef_20261010/DIAGNOSIS_r5.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag7_bluedef_20261010/sol_numaudit_r5C.md"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/diag7_bluedef_20261010/DRIVER_DESIGN_AUDIT_r5.md"},
    {"path":"scripts/det_nrb_judge_readout.py","flags_existing":["det_nrb_judge_public_block","det_nrb_judge_log10i"]},
    {"path":"scripts/det_siiii_repair_readout.py","flags_existing":["det_siiii_repair_public","det_siiii_repair_spec","det_siiii_repair_bands"]},
    {"path":"scripts/det_siiii_pop_readout.py","flags_existing":["class I","def printed"]},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/work/physics_comparison/physics_DET_iter0010.manifest.json"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/work/physics_comparison/physics_DET_iter0010.ion.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/work/physics_comparison/physics_DET_iter0010.shell.csv"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/work/det_obs_capture/manifest.json"},
    {"path":"/gpfs/kjhan/lumina/cmfgen_map/coll_presc_20261009/OMEGA/run1/work/lumina_spectrum_det_obs.csv"},
    {"path":"data/standart_data1/toy06/ionfrac_fe_toy06_cmfgen.txt","flags_existing":["fe1","fe2","19.480"]},
    {"path":"data/standart_data1/toy06/ionfrac_co_toy06_cmfgen.txt","flags_existing":["co1","co2","19.480"]},
    {"path":"data/standart_data1/toy06/ionfrac_ni_toy06_cmfgen.txt","flags_existing":["ni1","ni2","19.480"]},
    {"path":"data/standart_data1/toy06/spectra_toy06_cmfgen.txt"}
  ]
}
```
