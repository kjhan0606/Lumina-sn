# 외부 대기 J4와 진공 J37의 같은 CMFGEN 광자 대역 대조 — 2026-10-22

## 물리 판독

물질70·CMFGEN J71을 고정하고, 미완 진공 J37과 외부 대기 포함 J4를 같은 89개
산술 반경 중심·전 197,291개 CMFGEN 주파수에서 대조했다. 두 Lumina 장은 전
1,115,020개 주파수에서 log(ν)-선형으로 보간했으며 외삽·마스크·바닥값·재정규화는 없다.
각 대역 값은 `4π∫Jν/(hν)dν`의 참조 표본 사다리꼴 적분이다. PHOT 단면을 곱한
광이온화율이나 수렴·이온분포·스펙트럼 판정이 아니다.

| 셀 | 100–400Å J37→J4 | 400–800Å J37→J4 | 800–1600Å J37→J4 | 1600–3000Å J37→J4 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.999990→0.999990 | 0.999714→0.999652 | 0.999973→0.999973 | 0.999822→0.996912 |
| 45 | 0.982684→0.982684 | 0.970781→0.970781 | 0.263079→1.007024 | 0.658261→0.959932 |
| 75 | 0.211880→0.899011 | 0.348962→0.943464 | 0.066898→0.994770 | 0.248233→0.973351 |
| 88 | 0.156117→0.945813 | 0.187109→0.987675 | 0.049609→0.988548 | 0.217328→0.984185 |

표는 각각 Lumina/같은 CMFGEN 비율이다. **외곽의 큰 광자장 부족은 새 J4에서 크게
줄었다.** 그러나 전 셀 최소 비는 100–400Å에서 **셀37/0.647166**, 400–800Å에서
**셀37/0.763069**, 800–1600Å에서 **셀37/0.773575**, 1600–3000Å에서
**셀38/0.767779**다. 기존 생산 전체 주파수 격자 PHOT 광흡수 비교에서도 네 이온
Si II/III·S II/III의 최소가 모두 셀37이었다(각각0.873989/0.865335/0.710992/0.737935).
따라서 다음 물리 원인분리는 셀37~38의 약 UV/극자외선 형상·수송 공간 표현과
미수렴을 우선 구분해야 한다. 대역 광자 적분과 PHOT율은 가중치가 달라 수치 비율을
동일시하지 않는다.

J37과 J4는 경계뿐 아니라 수송 구현·반복 횟수도 다르므로 위 개선 전체를 외부 경계의
단독 효과로 귀속하지 않는다. J4는 4회만 계산했고 J3의 잔차는 약0.14126,
**J4 자신의 잔차·성분별 전방오차는 미측정**이다. J4·J37 모두 최종 자체장이 아니다.
현재 실행 중인 Ada 단일구간 수송도 별개이며 새 전체장·물질 갱신·SE/RE는 이번에
수행하지 않았다. 최종 Si/S 분포와 스펙트럼·물리량의 CMFGEN 동질성은 `NOT_MET`다.

## 증거·재현

- 새 증거 root: `/gpfs/kjhan/cmfgen_runs/lumina_outer_j4_band_20261022-XsIv0E`.
- 결과 `result/diagnostic.json` SHA256 `78ee4140cdf91fc9872810dc82de5d097ffd2db8bbda1a4175b53bd68bc375e1`.
  89셀 전체의 대역 최솟값·중앙값·최댓값과 대표 7셀을 저장했다.
- 실행 소스 `scripts/compare_lumina_outer_j4_bands.py` SHA256
  `8b575e45e11c9b4496ea8b972d8669eca233bc9b923f6fd8e06dc4a9d6c91363`.
  같은 root의 `scripts/`에 J37 진단 당시의 비교 함수 사본을 함께 보존했다.
- J4·CMFGEN J71 SHA256은 각각 `cfbdb7e592553b4fa37ba480237878a5abaf2da6206cb776cbd392c1fe538b30`·
  `d9073ac830d8a70beffc731dc919140b2b97ccff7e8d960060d42a7f7ca60563`.
  J37 선행 진단 SHA256은 `ab687f0047f233f1700c9b638f2d28dea447023cab08d0031f6f9b3216dd6fef`.
- 선행 J37 진단의 대표 7셀·4대역 **CMFGEN 참조 광자 적분과 새 계산이 모두 정확히 같았다**
  (`max_J37_J4_reference_relative_difference=0`). 이는 동일 기준·매핑의 교차 점검이지
  독립 수송 재실행은 아니다.
- Lageunha 사전 부하2.21/1.76/1.72·가용 메모리971615MiB를 확인하고 CPU1·GPU0,
  최대600초로 읽기 전용 계산했다. 로그 SHA256
  `7b3c65df8e2fe37a9e175c393b7a4dbdd8983942fdfcfdc4b4bdfd7d8b58c74f`.

기존 결과를 덮어쓰지 않도록 `--out`에는 존재하지 않는 새 경로를 준다. 고정 소스가
`<root>/scripts/`에 있으므로 다음은 Lageunha에서 실행한다.

```bash
timeout -k 5 600 taskset -c 26 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 -B /gpfs/kjhan/cmfgen_runs/lumina_outer_j4_band_20261022-XsIv0E/scripts/compare_lumina_outer_j4_bands.py \
  --config /gpfs/kjhan/cmfgen_runs/lumina_field0_checkpoint_20261020-EoA5Bi/input/config.json \
  --diagnostic /gpfs/kjhan/cmfgen_runs/lumina_outer_campaign_20261021-FAmNPQ/comparison-current/diagnostic.json \
  --old /gpfs/kjhan/cmfgen_runs/lumina_j37_band_20261022-Ow0ZBt/result/diagnostic.json \
  --out <존재하지_않는_새_결과_경로>
```
