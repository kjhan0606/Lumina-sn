# 미완 J37과 같은 물질 CMFGEN J71의 공간별 광자장 대조 — 2026-10-22

## 결론과 물리적 범위

물질70에서 저장된 Lumina J37은 안쪽 셀에서 CMFGEN J71과 가깝지만, 바깥 셀에서 크게 낮다.
셀88의 Lumina/CMFGEN 광자장 비는 100–400Å에서0.1561, 400–800Å에서0.1871,
800–1600Å에서0.04961이다. 89셀 합계의 앞 두 대역 비가 각각1.0034/0.9938인 것은
안쪽의 큰 값이 외곽 결손을 가리기 때문이다. 이 공간 차이는 같은 물질의 복사장
불일치 위치를 구체화하며, 이온분포나 스펙트럼 동질성 판정은 아니다.

J37은 Slurm1200705에서37반복까지 저장한 **미완·미검증** 장이다. 최종 상대변화는
0.004228852517, 최대 절대변화는8.919588468e-7이고 `FAILED_PRESERVED`/자식 시간초과로
끝났다. K(1)과 성분별 오차상한은 미계산이다. J36→J37의 100–1600Å 대역에서
상쇄 전 광자가중 변화도 보고된 정밀도에서0이지만, 한 반복의 정체를 고정점 오차상한이나
향후 반복의 수렴 증명으로 바꾸지 않는다. 전체 최대 절대변화는 셀9/6750.465Å에서 났다.

이 실행은 원래 물질90점의89셀 영역 바깥을 vacuum으로 끝낸다. 반면 실제 CMFGEN
`MODEL:136`/`VADAT:72`의 `THK_CONT=T`와 `OUTGEN:123–124`는 연장 대기를 사용하고
RMAX/R(1)=9.08235e15/6.05490e15=1.5다. J37을 만든 봉인 소스에는 이 외부 연장
수송이 없고, 현행 후보 코드는 `OUTER_BOUNDARY_NOT_IMPLEMENTED`로 그 경우를 거부한다.
따라서 경계·도메인 불일치는 외곽 결손을 설명할 유력 원인 후보다. 이 대조만으로
그 누락의 기여율을 분리하거나 계수·주파수/공간 이산화·미수렴의 기여를0으로 판정하지 않는다.

## 실제 대조

CMFGEN의 인접 원래점 J를 산술 셀 중심에서 반씩 합하고, Lumina의 전체1,115,020점 J를
log(ν)에 대해 CMFGEN의197,291주파수로 보간했다. 외삽·마스크·바닥값·재정규화는 없다.
각 대역의 `4π∫Jν/(hν)dν`를 같은 참조 주파수에서 사다리꼴 적분했다. 이는 광자장
적분이며 실제 Si/S PHOT 단면을 곱한 광이온화율이 아니다.

| 셀(안→밖) | 반경 cm | 100–400Å | 400–800Å | 800–1600Å | 1600–3000Å |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.72918e14 | 0.999990 | 0.999714 | 0.999973 | 0.999822 |
| 45 | 2.60206e15 | 0.982684 | 0.970781 | 0.263079 | 0.658261 |
| 88 | 6.05489e15 | 0.156117 | 0.187109 | 0.049609 | 0.217328 |

셀0/45/88의 앞 세 대역9개 비율을 별도의 `searchsorted` 보간과 명시적 사다리꼴 합으로
독립 재현했다. 첫 재현 시도는 정확한 최상단 격자 끝점을 외삽으로 잘못 거부해
결과를 쓰기 전에 종료했다. 끝점 처리만 수정한 고정 파일의9개 비율은 원 계산과
최대 상대차3.38936e-16으로 일치했다. 원 결과/첫 실패 코드/수정 코드를 모두 보존한다.

## 입력 신원·재현

- 증거 root: `/gpfs/kjhan/cmfgen_runs/lumina_j37_band_20261022-Ow0ZBt`
- `result/diagnostic.json` SHA256 `ab687f0047f233f1700c9b638f2d28dea447023cab08d0031f6f9b3216dd6fef`
- `step-detail/step.json` SHA256 `cc69dc8412015fa546244bb8556d0ec9fb81fbb20e59c8632036d3ed36e1c2c9`
- `independent/reproduced.json` SHA256 `c869ab76e43f5b091a20e21c678701c48c6010cbdc4181ff3247a7864666b5e3`
- J37/J36 SHA256: `2dfbd5152277a55b614284972b333bbab5238cd63049a1d17262e3cb0fcd434d` / `b5e2d35ec89a5e9c1034e37472b70bbb44902882ebe17d5d6e869ae985040cd2`
- CMFGEN J71 binary SHA256 `d9073ac830d8a70beffc731dc919140b2b97ccff7e8d960060d42a7f7ca60563`; 물질70 셀 binary와 계획·격자·checkpoint 해시는 JSON에 있다.

증거 root의 `scripts/`에는 실행 당시 비교기·checkpoint reader·의존성 사본과 세 진단
소스를 보존했다. `lageunha`에서 CPU1/OMP1·`taskset -c 26`으로
각 스크립트를 실행했고, 주 진단5.42초/RSS약1.89GiB, 직전 반복 대조4.83초/RSS약1.57GiB,
독립 재현은 exit0이었다. 기존 수송이나 CMFGEN 계산을 다시 실행하지 않았고 GPU 추가 사용0이다.

재현 시 기존 `result`를 덮어쓰지 않고 새 출력 경로를 사용한다. `lageunha`에서
증거 root를 현재 디렉터리로 잡고 CPU1/최대600초로 실행한다. 실제 peak RSS는4GiB 아래였으며
아래 명령에 별도 메모리 강제한도는 설정하지 않았다.

```bash
timeout -k 5 600 taskset -c 26 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 -B scripts/lumina_checkpoint_j_band_diagnostic.py \
  --config /gpfs/kjhan/cmfgen_runs/lumina_field0_checkpoint_20261020-EoA5Bi/input/config.json \
  --execution /gpfs/kjhan/cmfgen_runs/lumina_field0_checkpoint_20261020-EoA5Bi/run/execution.json \
  --out /gpfs/kjhan/cmfgen_runs/lumina_j37_band_20261022-Ow0ZBt/reproduce-new
```

다음 물리 작업은 이미 준비한 외부대기 결합 후보를 실제 전체장에 연결해 **같은 물질70**의
J를 다시 얻고, 외곽 대역과 실제 PHOT 가중 Si/S율을 같은 좌표에서 비교하는 것이다.
그 전에는 이 미완 J37을 최종 자체장 또는 SE/RE 입력으로 승격하지 않는다. 현재 Ada
단일구간 수송은 별도 봉인 실행으로 유지한다. 최종 Si II/III·S 이온분포와 스펙트럼·물리량
동질성 목표는 active/NOT_MET다.
