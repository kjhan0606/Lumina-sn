# 공개 CMFGEN toy06 19.48일과 완료 물질응답 비교

## 공개 비국소 침적과 현재 국소 붕괴 시나리오

공개 `edep_toy06_cmfgen.txt`의 19.48일 열은 γ선과 양전자에 의한 **총
에너지 침적**이다. 공개 논문은 이를 셀 중심 속도와 체적당 단위
`erg cm^-3 s^-1`로 정의하며, CMFGEN이 γ 수송과 비열 전자 처리를 사용한다고
설명한다. [StaNdaRT 자료 정의](https://arxiv.org/html/2209.11671)

완료 F2 고정-T 물질응답의 `evaluation-4/balance.npz`에 저장된
`instantaneous_scenario`는 같은 붕괴 에너지가 **생성된 곳에서 전부 국소적으로
침적된다고 가정한 열원**이다. 저장된 `stored_decay`는 기존 0시간 분기 때문에
전점 0이다. 두 항을 공개 침적과 같은 물리량이나 같은 수송 조건으로 보지
않는다. 공개 111개 속도점의 19.48일 열을 완료 응답의 반경을 속도로 바꾼
90개 점에 선형 보간했고, 공개 속도 범위 밖 한 점은 외삽하지 않았다.

| 원점 | 속도 km/s | 공개 총 침적 | 현재 국소 열원 가정 | 공개/국소 |
| ---: | ---: | ---: | ---: | ---: |
| 30 | 23909.69 | 2.46513e-8 | 8.12037e-16 | 3.04e7 |
| 40 | 18021.47 | 3.69927e-7 | 6.20850e-15 | 5.96e7 |
| 45 | 15178.76 | 1.61281e-6 | 1.65787e-14 | 9.73e7 |
| 50 | 12376.06 | 8.00060e-6 | 7.91744e-10 | 1.01e4 |
| 55 | 9610.02 | 6.80230e-5 | 7.78249e-5 | 0.874 |
| 60 | 7018.76 | 4.46357e-4 | 5.67862e-4 | 0.786 |
| 65 | 5066.17 | 1.10443e-3 | 1.13831e-3 | 0.970 |

수치는 모두 `erg cm^-3 s^-1`이고 원점 번호는 1부터 센다. 외곽에서는
국소 붕괴원이 작더라도 다른 곳에서 온 γ선이 침적될 수 있다. 이 결과는
**공개 toy06의 침적 프로필이 현재의 국소 가정과 큰 차이를 보이는 실제
자료**이며, 특히 S II/III 공개 불일치가 가장 큰 원점50도 여기에 포함된다.
그러나 두 시나리오는 γ 수송, 물질 상태와 열·비열 분배가 다르다. 공개 총
침적을 곧바로 전자 열원이나 S II 이온화율로 넣을 수 없고, 그 차이가 S
분율 불일치의 원인 또는 해결책이라는 증거도 아니다. 비열 이온화는 방향상
S III를 늘릴 수 있으므로 S II를 증가시키는 설명으로 단정하지 않는다.

원점50 완료 고정-T 장부에서 BF 가열/냉각은 각각 `3.15986e-6`/
`2.88238e-6`이며, 공개 **총** 침적 `8.00060e-6`은 이 규모와 비교할 만하다.
다만 공개 총 침적의 열 분배율은 미확인이고, 이 장부의 전체 순열잔차는
BB·FF·단열과 다른 항을 모두 포함한다. 공개 침적을 기존 계산에 주입하거나
재계산하지 않았고, 진행 중인 T/ne 작업의 최종 결과로도 취급하지 않는다.
후속 공개-물리 비교에서는 비국소 침적과 열/여기/이온화 분배를 함께 맞춰야
하며, matched-physics 정상 기준에서는 양쪽에 같은 열원 계약을 명시한다.

입력 SHA256: 공개 침적 파일
`cb107dae9b451c3e6d1faacec29dad4c03b5c2ae38b48c3601c740fb56ecfc4e`, 완료
`physical-profiles.npz` `766329d782f34a2c22bc6acaf4feb82b36c93d63ac74b5855a0272a0d858b29f`,
완료 `evaluation-4/balance.npz` `9d42fd13bef1291cf21f6387d68de0aa93876d34c9cf6920cbe061a37975d232`.
완료 응답 root 이름은 `lumina_second_field_gpu_20261030-DBmS99/response/run`이다.

재현은 `RESULT_RUN`에 그 완료 응답 root를 지정한 뒤 프로젝트 루트에서 아래
읽기 전용 명령을 실행한다. 배열과 epoch를 확인하며 재수송·새 물질 계산은 없다.

```bash
python3 - "$RESULT_RUN" <<'PY'
import sys
from pathlib import Path
import numpy as np
p = Path('data/standart_data1/toy06/edep_toy06_cmfgen.txt')
root = Path(sys.argv[1])
times = np.array(p.read_text().splitlines()[2].split(':')[1].split(), dtype=float)
table = np.loadtxt(p)
idx = np.flatnonzero(np.isclose(times, 19.48, rtol=0, atol=1e-9))
assert table.shape == (111, 57) and len(idx) == 1
assert np.isfinite(table).all() and np.all(np.diff(table[:, 0]) > 0)
with np.load(root / 'physical-profiles.npz') as profiles, \
     np.load(root / 'evaluation-4/balance.npz') as balance:
    velocity = profiles['radius_cm'] / (19.48 * 86400 * 1e5)
    public = np.interp(velocity, table[:, 0], table[:, idx[0] + 1],
                       left=np.nan, right=np.nan)
    local = balance['instantaneous_scenario']
    for node in (30, 40, 45, 50, 55, 60, 65):
        i = node - 1
        print(node, velocity[i], public[i], local[i], public[i] / local[i])
PY
```

### 침적 차이에 대한 기존 결합 행렬의 조건부 민감도

위 침적 비율이 곧 온도 변화 비율을 뜻하는 것은 아니다. 이미 완료된
고정 J의 기준 generation11과 그 T/ne 중심차분으로 저장된 180×180 결합
Jacobian을 재사용했다. 같은 기준점에서 **공개 총 침적이 전부 전자열로
바뀐다고 가정**하여, 현재 국소 순간열과의 차이 `ΔQ`를 열잔차에만 더했을
때 두 Newton 제안의 선형 차이 `Δx = −Jacobian⁻¹(0, ΔQ/열척도)`를
계산했다. 전하 행과 깊이 결합 단열항은 기존 행렬 그대로다.

이것은 실제 공개 모델의 열원 분배나 시간항이 아니며, 이미 수렴한 두
물리해의 차이도 아니다. 기준점 자체에 잔차가 있으므로 `Δx`는 **같은
기준점에서의 Newton 제안 차이**다. 공개 분포의 셀 중심을 원점으로 선형
보간했고, 공개 속도 범위 밖인 원점90은 `ΔQ=0`으로 둔 조건부 결과다.

| 원점 · 속도 km/s | 가상 열원 변화 erg cm^-3 s^-1 | `Δlog T` | `Δlog ne` |
| --- | ---: | ---: | ---: |
| 42 · 16875 | +6.58513e-7 | +0.003528 | +1.408e-5 |
| 45 · 15179 | +1.61281e-6 | +0.002824 | +4.625e-6 |
| 50 · 12376 | +7.99981e-6 | +0.001744 | +2.149e-6 |
| 55 · 9610 | −9.80185e-6 | −0.000420 | −6.372e-7 |
| 60 · 7019 | −1.21505e-4 | −0.003163 | −1.801e-4 |

원점50에서는 가상 열원 변화에 대한 선형 Newton 제안 차이가 약 0.17%의
로그온도다. 그렇다고 실제 열원 교체 시 온도가 그만큼만 바뀐다거나,
침적·비열이 Si/S 불일치에 중요하지 않다는 결론은 아니다. 특히 전체 90점
중 바깥 29점에서 `|Δlog T|>0.1`, 원점19에서는 `+0.883`이다. 이런 크기는
국소 선형 근사의 유효성을 넘어설 수 있다. 행 척도가 적용된 저장 행렬의
2-노름 조건수는 약 `1.28e9`이며 선형계 상대 잔차는 `1.77e-13`이다.
이는 선형계를 풀었다는 확인이지 비선형 물리 응답의 정확도나 오차 상한은
아니다. 공개 총 침적의 열·여기·이온화 분배와 새 물질/복사장 계산 없이
이 값을 실제 최종 결과로 사용하지 않는다.

같은 기준의 저장된 T±/ne± 응답에서 각 이온 분율의 중심차분을 구해,
위 `Δlog T/ne`에 곱한 **분율의 1차 제안 차이**는 다음과 같다.

| 원점 | Si II | Si III | S II | S III |
| ---: | ---: | ---: | ---: | ---: |
| 50 | +3.89e-9 | −4.56e-6 | −3.91e-6 | +2.98e-6 |
| 54 | +9.93e-9 | −6.01e-8 | −6.20e-6 | +6.19e-6 |
| 60 | −2.59e-10 | +2.91e-4 | +2.72e-8 | +1.68e-4 |

원점50의 공개/기준 S II 분율은 `0.845199/0.00271238`이므로 위
선형 제안은 그 차이보다 훨씬 작고 방향도 멀어지는 쪽이다. 그러나 이는
**현재 고정 J, 정상 물질, 국소 선형화, 공개 총 침적의 전량 열화 가정**에
한정된다. 공개 모델의 실제 열원 분배, 비열·시간항, 인구와 복사장 피드백이
없으므로 침적 처리의 실제 영향 전체나 최종 오차를 평가한 것이 아니다.
원점1–29는 위 `|Δlog T|>0.1` 때문에 이 선형 분율 예측의 해석 대상에서
제외한다. 새 물질을 풀거나 온도를 갱신하지 않았다.

선형화 입력의 SHA256은
`7f5949f94899b2485253f4a9ffbee4c5da9ea9a7f52ebf4393eddb1bcdf89bdb`,
기준 evaluation-1 balance는
`ab755cc6d7e5c08e1c42d41b6354b6d425eb4e624c8f3d6ca1806113a9610f29`다.
추가로 사용한 evaluation-4/5 `response.npz`의 SHA256은 각각
`29fd692f0af4acabb92b9d5074c445b46666264b955aa91ca6e9d90cd03873fe`,
`2bc3c3477d025d76bd82c263f8dc78f63135e5aa93e8ef84e21bef277ab90340`이고,
evaluation-1/2/3 응답의 SHA는 아래 generation18 열잔차 절에 있다.
공개 침적과 완료 반경 배열 SHA는 위 표의 신원과 같다. `TRIAL_RUN`은
진행 중 물질응답의 `lumina_second_field_thermal_replacement_20261030-fz3u4uV0/run`,
`RESULT_RUN`은 위 완료 응답 root다. 재현은 아래 읽기 전용 선형대수이며,
기존 작업이나 물질·열원을 변경하지 않는다.

```bash
OPENBLAS_NUM_THREADS=1 python3 - "$TRIAL_RUN" "$RESULT_RUN" <<'PY'
import sys
from pathlib import Path
import numpy as np
trial, result = map(Path, sys.argv[1:])
public = Path('data/standart_data1/toy06/edep_toy06_cmfgen.txt')
times = np.array(public.read_text().splitlines()[2].split(':')[1].split(), float)
table = np.loadtxt(public)
idx = np.flatnonzero(np.isclose(times, 19.48, rtol=0, atol=1e-9))
assert table.shape == (111, 57) and len(idx) == 1
with np.load(trial/'thermal-direction-1/linearization.npz') as jac, \
     np.load(trial/'evaluation-1/balance.npz') as bal, \
     np.load(trial/'evaluation-1/response.npz') as state, \
     np.load(result/'physical-profiles.npz') as profiles:
    assert np.array_equal(jac['temperature'], state['temperature'])
    assert np.array_equal(jac['ne'], state['ne'])
    key = 'native_instantaneous_scenario'
    scale = .5*(bal[key+'_gross'] + abs(bal[key+'_net']))
    assert np.array_equal(scale, jac['thermal_scale'])
    velocity = profiles['radius_cm'] / (19.48*86400*1e5)
    dep = np.interp(velocity, table[:, 0], table[:, idx[0]+1], left=np.nan, right=np.nan)
    delta = np.where(np.isfinite(dep), dep-bal['instantaneous_scenario'], 0.)
    forcing = np.r_[np.zeros(90), delta/scale]
    matrix = jac['jacobian']
    dx = np.linalg.solve(matrix, -forcing)
    print('uncovered:', np.flatnonzero(~np.isfinite(dep))+1,
          'large_T:', np.flatnonzero(abs(dx[:90]) > .1)+1)
    print('condition:', np.linalg.cond(matrix),
          'linear_relative_residual:', np.linalg.norm(matrix@dx+forcing, np.inf)/np.linalg.norm(forcing, np.inf))
    for n in (19, 42, 45, 50, 55, 60):
        i = n-1
        print(n, velocity[i], delta[i], dx[i], dx[90+i])
    others = [np.load(trial/f'evaluation-{k}/response.npz') for k in (2, 3, 4, 5)]
    try:
        all_responses = [state] + others
        cols = state['columns']
        assert all(np.array_equal(r['columns'], cols) for r in others)
        for k, sign in ((1, 1), (2, -1)):
            assert np.array_equal(all_responses[k]['ne'], state['ne'])
            assert np.allclose(all_responses[k]['temperature']/state['temperature'],
                               np.exp(sign*1e-4), rtol=1e-14, atol=0)
        for k, sign in ((3, 1), (4, -1)):
            assert np.array_equal(all_responses[k]['temperature'], state['temperature'])
            assert np.allclose(all_responses[k]['ne']/state['ne'],
                               np.exp(sign*1e-4), rtol=1e-14, atol=0)
        def fraction(r, z, charge, i):
            populations = r['population'][:, i]
            return populations[(cols[:, 0] == z) & (cols[:, 1] == charge)].sum() / populations[cols[:, 0] == z].sum()
        for n in (50, 54, 60):
            i = n-1
            for z in (14, 16):
                for charge in (1, 2):
                    f = [fraction(r, z, charge, i) for r in all_responses]
                    change = ((f[1]-f[2])*dx[i] + (f[3]-f[4])*dx[90+i])/(2e-4)
                    print('fraction', n, z, charge, f[0], change)
    finally:
        for r in others:
            r.close()
PY
```

## 해석 정정

공개 CMFGEN 이온분포·물리량·스펙트럼은 이미 저장소에 있으며 과거에도
사용했다. 자체 CMFGEN 물질70 또는 감쇠 진단이 수렴 기준해가 아니라는
사실을 공개 CMFGEN 결과가 없다는 뜻으로 일반화하지 않는다. 공개 자료의
반복별 상세 수렴 증거와 내부 J·준위인구는 이 비교에서 확인하지 않았다.

추가 공개 자료 확인: Dessart 등의 [DDC10 모델 배포](https://www.iap.fr/useriap/dessart/snia/snia_tech.html)는
toy06이 아닌 별도 시간의존 모델이다. 배포 압축파일의 목록과 README를
직접 확인한 결과 `0.976`일의 유체 초기구조와 `0.976–104.1`일의 관측자계
스펙트럼 50개가 있으며, 스펙트럼 범위는 `1500–25000 Å`다. 배포 목록에는
내부 `Jν(r)`, 준위인구나 CMFGEN 재시작 파일이 없다. 이 자료는 다른 모델의
스펙트럼 교차검사 후보일 뿐 toy06 기준장이나 자체 정상해의 대체물이 아니다.
파일 목록만 읽었고 원본을 풀거나 새 계산에 사용하지 않았다. 이 확인은 다른
별도 비공개·공개 배포의 존재까지 부정하지 않는다.

**후속 정정:** 아래 단일 S 응답의 초기 재현 실패는 동일한 물리 설정의
불일치로 보면 안 된다. 별도 실행 기록에 따르면 원 런처의
`LUMINA_NLTE_BF_COLLISION=1`이 초기 소형 재현에 누락됐고, 이 설정을 복원한
job `1245108`은 기준 응답을 재현한 뒤 공개 ne 사례도 계산했다. 해당 기록의
원점50 고정 J/T 결과는 현재 S II/S III `0.00271238/0.99624180`, 공개 ne로
변경한 결과 `0.00234511/0.99643824`다. 이는 전하·열·수송을 함께 푼 공개
CMFGEN 재현이 아니라 조건부 한 점 비교다. 이 인구 수치는 별도 진행 흐름이
작업 중인 인수인계서의 18:31 기록이며 이 문서 작성자가 새로 실행 재현한
수치는 아니다. Slurm `1245108`의 `COMPLETED`, 38초, 환경 변수는 회계
기록과 대조했다. 선택 response SHA256은
`e3a5b1f73ef7f2cc5785d92bc1937cbfbe0cef92330816c288a61bf315826b65`다.

## 실제 비교

두 번째 새 복사장에 대한 완료 고정-T 물질응답의 마지막 전하 수용
generation14를 공개 toy06 19.480일과 비교했다. 진행 중인 T/ne 작업
1243823의 초기 generation11은 같은 response SHA를 가지지만, 그것을
해당 작업의 최종 선택 후보로 부르지 않는다. 아래는 물질응답의 실제
`new_J_response_fraction`을 이용하며 `CMFGEN70_fraction`이나 직전 수용
물질 열을 공개 CMFGEN으로 착각하지 않는다.

### 공개 전자밀도만 바꾼 Si의 조건부 응답

#### 공개 온도와 전자밀도를 함께 적용한 추가 확인

Si II의 공개값과 기존 응답 차이가 가장 큰 원점34, 속도 `21533.88 km/s`에서
완료 F2 복사장과 원자모형을 유지하고 공개 온도와 전자밀도를 함께 적용했다.
온도가 바뀌므로 고정온도 적분을 재사용하지 않고 Si의 661개 bound-free
경로를 기존 전주파수 격자에서 직접 다시 적분했다. 충돌과 occupation도 새
온도·전자밀도에서 계산했다. 선 복사장 입력은 기존 값을 고정했다.

| Si 이온 | 공개 CMFGEN | 기존 T/ne | 공개 ne만 적용 | 공개 T/ne 적용 |
| --- | ---: | ---: | ---: | ---: |
| II | `0.15801146` | `4.28652498e-8` | `2.95620e-8` | `2.91299731e-8` |
| III | `0.7915622` | `0.705294338` | `0.6438464` | `0.650102346` |

기존 T/ne는 `9845.829669 K`/`1.86661132e7 cm^-3`, 공개 값은
`9243.838591 K`/`1.41029144e7 cm^-3`다. 같은 고정 복사장에서 공개
T/ne를 함께 적용해도 Si II의 큰 차이는 남는다. 다만 두 값을 동시에
변경했으므로 온도 단독의 영향은 이 시험으로 분리하지 않는다. 공개 모델은
시간의존성·비열 과정과 원자모형도 다르며, 한 원소의 고정장 응답은 공통
전하·열·수송 수렴이 아니다. 이 결과만으로 차이를 복사장 하나의 문제로
단정하지 않는다.

기존 조건에서는 저장된 Si 전체 개체수가 `rtol=2e-8`, `atol=1e-30`으로
재현됐고 Si II/III의 광이온화·유도/자발 재결합 흐름도 저장값과 일치했다.
새 T/ne의 정상상태 잔차는 `7.90e-17`이지만 이는 해당 단일 원소 방정식의
잔차이지 공동수렴의 증거가 아니다. grammar debug 작업 `1245601`은 CPU1,
8GiB, 최대15분으로 실행해 1분21초에 `COMPLETED`, exit `0:0`이었다.

실행 wrapper [compare_lumina_si_public_tne.py](../scripts/compare_lumina_si_public_tne.py)의
SHA256은 `bd8b3a1f58807e0fc3ee4aa6139efee552dc2cfcd2b3514b116ecb84c5f65c16`이다.
이 wrapper는 바로 아래에 기록한 **미추적 원본**의 SHA를 확인한 뒤 메모리에서
변환한다. 따라서 원본이 없는 다른 환경에서는 이 커밋만으로 재현되지 않는다.
원본의 결과 JSON에 있던 `scope`와 `state`의 `FIXED_J_T` 표기는 그대로 남아
있지만, 이번 `public_T_ne` 사례는 실제로 T와 ne를 모두 바꿨으므로 그 문구는
이 사례에 적용되지 않는다. 저장된 J와 선 복사장 입력만 고정이다.
동일 완료 결과와 준비 파일을 사용한 재현 명령은 다음과 같다.

```bash
env PYTHONPATH=scripts OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  LUMINA_ARTIS_PARITY=0 LUMINA_NLTE_BF_COLLISION=1 \
  python3 scripts/compare_lumina_si_public_tne.py \
  --result "$RESULT_JSON" --preparation "$PREP_JSON" \
  --public data/standart_data1/toy06 --node 34
```

아래의 두 지점 표는 이 추가 계산보다 먼저 실행한 공개 ne 단독 시험이다.

같은 완료 복사장·온도·원자모형을 유지하고, 앞의 S 시험과 같이 **Si 한
원소만** 공개 전자밀도에서 다시 풀었다. 원래 충돌 설정
`LUMINA_ARTIS_PARITY=0`, `LUMINA_NLTE_BF_COLLISION=1`을 사용했다. 먼저
각 점의 기존 Si 인구 전체가 저장된 선택 인구와 `rtol=2e-8`,
`atol=1e-30`에서 재현되는지 확인했고, Si II/III의 photo·유도/자발
재결합 흐름도 저장 장부와 일치했다. 바뀐 ne에서는 occupation·충돌
이온화·재결합율 등을 다시 계산했으며, 고정 온도의 bound 적분을
재사용하되 원 계산의 전90점 fallback 선택도 복원했다.

| 원점 · 속도 km/s | Si 이온 | 공개 CMFGEN | 현재 ne | 공개 ne만 적용 |
| --- | --- | ---: | ---: | ---: |
| 50 · 12376 | II | `0.0158594` | `5.19624e-6` | `4.46348e-6` |
| 50 · 12376 | III | `0.970772` | `0.995154` | `0.994369` |
| 34 · 21534 | II | `0.158011` | `4.28652e-8` | `2.95620e-8` |
| 34 · 21534 | III | `0.791562` | `0.705294` | `0.643846` |

원점50의 현재/공개 ne는 `3.78469e8/3.25389e8 cm^-3`, 온도는
`10302.01 K`다. 원점34는 `1.86661e7/1.41029e7 cm^-3`, `9845.83 K`다.
두 점 모두 공개 ne만 적용하면 Si II는 오히려 더 줄었다. Si III는 원점50에서
공개값에 약간 가까워졌지만 원점34에서는 더 멀어졌다. 따라서 **이 두 점에서
현재 J/T와 원자모형을 고정한 전자밀도 차이만으로 공개 Si II 결손을 설명할
수 없다.** 공개 분포와의 차이에는 시간 의존성·비열·원자모형·복사장 등
다른 조건도 섞여 있다. 이는 전 원소의 전하·온도·복사장을 새로 맞춘 해나
공개 CMFGEN 재현이 아니며, 다른 깊이 또는 다른 J에 일반화하지 않는다.

두 실행은 grammar `debug`에서 각각 CPU1·4GiB, 시간 상한10분이며 기존 J와
인구·자료는 변경하지 않았다. 완료 response SHA는
`e3a5b1f73ef7f2cc5785d92bc1937cbfbe0cef92330816c288a61bf315826b65`,
복사장 SHA는 `f8369b3553638c586f19c38ac4447860708dc1965a9278300e40617d75da83be`,
공개 물리표 SHA는 `a504a3d98284d8db8f72b53f1679bfa335cc3f51de40e351635fdbd78201ebef`다.
기존 사용자 작업인 미추적 `scripts/compare_lumina_public_ne_response.py`를
변경하지 않고 메모리에서만 S의 Z=16을 Si의 Z=14로 바꾸고 장부 원소 인덱스
1을 0으로 바꾸어 실행했다. 원본 SHA는
`ff5fb6b25e0a74f460e59b79cd778386f05792ec036e118f470e9c5e035eb0a6`,
변환된 입력 SHA는 `a8e0af40d0b8ebbb170861d5a28bc8027ca0fdf9e0072cbde3ba8442b082bf23`다.
이 미추적 원본은 이번 커밋에 포함하지 않았으므로 다른 환경에서 재현할 때는
해당 원본과 SHA가 필요하다. 같은 완료 결과와 준비 파일을 각각 `RESULT_JSON`,
`PREP_JSON`에 지정하고 원점 50 또는 34로 바꿔 아래 명령을 사용한다.

```bash
sed -E 's/\b16\b/14/g; s/output\[\x27ledger\x27\]\[1, i\]/output[\x27ledger\x27][0, i]/g; s/S_III_over_II/Si_III_over_II/g; s/SINGLE_S_RESPONSE/SINGLE_SI_RESPONSE/g' \
  scripts/compare_lumina_public_ne_response.py | \
  env PYTHONPATH=scripts OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  LUMINA_ARTIS_PARITY=0 LUMINA_NLTE_BF_COLLISION=1 \
  python3 - --result "$RESULT_JSON" --preparation "$PREP_JSON" \
  --public data/standart_data1/toy06 --node 50
```

원래 90개 점의 속도는 `radius_cm / (19.48*86400) / 1e5`다. 공개 100점
값을 각 속도에서 선형 보간했다. 최소 공개 속도보다 약0.029 km/s 낮은
내부점90은 외삽하지 않고 제외해 89점을 비교했다. 구간 평균이나 최종
v2의 질량 가중 판정이 아니다. 표의 수치는 **공개 CMFGEN / Lumina**의
0–1 이온분율이다.

| 속도 km/s | Si II | Si III | S II | S III |
| ---: | ---: | ---: | ---: | ---: |
| 5,066 | 3.789e-10 / 3.406e-10 | 0.002018 / 0.001963 | 4.030e-8 / 3.510e-8 | 0.015776 / 0.015254 |
| 7,983 | 8.036e-6 / 4.717e-6 | 0.882762 / 0.829266 | 0.000612 / 0.000357 | 0.970017 / 0.977026 |
| 10,164 | 0.000547 / 0.000109 | 0.985112 / 0.999512 | 0.182492 / 0.028866 | 0.811352 / 0.971099 |
| 12,376 | 0.015859 / 0.00000520 | 0.970772 / 0.995154 | 0.845199 / 0.002712 | 0.153768 / 0.996242 |
| 15,179 | 0.036999 / 0.00000285 | 0.945604 / 0.990041 | 0.809702 / 0.001241 | 0.188588 / 0.996911 |
| 19,769 | 0.146887 / 0.000000138 | 0.817306 / 0.739511 | 0.656345 / 0.0000372 | 0.336383 / 0.915100 |
| 28,108 | 0.027035 / 4.346e-12 | 0.824732 / 0.002763 | 0.315395 / 1.438e-9 | 0.629963 / 0.008021 |

89점의 최대 절대분율차는 Si II 0.158011(21,534 km/s), Si III
0.821969(28,108 km/s), S II 0.842487 및 S III 0.842474(둘 다
12,376 km/s)다. Si IV와 S IV도 각각 최대 0.880323과 0.941963이다.
이 값들은 실제 목표인 공개 결과와 비교할 때 차이가 크다는 진단이다.
이전 CMFGEN70 대비 작은 차이는 **비수렴 자체 재계산 물질70과의 차이**이며
공개 결과와의 차이로 전용하지 않는다.

예를 들어 12,376 km/s의 공개/Lumina 온도는 10286.44/10302.01 K,
전자밀도는 3.25389e8/3.78469e8 cm^-3이다. 이 위치에서 온도는 가깝지만
전자밀도와 이온화 분포는 다르다. 온도가 비슷하다는 사실만으로 어느 한
원인을 입증하지 않는다. 공개 계산은 시간 의존·비열을 포함하고, 여기의
Lumina는 고정 복사장·고정 온도의 정상 물질응답이므로 물리 조건과
이산화, 장 및 원자모형 차이가 혼합돼 있다. 공개 분율을 소스에 주입하거나
수렴된 공동해로 표기하지 않는다. 이 비교에는 새 BF/SE/수송이 없다.

### generation18 열잔차: gross 기준과 순채널 기준

2026-10-28의 [F2 열수지 기록](LUMINA_F2_THERMAL_STATE_2026-10-28.md)에서
이미 밝혔듯이 큰 선 가열과 냉각의 상쇄 때문에 gross 정규화와 채널
정규화는 매우 다를 수 있다. 이는 새로운 열정책 발견이 아니다. 이번에는
진행 중 작업의 **generation18 내부 시험점**에서도 그 상황이 지속되는지
확인했다. 활성 정책은 `native_instantaneous_scenario`다.

| 원점 | 전체 순열 | 최대(총 가열, 총 냉각)으로 나눈 절대 순열 | 순채널 절대값 합으로 나눈 절대 순열 |
| ---: | ---: | ---: | ---: |
| 42 | +4.13934e-5 | 3.12323e-6 | 0.999875 |
| 45 | +4.82685e-5 | 1.62615e-6 | 0.999716 |
| 50 | +3.44670e-5 | 2.44574e-7 | 0.997408 |
| 55 | −4.75302e-5 | 1.15058e-7 | 0.229617 |

순열의 단위는 `erg cm^-3 s^-1`이다. 순채널 정규화는 90점 중 69점에서
0.1보다 크고 85점에서 0.01보다 크다. 42점의 선 가열/냉각은 각각
`13.25338454 / 13.25334315`로 매우 비슷하지만 그 차이가 나머지 열항보다
크다. 채널 정규화는 기존 v2 합의 문턱이 아니므로 이 수치만으로 시험점이
기준에 실패했다고 판정하거나 새로운 차단 조건을 만들지 않는다. 반대로
기존 gross 기준을 만족한다는 사실만으로 순채널까지 잘 상쇄되거나 정확한
평형 온도에 도달했다고 주장하지 않는다.

기존 평가1의 고정장 기준과 평가2/3의 모든 점 동시 `T×exp(±1e-4)`,
동일 `ne` 섭동을 재사용하면, 42점의 순열은 `+4.44511e-5`, 동시
로그온도 방향 도함수는 약 `−1.86855e-4`, 50점은 각각 `+3.69490e-5`,
`−4.58654e-3`이다. 단열 항은 전 깊이에 결합되므로 이 값은 **프로필 전체를
같은 비율로 바꾼 방향 미분**이며, 점별 독립 Jacobian이나 전하를 재평형한
온도 보정량이 아니다. 선형 외삽만으로 실제 온도 오차를 정하지 않는다.
generation18의 순열은 42점 `+4.13934e-5`, 50점 `+3.44670e-5`로 기준보다
감소했지만 상당한 순채널은 남아 있다.

따라서 최종 선택 응답의 기존 합의 판정은 그대로 적용하면서, 결과 해석에는
같은 정책의 **절대 순열과 채널별 순열도 함께 제시**해야 한다. 더 엄격한
온도 수렴이 실제 Si/S와 스펙트럼에 어떤 차이를 주는지는 최종 선택 결과와
새 복사장 왕복 또는 명시된 민감도 계산으로 확인해야 하며, 현 자료만으로
CMFGEN 불일치의 원인이나 전역 온도 오차를 확정하지 않는다.

재현 입력은 진행 작업 root `lumina_second_field_thermal_replacement_20261030-fz3u4uV0/run`의
완료된 evaluation 1/2/3/8이다. `balance.npz` SHA256은 차례로
`ab755cc6d7e5c08e1c42d41b6354b6d425eb4e624c8f3d6ca1806113a9610f29`,
`a4d248fc3379a1e6e247dfcba073b890675a97dfd9736a5334274e9a4f628d8e`,
`02c002ef09adb20ee71593f6366ae3c9bc0f34b2a612198263a4ead7ffccef8a`,
`5acaddf8e9d1758ca18b8a1a70e13ae8115ba6625de7b9ac7eb01087fe01ca6c`다.
evaluation 1/2/3 `response.npz` SHA256은 각각
`e3a5b1f73ef7f2cc5785d92bc1937cbfbe0cef92330816c288a61bf315826b65`,
`5a0573c23543dc4bf35ba3a4a4a1e34826fa1ecfa29a3fa938d9c916e9056af9`,
`b5cfe9307cc14ea81a8171c0a1430702a0d8db40cb49d5546f02c0951a831138`이다.
새 물질·복사 수송 계산은 없고, 최종 응답 선택이나 작업 중단도 하지 않았다.

읽기 전용 재현은 위 진행 작업 root를 `TRIAL_RUN`에 지정하고 실행한다.

```bash
python3 - "$TRIAL_RUN" <<'PY'
import sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
balances = [np.load(root / f'evaluation-{n}/balance.npz') for n in (1, 2, 3, 8)]
responses = [np.load(root / f'evaluation-{n}/response.npz') for n in (1, 2, 3)]
T0, Tplus, Tminus = [r['temperature'] for r in responses]
assert np.array_equal(responses[0]['ne'], responses[1]['ne'])
assert np.array_equal(responses[0]['ne'], responses[2]['ne'])
assert np.allclose(Tplus / T0, np.exp(1e-4), rtol=1e-14)
assert np.allclose(Tminus / T0, np.exp(-1e-4), rtol=1e-14)
key = 'native_instantaneous_scenario'
net = balances[3][key + '_net']
gross = balances[3][key + '_gross']
channel = balances[3][key + '_channel_normalized']
scaled = abs(net) / (.5 * (gross + abs(net)))
print('channel >0.1, >0.01:', (channel > .1).sum(), (channel > .01).sum())
for n in (42, 45, 50, 55):
    i = n - 1
    derivative = ((balances[1][key + '_net'][i] - balances[2][key + '_net'][i])
                  / np.log(Tplus[i] / Tminus[i]))
    print(n, net[i], scaled[i], channel[i], balances[0][key + '_net'][i], derivative)
PY
```

### 진행 중 generation18의 물리 잔차와 공개 분율

진행 중 `1243823`의 generation18은 내부 선분에서 수용됐지만, 해당 작업의
최종 선택 응답은 아니다. 보존된 평가8 response·balance와 실행 바이너리의
단열항을 다시 사용해 후보 자체의 정규화 잔차를 확인했다. 활성 정책
`native_instantaneous_scenario`에서 최대 상대 전하 잔차는 `3.14582e-11`,
후보 자체의 가열/냉각 척도를 쓴 최대 열잔차는 `3.12323e-6`이며 각각
현행 고정장 기준 `1e-10`, `1e-3` 안에 있다. 별도 진단 정책
`physical_stored`의 같은 방식 열잔차는 `1.07516e-5`다. 저장된 전 원소
SE 최대 잔차는 `9.98888e-17`이고, 단열항의 재계산과 저장값은 전점
동일했다. 이는 고정 J에 대한 조건부 후보이지 수송 공동수렴이 아니다.

같은 원점50, 속도 약 12,376 km/s에서 후보 T/ne는
`10307.58 K / 3.78469e8 cm^-3`다. 아래 공개값은 위와 같은 toy06
19.48일 속도 보간 결과이며 물리 설정과 원자모형이 다르다.

| 분율 | 공개 CMFGEN | generation18 시험 후보 |
| --- | ---: | ---: |
| Si II | 0.015859 | 0.000005197 |
| Si III | 0.970772 | 0.995153 |
| S II | 0.845199 | 0.00271117 |
| S III | 0.153768 | 0.996243 |

따라서 이 후보의 전하·열잔차가 작더라도 공개 Si/S 차이가 해결된 것은
아니다. 또한 진행 중인 봉인 소스 `044df9ea`는 전하와 열 모두에 더
엄격한 초기 척도 `numerical_tolerance=1e-10`을 사용하므로 위 현행 물리
문턱과 다르게 다음 온도 평가를 시작했다. 이 차이는 이미 15:48 인계에
기록된 실행본/현행 코드 차이를 **실제 generation18 수치로 확인**한 것이다.
실행 중 코드를 바꾸거나 작업을 취소하지 않았고, 봉인 작업이 최종 선택한
후보로 이 시험점을 대체해 계수·수송에 넘기지 않았다.

현행 `solve_coupled()`의 실제 종료 경로도 이 보존 evaluation-8을 입력으로
확인했다. response·balance·실행 단열 라이브러리·입력 기하의 SHA를 확인하고,
후보 T/ne·인구와 전체90점 단열항을 사용했다. 단열항 재계산은 저장값과
배열 원소별 동일하다. 평가 callback이 호출되면 즉시 실패하도록 둔 상태에서
전하 `1e-10`, 열 `1e-3`을 전달한 결과 `numerical_converged=true`,
실제 신규 물질 평가 **0회**, 동일 후보 반환이었다. 실제 전하/열 잔차는
위 `3.14582e-11/3.12323e-6`을 재현했다. 이는 **같은 고정 J와 이 후보를
현행 해법의 시작 상태로 제공할 경우** 더 엄격한 구판 기준 때문에 추가
전주파수 적분을 할 필요가 없다는 직접 확인이다. 실행 중인 구판의 봉인
결과나 최종 선택을 변경하지 않았으며 이 시험점을 자동 게시하지 않았다.
다른 물질·새 복사장에서도0회로 끝난다는 뜻이 아니고 공동수렴·CMFGEN
동질성 판정도 아니다. 시험은 grammar `debug`, CPU1·4GiB·3분 한도다.

같은 후보를 공개 격자와 겹치는 89개 속도점 전부에서 비교하면 Si II,
Si III, S II, S III의 최대 절대분율차는 각각 `0.158011`, `0.821967`,
`0.842488`, `0.842475`다. 각 값은 점별 분율차이지 질량가중 또는 v2
전체 수용 판정이 아니다. 아래 표는 III 분율의 공개 CMFGEN / 이 시험
후보이며 공개 속도를 선형 보간했다.

| 속도 km/s | Si III 공개 / 후보 | S III 공개 / 후보 |
| ---: | ---: | ---: |
| 5,066 | 0.002018 / 0.001963 | 0.015776 / 0.015254 |
| 10,164 | 0.985112 / 0.999512 | 0.811352 / 0.971099 |
| 12,376 | 0.970772 / 0.995153 | 0.153768 / 0.996243 |
| 19,769 | 0.817306 / 0.737372 | 0.336383 / 0.914209 |
| 28,108 | 0.824732 / 0.002765 | 0.629963 / 0.008026 |

직전 완료 고정-T 응답의 같은 90점과 비교한 이 후보의 최대 절대분율 변화는
Si II `8.42e-9`, Si III `0.002744`, S II `6.30e-6`, S III `0.001852`다.
온도는 원점4에서 최대 상대 `9.516%` 바뀌고 ne 최대 상대 변화는
`0.184%`다. 따라서 온도가 모든 점에서 거의 안 변했다고 주장하지 않는다.
이 한 차례의 고정 J 보정만으로 큰 공개 분포 차이가 해소되지 않았다는
사실과, 다음 새 J를 포함한 왕복이 아직 없다는 사실을 함께 유지한다.

재현 자료 SHA256은 response
`73b214112785c73fd59f0249f776f1107abcfa45a4c52cd7011fa528b8482149`, balance
`5acaddf8e9d1758ca18b8a1a70e13ae8115ba6625de7b9ac7eb01087fe01ca6c`다.
`packet()`과 `residual()`에 이 자료와 같은 실행의 단열 바이너리 및 기존
90점 반경을 넣어 후보 ne와 각 정책의 실제 열 척도를 사용했다. grammar
debug CPU1·4GiB·3분 이내의 읽기 전용 판독이며 새 전리율·SE·수송은 없다.

## 공개 toy06 원자모형과 현재 사용 원자모형의 차이

19.48일 toy06의 [원 논문 CMFGEN 원자모형 표](https://arxiv.org/html/2209.11671)의
초준위(SL)·전체준위(FL) 개수를 현재 완료 새 장 응답에 사용한 원자 목록의
실제 행과 비교했다. 표의 순서는 `SL / FL`이다. 마지막 단일 준위 이온은
전리·재결합 경계용 종말 이온이며 통상의 상세 모델 이온과 구분한다.

| 이온 | 공개 CMFGEN toy06 | 완료 Lumina 응답의 원자 목록 |
| --- | ---: | ---: |
| Si II | 32 / 62 | 69 / 125 |
| Si III | 33 / 61 | 99 / 147 |
| Si IV | 37 / 48 | 50 / 61 |
| Si V | 1 / 1, 종말 | 52 / 203, 상세 |
| S II | 56 / 324 | 55 / 322 |
| S III | 48 / 98 | 79 / 256 |
| S IV | 27 / 67 | 65 / 176 |
| S V | 1 / 1, 종말 | 47 / 184, 상세 |

특히 S II의 FL 수는 비슷하지만 SL 수가 같지 않으며, Si III/S III의
준위 수는 크게 다르다. 같은 이온명만으로 같은 초준위 분할·준위 에너지·
전이·단면적이 보장되지 않는다. 준위 수 차이만으로 각 원자료가 다른지,
또는 이것이 공개 분율 오프셋의 원인이나 기여율인지는 판정할 수 없다.
공개 시간의존·비열 계산과 우리 정상 고정장 계산의 차이도 함께 남는다.
수렴한 matched-physics self-run은 현재 원자모형을 유지해 비교하고, 공개
toy06 재현은 논문 수준의 원자모형·시간/비열·격자 계약을 별도로 맞춰야 한다.

완료 응답 보고서의 실제 `levels.csv` SHA256은
`b3645052d00dbd535de4f9b3687eda3da02f3dd0584ccef4079c8f9e700daf67`다.
각 이온은 `atomic_number`, `ion_number+1`로 선택해 전체 행 수와 서로 다른
`super_level` 수를 세었다. 해당 응답 원 MODEL_SPEC의 SHA256은
`3e6f53861f16e2dac822d68e1412f91813c7a80f939fdbd0c4b1394e9055f2fb`이며
수용 manifest와 일치했다. 저장소 설정 MODEL_SPEC의 S V는 `39/163`이지만
실제 완료 응답 원 MODEL_SPEC은 `47/184`이므로 이 둘을 혼동하지 않는다.
기존 자료 읽기만 했으며 새 native·물질·수송 계산은 없다.

### S II 원자료는 이미 있고 마지막 두 준위가 선택에서 빠졌다

Hillier의 [공개 CMFGEN 원자자료 안내](https://sites.pitt.edu/~hillier/web/CMFGEN.htm)와
그 원자자료를 보존한 저장소에서
[S II `f_to_s_56.dat`](https://github.com/tardis-sn/carsus-data-cmfgen/blob/4ef0cb73b800479b1d0afa65b4cf54719356bab9/atomic/SUL/II/26sep14/f_to_s_56.dat)를
찾았다. 파일은 324 FL·56 SL이며, 마지막 준위명도 논문 toy06 표의 마지막
준위에 대응한다. 다만 이 일치만으로 해당 파일이 공개 toy06 계산의 실제
봉인 입력이었다는 증명은 아니다.

이 공개 파일의 SHA256 `775e506aa69271ff0b41baf0961479b295965fd5cea11a5c12514f938b72775c`는
현재 원 CMFGEN 실행 디렉터리에 연결된 S II F_TO_S 파일과 바이트 동일하다.
완료 응답이 실제 사용한 322개 S II FL 목록은 공개 파일과 **이름 322개 모두
일치**, 공통 FL의 통계가중치·SL 번호 불일치 0개다. `cm⁻¹→eV` 변환 후
공통 준위 에너지의 최대 절대차는 `6.15e-9 eV`다. 이 대조는 봉인된 완료
`levels.csv`에 대한 것으로, 현재 링크의 바이트가 옛 실행 당시에도 동일했는지
별도로 소급 증명하지 않는다.

현재 원 MODEL은 S II를 `322 FL / 55 SL`로 선택한다. 공개 후보의 나머지
두 준위는 원본 순번323 `3s2_3p2(1De)6d_2Se`(기존 SL54)와 순번324
`3s_3p3(5So)4p_6Pe`(새 SL56)다. 현재 연결된 전체 원 OSC 파일에는 이 두
준위에 연결되는 선이 각각19개와2개 있으며, 전체 PHOT 파일에도 두 이름이
존재한다. 이는 원자료가 전혀 없다는 뜻이 아니라 **현재 모델의 선택에서
마지막 두 준위가 제외된다는 증거**다. 원 OSC의 선 존재는 실제 유효 선율,
두 누락 준위의 인구 또는 이온분율 오차 기여를 증명하지 않는다.

완료 선택 응답의 원점50에서 기존 SL54 인구는 `2.91397e-5 cm^-3`, S II
총인구는 `169546.18 cm^-3`이다. **기존 SL54의 총인구를 고정한다는 가정**
아래, 여기에 추가될 323번 준위의 인구는 SL54 전체보다 클 수 없고 그 상한은
현재 S II의 `1.72e-10`배다. 그러나 새 준위를 넣어 풀면 SL54 자체와 다른
준위의 인구·율이 바뀔 수 있고, 작은 인구가 큰 전이율과 곱해 만드는 흐름도
이 상한만으로 제한되지 않는다. 특히 별도 SL56을 형성하는 324번 준위에는
이 상한을 적용할 수 없다. 따라서 두 준위의 동적 영향이 작다고 판정하지 않는다.
선택 `response.npz`의 `columns=(16,1,53)`인 행과 `Z=16, charge=1` 행들의
`population[:,49]`를 비교하면 된다. SL/원점 주소는 이 배열에서 0부터 센다.

기존 322개 S II에 대해서는 SL 구성 자체가 전혀 다르다는 가설보다 **공통
준위의 분포·율과 두 준위의 생략 효과**를 확인하는 편이 근거에 맞다. 324개로
바꾸려면 원 MODEL_SPEC뿐 아니라 실제 MODEL, 초준위 주소, 일관된 원자자료·
물질·수송 계보를 함께 재구성해야 한다. 현재 진행 중인 계산을 수정하거나
한쪽만 바꾸어 맞춘 결과를 동등성으로 취급하지 않는다.

### Si II–IV와 S III–IV도 공개 표의 준위·SL 수를 구성할 후보가 있다

동일 공개 원자자료 저장소 commit `c7e332227de2aa02a5ee2fbf0399ab2e227af9bd`의
F_TO_S를 현재 원 MODEL의 이온별 FL과 대조했다. 아래 `prefix`는 원 논문의
전체준위 수만큼 원자 파일 앞에서 선택한 행이며, 마지막 준위의 구성도 논문
표와 대응한다. 공통 행의 순서·정확한 이름·통계가중치·SL 번호가 모두 같다.
에너지의 `cm⁻¹→eV` 변환 후 최대 차이는 다섯 이온 중 `1.16e-8 eV`다.

| 이온 | 공개 표 FL / SL | 현재 FL 중 일치 행 | 원 파일과 현재 링크 SHA256 |
| --- | ---: | ---: | --- |
| Si II | 62 / 32 | 62 / 62 | `3fde7ba1f84a7fabcd58f62387203878bcebaa30d46f3e0a998958aeb48e81b5` |
| Si III | 61 / 33 | 61 / 61 | `d9b933d49b094676279365dfe97e67831be960282b8d6e4cd4570d69bef9764c` |
| Si IV | 48 / 37 | 48 / 48 | `5f36c8c6928f34a505e5047b7c26a6691bfe205d4462624e0c3cd8d3fc70131b` |
| S III | 98 / 48 | 98 / 98 | `437befd40710cbb4ebaa2c6a2607e449c7cf15087efc635329bdc7162a57077f` |
| S IV | 67 / 27 | 67 / 67 | `04c7c6a2a77429bec1ec136100d6c60817841cd9fb08d6f3ab56b64b786bbbd5` |

공개 원 파일은 해당 commit의 `atomic/SIL/{II,III,IV}/19apr23/`의
`f_to_s_79`, `f_to_s_ls`, `f_to_s_split`과
`atomic/SUL/{III,IV}/19apr23/`의 `f_to_s_127`, `f_to_s_69`다. 위 SHA는
각 원 파일과 현재 원 실행 디렉터리의 대응 F_TO_S 링크가 바이트 동일함을
확인한 값이다. 완료 `levels.csv`의 공통 행은 위 다섯 이온 336개와 S II
322개, 총658개다. 정확한 0기준 SL ID·g·이름·순서와 에너지를 검증했다.

이는 **공개 표의 Si/S 준위 구성을 기존 자료에서 만들 수 있다는 후보 증거**다.
논문 공개 계산의 실제 OSC/PHOT/충돌 파일·전체 SL 묶음·진화 상태의 바이트
동일성을 증명하지 않으며, 아직 candidate MODEL을 만들거나 수렴 계산하지 않았다.
준위가 더 많다는 이유만으로 현재 구현을 잘못됐다고 판단하지 않는다.

철족도 복사장을 바꾸므로 함께 확인했다. 현재 파일 그대로 논문 접두 행의
SL 개수가 맞는 후보는 Fe III `698/83`, Co II `2747/136`, Co III
`3917/124`, Co IV `314/37`, Co V `387/32`, Ni II `1000/59`, Ni III
`1000/47`, Ni IV `254/28`이다(순서는 FL/SL). 현재 링크의 Fe II·Fe IV·
Fe V·Ni V는 논문과 SL 개수가 다르지만, 같은 원자자료 안에 각각 논문
표의 개수·마지막 준위에 대응하는 `f_to_s_228`, `f_to_s_100`,
`f_to_s_139`의 앞191개, `f_to_s_152`의 앞183개가 있다. 현재 모델에 없는
Fe I도 별도 `f_to_s`의 앞136개가 44 SL·논문 마지막 준위와 대응한다.
이들은 파일 기반 후보일 뿐 공개 실행의 확정 입력이나 물리적 효과가 아니다.
새 native·BF·SE·수송 계산과 원 파일 교체는 하지 않았다.

### 공개 후보 밖 추가 준위의 현재 광이온화 흐름

위 접두 준위 대응을 완료된 새 J 고정-T 응답의 기존 BF 율에 적용했다.
각 이온의 현재 bound photo 흐름 중 공개 표의 준위 수를 넘어서는 FL에서
나온 비율이다. 322개만 포함된 현재 S II에는 이 기준의 추가 FL이 없다.

| 원점(1기준) · 속도 km/s | Si II→III 추가 FL | Si III→IV 추가 FL | S III→IV 추가 FL |
| --- | ---: | ---: | ---: |
| 50 · 12,376 | 0.657% | 3.259% | 1.237% |
| 54 · 10,164 | 1.598% | 11.926% | 3.260% |
| 60 · 7,019 | 4.269% | 18.025% | 5.034% |

예를 들어 원점60의 Si III bound photo 총흐름은 `174609.62`, 그중 추가
FL은 `31473.80 cm^-3 s^-1`다. S III는 `59197.60` 중 `2979.80`이다.
이 값은 **현재 J/T/ne와 현재 FL 인구를 고정한 방향별 bound photo의
분해**이며, 공개 계산의 rate가 아니다. 이 표의 S III는 S III→IV이지
관심 비율을 직접 정하는 S II→III가 아니다. 총 BF photo에는 bound 외
pseudo 항도 포함되며 원점60에서 위 두 bound 합은 총 BF photo의 각각
`99.77%`, `99.61%`다.

따라서 실제 현 상태에서 Si III 추가 준위의 해당 광이온화 흐름은 일부 깊이에서
무시하기 어렵고, 원점50의 S III 추가준위 직접 photo 비중은 작다. 그러나
준위를 실제로 빼거나 추가하면 SL 분할·다른 인구·복사장과 역방향 율이 함께
바뀌므로, 위 수치만으로 재계산 결과나 공개 S II/III 불일치의 기여율을
예측하지 않는다. S II의 빠진323·324번 준위는 현재 응답에 없으므로 이
분해의 `0% 추가`에 포함되지 않는다.

재현은 완료 `route-roster.npz`의 `(Z,stage,lower,probability)`,
`bound-rates.npy`의 photo열, 선택 `response.npz`의 `population`, `fraction`,
`columns`, 실제 `levels.csv`를 사용한다. 각 경로에서
`population[lower의 SL,node] * fraction[lower,node] * probability *
bound_rates[route,node,0]`을 더하고, 원자번호별 `level_number`를 공개 표의
FL 개수와 비교한다. 파일 SHA는 아래 신원과 S II 분해 절에 기록된 것을
사용하며, 분모를 선택 응답의 `balance.npz` BF photo 총흐름과 대조했다.
입력은 읽기만 했고 새 BF 적분·SE·수송은 없다.

## 공개 오프셋과 마지막 새 장 응답을 분리

같은 완료 분율표의 `CMFGEN70_fraction`, `accepted_fraction`,
`new_J_response_fraction`을 공개 19.48일과 같은 속도에서 대조했다.
물질70은 자체 비수렴 CMFGEN, accepted는 새 수송의 이전 수용 Lumina 물질,
마지막 열은 새 장의 고정-T 전하/SE 응답이다. 이들은 같은 물리 모델에서
완전 수렴한 세 해가 아니며 각각의 계보를 유지한다.

| 12,376 km/s | 공개 CMFGEN | 자체 물질70 | 이전 Lumina 물질 | 새 장 응답 |
| --- | ---: | ---: | ---: | ---: |
| Si II | 0.015859 | 0.00000417 | 0.00000506 | 0.00000520 |
| Si III | 0.970772 | 0.994761 | 0.995016 | 0.995154 |
| S II | 0.845199 | 0.002709 | 0.002683 | 0.002712 |
| S III | 0.153768 | 0.996547 | 0.996259 | 0.996242 |

S II/III의 공개 대비 큰 차이는 자체 물질70 및 이전 수용 Lumina 물질에도
존재했다. 이전 물질→새 장 응답의 전90점 최대 절대 변화는 S II
`1.97123e-4`, S III `3.69060e-3`이며, 공개→이전 Lumina의 89점 최대는
각각 `0.842516`과 `0.842492`다. Si III도 이전→새 최대 `0.005753`이고
공개→이전 최대 `0.822078`이다. 이 결과는 **이번 한 번의 새 장 응답이
이미 존재하던 차이를 만들었다는 설명을 지지하지 않는다.** 그러나 원래
물질70의 미수렴, 공개 시간진화·비열 과정, 다른 복사장·열원·원자모형 등이
함께 다르므로 특정 원인이나 해당 항의 기여율을 결정하지 않는다.

물질70 원 인구와 이전 수용 response 원 인구에서 각각 Si/S II–V의 720개
분율을 독립 복원해 CSV 해당 열과 대조했다. 최대 절대차는 각각
`6.66134e-16`, `7.77156e-16`이다. 두 입력의 SHA256은 각각
`50e7b2c6f810b1c4fcfb4a59812ea88d07d17e985c878adcbe3974a821f99a3a`,
`d5d3735b90e88b45a9e0e85572cf28717c37cd01a2913017e107b9d18b3058bc`다.
수정한 공개 비교기는 위 세 열을 구분해 재현하며 직접시험 4개가 통과했다.

### 공개 S II의 주변 시기 비교

공개 CMFGEN 시계열의 같은 속도에 각 epoch를 선형 보간했다. 원점50의
큰 공개 오프셋이 19.48일의 일시적인 점프인지 확인하기 위한 판독이며,
새 물리 계산이나 같은 고정 물질에서의 시간항 분해가 아니다.

| 폭발 후 일수 | 10163.5 km/s | 12376.1 km/s | 15179 km/s |
| ---: | ---: | ---: | ---: |
| 14.64 | 0.006483 | 0.830133 | 0.905418 |
| 17.71 | 0.062517 | 0.860146 | 0.844398 |
| 19.48 | 0.182492 | 0.845199 | 0.809703 |
| 21.43 | 0.407011 | 0.819001 | 0.775186 |
| 23.57 | 0.606376 | 0.797770 | 0.732697 |

표는 S II 분율이다. 원점50에 가까운 12376.1 km/s의 공개 분율은 이
기간에 0.798–0.860이므로, 공개 S II `0.845199` 대 고정장 Lumina
`0.002712`의 큰 차이는 공개 분포의 19.48일 한 점에서만 발생하는 급변은
아니다. 동시에 더 안쪽의 S II는 크게 바뀌며, 고정 속도에서 온도·ne도
변한다. 공개 내부 복사장은 이 자료에 없다. 따라서 이것을 시간의존의 영향이
작다는 증거나 시간항의 개별 기여율로 해석하지 않는다. 공개 파일 SHA256은
아래 신원 항목의 S SHA와 같다.
재현은 `public_block(path, epoch='14.640')` 등으로 각 epoch 블록을 읽고
`interp_columns(table, [velocity], 2)`를 적용한다.

## 가장 큰 S II/III 분율차 점의 기존 BF 흐름

12,376 km/s, 원래점50에서 완료 고정-T 물질응답의 기존 `balance.npz`를
읽었다. 새 율 적분·SE·수송은 없다. `bf_ion_Z_stage=(16,2)`는 S II에서
시작하는 BF 경로이고, 분모의 `bf_ion_Z_charge=(16,1)/(16,2)`는 각각
S II/S III 이온 총밀도다. `photo`, `spontaneous`, `stimulated`는 현재 후보
전체 준위인구로 가중된 방향별 체적 입자 흐름이다.

| 항목 | 현재 Lumina 고정장 응답 |
| --- | ---: |
| S II / S III 밀도 (cm^-3) | 169546.18 / 62273358.03 |
| S II→III 광이온화 (cm^-3 s^-1) | 56827.7432 |
| S III→II 자발 / 유도 재결합 (cm^-3 s^-1) | 56713.3174 / 114.3765 |
| BF 순흐름 / (정방향+역방향) | +4.3413e-7 |
| 광이온화율 / 복사 재결합율 (해당 이온당 s^-1) | 0.335176 / 0.000912552 |
| S III / S II 분율비 | 367.294 |

공개 이온분율의 같은 속도 보간 S III/S II는 `0.181931`이다. 두 분율비는
약 `2018.87`배 차이 난다. 현재 응답의 BF **정·역 체적 흐름 자체는 거의
상쇄**하므로, 단순히 잔차가 작다는 것은 공개 분율과 같다는 뜻이 아니다.

이 데이터는 현재 상태의 BF 채널만 보여준다. 충돌·비열·시간 변화·다른
이온의 전체 유출입, 공개 계산의 동일 채널 율과 준위인구는 아직 없다.
특히 정·역 흐름 상쇄만으로 다른 큰 유출입이 없다고 결론내리지 않는다.
현재 장과 내부 분율을 고정하고 두 이온만 남긴 가상 선형계에서의 역률 합
시간은 약 2.98초지만, 이것은 전원소·다준위·복사장 결합계의 실제 완화
시간도 공개 CMFGEN의 이온화 동결 여부도 아니다. 공개 분율을 맞추기 위한
율 배율이나 임의 튜닝값으로 사용하지 않는다.

### 같은 상태의 충돌 전리·3체 재결합

위 `balance.npz`는 복사 BF만 저장하므로, 완료 응답의 T/ne·SL 인구·
FL 분율·occupation과 동일한 PHOT 경로를 사용해 실행 소스의 충돌 전리와
Saha 역률을 읽기 전용으로 재구성했다. 조건은 원 실행과 같은
`LUMINA_NLTE_BF_COLLISION=1`이다. 각 행은 `cm^-3 s^-1`이며 원래점 번호다.

| S II→III | 31 | 34 | 50 | 60 |
| --- | ---: | ---: | ---: | ---: |
| 광전리 | 3.86403509 | 56.62737987 | 56827.74322 | 21872.19278 |
| 복사 재결합 | 3.86403496 | 56.62737690 | 56827.69388 | 21872.19378 |
| 충돌 전리 | 2.68587e-7 | 6.50219e-6 | 0.1360313 | 0.2550326 |
| 3체 재결합 | 4.03527e-7 | 9.46283e-6 | 0.1853728 | 0.2540305 |

원점50에서 충돌 전리/광전리는 `2.39375e-6`, 3체/복사 재결합은
`3.26201e-6`이다. 그러나 복사 BF 순흐름 `+0.04934150`과 충돌 BF
순흐름 `−0.04934147`이 상쇄된다. 합 `2.91e-8`은 이 네 방향을 저장
인구로 재구성한 결과이지 독립적인 전체 SE 잔차나 공개 모델의 값은 아니다.
S III→IV도 원점50에서 광전리/복사 재결합 `372.186076/372.186046`,
충돌 전리/3체 재결합 `5.63771e-5/8.62874e-5`로 재구성됐다.

따라서 **현재 저장된 정상 고정장 상태의 이 점들에서** 충돌 전리·3체
재결합은 개별 주흐름을 지배하지 않으며 복사 순흐름과 함께 보아야 한다.
공개 CMFGEN의 충돌율, 비열·시간항, 다른 J/T/ne나 재계산된 인구에 대해
동일하다는 뜻은 아니다. 공개 이온화 차이의 전역 원인·공동수렴·스펙트럼
동등성은 여전히 미판정이다.

재현은 이 절 하단에 명시한 다섯 파일의 SHA를 사용한다. `route-roster`에서
S의 stage2/3과 `phot_grid=0`을 고르고, 각 경로의 FL/SL 주소와 저장 인구를
대응시킨다. 원 실행 source commit `95b785b394bd0e95457424fbe79eae040f97e72d`의
`nlte_bf_cmfgen_collisional_ionization_rate`와 `ModelMaterialRates.prepare`의
Saha식을 적용해 경로 확률·하위 FL 인구/상위 SL 인구로 각각 합한다.
선택 경로는 stage2 322개, stage3 256개이며 상위 terminal 경로는 없다.
복사값은 같은 선택 `balance.npz`에서 읽는다. 이는 소스식을 이용한 별도
후처리이며 실행 바이너리의 충돌 행렬을 직접 추출한 결과는 아니다. 새 BF
적분·SE·수송은 없다.

### S III 내부 재분배만으로 가능한 BF 변화의 조건부 상한

현재 등록된 S II 시작 BF 경로 322개는 모두 S III의 FL0, 즉 S III
SL0에 도착한다. 같은 선택 응답의 원점50에서 S III 총인구는
`62273358.03`, 그중 SL0 인구는 `55232017.25 cm^-3`로 `88.69285%`다.
FL0 하나의 분율 `10.64963%`와 그 fine-structure 묶음인 **SL0 전체의 분율**을
혼동하지 않는다.

다음은 원 모델을 바꾼 물리 계산이 아니라, 완전히 명시된 BF 전용 가정의
대수적 한계다. J/T/ne, S II의 FL별 인구비, BF 계수 및 S III SL 내부
FL 비율을 고정하고 S III 총인구만 SL 사이에서 재분배한다고 가정한다.
S III 전부를 SL0에 몰더라도 S II←III 복사 BF 재결합률/전체 S III는
현재 `0.000912552 s^-1`에서 최대 `0.001028890 s^-1`까지다.
같은 가정의 S II→III 광이온화율/전체 S II는 `0.335176 s^-1`로 고정된다.
이 **BF 두 방향만 평형**이면 S III/S II는 최소 `325.764`이고, 공개
비율 `0.181931`의 약 `1791`배다.

즉 고정된 현재 S II 여기분포와 장에서 **S III 내부 인구를 SL0으로
옮기는 것만으로** 공개 비율을 설명할 수 없다. 그러나 S II 여기분포·J·T/ne가
바뀌거나 충돌·비열·시간항·다른 이온 경로가 관여하면 이 상한은 적용되지
않는다. 별도 SL 수 변경이 역으로 S II 여기와 J를 바꾸는 효과도 배제하지
않는다. 공개 계산의 실제 BF율과 SL 인구는 없으므로 이것을 공개 해의
평형 증명이나 실제 기여율로 사용하지 않는다.

재현: `route-roster.npz`에서 `(Z,stage)=(16,2)`의 `target` 322개를 실제
`levels.csv`에 대조하고, 선택 `response.npz`의 `columns=(16,2,0)`과
`Z=16, charge=2` 전체 인구를 읽는다. `balance.npz`의 S II photo/자발·유도
BF 흐름을 각 이온 총인구로 나누고, `photo_per_SII * SL0_fraction /
recomb_per_SIII`를 계산한다. 세 입력과 원자 목록의 SHA는 아래 신원 및
하위준위 절에 기록되어 있다. 새 BF 적분·SE·수송은 없다.

현재 응답의 원 MODEL은 `DO_DDT=F`, `TRT_NON_TE=F`이고 실제 응답
`ModelResponse.evaluate`는 정상 SE를 푼다. 공개 시간의존 결과와 주 비교할 때
이 범위 차이를 분리해야 한다. 기존 v2의 주 acceptance는 같은 물리 조건의
정상 CMFGEN self-run이며 공개 시간의존 19.48일은 과학 교차검사다. 사용자의
최종 스펙트럼·물리량 비교 목표는 그대로 유지하며 이 정량 진단만으로
어느 경로의 수렴·동등성을 선언하지 않는다.

### S II와 S III의 초준위 인구를 모두 재배치한 조건부 한계

앞의 S III만 재배치하는 가정보다 넓게, 같은 원점50에서 S II의 55개
SL 인구도 임의로 재배치한다고 가정했다. 실제 준위 인구를 수정하거나
평형을 다시 푼 것은 아니다. J/T/ne, 원 모델의 BF 계수와 각 SL 안의
FL 인구비는 고정한다. S II의 322개 BF 경로는 55개 SL을 모두 포함하며,
대상은 모두 S III SL0이다. 선택된 응답의 각 S II SL 내부 FL 분율 합은
1과 최대 `2.22e-16` 차이 난다.

각 S II SL의 bound photo 계수를 소속 FL의 분율과 경로 확률로 가중해
합하면 최솟값은 바닥 SL0의 `0.0208996826 s^-1`이다. 전체 광이온화율은
이 비음수 bound 기여에 용해항을 더하므로, 용해항이 비음수인 조건에서
S II를 어떻게 재배치하더라도 전체율은 이보다 작을 수 없다. S III를
SL0에 전부 몰았을 때 앞 절의 최대 복사 재결합률은 `0.00102889043 s^-1`다.
따라서 **두 방향의 BF만으로 정상 평형을 이루는 가상계**의 S III/S II
비율은 최소 `20.31284`이며, 공개 19.48일 같은 속도의 `0.18193056`보다
약 `111.65`배 크다.

이는 현재 장과 원자계수에서 S II 여기 및 S III SL 인구의 재배치만으로
공개 비율을 만드는 설명을 더욱 제한한다. 그러나 공개 계산은 시간의존·
비열을 포함하고, 위 가상계는 충돌·다른 이온·시간항을 제외했다. 또한
J/T/ne 또는 SL 내부 분율과 원자모형 자체를 바꾸면 이 한계는 적용되지
않는다. 따라서 이 결과는 실제 전체 평형의 하한, 복사장 오차의 증명,
공개 해의 BF 평형 증명 또는 목표 달성이 아니다.

재현은 아래 신원의 route·bound rates·response·원자 목록과 앞 절의
`balance.npz`를 사용한다. `(Z,stage)=(16,2)`의 `lower`를 SL로 묶어
`sum(fraction_FL * probability * bound_photo_rate)`의 최솟값을 구하고,
앞 절의 전체 S III당 복사 재결합률을 S III SL0 분율로 나눈 상한으로
나눈다. 공개 분율은 같은 비교의 속도 보간값이다. 저장 자료의 읽기만
했으며 새 물리 계산은 수행하지 않았다.

재현 자료는 완료 response의 `evaluation-4/balance.npz`이고 SHA256은
`9d42fd13bef1291cf21f6387d68de0aa93876d34c9cf6920cbe061a37975d232`다.
`bf_ion_Z_stage`, `bf_ion_Z_charge`, `bf_ion_number_flows`,
`bf_ion_number_density`, `bf_ion_number_flow_channels`를 읽고 위 정의로
나누어 재현한다. 작업은 grammar-debug CPU1·4GiB·3분 범위의 읽기만 했다.

### 고정된 현재 장에서 공개 분율의 조건부 지속시간

위 BF 계수의 상·하한을 시간 방향으로도 사용하면, 현재 장에서 공개
분율이 단순한 과도 상태로 얼마나 유지될 수 있는지 **조건부**로 볼 수
있다. 원점50에서 초기 S II/S III 분율을 공개 19.48일의
`0.8451990006 / 0.1537675248`로 두고, 두 단계의 합 `0.9989665254`이
보존된다고 가정한다. 같은 J/T/ne와 원 원자모형, 각 SL 내부 FL 분율,
비음수 용해항 및 위 BF 율의 상·하한이 시간 내내 유지되며, 다른 이온·
충돌·비열·시간적으로 변하는 원천은 없다고 가정한다.

이때 S II 분율 `x`는 `dx/dt ≤ −g*x + a*(N−x)`를 만족한다. `N`은 위 두
단계 분율의 합이고, `g=0.020899682592 s^-1`,
`a=0.0010288904282 s^-1`는 앞 절의 광이온화 하한과 재결합 상한이다.
따라서 공개 초기값에서 출발하는 이 가상계의 S II 분율 상한은 다음과 같다.

| 경과 시간 | S II 분율 상한 |
| ---: | ---: |
| 0초 | 0.845199 |
| 60초 | 0.261049 |
| 300초 | 0.047981 |
| 무한시간 | 0.046872 |

상한식의 완화 시간은 `1/(g+a)=45.60초`다. 실제 공개 파일에서 같은
속도의 S II 분율은 17.71일 `0.860145`, 19.48일 `0.845199`, 21.43일
`0.819001`이다. 인접 출력 사이 평균 변화율은 각각 `−9.77e-8`,
`−1.55e-7 s^-1`이며, 이는 각 epoch 사이의 평균이지 19.48일 순간 변화율
또는 그 사이의 급변을 제한하는 증거는 아니다.

결론은 **현재 Lumina의 J와 BF 계수를 그대로 유지하면서 다른 과정 없이
시간항만 추가하는 두 단계 가상계는 공개의 높은 S II를 수분간 유지하지
못한다**는 것이다. 공개 CMFGEN 자체의 장이나 BF 율은 확보하지 못했다.
공개 모델의 비국소 침적, 비열·충돌, 다른 이온, 시간에 따라 변하는 장과
물질·준위분율은 모두 위 가정 밖이다. 따라서 이 비교로 공개 모델의
시간의존 효과가 작다거나 복사장이 틀렸다고 판정하지 않는다. 실제 해결을
위해서는 같은 물리 계약의 CMFGEN 내부 장·율과 새 공동수렴 상태가 필요하다.

재현은 공개 S 파일과 위 절에서 신원 확인한 BF 한계만 읽는다. 새 율 적분,
물질응답 또는 시간진화 계산은 없다.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=scripts python3 - <<'PY'
from pathlib import Path
import numpy as np
from compare_lumina_public_toy06 import public_block, interp_columns
p = Path('data/standart_data1/toy06/ionfrac_s_toy06_cmfgen.txt')
v = np.array([12376.055805864145])
epochs = ('17.710', '19.480', '21.430')
values = [(t, float(interp_columns(public_block(p, t), v, 2)[0]),
           float(interp_columns(public_block(p, t), v, 3)[0])) for t in epochs]
g, a = 0.020899682592, 0.0010288904282
x, N = values[1][1], values[1][1] + values[1][2]
equilibrium = N * a / (g + a)
print('epochs:', values, 'equilibrium:', equilibrium, 'seconds:', 1/(g+a))
for t in (0, 60, 300):
    print(t, equilibrium + (x-equilibrium)*np.exp(-(g+a)*t))
for left, right in zip(values[:-1], values[1:]):
    print(left[0], right[0], (right[1]-left[1])/((float(right[0])-float(left[0]))*86400))
PY
```

## S II 광이온화의 실제 하위준위 구성

위 원래점50의 새 장 응답에 대해 기존 bound-rate, PHOT route, 선택된
FL/SL 인구·분율을 읽었다. 같은 이온의 322개 경로를
`n_SL * fraction_FL * route_probability * bound_photo_rate`로 합했다.
T와 J가 같은 bound photo 적분은 고정온도 캐시에서 보존되며, 다른 ne의
Milne 재인수는 역방향 율에만 적용된다. 각 준위의 인구·분율은 마지막
선택 응답의 것을 사용했다. 원래 총 BF photo에는 bound 외 용해항이 있어
준위별 bound만의 합과 구분한다.

| S II 하위 FL (0기준) | SL | 들뜬 에너지 eV | bound photo 체적 흐름 | 전체 BF photo 대비 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 1.8455 | 6483.49 | 11.41% |
| 4 | 2 | 3.0465 | 5949.91 | 10.47% |
| 5 | 3 | 9.8438 | 4955.97 | 8.72% |
| 1 | 1 | 1.8415 | 4322.68 | 7.61% |
| 6 | 3 | 9.8886 | 3203.16 | 5.64% |
| 0 (바닥) | 0 | 0 | 2729.37 | 4.80% |

단위는 `cm^-3 s^-1`이다. bound 합 `56631.7273`은 저장된 총 BF photo
`56827.7432`의 `99.6551%`다. bound 바닥준위는 bound 합의 `4.8195%`이며
나머지 약 `95.18%`는 들뜬 하위준위에서 온다. 따라서 이 조건에서 바닥준위의
한 전리율만으로 총 photo 흐름을 대표할 수 없다. 이는 이번 새 장·후보의
정량 확인이며, 원 물질70/기존 장에서 이미 10월27일 확인한 들뜬준위 기여를
새로운 기작의 최초 발견으로 재명명하지 않는다.

가장 큰 FL2·FL4·FL5 경로의 문턱 파장은 각각 약 576.87·611.02·918.80 Å다.
이는 그 경로의 **전리 문턱**이며 실제 광흡수 기여가 그 파장 자체나 특정
좁은 대역에 집중됐다는 증거가 아니다. 앞서 비교한 600–750 Å 하나만으로
S II 전체 율이나 공개 CMFGEN 차이를 설명하지 않는다. 공개 계산의 준위인구와
내부 복사장은 없으므로 어느 준위·주파수가 두 계산의 오프셋을 일으켰는지도
여기서는 단정할 수 없다.

결과는 완료 실행의 기존 `route-roster.npz`, `bound-rates.npy`,
`evaluation-4/response.npz`, `evaluation-4/balance.npz` 및 해당 실행의
`atomic/levels.csv`에서 재현한다. 각 SHA256은 각각
`478d0e9c6902e1a2695eeda4b7d06780db441aec90c82908172c936b76b7df30`,
`81c02406c6c6c7f840afc65418a13afa383271ffe67c382786c4bfe97591c705`,
`e3a5b1f73ef7f2cc5785d92bc1937cbfbe0cef92330816c288a61bf315826b65`,
`9d42fd13bef1291cf21f6387d68de0aa93876d34c9cf6920cbe061a37975d232`,
`b3645052d00dbd535de4f9b3687eda3da02f3dd0584ccef4079c8f9e700daf67`이다.
grammar-debug CPU1·4GiB·3분 상한의 읽기만 수행했고 새 적분·SE·수송은 0이다.

## 신원·재현

- 완료 응답 보고서 SHA256 `e022dda77f1a66ce4d9269359079b67a76e134d8a70b9581e5c08db3d82b71cd`
- 생성 분율표 SHA256 `045652b21018fd42368e102db6c95385198700c2879ebadd893a559aa35aa5b2`
- 물리 프로필 SHA256 `766329d782f34a2c22bc6acaf4feb82b36c93d63ac74b5855a0272a0d858b29f`
- 선택 response SHA256 `e3a5b1f73ef7f2cc5785d92bc1937cbfbe0cef92330816c288a61bf315826b65`
- 공개 Si/S/phys SHA256 각각 `671fcc3b6ceb6c765741f27771b1b062757ebbbce8f1890bd1b8078b18fb501d`, `aeb7e689dd8c8a7e41ea2b11573df9a82bb1d28f84476423db584fc48b601a0b`, `a504a3d98284d8db8f72b53f1679bfa335cc3f51de40e351635fdbd78201ebef`.
  Si/S는 공개 원본의 현재 본문과도 해시 일치했다.

후보 response 배열의 원소·이온별 인구 합에서 Si/S II–V의 720개 분율을
독립 복원해 완료 CSV와 대조했다. 최대 절대차는 `7.77156e-16`이다.
비교기의 격리 직접시험은 grammar-debug CPU1·4GiB·3분 상한에서
`4 passed`였다. 최초 독립 대조 명령은 원격 인용 처리 오류로 Python 구문
검사에서 실패해 물리 자료를 읽지 않았고, 정정한 단회가 위 결과를 냈다.

완료 보고서에 기록된 정확한 result 경로를 `RESULT_JSON`에 지정해 재현한다.
실행기는 지정 결과의 완료 상태와 CSV·물리 프로필 SHA를 확인하고 JSON을
stdout에만 출력한다. 기존 결과를 쓰거나 덮어쓰지 않는다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/compare_lumina_public_toy06.py \
  --result "$RESULT_JSON" --public data/standart_data1/toy06
```

최종 물리·스펙트럼 동질성은 `NOT_MET`이다.

## 점값과 근사 구간평균의 차이

사용자가 제기한 점값 대 bin 평균 가능성을 기존 공개 이온분율로 확인했다.
공개 100점 분포를 속도에서 선형 보간하고, 기존 90 원점의 이웃 중간점을
가상 구간 경계로 삼아 `v²`로 체적 가중해 적분했다. 구간이 공개 범위 밖이면
외삽하지 않았고 양 끝 두 점은 경계를 임의로 만들지 않아, 88개 구간을 보았다.
이 계산은 공개 분포의 **조건부 체적 평균**과 그 중심점 보간값을 비교하며,
실제 Lumina MC estimator나 원소 밀도·조성 가중 이온분율을 재구성하지 않는다.

가장 큰 S 차이 지점인 원점50(12376.06 km/s)의 가상 구간은
12095.32–12656.08 km/s이다. 공개 점값과 가상 체적 평균은 다음과 같다.

| 이온 | 공개 점값 | 공개 근사 구간평균 | 평균 − 점값 |
| --- | ---: | ---: | ---: |
| Si II | 0.0158594 | 0.0165335 | +0.0006741 |
| Si III | 0.9707716 | 0.9700782 | −0.0006934 |
| S II | 0.8451990 | 0.8448433 | −0.0003557 |
| S III | 0.1537675 | 0.1541168 | +0.0003493 |

같은 점에서 공개 S II와 Lumina 점값의 기존 차이 약 `0.842487`에 비해,
공개 프로필을 이 가상 평균으로 바꾼 효과는 `0.000356`이다. **이 조건의
공개 격자 평균만으로는 그 큰 오프셋을 설명하지 못한다.** 전88개 구간에서
점값과 체적 평균의 최대 절대차는 Si II `0.000858`, Si III `0.029056`,
S II `0.010032`, S III `0.009994`다. Si III처럼 이온화 전선 근처에서는
평균 방식이 상당히 중요할 수 있으므로 무시하지 않는다.

이는 실제 MC bin의 경계·질량/원소밀도 가중·격자 이하 구조를 포함한
비교가 아니다. 공개 100점 이하의 급격한 구조가 선형 보간에 드러나지 않는
가능성이나 Lumina estimator 자체의 평균 효과는 이 결과로 배제하지 않는다.
공개 계산과 현재 Lumina의 시간의존·비열·원자모형 차이도 남아 있다.
새 수송·SE·BF 물리 계산은 없으며, 합성 직선의 해석적 평균과 일정 분율,
내림차순·범위 밖 비외삽을 포함한 관련 직접시험은 `6 passed`였다.

기존 비교기에 추가한 선택적 진단이며 기존 기본 출력은 유지한다. 이전 절의
완료 결과에 `RESULT_JSON`을 지정해 읽기 전용으로 재현한다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/compare_lumina_public_toy06.py \
  --result "$RESULT_JSON" --public data/standart_data1/toy06 --shell-average-check
```

## 공개 전자밀도 단일 S 응답 시도의 재현 실패

이 절은 초기 시도의 이력이다. 이후 환경을 맞춘 재현과 공개 ne 계산 결과는
맨 위의 **후속 정정**을 따른다.

공개 전자밀도 차이가 원점50의 S 분율에 미치는 영향을 같은 J/T에서
직접 보려 했으나, 먼저 필요한 기존 선택 응답의 재현에 실패했다.
따라서 공개 전자밀도 사례는 계산하지 않았고, 이 시도에서 전자밀도의
영향이나 공개 분포와의 물리적 원인을 판정할 수 없다.

- 첫 작업 `1245074`는 바뀌지 않는 다른 점들의 ne와 선택 occupation을
  섞어 C 파생상태 검사에서 중단됐다. S 정상해 계산 전 실패다.
- ne 배열을 일치시킨 `1245078`은 기존 ne에서 S 정상해를 풀었지만,
  저장된 선택 인구 247개 중 140개가 `rtol=2e-8`을 벗어났다.
  위반 성분의 최대 상대차는 `1.0018e-4`, 최대 절대차는 `0.11330701`이다.
- 원 계산의 bound 캐시는 어느 한 점에서 underflow가 발생하면 해당 경로를
  전90점에서 직접 적분한다. 이 전점 fallback을 복원한 `1245087`도 같은
  인구 차이를 냈다. 따라서 전점 fallback 선택 누락만으로 이 차이를 설명할
  수 없다. 원점50의 선택 상태 FL/SL 분율·occupation은 재현됐다.

세 작업의 실제 경과는 각각 32초, 29초, 30초이며 CPU1·4GiB의 소형
시험이다. 새 전90점 BF 적분·새 수송은 없고 실행 중인 열응답을 건드리지
않았다. 완료 응답 `result.json` SHA256은
`e022dda77f1a66ce4d9269359079b67a76e134d8a70b9581e5c08db3d82b71cd`다.
직접 시험용 미완 스크립트는 `scripts/compare_lumina_public_ne_response.py`,
마지막 시험 당시 SHA256은
`1268dd9484eb4854056be1afcf742c5470f2a880ef9cad56a2cd243654ab86a6`다.
스크립트는 기준 재현에 실패하면 공개 ne 결과를 내지 않도록 유지하며,
원인을 밝히기 전 과학적 비교기로 취급하거나 허용오차만 완화하지 않는다.

### 기존 S II 광이온화 흐름의 재구성

같은 선택 인구와 새 장에서 원점50의 S II 경로별 photo 율에 FL/SL
분율·인구·경로 확률을 곱해 합했다. `1245091`의 결과는 재구성
`56827.74321735174`, 저장 흐름 `56827.74321735181 cm^-3 s^-1`로,
비율 `0.9999999999999987`이다. 해당 점 S 경로의 전90점 fallback 수는
0이다. 따라서 **기존 S II photo 총흐름**은 재현되지만, 이것만으로
BF 역방향·다른 이온·BB·충돌·전체 S 정상해가 재현된다는 뜻은 아니다.
해당 정상해는 여전히 위 기준 재현에서 실패해 공개 ne 계산은 실행되지 않았다.

흐름 출력 확장 전 `1245090`은 Python 문법 오류로 물리 계산 시작 전
1초 만에 실패했고 정정 후 `1245091`은 30초에 위 결과를 낸 뒤 기준
인구 재현 실패로 종료됐다. S II/III의 세 BF 흐름으로 확장한 `1245099`는
출력 직렬화 오류로 22초 뒤 종료해 그 흐름은 기록하지 못했다. 이는
진단 스크립트 오류이지 생산 계산 오류로 분류하지 않는다. 직렬화 코드만
정정한 현재 미완 스크립트 SHA256은
`b2d25dcccbc80955c1635d61a2766ca4feb911447a9501f1745a986e810eddda`이며
문법 확인만 했고 정정 후 실제 재실행은 하지 않았다.
