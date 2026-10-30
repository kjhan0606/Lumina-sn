# 같은 주파수의 formal 입력과 MOM 반환 상태

## 판독 결과와 해석 범위

기존 78회 이전 주파수 진입 포획과 완료된 단일 MOM 재생을 읽었다.
주파수는 양쪽 모두 `6.107497416899375e15 Hz`다. 입력의 formal
`JNU_STORE/HNU_STORE/KNU_STORE/NNU_STORE` 90점 배열은 재생 반환까지
바이트가 아니라 배열 원소별로 완전히 동일하다. 새 수송·물질 갱신은 없다.

formal 모멘트는 기존 `angular_support()`의 두 각도 지지 행렬 검사에서
최소 고유값 `+0.0105095485`이며 `max(H²/(J*K))=0.949544915`다.
이는 저장된 formal 모멘트가 해당 필요조건을 통과한다는 뜻이지, formal
수송이나 CMFGEN 전체가 수렴했다는 증거는 아니다.

같은 원점에서 formal J를 native MOM의 `γr²J` 정규화로 변환했다.

| 원점, 1부터 | formal `γr²J` | MOM 반환 `γr²J` | 반환/formal |
| ---: | ---: | ---: | ---: |
| 1 | 35595.6109063 | 35595.6109063 | 1 |
| 2 | 35599.8052537 | 35599.8052537 | 1 |
| 3 | 35565.1539161 | 276.6559705 | 0.00777884924 |
| 4 | 35533.8233602 | 481.4712019 | 0.01354965935 |
| 5 | 35488.6335859 | 2296.3784578 | 0.06470743519 |

첫 두 값의 일치는 앞서 확인한 음수 raw J의 `FORM_VAL` 대체와 일치한다.
나머지 반환값까지 formal 해와 같다는 뜻은 아니다. 반환 J와 저장 closure의
경계 K를 결합한 보수적 필요조건 `H²≤max(J_left,J_right)*K`는 경계
3–7에서 깨지고 최대 비는 `7.598813802`다. 이 계산에서는 배열 끝의
비경계 H 원소를 제외하고 실제 89개 경계만 사용했다.

따라서 **현재 주파수의 저장 formal 모멘트 자체가 이미 같은 비물리성을
가지고 있다는 설명은 지지되지 않는다.** MOM 계산·closure 연결과 이미
불일치한 이전 주파수 상태의 전달을 함께 조사해야 한다. MOM 갱신 전후의
차이만으로 formal을 정답으로 채택하거나 최초 실패 지점을 확정하지 않는다.
재생은 원래 실패 이력과 가까운 독립 실행이지 비트 동일한 전체 이력이
아니다. 그 범위는 [기존 재생 기록](CMFGEN_MOM78_FIRST_FAILURE_2026-10-30.md)에
기술돼 있다.

## 음수 불투명도와 비선형 이류 계수의 구분

이 반환 snapshot의 `CHI`, `CHI_H`, `CHI_J`, `ETA`, `SOURCE`는 전점
양수다. 그 최솟값은 각각 `2.06035417e-9`, `6.02399489e-7`,
`2.00233066e-7`, `1.10800366e-18`, `1.10513742e-12`다. 이것은 이번
호출의 음수 불투명도·방출률 가설을 제한하며, 더 앞선 이력이나 원자자료
전체의 정확성을 증명하지는 않는다.

원 소스는 내부 반복마다 `P_J=1+(CON_DELTA/CHI_J)*dln(γr²J)/dlnR`를
갱신한다. 원래 `LINMON` 분기로 반환 J에서 다시 계산한 P_J는 저장값과
최대 상대차 `3.69e-8` 이내다. 같은 식에 formal J를 넣으면 다음과 같다.

| 원점 | 저장 P_J | 반환 J로 재계산 | formal J를 넣은 조건부 값 |
| ---: | ---: | ---: | ---: |
| 4 | −2080.43757 | −2080.43761 | +3.12392792 |
| 5 | −1454.80451 | −1454.80451 | +2.89259555 |
| 6 | −473.611812 | −473.611812 | +2.87829386 |
| 7 | −96.6857572 | −96.6857572 | +2.96046435 |

이는 반환 J의 급한 공간 변화가 다음 내부 반복의 비선형 계수에도 반영되는
경로를 확인한다. P_J는 불투명도가 아니며 음수라는 이유만으로 오류라고
판정하거나 floor를 적용하지 않는다. formal을 넣은 열도 원식의 새로운
해가 아니라 같은 도함수 식의 조건부 평가다. 원점1의 formal P_J도
음수이지만 `HONJ` 외부 경계행은 P_J를 사용하지 않으므로 별도로 구분한다.

## 입력 신원과 재현

공통 실행 자료 루트를 `CMFGEN_RUNS_ROOT`로 지정한다. 이번 판독은 작은
snapshot 두 개와 inventory만 읽으며, 파일 쓰기·행렬 풀이·새 실행은 없다.
원 소스 대조 위치는 entry build의 `mom_jrel_v9.f` 내부 반복, J 대체,
도함수 갱신 및 H 식이다. 판독 소스 기준 커밋은 `035e4f90`이다.

```bash
PYTHONPATH=scripts python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
import numpy as np
from analyze_cmfgen_mom_capture import capture
from analyze_cmfgen_initial_closure import angular_support
from solve_cmfgen_mom_kernel import linmon
root = Path(os.environ['CMFGEN_RUNS_ROOT'])
files = [
    ('lumina_mom78_blue_entry_20261030-OClXsFrM/build/state_inventory.json',
     '2c5cada170a9731c2870ae66f71c67ccb698a181731d9c3d6effb82a70610d83'),
    ('lumina_mom78_blue_job_20261030-sQ1UyRxn/run-parent/run/material/MOM_CAPTURE.bin',
     '9fece58cfcceb1077cd892fb8d7f83437fae5ec8e44a76cc623369b1133426f1'),
    ('lumina_mom78_blue_actual_replay_20261030-EQEupjhB/MOM_FINAL.bin',
     'd949a4e4bbf550bf352a57b84f42febc264b2c23aadf0d276ac3515765fbe43f')]
for name, expected in files:
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
inventory = json.loads((root/files[0][0]).read_text())
entry, final = [capture(root/name, inventory) for name, _ in files[1:]]
q, j, qf = entry['modules']['ray'], final['modules']['jrel'], final['modules']['ray']
v = lambda group, name: group[name]['values']
names = ('JNU_STORE', 'HNU_STORE', 'KNU_STORE', 'NNU_STORE')
formal = [v(q, name) for name in names]
assert entry['args']['FREQ'] == j['FREQ_SAVE']
assert all(np.array_equal(v(q, name), v(qf, name)) for name in names)
print('formal_support', angular_support(*formal))
print('formal_max_H2_JK', np.max(formal[1]**2/(formal[0]*formal[2])))
J = v(j, 'GAM_RSQJNU_SAVE'); H = v(j, 'GAM_RSQHNU_SAVE')[:-1]
K = v(qf, 'KMID_ON_J_SAVE')[:-1]*(J[:-1]+J[1:])
ratio = H**2/(np.maximum(J[:-1], J[1:])*K)
print('MOM_conservative_max', ratio.max(), 'edges', np.flatnonzero(ratio > 1)+1)
r = v(j, 'R'); f = v(j, 'GAM_REL')*r*r*formal[0]
for node in range(1, 6):
    i = node-1; print('J', node, f[i], J[i], J[i]/f[i])
factor = v(j, 'CON_DELTA')/v(j, 'CHI_J')
p = 1+factor*r*linmon(J, r)/J
pf = 1+factor*r*linmon(f, r)/f
scale = np.maximum(abs(v(j, 'P_J')), abs(p))
print('P_J_max_relative', np.max(np.divide(abs(v(j, 'P_J')-p), scale,
      out=np.zeros(len(p)), where=scale > 0)))
for node in range(4, 8):
    i = node-1; print('P_J', node, v(j, 'P_J')[i], p[i], pf[i])
for name in ('CHI', 'CHI_H', 'CHI_J', 'ETA', 'SOURCE'):
    print(name, v(j, name).min()); assert np.all(v(j, name) > 0)
PY
```

운전자 재현 상태는 `REPRODUCED`다. 고정장 물질응답 1243823은 22:55 KST
조회에서 실행 중이었으며 이 판독으로 중단·재시작·추가 수송을 하지 않았다.
Si/S 분포·스펙트럼·자기일관 CMFGEN 동질성의 최종 목표는 여전히 미달이다.
