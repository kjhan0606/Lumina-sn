# NLTE 생성자 GTH 수리 판정 — 2026-08-09

## 결론

Stage-IV Si III–IV lower endpoint의 미세 음수 population은 물리적인 음수가
아니다. 조립된 전이율의 큰 outflow 대각합이 float64에서 반올림되고, 그 행렬을
일반 LU로 푸는 과정에서 forward sign이 깨진 것이다.

생산 수리는 population을 자르거나 `exp(z)`로 양수화하지 않는다. 단일 원소총량으로
닫힌 irreducible continuous-time rate generator에 한해, nonnegative
off-diagonal 전이율만 사용하는 long-double GTH 정상상태 해법을 적용한다.

## Cholesky 또는 지수변환을 쓰지 않은 이유

- preconstraint SE 행렬은 대칭 positive-definite 행렬이 아니라 열합이 0인
  비대칭 singular generator다. 한 conservation row를 넣은 뒤에도 일반적으로
  SPD가 아니므로 Cholesky의 적용 대상이 아니다.
- `A=L L^T`나 `q=||L^T x||^2`는 SPD quadratic form의 비음수성을 검사하는
  도구이지, 비대칭 정상상태 방정식의 population을 구하는 해법이 아니다.
- `n=exp(z)`는 임의의 일반 선형계가 요구하는 음수까지 숨길 수 있어 부적합성을
  검출하지 못한다.
- `exp(tQ)`/uniformization은 유효 generator의 독립 진단에는 유용하지만, 생산
  정상상태를 얻기 위해 큰 `t`까지 반복할 필요가 없다. GTH가 같은 generator
  정상상태를 직접 계산한다.

## 실측 판정

두 행렬 모두 `N=162`, `n_lo=101`, `Z=14`, lower ion0=2, shell=4이다.

| endpoint | generator topology | 일반/고정밀 post 해 | exact-generator 80자리 및 C GTH |
|---|---|---|---|
| 3500 K | 1 SCC, 1 closed class, 3666 edges | 80자리 음수 11, min `-1.2695790142e-12` | 음수 0, min `1.4276152500892796e-16`; III/IV=`26458377.684286512/974287.3570324968` |
| 140000 K | 1 SCC, 1 closed class, 3666 edges | 음수 0, min `3.2013992859311626e-18` | 음수 0, min `3.2013992855783896e-18`; III/IV=`1.7119317692731112e-6/1.4283550387855277e-5` |

실제 dump를 새 C GTH 커널에 넣은 exact-generator 성분별 잔차는 lower
`6.898489835600487e-17`, upper `7.785232791286191e-17`이다. 두 C 결과는
80자리 exact-generator 해와 출력 정밀도까지 일치한다.

## 생산 적용 조건

GTH는 아래를 모두 만족할 때만 사용한다.

1. preconstraint RHS가 모두 정확히 0이다.
2. off-diagonal이 전부 finite하고 0 이상이며 diagonal이 0 이하이다.
3. float64 diagonal과 long-double off-diagonal outflow 합의 상대 불일치가
   `1e-12` 이하이다.
4. postconstraint 행렬에는 positive RHS를 가진 all-ones normalization row가
   정확히 하나뿐이다.
5. GTH state reduction이 유일한 strictly-positive 정상상태를 만들고,
   exact-generator 성분별 잔차가 `1e-12` 이하이다.

음의 off-diagonal, time-dependent/nonzero RHS, two-stage lock, b-space transform,
anchor/pin 행은 GTH 자격이 없다. 비생성자는 기존 general dense solve로 간다.
generator로 인정된 뒤 multiple closed class 또는 수치 실패가 나타나면 fallback하지
않고 fail-closed한다.

## OpenMP assembly-status 경합

기존 CPU shell solve는 assembly 전후의 전역 `population_error_count`를 비교했다.
50 shell이 OpenMP로 동시에 도는 동안 다른 shell이 이 counter를 올리면, 정상적으로
조립된 target shell도 `ASSEMBLY_FAILED`로 잘못 분류됐다.

assembler는 실제 local 실패를 return code로 이미 전달한다. 따라서 local assembly
판정은 이 return code만 사용하도록 바꿨고, 전역 counter snapshot 비교를 제거했다.
job 251596의 lower/upper 행렬 파일이 완전히 생성됐는데 두 target이 모두
`ASSEMBLY_FAILED`로 표시된 현상과 일치한다.

## 봉인 자산

- job 251596 run root:
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260809T005908Z_91dfc87a9c45`
- lower pre/post SHA:
  `c5362106fb60d74a7d9c759fea9ba6b510084a0f1b40e1c9ced78c99c4e37a20`,
  `75660481e8e2ad04463aff29f1b8d0849f830794b964be9621f8b9ba8f7f65f2`
- upper pre/post SHA:
  `030daedbb55a85a34784555baefcffdffb7408650f7940dc0357609f3da8e8fb`,
  `f8b909abc83b62e7e4961f753a60e28ce8eb0062de5851cbb1db336d60a2c412`
- lower/upper JSON SHA:
  `81e25bc9d710d87ee6099d0b06240ef7422ec258f89785e77609788714ea6966`,
  `0980ffd8dc0821bf9f03a6870ddaa5aa8d654ea8387cb346de79cefe2e0209b2`
- repaired CUDA SHA:
  `a403d49d8f610b7a6d1f94999209b3c3eb01e3b38823aa6370b12f7d5e35c32b`
- repaired full-gate log SHA:
  `2947c8b0e86ed96dd2c12030f6d82c57ac3ef8808f24b802b3522c494271bf5b`
- repair flight: job 251599, same active deck/sigma/single-total/Stage-IV,
  run root
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260809T013602Z_a403d49d8f61`

Job 251599는 lower/upper 모두 생산 GTH로 strictly-positive 정상상태를 재현했다.
lower/upper 최소 population은 각각 `1.4276152500892798e-16`,
`3.2013992855783896e-18 cm^-3`, exact-generator residual은 각각
`6.898489835600487e-17`, `7.7852327912861914e-17`이다. 이전 Si 미세 음수와
OpenMP 허위 `ASSEMBLY_FAILED`는 양 endpoint에서 모두 소실했다. 뒤이어 보인
Fe III-IV `RANK_INCOMPLETE`(rank `102/202`)는 후속 job 251601의 실제 행렬로
재분류했다. 여기서 202는 runtime `SUPER_CUTOFF=100` 투영으로 Fe III/IV가
각각 101 state인 경우다. lower/upper shell 44 모두 pair total과 RHS가 정확히
0이고, rate graph는 3082 directed edge, 단일 202-state SCC와 단일 closed
class를 갖는다. 즉 Fe rate topology나 GTH 결손이 아니라 exact-zero total을
일반 dense solve로 보낸 경계 오류다. production은 exact equality에서만 두
stage를 0으로 반환하도록 수리했으며 positive subnormal은 solve에 남긴다.

## 아직 닫히지 않은 물리 문제

GTH는 주어진 rate generator의 정상상태를 정확하고 양수로 구하는 수치 수리다.
Fe/Co/Ni V–VI collision 근사식 부재, overlapping II–III/III–IV pair의 최종
stage ownership, upper/midpoint에서 LTE-unmapped 고이온 선이 지배하는 문제,
shell 0–11의 동일부호 온도 residual은 별도의 물리 폐합 과제로 남는다.
