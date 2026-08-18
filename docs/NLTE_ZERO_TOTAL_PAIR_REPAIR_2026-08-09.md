# NLTE active-pair exact-zero total repair — 2026-08-09

## 판정

Job 251599에서 Si III–IV exact-generator GTH는 양 endpoint 모두 finite,
strictly-positive 정상상태를 재현했다. 그 뒤 Fe III–IV에서 보인
`RANK_INCOMPLETE`, rank `102/202`는 rate topology나 GTH 결손이 아니다.

Job 251601의 lower/upper Fe III–IV shell 44 덤프는 모두 다음을 만족한다.

- N=`202` (`SUPER_CUTOFF=100`, Fe III/IV 각 101 state)
- prelock RHS nonzero=`0`
- single-total normalization RHS=`0.0` exact
- negative off-diagonal=`0`
- directed edges=`3082`
- SCC=`1×202`, closed class=`1`
- zero incoming/outgoing state=`0/0`

비음수 population과 `sum_i n_i = 0`을 함께 요구하면 물리적 해는 모든 state의
exact zero 하나뿐이다. 영 RHS homogeneous generator를 일반 dense rank test로
보낸 것이 `rank 102`의 정체다.

## GTH 반증 시험

동일 lower pre-generator를 production C GTH에 직접 넣었다.

| normalization total | rc | finite positive states | residual |
|---:|---:|---:|---:|
| `1` | 0 | 202/202 | `7.79310569895104e-17` |
| `1e20` | 0 | 202/202 | `8.764088001974752e-17` |

total=`1`의 min/max는 `4.235764078220075e-15` / `0.9204718511819489`다.
따라서 generator kernel과 연결성은 정상이다.

## Production 수리

`nlte_zero_total_pair_exact_zero()`를 pair layout 검증 뒤, matrix allocation/assembly
전에 둔다.

- `conservation_total == 0.0`인 경우에만 해당 shell의 pair full-level population과
  두 ion density를 정확히 0으로 쓴다.
- tolerance, clamp, floor, jitter가 없다.
- 양의 subnormal `1e-300`은 exact zero로 분류하지 않고 기존 solve 경로에 남긴다.
- 음수·비유한 density, 누락 ion slot, 비어 있거나 잘못된 level layout은 fail-closed다.
- private A2-10 candidate 안에서 실행되므로 실패 trial은 공개 state를 오염시키지 않는다.
- 기존 Z-inert 경계와 overlapping-pair 저장/복원 순서는 바꾸지 않는다.

표적 forensic gate에서는 다음을 남긴다.

```text
[A2-07][PAIR-EXACT-ZERO] Z=26 ion=2 shell=44 ... total=0
```

## 비채택안

- Cholesky+jitter: 보존 generator의 의도된 영 고유값을 양수로 이동시켜 다른 문제를
  푼다. 비대칭 rate matrix에도 직접 적용할 수 없다.
- `L.T @ x; dot(y,y)`: PSD quadratic form 검증에는 유효하지만 정상상태 선형계의
  보존 제약을 대체하지 않는다.
- elementwise `exp(A)` 또는 log-space matrix: 부호 있는 generator/보존 합을
  보존하지 않으며 overflow/underflow와 조건수 문제를 바꿔 쓸 뿐이다.
- 음수 결과 clamp: 큰 수 상쇄의 원인을 숨기고 population 총량·rate residual을
  깨뜨린다.

## 증거

- run root:
  `/gpfs/kjhan/lumina/det_convergence/det1234_20260809T020237Z_a403d49d8f61`
- lower JSON SHA:
  `c8828dd7a2fd45454e44efcfbbea958634e1b8016bb74be093dd10bbcdeb75b2`
- upper JSON SHA:
  `dd25e877f039ac5cd099d051595020e5c4d07408f51c4911049e2665b8ee261b`
- repaired full-gate log SHA:
  `61d123ab23a0808feeb1023ce540c225cbf931c227d2908a250ffca623a0ee80`
- repaired CUDA SHA:
  `3e38b9cd0750d4d36e1fac0b94c39b94bb84eca029522aa42c305360a7c26a98`
- H200 production verification: job 251622 (실행 중)

이 exact-zero 셀은 CMFGEN finite 물리값 재현 증거가 아니다. finite 재현 증거는
별도로 보존된 CMFGEN `*PRRR` 23-cell Γ 비교와 job 251599의 positive Si III–IV
population이다.

## Stage-IV ownership 한계

현재 adjacent overlapping pair는 `single-total` 조건에서도 `pair_shares_slot`이면
최종 full-level 저장 때 각 이온을 upstream `ion_number_density` 총량으로 rescale한다.
뒤 `(III,IV)` pair의 lower-III block도 앞 `(II,III)` pair가 쓴 값으로 복원된다.
따라서 GTH 벡터는 pair 내부 정상상태의 양수성과 level shape를 증명하지만, 현 구조가
II/III/IV 전체 ion fraction을 하나의 rate system으로 결정한다는 증거는 아니다.

이번 Fe exact-zero 셀은 upstream Fe III와 Fe IV 총량이 모두 정확히 0이므로 이
한계와 모순되지 않는다. finite Stage-IV ion partition의 최종 폐합은 shared stage의
단일 final owner를 정하고 multi-stage/element-wide rate solve로 검증해야 한다.
