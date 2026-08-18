# CMFGEN negative-Sobolev / radiative-equilibrium audit — 2026-08-09

## 판정

현재 Lumina A2-09의 line 식

```text
n_upper * A_ul * h * nu * beta(tau)
```

은 방출 진단 또는 transport 재료로는 유한한 의미가 있지만, CMFGEN의 line
radiative-equilibrium net term과 같은 물리량이 아니다. 특히 `tau < 0`에서
`beta=(1-exp(-tau))/tau`가 maser 증폭을 포함하므로, 이 양을 그대로 양의 cooling에
더한 endpoint residual은 CMFGEN과 비교할 수 없다.

CMFGEN Sobolev 경로의 열평형 line 항은 radiation field가 들어간 net bracket
`ZNET`을 구성한 뒤 `ETAL_MAT * ZNET`으로 `STEQ_T`를 갱신한다. 따라서 Lumina의
다음 생산 구현 단위는 standalone escape emissivity가 아니라 동일 세대의
population, signed opacity, line-resolved radiation field로 만든 net line rate다.

## CMFGEN 원문 증거

정본 source root:

```text
/gpfs/kjhan/cmfgen_src/cur_cmf
```

`subs/sobjbar_sim.f`:

- 158–177행: `EXPONX(GAM)`으로 escape probability를 계산하지만, net bracket은
  `T2=AV*CHIL/ETAL`, `T1=1-T2`, `ZNET_BL += T1*EX_VEC`로 radiation solution `AV`와
  결합한다.
- 187–202행: 각 simultaneous line의 `ZNET`과 그 opacity/emissivity 미분을 만든다.
- 209행: `JBAR=(1-ZNET_BL)*ETAL/CHIL`이다. 즉 `JBAR`와 `ZNET`은 같은 line
  radiation solution의 두 표현이다.

`new_main/cmfgen_sub.f`:

- 2476–2480행: rate equation에는
  `EINA*population*ZNET_SIM`, temperature equation에는
  `ETAL_MAT*ZNET_SIM`을 넣고 `STEQ_T -= T3`로 적용한다.
- 2769–2778행의 line-heating/cooling diagnostic도 같은
  `ETAL_MAT*ZNET_SIM`을 사용한다.

따라서 `n_upper*A_ul*h*nu*beta`만으로 CMFGEN line cooling을 재현했다고 말할 수
없다. `beta`는 net bracket의 한 구성요소이지 최종 net energy exchange가 아니다.

## CMFGEN RE 단위와 finite known answer

`new_main/cmfgen.f:121-122`의 `OPLIN=1e10*pi*e^2/(m_e*c)`와
`EMLIN=1e25*h/(4*pi)`, `web/full_descr.tex:489`의 `1e15 Hz` frequency unit,
cm^-3 number population을 함께 복원하면 다음 변환을 얻는다.

```text
q_line_internal = ETAL_MAT * ZNET
q_line_cgs = q_line_internal * 4*pi*1e-10  [erg cm^-3 s^-1]
ETAL_MAT * 1e-10 = integrated emissivity [erg cm^-3 s^-1 sr^-1]
```

`eval_temp_ddt_v2.f:237-248`와 `eval_adiabatic_v3.f:143-170`도 RE 내부
단위가 cgs보다 `1e10/(4*pi)`만큼 크다고 독립적으로 명시한다.

O-PHYS의 line 76887, depth 90 finite witness는 raw
`ETAL_MAT=3.657273805148772e11`, `ZNET=1.45662e-3`이며 raw cgs net은
`0.6694430052329409 erg cm^-3 s^-1`다. deck line scale `0.997943`을 적용한
실제 production comparison 값은 `0.6680659609711768 erg cm^-3 s^-1`다.
따라서 이번 비교는 near-zero 여부가 아니라 명시적인 finite cooling 재현을
요구한다.

fixture 정본:

```text
validation/a2_10/CMFGEN_LINE_NET_KNOWN_ANSWER_2026-08-09.json
schema = lumina-cmfgen-line-net-known-answer-v2
SHA256 = 5a967bbbf6f374c69c6ae5fd63d420d1fadc002c04ddf2fbbef24192a81951a0
```

## CMFGEN의 음의 opacity 처리

현재 O-PHYS benchmark deck은 다음을 명시한다.

```text
T          [CHK_L_POS]
SRCE_CHK   [NEG_OPAC_OPT]
```

`new_main/cmfgen_sub.f`의 Sobolev branch 3551–3580행은
`tau_sob < -0.5`이면 `NEG_OPACITY`를 세우고 합성 opacity `CHIL=1`, 개별
`CHIL_MAT=1/NUM_SIM_LINES`로 바꾸어 `SOBJBAR_SIM` 입력을 일관되게 만든다.
`new_main/mod_subs/sub_sob_line_v3.f` 153–162행과 230–236행은 이 depth에서
opacity에 대한 `ZNET/JBAR` variation을 0으로 만든다.

이것은 CMFGEN benchmark parity의 명시적 정책이다. 다만 Lumina A2-08의 signed
`tau` publication을 조용히 고치거나 0으로 clamp하라는 뜻은 아니다. 두 경로를
구분한다.

1. A2-08은 signed `tau`를 원형 그대로 발행한다.
2. 현 A2-10은 완전한 net-rate owner가 없으므로 `tau < 0`의 energy 소비를 모두
   fail-closed한다. 임의 tolerance, floor, clamp는 없다.
3. 향후 CMFGEN-parity lane은 sealed deck의 `CHK_L_POS/SRCE_CHK` 정책을 명시적으로
   재현한다.
4. 완전 maser lane을 열려면 saturation과 radiation feedback을 포함한 별도 물리
   계약이 필요하다. parity lane과 섞지 않는다.

CMFGEN의 `EXPONX` 자체는 `|X|<1e-3`에서 급수식을 써 작은 수 상쇄를 피한다
(`subs/exponx.f` 15–20행). 이는 사용자가 지적한 `큰 수-큰 수` roundoff 문제와
직접 관련된 좋은 선례지만, net-rate 항을 standalone emission으로 바꾸어도 된다는
근거는 아니다.

## job 251622이 드러낸 수치

run root:

```text
/gpfs/kjhan/lumina/det_convergence/det1234_20260809T022526Z_3e38b9cd0750
```

- lower/upper 각각 50개 endpoint ledger가 모두 finite였다.
- Fe III–IV shell 44의 element total exact-zero 경계는 lower, upper, geometric-mid
  세 trial에서 모두 exact-zero로 통과했다.
- endpoint sign은 bracket 35 shells(12–46), same-negative 12 shells(0–11),
  same-positive 3 shells(47–49)였다.
- geometric midpoint `22135.943621178667 K`에서 바깥 3 shells(47–49)은 interior
  sign change를 보였고, 안쪽 12 shells는 여전히 same-negative였다.
- lower shell 0의 signed-tau forensic은 line emission의
  `99.9997982385%`가 negative-tau cell에서 왔다. 최대 항은 Co II
  `tau=-23.4290436825`, `beta=6.3877781636e8`이었다.
- 이 flight는 model rc=1, wrapper `FAILED 70:0`, material publication 보존으로
  종료했다. finite endpoint 생성과 exact-zero repair는 실증했지만 CMFGEN-equivalent
  line energy를 실증하지 않았다.

## 적용한 안전 경계

생산 candidate가 A2-09 bundle을 만든 뒤 A2-10 ledger로 들어가기 직전에, 실제
A2-08 `A208_BLOCK_UNSUPPORTED` capability checker를 사용해 in-grid active line의
모든 negative `tau`를 검사한다.

- 하나라도 있으면 `RADEQ_SIGN_MISMATCH`로 private candidate를 폐기한다.
- 가장 음의 line/shell/Z/ion/level과 전체 negative line-shell 수를 기록한다.
- `tau`, population, opacity, emissivity byte를 바꾸지 않는다.
- A2-09 signed publication과 forensic은 유지하고, A2-10 energy use만 차단한다.

이는 최종 물리 구현이 아니라 잘못된 finite 값을 생산 결과로 오인하지 않게 하는
임시 safety closure다.

검증:

```text
focused selftests = PASS
D/K/Z/CP = 19/19, 7/7, 12/12, 4/4 PASS
gate log SHA256 = 606b9f4b80d46c4317f41b0865425bb979ce5509fbc47ab1f3a52230f63b007b
CUDA SHA256 = a54d2600542a53c002deaf15e1a172cce93df9e94ffc2e19fa8638f4aac218ae
CUDA targets = sm_80, sm_86, sm_90
```

## 다음 생산 계약

상세 정본은
`docs/CMFGEN_LINE_NET_DATA_CONTRACT_2026-08-09.md`다. 핵심은 다음과 같다.

1. 모든 BB-domain line을 담는 energy/radiation 집합 `Q_E`를 만들고 현재 population
   rate graph `Q_g`를 그 subset으로 보존한다. 중복 cache 대신 단일 `Q_E Jbar`
   cache를 continuum과 같은 transaction으로 발행한다.
2. component 경로는 `fma(-chi_int,Jbar,eta_int)`로 signed net을 먼저 만들고,
   emission/absorption을 별도 거대 합으로 만든 뒤 빼지 않는다.
3. MC standard error 또는 DET formal-solution bound보다 작은 net sign은
   `UNRESOLVED_CANCELLATION`으로 fail-closed한다. floor/clamp/jitter는 없다.
4. CMFGEN `tau<-0.5`의 `SRCE_CHK` effective material, `AV/Jbar` 의미와 O-PHYS
   `SCL_LN` scale을 재현한 뒤에만 A2-10 binned line owner를 교체한다.
5. CPU/CUDA gate와 동일 line/depth H200 finite 비교 뒤 DET master / lagged MC
   feedback의 coevolution generation barrier에 재연결한다.

## 입력 봉인

```text
3997846d5a4041def4388852e8d0c6711df4c162814dd0b64c033dc6611b1768  subs/exponx.f
dcaae6b4b3e1154acf2d4bb60cc2d1ccd4e51cc27f72b16d113d4ec2bdad8572  subs/sobjbar_sim.f
7e463675e5fbf90728698a96b3a57916fd751df48494c92326a21bd99248504b  new_main/mod_subs/sub_sob_line_v3.f
092f8526661b1f9a5eaeb7a875f07f55623f427ad07c383f2659f2fb67143374  new_main/cmfgen_sub.f
992fba38c8d786b880f345dd91b25103ad1028897c412a8d36bd281be8f7aa47  new_main/cmfgen.f
7f97c601c7b861efb0cf93bab41c943c26ac3388a83fef937e2d9684fb53f3af  new_main/subs/eval_temp_ddt_v2.f
b9c148098f009fb2594d97ada99a05ccc1d413f52384d877f6e76602db1a286a  new_main/subs/eval_adiabatic_v3.f
b670330b5411831649b675edc19bea787d0f1d47fdcf66a8e03e04014e83301d  web/full_descr.tex
874dd9b88d7b509068ad69dc03ff269ece12dad22efc099804089d90fb0d9651  toy06_19p48d_ophys/VADAT
```
