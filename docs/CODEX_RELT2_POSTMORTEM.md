# relT2 오프라인 부검 보고서

## 판정 요약

- **종료는 `NUM_ITS` 소진이 아니라 강제 종료**다. 60회 설정 중 great-iteration 41–54만 완료하고 55의 STEQ 구성 중 끊겼으며, Slurm은 `Killed`, wrapper는 `CMFGEN_EXIT=137`을 기록했다. kill의 주체와 원인(OOM·관리자·사용자 등)은 **UNRESOLVED**다. [IN_ITS](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/IN_ITS:1), [OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:611), [Slurm out](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/seq_logs/modern1948_slurm-397936.out:2), [batch.log](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/batch.log:1)
- **발산은 it54에서 시작된 것이 아니다.** continuation 첫 full step인 it41부터 반환 `MAXCH=1.00×10^7%`, depth 21 증가 `8.31×10^6%`였고, it42–43에서 `1.12×10^8 → 2.35×10^9%`로 악화됐다. [OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:136)
- it54의 직접 변수는 **Si III, superlevel 97, STEQ 166, `3s10g3Ge`, depth 21**이다. 보정값 `−234.95`가 `+2.3495×10^4%` 인구 증가로 출력됐다. [CORRECTION_LINK](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/CORRECTION_LINK:12), [LEVEL_SL_STEQ_LINKS](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/LEVEL_SL_STEQ_LINKS:271), [STEQ_VALS](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/STEQ_VALS:674806)
- 385770과는 **엄밀한 동형 실패가 아니다.** 385770의 NaN 직전 폭주는 depth 25의 **Si V `2p5_4d_1Po`**, relT2는 depth 21의 **Si III `3s10g3Ge`**다. 다만 둘 다 고정-T LAMBDA 인구 보정이 외곽의 Si 고이온/고준위에서 폭주했다는 계열 유사성은 있다. [385770 STEQ_VALS](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_conv/STEQ_VALS:258144), [385770 level map](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_conv/LEVEL_SL_STEQ_LINKS:375)
- **“modern은 수렴점 근방이 아니라 불안정 궤적 위였다”는 가설은 지지된다.** modern 종점의 실제 반환 MAXCH는 `3.46×10^3%`가 아니라 `1.00×10^7%`였고, 49개 변수가 depth 9에서 이미 100% 이상 보정 대상이었다. 그 continuation의 첫 full step은 즉시 `MAXCH=10^7%`로 폭주했다. [modern OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/OUTGEN:3093), [modern CORRECTION_SUM](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/CORRECTION_SUM:14), [relT2 OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:139)
- 처방 순위는 **(b) 인구 trust-region > (a) 장기 LAMBDA 연마 > (d) fixed-T 조건부 앵커 > (c) 더 이른 SN_HYDRO 재시작**이다.

---

## 1. 종료 방식

파일상 시작은 `2026-07-31 21:59:26`, 종료 기록은 `2026-08-01 04:23:39`다. Slurm 출력에는 executable이 `Killed`됐고 exit 137이 기록됐다. 별도 `#SBATCH --error` 지시 없이 `--output`만 정의돼 있으며 kill 메시지는 그 `.out`에 들어 있다. [Slurm script](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/slurm_cmfgen_modern.sh:2), [Slurm out](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/seq_logs/modern1948_slurm-397936.out:1)

`OUTGEN`은 it54 결과를 완전히 쓴 뒤 it55에서 `Zeroing BA matrices → Call STEQ routines → T 15 F`까지만 기록하고 끝난다. 정상 종료 footer나 it55 correction/MAXCH는 없다. [OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:600), [OUTGEN 끝](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:611)

따라서:

- `NUM_ITS=60` 소진: **아님**. [MODEL echo](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/MODEL:50)
- 완료 iteration: 41–54, 총 14회.
- it55: 진입했으나 미완료.
- 종료 분류: **외부 kill/비정상 종료**.
- `137`의 구체 원인: 로그에 OOM 등의 명시가 없어 **UNRESOLVED**. 요청 메모리는 200 GB였다. [Slurm script](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/slurm_cmfgen_modern.sh:7)

---

## 2. 발산 연대기

표의 `MAXCH`는 화면의 “Maximum % increase”가 아니라 `SOLVEBA_V13` 반환값이다. 99.999% 이상 감소는 반환 MAXCH에서 `10^7%`로 치환된다. [solveba_v13.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:201)

| great-it | 방식 / T | 최대 증가 | 최대 감소 | 반환 MAXCH |
|---:|---|---:|---:|---:|
| 42 | LAMBDA / fixed | d45, `1.12e8%` | d11, `1.09e2%` | `1.116342655e8%` [근거](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:170) |
| 43 | LAMBDA / fixed | d45, `2.35e9%` | d1, `1.10e2%` | `2.346375558e9%` [근거](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:221) |
| 44 | LAMBDA / fixed | d46, `3.74e8%` | d4, `1.00e2%` | `3.736424467e8%` [근거](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:254) |
| 45 | LAMBDA / fixed | d10, `4.58e5%` | d41, `1.00e2%` | `6.145742e6%` [근거](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:286) |
| 46 | LAMBDA / fixed | d25, `7.18e4%` | d34, `1.04e2%` | `1.00e7%` [근거](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:322) |
| 47 | LAMBDA / fixed | d45, `2.50e8%` | d9, `1.01e2%` | `2.502965782e8%` [근거](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:354) |
| 48 | LAMBDA / fixed | d48, `5.17e4%` | d15, `99.8%` | `5.316126e4%` [근거](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:386) |
| 49 | full/BA / fixed | d31, `3.46e7%` | d31, `5.15e5%` | `3.456653e7%` [근거](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:435) |
| 50 | LAMBDA / fixed | d30, `2.57e5%` | d11, `1.02e2%` | `1.00e7%` [근거](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:466) |
| 51 | LAMBDA / fixed | d33, `1.61e5%` | d10, `1.04e2%` | `1.00e7%` [근거](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:498) |
| 52 | LAMBDA / fixed | d37, `4.14e5%` | d33, `1.04e2%` | `1.00e7%` [근거](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:534) |
| 53 | LAMBDA / fixed | d36, `3.81e3%` | d38, `1.01e2%` | `1.00e7%` [근거](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:566) |
| 54 | LAMBDA / fixed | d21, `2.35e4%` | d25, `99.5%` | `2.349524e4%` [근거](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:598) |
| 55 | 미완료 | **UNRESOLVED** | **UNRESOLVED** | **UNRESOLVED**; STEQ 진입 직후 종료 [근거](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:611) |

악화 시작점은 표의 it42가 아니라 **첫 continuation full step it41**이다. great counter 41 승계가 직접 기록됐고, 그 step에서 depth 21 증가 `8.31×10^6%`, 반환 `10^7%`가 나왔다. [OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:110), [it41 correction](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:139)

auto-LAMBDA는 **it42부터** 걸렸다. 입력은 `DO_LAM_IT=F`, `DO_LAM_AUTO=T`, `LAM_VAL=400`, `NUM_LAM=2`였고, it41 MAXCH가 400보다 훨씬 컸다. [IN_ITS](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/IN_ITS:2), [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/MODEL:326) 소스는 full step의 MAXCH가 문턱 이상이면 다음 step을 LAMBDA/fixed-T로 전환하고, LAMBDA MAXCH가 `10^5%`보다 크면 예정 횟수를 넘어 LAMBDA를 지속한다. [solve_for_pops.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/mod_subs/solve_for_pops.f:285)

it48이 `5.32×10^4%`로 `10^5%` 아래로 내려온 뒤 it49 full이 재시도됐지만, 곧바로 `3.46×10^7%`로 재폭주했다. 이에 it50부터 다시 LAMBDA가 적용됐다. [OUTGEN it48](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:389), [OUTGEN it49](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:435)

---

## 3. it54 변수와 depth 21/25 조건

### 직접 변수

`CORRECTION_LINK`의 depth-21 최대 증가 항은:

```text
-2.3495E+02  SkIII  SL=97  I(STEQ)=166
```

이다. [CORRECTION_LINK](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/CORRECTION_LINK:12)

`SkIII`은 원자번호 14, 질량 28.1의 Si III이고, MODEL의 해당 데이터 블록도 `Energy levels ... Si III`로 명시한다. [MODEL species](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/MODEL:16), [MODEL Si III](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/MODEL:492)

준위 매핑은:

- ion: Si III
- physical level: 145
- superlevel: 97
- STEQ equation: 166
- label: `3s10g3Ge`

이다. [LEVEL_SL_STEQ_LINKS](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/LEVEL_SL_STEQ_LINKS:271)

최종 solution 배열도 equation 166의 depth 21 성분을 `−2.3495×10^2`로 기록한다. 음수 correction은 인구 증가를 뜻하므로 출력의 `+2.3495×10^4%`와 일치한다. [STEQ_VALS](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/STEQ_VALS:674806), [부호 정의](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:305)

depth 25의 `99.5%` 감소는 **Co IV SL13, STEQ 1393, `3d5(6S)4s_5Se[2]`**이다. [CORRECTION_LINK](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/CORRECTION_LINK:20), [level map](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/LEVEL_SL_STEQ_LINKS:16578), [STEQ_VALS](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/STEQ_VALS:676033)

단일 trace level만의 문제는 아니다. it54에는 depth 21에서 262개 변수가 100% 이상, 1,434개가 10% 이상 변했고, depth 25에서는 각각 227개와 1,461개였다. Si III/Co IV 항은 넓게 무너진 correction field의 극값이다. [CORRECTION_SUM](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/CORRECTION_SUM:26)

### 물리 조건

| depth | R | V | T | 전자밀도 `n_e` | 원자밀도 | 질량밀도 | Rosseland τ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 21 | `4.93408e15 cm` | `2.93159e4 km/s` | `1.02613e4 K` | `1.58247e6 cm⁻³` | `5.43276e5 cm⁻³` | `2.73532e−17 g/cm³` | `1.243e−3` |
| 25 | `4.52809e15 cm` | `2.69037e4 km/s` | `9.97430e3 K` | `3.42140e6 cm⁻³` | `1.25013e6 cm⁻³` | `6.29424e−17 g/cm³` | `1.915e−3` |

R·V는 [RVTJ radius](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/RVTJ:13)와 [velocity](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/RVTJ:26), T·`n_e`는 [RVTJ 전자밀도](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/RVTJ:52)와 [temperature](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/RVTJ:65), 밀도는 [atom density](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/RVTJ:195)와 [mass density](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/RVTJ:221)에서 읽었다. 이번 런에서 재생성된 MEANOPAC도 depth 21/25의 R·V와 `τRoss=1.243×10⁻³/1.915×10⁻³`를 확인한다. [MEANOPAC d21](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/MEANOPAC:22), [MEANOPAC d25](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/MEANOPAC:26)

단, RVTJ의 완료 시각은 7월 30일로 relT2 시작 전이다. T는 전 iteration에서 고정됐으므로 앵커값으로 사용할 수 있지만, 인구와 함께 변할 수 있는 **최종 relT2 `n_e`는 crash 후 RVTJ가 다시 쓰이지 않아 UNRESOLVED**다. [RVTJ timestamp](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/RVTJ:2), [RVTJ fixed-T flag](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/RVTJ:11)

---

## 4. 385770과의 동형성

385770의 마지막 유한 correction은 it46 LAMBDA/fixed-T에서 depth 25 증가 `2.14×10^6%`, 반환 `10^7%`였다. [385770 OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_conv/OUTGEN:354)

그 극값은 STEQ 241의 depth 25 성분 `−2.1397×10^4`, 즉 **Si V SL23 `2p5_4d_1Po`**였다. [385770 STEQ_VALS](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_conv/STEQ_VALS:258144), [level map](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_conv/LEVEL_SL_STEQ_LINKS:375), [SkV=Si V](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_conv/MODEL:384)

NaN은 it49에서 처음 생긴 것이 아니다.

- it46: 유한하지만 Si V population correction 폭주. [OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_conv/OUTGEN:369)
- it47: luminosity가 처음 `NaN`; 이후 모든 depth에서 CMF solve가 실패한다. [OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_conv/OUTGEN:396), [depth failure](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_conv/OUTGEN:400)
- it48: luminosity와 spectrum change가 계속 NaN. [OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_conv/OUTGEN:797)
- it49: DTDR·opacity·luminosity가 NaN, grey solution 실패, BA 미계산. [OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_conv/OUTGEN:836)
- 최종 exit 2는 `wr_x_info.f`의 `(F*.3)` 포맷 오류다. [batch.log](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_conv/batch.log:1), [Slurm out](/gpfs/kjhan/cmfgen_runs/seq_logs/conv1948_slurm-385770.out:2)

따라서 판정은 다음과 같다.

- 같은 depth: **아님** — 25 대 21.
- 같은 변수: **아님** — Si V `2p5_4d_1Po` 대 Si III `3s10g3Ge`.
- 같은 종료 형태: **아님** — NaN 전파 후 Fortran error 대 유한값 상태에서 external kill.
- 공통 현상: 고정-T LAMBDA 단계에서 외곽 Si 고이온/고준위 population correction이 폭주.
- 두 폭주가 같은 Jacobian mode나 동일한 방사선장 feedback에 의한 것인지: condition number/eigenmode 자료가 없어 **UNRESOLVED**.

또한 385770의 전역 입력은 `FIX_T=F`였고, LAMBDA step 자체만 T를 고정했다. relT2처럼 런 전체를 fixed-T로 묶은 실험과 완전히 동일한 조건은 아니다. [385770 MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_conv/MODEL:306), [385770 it46](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_conv/OUTGEN:356)

---

## 5. modern 상태의 불안정성

modern it40의 `3.46×10^3%`는 “잔차”나 반환 MAXCH가 아니라 **가장 큰 양의 population correction**이다. 같은 step의 최대 감소는 `1.01×10^2%`, 반환 MAXCH는 `1.00×10^7%`다. [modern OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/OUTGEN:3091)

correction의 두 집중점은:

1. depth 9: Si III SL69, STEQ 138, `3s9s1Se`, solution `−34.603` → `+3.4603×10^3%`. [CORRECTION_LINK](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/CORRECTION_LINK:12), [level map](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/LEVEL_SL_STEQ_LINKS:243), [solution](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/STEQ_VALS:1947490)
2. depth 22: terminal nickel balance `NkSEV`, STEQ 1798, solution `+1.0064`, 즉 100%를 넘는 감소. 이것이 반환 MAXCH를 `10^7%`로 만든 항이다. [CORRECTION_LINK](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/CORRECTION_LINK:20), [solution](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/STEQ_VALS:1952752)

마지막 raw STEQ에서 Si III local equation 69의 depth 1–10 값은 depth 9에서 `7.7898×10⁻4`로 그 구간 최대이며 correction 위치와 일치한다. 출력 루틴은 한 행에 연속 depth 10개를 기록한다. [raw STEQ](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/STEQ_VALS:1914640), [wr_asci_steq.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/wr_asci_steq.f:21)

반면 nickel terminal equation의 depth-22 raw residual은 `−6.9527×10⁻56`인데 solution correction은 `+1.0064`다. 즉 문제는 단순히 “큰 절대 residual 하나”가 아니라, 작은/trace-population 방정식이 선형 solve에서 order-unity correction으로 증폭되는 민감한 방향에 있다. 정확한 Jacobian condition number는 출력되지 않아 **UNRESOLVED**다. [raw terminal row](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/STEQ_VALS:1920043), [solution](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/STEQ_VALS:1952752)

또한 modern 종점에는 depth 9에서 49개 변수가 100% 이상, 1,280개가 10% 이상 변했고, depth 22에서도 각각 39개와 1,611개였다. 고립된 한 준위의 오차라고 보기 어렵다. [CORRECTION_SUM](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/CORRECTION_SUM:14)

별도 radiative-equilibrium diagnostic도 안쪽에서 `|6.19×10^6|`까지 남아 있지만, fixed-T 상태라 electron-energy equation은 “not computed”와 0으로 기록됐다. 따라서 이것은 활성 수렴 norm이 아니라, 해당 snapshot이 열적 coupled solution이 아님을 보여주는 보조 근거다. [STEQ_VALS](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/STEQ_VALS:1914473), [energy equation](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/STEQ_VALS:1914485)

종합하면 가설은 **지지**된다.

- 종료 문턱은 `EPS_TERM=0.1%`인데 실제 반환 MAXCH는 `10^7%`였다. [modern MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/MODEL:312), [modern OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/OUTGEN:3096)
- 마지막 LAMBDA correction은 depth-9 Si III 고준위와 depth-22 terminal balance에 집중됐다.
- 그 true continuation의 첫 full step it41은 즉시 depth-21 `8.31×10^6%`, MAXCH `10^7%`로 폭주했다. [relT2 OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:139)
- 이후 유일한 full 재시도 it49도 `3.46×10^7%`로 실패했다. [relT2 OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:435)

다만 이를 특정 불안정 고유모드나 음의 Jacobian eigenvalue로 확정하는 것은 **UNRESOLVED**다.

---

## 6. 처방 후보 순위

### 1위 — (b) T_e/인구 언더릴랙세이션, 특히 population trust region

직접 병목은 T가 아니라 population correction이다. 모든 완료 step에서 T가 고정됐는데도 correction은 `10^3–10^9%`였고, 기존 `MAX_LIN=MAX_LAM=10` 제한도 이를 안정화하지 못했다. [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/MODEL:308), [MAX limits](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/MODEL:315)

설계상 우선순위는:

- trace/high-n population에 대한 `Δln n` 또는 fractional-change trust radius,
- residual이 악화되면 correction reject/축소,
- 감소가 100%에 접근하는 terminal-ion row의 별도 floor/스케일링,
- population 안정화 뒤에만 T를 해제하고 기존 `MAX_dT=0.05` 적용

이다. 현재 `MAX_dT`는 T correction에만 작용하므로 fixed-T 발산에는 효력이 없었다. [solveba_v13.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/solveba_v13.f:311), [MODEL MAX_dT](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/MODEL:317)

### 2위 — (a) LAMBDA 전용 장기 연마 후 점진 full 전환

modern 말단의 증가 극값은 `4.43e4 → 1.38e4 → 8.98e3 → 3.46e3%`로 내려갔으므로 장기 LAMBDA 연마에는 일부 경험적 근거가 있다. [modern OUTGEN it37](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/OUTGEN:2972), [it40](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/OUTGEN:3091)

그러나 반환 MAXCH는 계속 `10^7` 또는 `6.54×10^6%`였고, relT2에서도 LAMBDA가 it53 `3.81e3%`까지 내려갔다가 it54 `2.35e4%`로 역행했다. [modern it39](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/OUTGEN:3052), [relT2 it53–54](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/OUTGEN:566)

따라서 단순 “N회 후 full”이 아니라 반환 MAXCH·100% 초과 변수 수·Si III/terminal-ion correction이 여러 step 연속 감소할 때만 작은 full correction을 허용하는 방식이어야 한다. 기존 `NUM_LAM=2` 자동 순환은 이 안정성 조건을 보장하지 않았다. [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/MODEL:326)

### 3위 — (d) fixed-T 앵커 영구 채택 + 민감도 오차봉

이것은 수렴 처방이 아니라 **조건부 과학 산출물의 fallback**이다. fixed-T 상태에서도 population solve가 발산했으므로 현재 snapshot을 “수렴한 모델”이라고 부를 수는 없다. [relT2 correction summary](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_relT2/CORRECTION_SUM:26)

채택한다면 결과를 “주어진 T profile과 선택한 population snapshot에 조건부”라고 명시하고, T-profile·population damping·atom-set 변화에 대한 스펙트럼 변동을 정식 오차봉으로 보고해야 한다. 열평형 방정식이 계산되지 않았다는 점도 함께 공개해야 한다. [modern STEQ energy block](/gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern/STEQ_VALS:1914485)

### 4위 — (c) 더 이른 SN_HYDRO 재시작점

현재 파일에는 어느 이전 checkpoint가 안정한 population/radiation state를 갖는지 보여주는 증거가 없다. SN_HYDRO만 더 이르게 되돌리고 호환 가능한 SCRTEMP/POINT/EDDFACTOR를 갖추지 못하면, 선행 relT1에서 확인된 LTE cold-start 문제를 다시 만들 가능성이 있다. [선행 부검](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_RELT_POSTMORTEM.md:78)

선행 부검도 restart 삭제 후 immediate forced-LAMBDA cold start를 relT1 실패의 핵심으로 판정했다. [선행 결론](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_RELT_POSTMORTEM.md:84), [기전](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_RELT_POSTMORTEM.md:96) 따라서 이 안은 “더 이른 시점” 자체가 아니라 **완전하고 구조적으로 호환되는 이전 population/transfer checkpoint가 실제로 존재할 때만** 검토할 수 있으며, 현재는 그 존재와 우월성이 **UNRESOLVED**다.

신규 계산·제출·파일 수정은 수행하지 않았다.