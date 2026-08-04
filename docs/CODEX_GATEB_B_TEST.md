## Gate B Oracle Phase 1 테스트 보고

소스 수정 없이 CPU 빌드·실행 artifact만 생성했습니다. GPU 실행, 큐 제출, git 명령은 수행하지 않았습니다. CMFGEN 비교 수치에 대한 게이트 판정도 하지 않았습니다.

### 1. Oracle 빌드·결정성

| 명령 | exit | 결과 |
|---|---:|---|
| `make bench_frozen_oracle` | 0 | 기존 바이너리 up-to-date |
| `make -B bench_frozen_oracle` | 0 | 강제 재빌드 완료; 기존 compiler warning만 발생 |
| `./bench_frozen_oracle … /tmp/gateb_B_phase1.qCtpnL/run1` | 0 | s0/s8/s45 생성 |
| `./bench_frozen_oracle … /tmp/gateb_B_phase1.qCtpnL/run2` | 0 | s0/s8/s45 생성 |
| `sha256sum run1/*.csv run2/*.csv` | 0 | 각 셸 쌍별 동일 |
| `cmp …s0…` | 0 | byte-identical |
| `cmp …s8…` | 0 | byte-identical |
| `cmp …s45…` | 0 | byte-identical |

| 셸 | run1/run2 공통 SHA-256 |
|---|---|
| s0 | `e8c6e300b3ed68411ac6137e1f08605462283a04d785fc424ba6b37957a725cd` |
| s8 | `d63c7225012d177b249224d93b7a398e28354158d9e94ca88048a186ad18d8e0` |
| s45 | `8129bc1ab71039d2ea9081e5c1424c169126855576161edba3c7dc5dd7be03ee` |

CSV 완결성 검사 exit 0:

| 셸 | 행 | available | unavailable | 범주 |
|---|---:|---:|---:|---|
| s0 | 158 | 93 | 65 | bf 64, ff 5, bb 38, collisional 16, thermal 7, state 28 |
| s8 | 166 | 127 | 39 | bf 64, ff 5, bb 46, collisional 16, thermal 7, state 28 |
| s45 | 160 | 102 | 58 | bf 64, ff 5, bb 40, collisional 16, thermal 7, state 28 |

malformed 행, status 도메인 위반, unavailable note 누락은 모두 0건입니다.

### 2. CMFGEN 첫 대조표 — REPORT-ONLY

실행 명령:

```sh
PYTHONPYCACHEPREFIX=/tmp/gateb_B_phase1.qCtpnL/pycache \
python3 -m py_compile scripts/oracle_compare_cmfgen.py

PYTHONPYCACHEPREFIX=/tmp/gateb_B_phase1.qCtpnL/pycache \
python3 scripts/oracle_compare_cmfgen.py \
  --oracle-dir /tmp/gateb_B_phase1.qCtpnL/run1 \
  --model-dir data/tardis_reference_toy06_19p48d_sivcaiv \
  --cmfgen-dir /gpfs/kjhan/cmfgen_runs/toy06_19p48d_modern \
  --edd-dir /gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4 \
  --out-dir /tmp/gateb_B_phase1.qCtpnL/compare
```

두 명령 모두 exit 0입니다. 전체 484행 중 수치 대응 33행, `lumina_unavailable` 162행, CMFGEN 측 `unavailable` 289행입니다.

| 셸 | 분류 | 수량 | Z:stage | Lumina | CMFGEN | L/C | 앵커 |
|---:|---|---|---:|---:|---:|---:|---|
| s0 | state | T_e | 0:-1 | 2.120268337e+04 | 1.853588700e+04 | 1.143872067e+00 | RVTJ:Temperature (10^4K)*1e4 |
| s0 | state | n_e | 0:-1 | 4.621814000e+09 | 4.852872100e+09 | 9.523873502e-01 | RVTJ:Electron density |
| s0 | bf | Gamma_photoion_total | 14:1 | 0.000000000e+00 | 2.079745202e-02 | 0.000000000e+00 | Sk2PRRR:sum(PR)/Ion Density |
| s0 | bf | Gamma_photoion_total | 16:1 | 0.000000000e+00 | 1.274938583e-02 | 0.000000000e+00 | S2PRRR:sum(PR)/Ion Density |
| s0 | bf | Gamma_photoion_total | 26:1 | 8.330218723e+05 | 7.624200139e-03 | 1.092602315e+08 | Fe2PRRR:sum(PR)/Ion Density |
| s0 | bb | jbar_representative | 26:1 | 1.501866504e-03 | 9.379098742e-04 | 1.601290854e+00 | EDDFACTOR:1.258150482e+15 Hz |
| s0 | bb | jbar_representative | 26:2 | 1.793868012e-03 | 8.027568538e-04 | 2.234634315e+00 | EDDFACTOR:1.581590788e+15 Hz |
| s0 | bb | jbar_representative | 27:2 | 3.226170349e-04 | 9.231811683e-05 | 3.494623222e+00 | EDDFACTOR:3.192526570e+15 Hz |
| s8 | state | T_e | 0:-1 | 1.200772780e+04 | 1.032347600e+04 | 1.163147742e+00 | RVTJ:Temperature (10^4K)*1e4 |
| s8 | state | n_e | 0:-1 | 7.496125000e+08 | 7.341043400e+08 | 1.021125280e+00 | RVTJ:Electron density |
| s8 | bf | Gamma_photoion_total | 14:1 | 4.993938812e+01 | 1.968552105e-03 | 2.536858841e+04 | Sk2PRRR:sum(PR)/Ion Density |
| s8 | bb | jbar_representative | 14:1 | 2.034428000e-05 | 1.459557936e-05 | 1.393865875e+00 | EDDFACTOR:1.955062180e+15 Hz |
| s8 | bb | jbar_input_raw | 14:1 | 2.034428000e-05 | 1.459557936e-05 | 1.393865875e+00 | EDDFACTOR:1.955062180e+15 Hz |
| s8 | bb | jbar_representative | 14:2 | 7.490040000e-05 | 7.959250361e-05 | 9.410484229e-01 | EDDFACTOR:1.179078297e+15 Hz |
| s8 | bb | jbar_input_raw | 14:2 | 7.490040000e-05 | 7.959250361e-05 | 9.410484229e-01 | EDDFACTOR:1.179078297e+15 Hz |
| s8 | bf | Gamma_photoion_total | 16:1 | 1.725658118e+00 | 2.002659369e-03 | 8.616832923e+02 | S2PRRR:sum(PR)/Ion Density |
| s8 | bb | jbar_representative | 16:1 | 2.193058971e-06 | 1.092832775e-07 | 2.006765372e+01 | EDDFACTOR:3.305787835e+15 Hz |
| s8 | bb | jbar_representative | 16:2 | 1.077846238e-05 | 1.790743976e-06 | 6.018985698e+00 | EDDFACTOR:2.496300635e+15 Hz |
| s8 | bf | Gamma_photoion_total | 26:1 | 9.398721146e+02 | 1.141949575e-03 | 8.230416958e+05 | Fe2PRRR:sum(PR)/Ion Density |
| s8 | bb | jbar_representative | 26:1 | 6.640636433e-05 | 6.789423149e-05 | 9.780855145e-01 | EDDFACTOR:1.258150482e+15 Hz |
| s8 | bb | jbar_representative | 26:2 | 4.033624221e-05 | 3.084237016e-05 | 1.307819146e+00 | EDDFACTOR:1.581590788e+15 Hz |
| s8 | bb | jbar_representative | 27:2 | 3.326616389e-05 | 2.209465584e-05 | 1.505620369e+00 | EDDFACTOR:1.702989105e+15 Hz |
| s45 | state | T_e | 0:-1 | 1.189706591e+04 | 1.649763900e+04 | 7.211374856e-01 | RVTJ:Temperature (10^4K)*1e4 |
| s45 | state | n_e | 0:-1 | 1.017327000e+05 | 2.004388500e+05 | 5.075498088e-01 | RVTJ:Electron density |
| s45 | bf | Gamma_photoion_total | 14:1 | 4.094554828e-01 | 6.651445554e-07 | 6.155887159e+05 | Sk2PRRR:sum(PR)/Ion Density |
| s45 | bb | jbar_representative | 14:1 | 6.524693000e-07 | 4.964944612e-06 | 1.314152223e-01 | EDDFACTOR:1.955062180e+15 Hz |
| s45 | bb | jbar_input_raw | 14:1 | 6.524693000e-07 | 4.964944612e-06 | 1.314152223e-01 | EDDFACTOR:1.955062180e+15 Hz |
| s45 | bb | jbar_representative | 14:2 | 1.034507000e-07 | 2.987722140e-06 | 3.462527475e-02 | EDDFACTOR:2.484850976e+15 Hz |
| s45 | bb | jbar_input_raw | 14:2 | 1.034507000e-07 | 2.987722140e-06 | 3.462527475e-02 | EDDFACTOR:2.484850976e+15 Hz |
| s45 | bf | Gamma_photoion_total | 16:1 | 1.317759645e-03 | 6.446397152e-07 | 2.044180050e+03 | S2PRRR:sum(PR)/Ion Density |
| s45 | bb | jbar_representative | 16:1 | 1.697185642e-07 | 1.515018823e-06 | 1.120240631e-01 | EDDFACTOR:2.380247636e+15 Hz |
| s45 | bb | jbar_representative | 16:2 | 1.425776769e-07 | 2.786116394e-06 | 5.117434333e-02 | EDDFACTOR:2.496300635e+15 Hz |
| s45 | bf | Gamma_photoion_total | 26:1 | 0.000000000e+00 | 3.357747661e-07 | 0.000000000e+00 | Fe2PRRR:sum(PR)/Ion Density |

새로 생성된 네 비교 artifact는 저장소에 있던 Phase 1 산출물과 각각 `cmp` exit 0으로 동일합니다.

### 3. 단위 왕복

RVTJ 원문 physical line을 직접 읽어 9건 모두 `raw_on_line=True`, `conversion_ok=True`로 확인했습니다.

| 셸 | 수량 | RVTJ line | raw | 변환 후 | 변환 |
|---|---|---:|---:|---:|---|
| s0 | velocity | 35 | 4.3941823505e+03 | 4.3941823505e+03 | identity km/s |
| s0 | n_e | 61 | 4.8528721000e+09 | 4.8528721000e+09 | identity cm⁻³ |
| s0 | T_e | 74 | 1.8535887000e+00 | 1.8535887000e+04 | ×10⁴ K |
| s8 | velocity | 33 | 1.0163505750e+04 | 1.0163505750e+04 | identity km/s |
| s8 | n_e | 59 | 7.3410434000e+08 | 7.3410434000e+08 | identity cm⁻³ |
| s8 | T_e | 72 | 1.0323476000e+00 | 1.0323476000e+04 | ×10⁴ K |
| s45 | velocity | 27 | 3.5975288045e+04 | 3.5975288045e+04 | identity km/s |
| s45 | n_e | 53 | 2.0043885000e+05 | 2.0043885000e+05 | identity cm⁻³ |
| s45 | T_e | 66 | 1.6497639000e+00 | 1.6497639000e+04 | ×10⁴ K |

### 4. default-OFF 무접촉 검사

다음 두 오브젝트를 같은 조건에서 만들었습니다.

```sh
gcc -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE \
  -c src/lumina_plasma.c -o lumina_plasma_default_off.o

gcc -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE \
  -ULUMINA_FROZEN_ORACLE \
  -c src/lumina_plasma.c -o lumina_plasma_explicit_off.o
```

둘 다 exit 0이며:

- SHA-256 공통: `4c634dd95fb938007c73e48013e76c410e3b394dec7253a7f7cea06338908a4e`
- 오브젝트 `cmp`: exit 0
- `nm | rg 'lumina_oracle|g_oracle'`: exit 1, 매치 없음
- gate-OFF 전처리 결과 동일 검색: exit 1, 매치 없음

따라서 가능한 오브젝트/전처리 수준에서 기본 미정의 경로와 명시적 OFF 경로가 동일하고 Gate B observer 코드가 제거됩니다. Oracle 이전 clean production snapshot이 없으므로 전체 실행의 before/after byte 비교는 수행할 수 없었습니다.

### 5. 전체 make 재확인

```sh
make -B lumina
```

exit 2입니다. 첫 오류는 알려진 strict-C11 결함입니다.

```text
src/lumina_plasma.c:1784:18: error: 'M_PI' undeclared
```

`src/lumina_cmfgen.c`에서도 같은 `M_PI` 미정의가 여러 위치에서 재현됐습니다. 링크나 production 실행 단계에는 도달하지 않았습니다.

### 이상·제한 항목

- s45는 CMFGEN 속도 범위 밖입니다. depth 1, Δv `-1048.712 km/s` 최근접 대응만 사용했고 외삽하지 않았습니다.
- 비율표에는 큰 Gamma 비율과 Lumina 0/CMFGEN 비영점 행들이 그대로 보존돼 있습니다. 수치 판정은 적용하지 않았습니다.
- 생성 CSV에는 판정용 필드나 판정 문자열이 없습니다. 생성 Markdown에는 “그러한 판정을 포함하지 않는다”는 설명문 한 줄 때문에 엄격한 키워드 검색은 그 설명문을 매치합니다.
- 전체 production OFF 실행 비교는 기존 `M_PI` 빌드 결함과 clean baseline 부재로 제한됩니다.

산출물: [비교 CSV](/tmp/gateb_B_phase1.qCtpnL/compare/oracle_vs_cmfgen.csv), [REPORT-ONLY Markdown](/tmp/gateb_B_phase1.qCtpnL/compare/oracle_vs_cmfgen.md), [depth map](/tmp/gateb_B_phase1.qCtpnL/compare/shell_cmfgen_depth_map.csv), [단위 왕복표](/tmp/gateb_B_phase1.qCtpnL/compare/cmfgen_parser_roundtrip.csv).