정식 판정은 **제거 1건, 잔류 3건**입니다. 읽기 전용 지시에 따라 [docs/CODEX_REMOVAL_CONFIRM.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_REMOVAL_CONFIRM.md)는 생성하지 않았으며, 아래가 완성 보고서 본문입니다. 파일·commit·모델·GPU 실행은 없었습니다.

# CODEX_REMOVAL_CONFIRM

## -o 요약

| 항목 | 판정과 한 줄 근거 | 결합 실패·제외 | 임계 `1e-6 / 1e-9 / 1e-12` |
|---|---|---:|---:|
| I2a Fe IV `A_ul` `(Z=26, ion0=3 → Fe IV)` | **잔류** — 2선은 실제 10배 차이이며, CMFGEN 전체 72,223선 중 4,336선만 양쪽에 존재 | Lumina측 실패 0; CMFGEN 단독 **67,887선** | **2 / 2 / 2** |
| I2d Fe III `A_ul` `(Z=26, ion0=2 → Fe III)` | **제거** — 양쪽 전체 1,500준위·136,263선이 엄격결합되고 값이 전부 동일 | **0** | **0 / 0 / 0** |
| I3c-1 Fe IV σ `(Z=26, ion0=3 → Fe IV)` | **잔류** — 200/1,000준위만 존재하며 `1e-12`에서 22,904점 발생 | 800준위=`800,000` grid slot 결합 불가; 양쪽 0이라 비율 제외 **149,993점** | **0 / 1 / 22,904** |
| I3c-2 Ni IV σ `(Z=28, ion0=3 → Ni IV)` | **잔류** — 200/1,000준위만 존재하며 `1e-12`에서 36,746점 발생 | 800준위=`800,000` grid slot 결합 불가; 양쪽 0이라 비율 제외 **149,603점** | **0 / 0 / 36,746** |

## 1. 기준과 정본

`A_ul`의 상대차는

\[
r_A=|A_\mathrm{Lumina}-A_\mathrm{CMFGEN}|/|A_\mathrm{CMFGEN}|
\]

이며 준위 결합은 configuration 정규화, `g` 일치, `|ΔE|≤1e-6 cm⁻¹`을 요구했다.

- Lumina 입력: [line_list.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/line_list.csv), 필드 `line_id,wavelength,atomic_number,ion_number,level_number_lower,level_number_upper,f_lu,A_ul`
- 준위: [levels.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/levels.csv), 필드 `atomic_number,ion_number,level_number,energy_eV,g`
- 로더가 실제 읽는 필드: [lumina_atomic.c:649](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:649)
- CMFGEN `osc` 파서 필드 `i,j,f,A,lam_A,trans_id`: [cmfgen_parser.py:59](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmfgen_parser.py:59), [cmfgen_parser.py:196](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmfgen_parser.py:196)

σ 비교는 `cmfgen_sigma_bf.bin:sigma_cm2[level,bin]`과 CMFGEN `PHOT*_A`의 `cs_type,energy,sigma_Mb`를 같은 1,000개 중심주파수에서 평가했다.

- Lumina binary: [cmfgen_sigma_bf.bin](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/cmfgen_sigma_bf.bin)
- binary 형식 및 로더: [lumina_atomic.c:1007](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_atomic.c:1007)
- CMFGEN phot 파서: [cmfgen_parser.py:245](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmfgen_parser.py:245), [cmfgen_parser.py:268](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/scripts/cmfgen_parser.py:268)

## 2. I2 결합 커버리지

### I2a Fe IV

- Lumina: 200준위, 4,336선.
- CMFGEN 런: 1,000준위, 72,223선 — [feiv_osc.dat:13](/gpfs/kjhan/cmfgen_21jun23/atomic/FE/IV/18oct00/feiv_osc.dat:13), [feiv_osc.dat:16](/gpfs/kjhan/cmfgen_21jun23/atomic/FE/IV/18oct00/feiv_osc.dat:16).
- Lumina의 200준위는 CMFGEN 앞 200준위와 엄격결합: **200/200**, 실패 0.
- Lumina 4,336선 전부 CMFGEN 전이에 유일 결합: **4,336/4,336**, 실패 0.
- 그러나 CMFGEN 전체 기준으로는 **4,336/72,223**만 양쪽에 존재한다. 나머지 **67,887선**은 적어도 한 끝준위가 Lumina의 200준위 범위 밖이어서 결합할 Lumina 선이 없다.

따라서 4,336은 “Fe IV 전체”가 아니라 **Lumina에 남은 200준위 부분집합 전체**다.

### I2d Fe III

- Lumina: 1,500준위, 136,263선.
- CMFGEN: 1,500준위, 136,263선 — [osc_data:10](/gpfs/kjhan/cmfgen_21jun23/atomic/FE/III/19apr23/osc_data:10), [osc_data:13](/gpfs/kjhan/cmfgen_21jun23/atomic/FE/III/19apr23/osc_data:13).
- 엄격 준위 결합: **1,500/1,500**, 실패 0.
- 선 결합: **136,263/136,263**, 실패·단독선·중복결합 모두 0.

136,263은 양쪽 Fe III 이온의 전체 전이 수다.

## 3. Fe IV 불일치 2선

| Lumina line id / CMFGEN trans id | λ(Å) | 준위 `lower→upper` | Lumina `A_ul` | CMFGEN `A_ul` | `r_A` | Lumina/CMFGEN `f_lu` | Lumina/CMFGEN `gf` | 600–3000Å |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 60943 / 4252 | 584.368 | 18→183 | 6.8198e6 | 6.8198e7 | 0.9 | 3.491422e-4 / 3.4914e-3 | 3.491422e-3 / 3.4914e-2 | 밖 |
| 60960 / 3161 | 584.397 | 14→166 | 4.2231e7 | 4.2231e8 | 0.9 | 2.882995e-3 / 2.8830e-2 | 1.729797e-2 / 1.7298e-1 | 밖 |

원시 행:

- Lumina: [line_list.csv:60945](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/line_list.csv:60945), [line_list.csv:60962](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/line_list.csv:60962)
- CMFGEN: [feiv_osc.dat:5280](/gpfs/kjhan/cmfgen_21jun23/atomic/FE/IV/18oct00/feiv_osc.dat:5280), [feiv_osc.dat:4189](/gpfs/kjhan/cmfgen_21jun23/atomic/FE/IV/18oct00/feiv_osc.dat:4189)

고정 표지 `f_lu≥0.1`로는 두 선 모두 강선이 아니다. `gf≥0.1`을 쓰면 두 번째 선의 CMFGEN 값만 해당한다.

## 4. 임계 의존성

| 항목 | 분모 | `r>1e-6` | `r>1e-9` | `r>1e-12` |
|---|---:|---:|---:|---:|
| Fe IV `A_ul` | 4,336 | 2 | 2 | 2 |
| Fe III `A_ul` | 136,263 | 0 | 0 | 0 |
| Fe IV σ | 50,007 | 0 | 1 | 22,904 |
| Ni IV σ | 50,397 | 0 | 0 | 36,746 |

σ 최대 상대차는 Fe IV `1.0799575e-9`, Ni IV `3.5713352e-12`였다. 원인 해석은 하지 않는다.

## 5. σ 0건의 의미와 제외점

### Fe IV

- Lumina 200준위와 CMFGEN 앞 200준위 결합: **200/200**, 실패 0.
- 해당 200준위의 binary `has_cmfgen`: **200/200**.
- CMFGEN evaluator: type 20이 180준위, type 1이 20준위. 미지원·미결합 준위 0.
- 전체 200,000 grid point 중 CMFGEN σ가 양수인 **50,007점만** 상대차 분모로 사용.
- 나머지 **149,993점**은 문턱 아래로 양쪽 σ가 정확히 0이었다. 비교 불가능한 점은 아니지만, 0 분모 때문에 상대차 집계에서 제외됐다.
- CMFGEN의 나머지 800준위는 Lumina에 없으므로 `800×1,000=800,000` grid slot은 결합 자체가 불가능했다.

`0/50,007`은 값의 완전 동일을 뜻하지 않는다. 양수점 중 정확히 같은 점은 2점뿐이고 50,005점은 비영 상대차였으며, `1e-6` 아래였다는 뜻이다.

### Ni IV

- Lumina/CMFGEN 준위 결합: **200/200**, 실패 0.
- binary `has_cmfgen`: **200/200**.
- 200준위 모두 CMFGEN type 1 evaluator 사용. 미지원·미결합 준위 0.
- 200,000점 중 양수점 **50,397점**만 상대차 분모.
- 양쪽 0으로 비율에서 제외: **149,603점**.
- Lumina에 없는 CMFGEN 800준위의 **800,000 grid slot**은 결합 불가.

`0/50,397` 역시 완전 동일이 아니다. 50,397점 전부 비영 상대차였지만 `1e-6` 아래였다.

## 6. 최종 판정

- **I2a Fe IV `A_ul`: 잔류.** 2개의 완결된 불일치가 존재하고, CMFGEN 전체 전이 중 67,887선은 비교되지 않았다.
- **I2d Fe III `A_ul`: 제거.** 양쪽 전체 준위·전체 전이가 실패 없이 결합되며 세 임계에서 모두 0건이다.
- **I3c-1 Fe IV σ: 잔류.** 이온 전체가 아니라 200/1,000준위만 비교됐고 `1e-9`, `1e-12`에 민감하다.
- **I3c-2 Ni IV σ: 잔류.** 이온 전체가 아니라 200/1,000준위만 비교됐고 `1e-12`에 민감하다.