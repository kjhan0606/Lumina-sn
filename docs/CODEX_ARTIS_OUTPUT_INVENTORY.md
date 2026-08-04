## 결론

기존 두 런만으로 Gate 4의 “완전한 인과적 fate/energy census”는 만들 수 없습니다. 다만 다음 범위까지는 결정적으로 가능합니다.

- 가능: ts27 탈출 packet별 `e_rf/e_cmf`, 탈출 주파수·방향·시각, 마지막 bb/fb/ff 방출 ID, 마지막 bb 흡수선, 마지막 방출 이후 전자산란 여부, 최종 저장 에너지, timestep별 deposition과 전역 사건 수.
- 부분 가능: Kromer식 “마지막 흡수선 ↔ 마지막 방출선/continuum” 탈출 에너지 분해.
- 불가: bf 흡수 continuum ID, bf 당시 `nu_edge/nu`, bf→MA와 bf→K 분기별 escaped energy, 정확한 K 재방출 lineage, 독립적인 adiabatic/numerical-loss 분리.

즉 기존 자료는 “Kromer식 last-interaction spectrum decomposition”에는 충분하지만, Gate 4의 상호배타적 causal fate census에는 부족합니다.

---

## 1. 파일 전수·크기·timestep 범위

실제 경로는 서로 형제 디렉터리입니다.

- `../artis-ref/tests/toy06_nlte_bk/`
- `../artis-ref/tests/toy06_whitebox_run/` — `toy06_nlte_bk/` 하위가 아님.

`find -maxdepth 1 -type f -printf '%f\t%s\n'` 전수 실측 결과:

| 파일군 | `toy06_nlte_bk` | `toy06_whitebox_run` |
|---|---:|---:|
| 전체 | 138 files, 667,232,750 B | 137 files, 666,776,848 B |
| `packets00_*.out` | 8, 213,660,053 B | 8, 213,654,382 B |
| `packets_*_ts29.tmp` | 8 × 24,000,008 B | 8 × 24,000,008 B |
| `estimators_*.out` | 8, 2,876,322 B | 8, 2,876,145 B |
| `nlte_*.out` | 8, 80,352,647 B | 8, 80,351,889 B |
| `radfield_*.out` | 8, 2,000,369 B | 8, 1,999,146 B |
| `output_*-*.txt` | 64, 73,766,586 B | 64, 73,338,446 B |
| `gridsave_ts29.tmp` | 5,044,220 B | 5,028,243 B |
| `spec.out` | 215,176 B | 213,953 B |
| `light_curve.out` | 928 B | 929 B |
| `gamma_spec.out` | 124,351 B | 123,833 B |
| `gamma_light_curve.out` | 684 B | 680 B |
| `deposition.out` | 4,452 B | 4,457 B |
| `emission.out` | 5,646,429 B | 5,642,426 B |
| `emissiontrue.out` | 5,939,402 B | 5,937,497 B |
| `absorption.out` | 2,993,367 B | 2,992,477 B |
| `bflist.out` | 116,821 B | 116,821 B |
| `linestat.out` | 11,004,998 B | 11,004,998 B |

나머지까지 포함한 전수 파일군:

- 입력·원자자료: `adata.txt`, `phixsdata.txt`, `transitiondata.txt`, `abundances.txt`, `compositiondata.txt`, `model.txt`, `input.txt`, `input-newrun.txt`, `exspec.txt`, `syn_dir.txt`.
- 격자·핵종: `grid.out`, `nuclides.out`, `gammalinelist.out`, `modelgridrankassignments.out`, `timesteps.out`.
- 설정·실행물: `artisoptions.h`, `sbatch_*.sh`, 각 `sn3d*`, `exspec*`; whitebox에만 `build_toy06_model.py`.
- 로그: `sn3d_run.log`, `exspec_run.log`; bk에 추가로 `slurm_*.out/.err`.

압축 파일은 두 디렉터리 모두 0개입니다. `*.gz, *.bz2, *.xz, *.zst, *.zip, *.tar, *.tgz` 전수 `find` 결과가 비어 있었습니다.

### rank별 핵심 크기

`packets00_0000..0007.out`, 바이트 순서:

- bk: `26723726, 26713563, 26707153, 26707894, 26701716, 26702613, 26695465, 26707923`
- whitebox: `26704934, 26711302, 26707752, 26702948, 26707962, 26703340, 26717698, 26698446`

각 파일은 `wc -l = 100001`, 즉 헤더 1 + packet 100,000개이며 전체 800,000 packet입니다. 실제 헤더는 [bk packets00_0000.out:1](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk/packets00_0000.out:1), [whitebox packets00_0000.out:1](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_whitebox_run/packets00_0000.out:1)에 있습니다.

### timestep과 19.48 d

두 런의 timestep 표는 동일합니다.

- ts27: start `19.42 d`, mid `20.2549 d`, width `1.70579 d`
- ts28 start: `21.1258 d`

따라서 `19.48 d`는 명백히 ts27 구간 `[19.42, 21.1258)` 안입니다. 근거: [timesteps.out:29](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk/timesteps.out:29), [timesteps.out:30](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk/timesteps.out:30). 다만 산출물의 대표 epoch는 정확한 `19.48 d`가 아니라 ts27 midpoint `20.2549 d`입니다.

| 산출물 | 범위/실측 |
|---|---|
| `timesteps.out` | header + ts0–29, 31 lines |
| `spec.out` | header + 1000 frequency bins, 1001 lines; 30 timestep columns |
| `light_curve.out`, `gamma_light_curve.out` | ts0–29, 각 30 lines |
| `deposition.out` | header + ts0–29, 31 lines |
| `emission*.out`, `absorption.out` | 1000 bins × 30 timesteps = 30,000 lines |
| `estimators_0000..0007.out` | 각 파일 모두 ts0–29 |
| `radfield_0002..0007.out` | ts5–29; rank1 ts11–29; rank0 header only |
| `nlte_0002..0007.out` | ts5–29; rank1 ts10–29; rank0 header only |
| `packets00_*.out` | ts29 종료 뒤 작성된 최종 스냅샷이나 과거 escape time 보존 |
| restart packets | ts29만 존재; ts27 binary snapshot 없음 |

최종 packet 텍스트는 마지막 timestep에만 쓰입니다: [sn3d.cc:716](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/sn3d.cc:716)-[718](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/sn3d.cc:718). `KEEP_ALL_RESTART_FILES=false`라 이전 checkpoint를 지우므로 ts29만 남은 것이 소스와 일치합니다: [bk artisoptions.h:136](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk/artisoptions.h:136), [sn3d.cc:585](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/sn3d.cc:585)-[599](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/sn3d.cc:599).

관측자 도달시각 `t_arrive = escape_time - pos·dir/c`를 적용한 ts27 escaped r-packet 실측:

| 런 | ts27 escaped r-packets | raw `Σe_rf` | 물리 정규화 `/8` |
|---|---:|---:|---:|
| bk | 52,576 | `1.150832402e49` erg | `1.438540503e48` erg |
| whitebox | 52,238 | `1.146440698e49` erg | `1.433050873e48` erg |

도달시각 계산과 binning은 [spectrum_lightcurve.cc:570](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/spectrum_lightcurve.cc:570)-[580](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/spectrum_lightcurve.cc:580)에 정의되어 있습니다. `/8`은 exspec가 rank별 MC ensemble을 `nprocs_exspec`으로 나누기 때문입니다: [spectrum_lightcurve.cc:578](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/spectrum_lightcurve.cc:578)-[580](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/spectrum_lightcurve.cc:580).

---

## 2. `packets` 32-column 스키마

실제 헤더와 writer가 정확히 일치합니다: [packet.cc:32](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/packet.cc:32)-[49](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/packet.cc:49), [packet.cc:218](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/packet.cc:218)-[241](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/packet.cc:241).

| 열 | 의미 |
|---|---|
| 1 `number` | packet 고유 번호 |
| 2 `where` | 최종 `cellindex`; 탈출 packet은 탈출 직전 cell |
| 3 `type_id` | 최종 packet 상태 |
| 4–6 `posx/y/z` | 최종/탈출 위치 |
| 7–9 `dirx/y/z` | rest-frame 진행 단위벡터 |
| 10 `tdecay` | pellet decay time |
| 11 `e_cmf` | comoving-frame packet energy |
| 12 `e_rf` | rest-frame packet energy |
| 13 `nu_cmf` | comoving frequency |
| 14 `nu_rf` | rest-frame/observer spectrum에 쓰는 frequency |
| 15 `escape_type_id` | 탈출 직전 종류: r=11, gamma=10 등 |
| 16 `escape_time` | grid 경계를 벗어난 simulation time, seconds |
| 17 `emissiontype` | 가장 최근 r-packet 방출 process/transition |
| 18 `trueemissiontype` | “last thermal emission”용 별도 표식 |
| 19–21 `em_pos*` | 마지막 방출 위치; 단 전자산란도 이 위치를 갱신 |
| 22 `absorption_type` | 마지막 기록 흡수 ID |
| 23 `absorption_freq` | 마지막 기록 흡수의 `nu_rf` |
| 24 `nscatterings` | 마지막 방출 이후 전자산란 횟수 |
| 25 `em_time` | 마지막 방출/전자산란 위치가 기록된 시각 |
| 26 `originated_from_particlenotgamma` | pellet의 최초 decay product가 particle인지 |
| 27–29 `trueem_pos*` | true thermal emission 위치 |
| 30 `trueem_time` | true thermal emission 시각 |
| 31 `pellet_nucindex` | decay nuclide index |
| 32 `pellet_decaytype` | decay channel |

구조체 원정의는 [packet.h:37](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/packet.h:37)-[71](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/packet.h:71), type ID는 [packet.h:11](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/packet.h:11)-[25](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/packet.h:25)에 있습니다.

### emission/absorption ID 의미

`emissiontype`:

- `>=0`: 마지막 bb 방출의 `linelist` index. 방출 때 직접 설정됩니다: [macroatom.cc:227](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:227)-[235](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:235).
- 일반 음수: free-bound continuum ID `-1-bflist_index`. 매핑 공식: [atomic.h:534](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/atomic.h:534)-[542](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/atomic.h:542).
- `-9999999`: free-free.
- `-9999000`: not set. 상수: [packet.h:27](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/packet.h:27)-[28](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/packet.h:28).
- `bflist.out` 행은 `index element ion lowerlevel upperionlevel`: [input.cc:1575](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/input.cc:1575)-[1585](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/input.cc:1585).

`absorption_type`:

- `>=0`: 마지막 bb 흡수 linelist index; 이때만 `absorptionfreq=nu_rf`를 명시적으로 씁니다: [rpkt.cc:584](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:584)-[590](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:590).
- `-1`: ff absorption → K.
- `-2`: bf absorption.
- `-3/-4/-5`: gamma Compton/photoelectric/pair production.
- `-6/-7/-10`: decay/pellet 계열 표식. 정의: [packet.h:51](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/packet.h:51)-[57](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/packet.h:57).

중요한 결함은 bf 경로가 `absorptiontype=-2`만 쓰고 `allcontindex`, `nu_edge`, bf 사건 주파수를 packet에 보존하지 않는다는 점입니다. 선택된 continuum과 `nu_edge`는 지역변수로만 존재합니다: [rpkt.cc:407](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:407)-[435](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:435). 따라서 `absorption_type=-2` 행의 `absorption_freq`가 0이 아닐 수 있어도, 이는 이전 bb 흡수에서 남은 stale 값일 수 있으며 bf 당시 `nu`로 사용할 수 없습니다.

`nscatterings`는 전자산란 때 증가하고 방출 때 0으로 reset됩니다: [rpkt.cc:383](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:383)-[400](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/rpkt.cc:400), [macroatom.cc:232](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:232)-[235](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:235), [kpkt.cc:449](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/kpkt.cc:449)-[452](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/kpkt.cc:452). 따라서 “마지막 방출 뒤 e-scattering을 한 번 이상 겪었는가”에는 정확합니다.

`trueemissiontype`은 독립적인 “K-packet origin boolean”이 아닙니다. macro-atom 종료 때 비어 있으면 채우며 소스 자체에 의미상 TODO가 있습니다: [macroatom.cc:541](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:541)-[547](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:547). 이를 K 재방출 lineage의 결정적 표식으로 쓰면 안 됩니다.

---

## 3. 기존 자료로 가능한 실제 census

전체 escaped r-packet의 최종 방출 process 분해:

| 런 | bb line | fb continuum | ff | escaped-r 전체 |
|---|---:|---:|---:|---:|
| bk | 442,592 | 7,692 | 157,819 | 608,103 |
| whitebox | 442,116 | 7,758 | 157,927 | 607,801 |

`Σe_rf`:

- bk raw `1.253643701e50`, 물리 `/8 = 1.567054626e49 erg`
- whitebox raw `1.252561413e50`, 물리 `/8 = 1.565701766e49 erg`

ts27에서 만들 수 있는 last-interaction proxy 분해는 다음과 같습니다. 이는 완전한 causal fate가 아니라 현재 열로 정의한 상호배타적 proxy입니다.

| proxy 규칙 | bk count / raw energy | whitebox count / raw energy |
|---|---:|---:|
| e-scatter: `nscatterings>0` | 27,547 / `5.940073276e48` | 27,194 / `5.881048562e48` |
| resonance: `nscat=0`, `abs>=0`, `emit==abs` | 3,699 / `8.051773638e47` | 3,602 / `7.777410319e47` |
| bb-MA: `nscat=0`, `abs>=0`, `emit!=abs` | 20,834 / `4.605794907e48` | 20,970 / `4.653388642e48` |
| bf-lastabs: `nscat=0`, `abs=-2` | 30 / `5.602790200e45` | 13 / `2.642895400e45` |
| origin/other | 466 / `1.516756840e47` | 459 / `1.495858460e47` |

resonance counter 자체는 macro-atom이 activating line과 동일한 선을 방출할 때 증가합니다: [macroatom.cc:205](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:205)-[211](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:211). 그러나 packet의 `absorptiontype`은 K/비열적 경로에서 항상 reset되지 않으므로 위 equality 분해는 escaped-energy proxy이지 완전한 causal event log가 아닙니다.

기존 rank-0 thread 로그를 ts27에서 8 MPI rank 합산한 사건 수:

| 사건 | bk | whitebox |
|---|---:|---:|
| electron scattering | 7,379,267 | 7,421,293 |
| resonance scattering | 85,919 | 85,983 |
| bb MA activation | 1,704,121 | 1,709,607 |
| bf→MA | 471 | 435 |
| bf→K | 33 | 38 |
| ff→K | 0 | 2 |
| K→r bb | 29,197 | 29,078 |
| K→r fb | 133 | 106 |
| K→r ff | 10 | 11 |

이 counter들은 표준적으로 timestep마다 출력됩니다: [stats.cc:45](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/stats.cc:45)-[80](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/stats.cc:80). 단, 사건 수일 뿐 energy weight와 escaped-packet linkage가 없습니다.

---

## 4. Gate 4 가능/불가 판정

| Gate 4 항목 | 판정 | 이유 |
|---|---|---|
| escaped r/gamma energy | 가능 | `escape_type`, `e_rf/e_cmf`, `escape_time`, pos/dir 보존; gamma도 현재 유지됨 |
| escape frequency/time | 가능 | `nu_rf`, `nu_cmf`, `escape_time`; observer arrival 보정 가능 |
| final bb/fb/ff emission | 가능 | `emissiontype`과 `bflist.out`로 결정적 |
| no interaction | 부분 | escaped r 중 `EMTYPE_NOTSET`는 두 런 모두 0; packet별 총 interaction count가 없어 일반적인 무상호작용 증명에는 부족 |
| electron scattering | 가능(최종 fate 의미) | 마지막 방출 이후 횟수는 정확; 이전 방출 전 history는 소실 |
| resonance scattering | 부분/proxy | `emit==abs`와 `nscat=0` 가능; stale absorption 때문에 완전 인과 표식은 아님 |
| bound-bound MA | 부분/proxy | last absorbed/emitted line 비교 가능; MA 내부에서 K를 거쳤는지 분리 불가 |
| bf absorption/recombination | 부분 | `abs=-2` 및 최종 fb emission ID는 가능; 흡수 continuum ID와 bf→MA/K escaped linkage 불가 |
| ff | 부분 | 최종 ff emission 및 `abs=-1` 가능; K lineage와 전체 사건 history 불가 |
| thermal/K 재방출 | 불가(escaped energy) | 로그에는 사건 수만 있고 packet별 K 진입/이탈 lineage 없음 |
| bf continuum ID 분포 | 불가 | 선택된 `allcontindex`가 packet/writer에 기록되지 않음 |
| bf `nu_edge/nu` 분포 | 불가 | `nu_edge`와 bf 사건 당시 `nu` 모두 출력되지 않음 |
| injected energy | 가능 | 두 런 모두 `3.24586e49 erg`; [bk output_0-0.txt:418](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk/output_0-0.txt:418)-[422](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk/output_0-0.txt:422) |
| deposited energy | 가능 | `deposition.out`에 total/gamma/particle components가 ts0–29 존재; writer [sn3d.cc:229](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/sn3d.cc:229)-[301](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/sn3d.cc:301) |
| stored radiation/gamma | 가능, ts29만 | 최종 non-escape type 10/11의 `e_cmf` 합 |
| thermal pool | 가능, ts29만 | final `TYPE_KPKT=12`의 `e_cmf` 합 |
| adiabatic loss | 불가, 잔차만 | K energy 감소는 수행되지만 누적 loss field/counter가 없음: [kpkt.cc:377](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/kpkt.cc:377)-[384](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/kpkt.cc:384) |
| numerical loss | 불가 | 별도 accumulator가 없어 input−final 잔차에서 adiabatic과 분리 불가 |
| ts27 stored/thermal snapshot | 불가 | ts27 checkpoint가 삭제되고 ts29만 남음 |
| cascade-cap fallback | N/A | 이는 Lumina 전용 Gate 항목이며 조사한 ARTIS packet schema에는 대응 표식이 없음 |

참고로 최종 `e_cmf` 기반 제한적 잔액은 다음처럼 산출할 수 있습니다.

| 런 | escaped | active non-K stored | K thermal pool | input−final residual |
|---|---:|---:|---:|---:|
| bk | `1.422635126e49` | `4.493243914e48` | `6.163519861e47` | `1.312265284e49` |
| whitebox | `1.421359921e49` | `4.490350268e48` | `6.178459218e47` | `1.313680460e49` |

모두 rank ensemble `/8` 후 값입니다. 마지막 residual은 약 40.4%이지만 이것을 곧바로 “adiabatic loss”나 “numerical loss”라고 부를 근거는 없습니다. 일관된 frame/time 회계와 독립 loss accumulator가 없기 때문입니다.

---

## 5. 표준 옵션인가, 계측 개조인가

이 ARTIS 버전의 출력 관련 옵션은 런타임 `input.txt`가 아니라 대부분 `constexpr` C++ 설정입니다. 예를 들어 `WRITE_EMISSIONABSORPTION_SPEC_AT_END`, `KEEP_ESCAPED_GAMMAS`, `KEEP_ALL_RESTART_FILES`가 모두 [artisoptions.h:126](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk/artisoptions.h:126)-[136](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk/artisoptions.h:136)에 있습니다. 즉 값을 바꾸려면 재빌드가 필요하며 “재컴파일 없는 config toggle”은 아닙니다.

| 부족 항목 | 재컴파일 없는 runtime config | 표준 compile option | 계측 개조 |
|---|---|---|---|
| ts27 종료 packet text | 가능: `timestep_finish=28`로 target 뒤 종료 | 불필요 | 불필요 |
| 모든 timestep packet snapshot | 불가 | `KEEP_ALL_RESTART_FILES=true` | 불필요; 단 raw ABI binary |
| escaped gamma | 현재 이미 가능 | `KEEP_ESCAPED_GAMMAS=true` | 불필요 |
| emission/absorption spectra | 현재 이미 가능 | `WRITE_EMISSIONABSORPTION_SPEC_AT_END=true` | 불필요 |
| detailed line/BF estimators | 불가 | `DETAILED_LINE_ESTIMATORS_ON`, `DETAILED_BF_ESTIMATORS_ON` | event fate에는 여전히 부족 |
| virtual-packet contributions | 불가 | `VPKT_ON`, `VPKT_WRITE_CONTRIBS`; 문서 [artisoptions_doc.md:61](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/artisoptions_doc.md:61)-[71](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/artisoptions_doc.md:71) | real-packet fate 대용은 아님 |
| macro-atom transition log | 불가 | 표준 artisoptions가 아님; `LOG_MACROATOM=false`가 소스에 고정 | hardcoded switch 변경+재빌드 필요; packet number도 없어 충분치 않음 |
| bf continuum ID/`nu_edge/nu` | 불가 | 없음 | 필수 |
| exact K lineage | 불가 | 없음 | 필수 |
| adiabatic/numerical loss 분리 | 불가 | 없음 | 필수 |

`LOG_MACROATOM`의 위치와 현 출력 열은 [macroatom.cc:34](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:34)-[39](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:39), [macroatom.cc:557](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:557)-[566](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:566)입니다.

---

## 6. epoch-정렬 재런 권고

재런을 하게 된다면 최소 권고는 다음입니다.

1. 런타임 epoch 설정

- `ntimesteps/tmin/tmax`를 조정해 목표 epoch `19.48 d`가 정확히 `tmid`가 되도록 timestep grid를 정의.
- target이 ts27이면 `timestep_finish=28`로 하여 target 직후 `packets00_*.out`을 쓰게 함.
- `nprocs_exspec`를 실제 MPI rank 수와 동일하게 유지.
- 기존 `input.txt`에서 이 항목들은 [input.txt:2](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk/input.txt:2)-[4](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk/input.txt:4), [input.txt:22](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk/input.txt:22)에 있습니다.

2. 표준 compile-time 출력 옵션

- `KEEP_ESCAPED_GAMMAS=true`
- `WRITE_EMISSIONABSORPTION_SPEC_AT_END=true`
- `KEEP_ALL_RESTART_FILES=true`
- `DETAILED_BF_ESTIMATORS_ON=true`
- `DETAILED_BF_ESTIMATORS_USEFROMTIMESTEP <= target`
- `DETAILED_LINE_ESTIMATORS_ON=true`
- `VPKT_ON/VPKT_WRITE_CONTRIBS`는 별도 virtual-packet 진단 lane에서만 사용. real-packet Gate 4와 혼합하지 않는 편이 안전합니다.

3. Gate 4 완성을 위한 필수 계측 필드

- packet별 `total_interaction_count`와 `last_event_class`.
- bf 사건마다 `allcontindex`, element/ion/lower level/upper target, `nu_cmf_event`, `nu_edge`, `nu_edge/nu`, 선택 branch `BF_TO_MA/BF_TO_K`.
- K 진입 원인과 K 이탈 channel: ff/bf/collisional/earlier-decay → bb/fb/ff/MA.
- 사건 전후 `type`, `e_cmf/e_rf`, timestep, event time, packet number.
- timestep별 동일 frame의 `injected`, `escaped-r`, `escaped-gamma`, `stored-r/gamma`, `stored-K`, `adiabatic loss`, `numerical residual`.
- target epoch packet text snapshot.

이 계측이 있어야 Gate 4의 식

`injected/deposited = escaped + stored + adiabatic + thermal pool + numerical loss`

을 독립 항목으로 닫고, escaped energy를 요청한 모든 fate로 상호배타적으로 분해할 수 있습니다.

조사 중 파일 수정, ARTIS/exspec 실행, git 명령은 하지 않았습니다. 사용한 것은 `find/stat/head/wc/awk/rg/nl` 기반 읽기 전용 실측뿐입니다.