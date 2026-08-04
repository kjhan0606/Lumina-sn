결론부터 말하면, 입력-only 최소 해는 `ntimesteps=30`, `tmin=2.0 d`, 목표 timestep `ts27`을 유지하고 `tmax` 하나만 `25.0 → 23.958407567406027 d`로 바꾸는 것입니다. 그러면 로그 그리드의 `ts27 midpoint = 19.48 d`가 됩니다. 목표 직후 종료·packet text 출력을 위해 `timestep_finish`도 `030 → 028`로 바꿉니다.

단, `DETAILED_LINE_ESTIMATORS_ON=true`는 출력만 추가하는 옵션이 아니라 NLTE 선 rate를 바꾸므로 기존 런과 직접 비교할 primary lane에서는 켜면 안 됩니다. 별도 diagnostic lane으로 분리해야 합니다.

## 1. 그리드 산식과 입력 diff

현재 compile-time 방식은 `LOGARITHMIC`입니다: [artisoptions.h](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk/artisoptions.h:130).

ARTIS 구현은 [input.cc](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/input.cc:1917)에 따라, timestep 수를 \(N\), index를 \(n\)이라 하면

\[
\Delta\log t = {\log(t_{\max})-\log(t_{\min})\over N}
\]

\[
t_{\mathrm{start},n}=t_{\min}\exp(n\Delta\log t)
\]

\[
t_{\mathrm{mid},n}=t_{\min}\exp[(n+0.5)\Delta\log t]
\]

\[
\Delta t_n=t_{\min}\exp[(n+1)\Delta\log t]-t_{\mathrm{start},n}
\]

입니다. 즉

\[
t_{\mathrm{mid},n}=t_{\min}
\left({t_{\max}\over t_{\min}}\right)^{(n+0.5)/N}.
\]

`N=30`, `n=27`, `tmin=2.0 d`, `tmid=19.48 d`를 대입하면

\[
t_{\max}
=2\left({19.48\over2}\right)^{30/27.5}
=\boxed{23.958407567406027\ {\rm d}}.
\]

`tmin`을 유지하는 이유도 물리적입니다. `model.txt`의 model epoch 자체가 `2.000000 d`이고 [model.txt](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk/model.txt:2), ARTIS는 `tmin`이 model epoch 이상인지 검사하고 밀도를 \((t_\mathrm{model}/t_\min)^3\)로 재조정합니다: [grid.cc](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/grid.cc:1844), [grid.cc](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/grid.cc:1918). 따라서 `tmin` 변경보다 `tmax` 하나를 바꾸는 편이 초기 ejecta 상태를 보존합니다.

기준 파일은 실행 후 restart 상태인 `input.txt`가 아니라 fresh-run 사본 [input-newrun.txt](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk/input-newrun.txt:1)이어야 합니다.

```diff
--- input-newrun.txt
+++ input.txt (epoch-aligned fresh deck)
@@
-000 030                  # timestep_start timestep_finish
-2.0 25.0                 # tmin_days tmax_days
+000 028                  # timestep_start timestep_finish
+2.0 23.958407567406027                 # tmin_days tmax_days
@@
-0                        # simulation_continued_from_saved
+0                        # 그대로 유지: fresh run
@@
-8                        # nprocs_exspec
+8                        # 그대로 유지: MPI rank 수와 동일
```

`000 028`은 그리드를 바꾸지 않고 ts0–27만 실행합니다. ARTIS는 `timestep_finish-1`, 즉 ts27 종료 후 `packets00_*.out`을 씁니다: [sn3d.cc](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/sn3d.cc:716). `ntimesteps=30`은 그대로이므로 `timesteps.out`에는 여전히 ts0–29 정의가 기록될 수 있지만, 실제 transport 산출물은 ts27까지입니다.

계산된 목표 timestep은 다음입니다.

- ts27 start: `18.6902518421 d`
- ts27 mid: `19.4800000000 d`
- ts27 width: `1.61286676782 d`
- ts27 end/ts28 start: `20.3031186099 d`

ARTIS의 `{:g}` timestep writer는 midpoint를 `19.48`로 출력합니다: [sn3d.cc](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/sn3d.cc:314). IEEE double 내부에서는 플랫폼의 `log/exp` 반올림 때문에 약 \(10^{-14}\) d 수준 차이가 생길 수 있으므로, “bitwise decimal equality”가 아니라 ARTIS 출력 그리드의 `19.48 d`를 Gate 조건으로 삼아야 합니다.

### 다른 timestep 영향

`tmax`만 바꾸더라도 로그 그리드 전체가 바뀝니다. 기존 대비 새 midpoint 비율은

\[
{t'_{\mathrm{mid},n}\over t_{\mathrm{mid},n}}
=\left({23.958407567406027\over25}\right)^{(n+0.5)/30}.
\]

| ts | 기존 start | 신규 start | 기존 mid | 신규 mid | 신규−기존 mid |
|---:|---:|---:|---:|---:|---:|
| 0 | 2.000000 | 2.000000 | 2.085988 | 2.084509 | −0.001479 d |
| 4 | 2.800817 | 2.784969 | 2.921235 | 2.902647 | −0.018588 d |
| 10 | 4.641589 | 4.576210 | 4.841150 | 4.769576 | −0.071574 d |
| 20 | 10.772173 | 10.470851 | 11.235313 | 10.913292 | −0.322021 d |
| 26 | 17.851937 | 17.205510 | 18.619464 | 17.932521 | −0.686943 d |
| 27 | 19.419990 | 18.690252 | 20.254934 | 19.480000 | −0.774934 d |
| 28 | 21.125776 | 20.303119 | 22.034059 | 21.161018 | −0.873041 d |
| 29 | 22.981393 | 22.055167 | 23.969456 | 22.987098 | −0.982358 d |

비교 시 추가 주의점이 있습니다. `tmax`는 단순 grid endpoint가 아니라 simulation interval의 총 decay energy와 pellet당 에너지, decay-time sampling에도 사용됩니다: [decay.cc](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/decay.cc:1072), [packet.cc](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/packet.cc:114), [decay.cc](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/decay.cc:1316). 따라서 finite-MC realization은 기존 런과 달라집니다. 이것은 epoch 정렬을 위한 불가피한 input-only 효과이며 bitwise 재현 런으로 취급하면 안 됩니다.

계획된 정확한 deck byte stream의 SHA-256은 다음과 같습니다.

```text
cf59b7156666efeaa633057fc1b08b4568fb1cec27635655470182fbaf860b68
```

아직 파일로 만들지 않은 “제안본”의 checksum입니다.

## 2. §5 출력 옵션 diff와 물리 정합성

현재 상태와 권고 판정은 다음과 같습니다.

| 옵션 | 현재 | primary lane | diagnostic lane | 물리 영향 |
|---|---:|---:|---:|---|
| `KEEP_ESCAPED_GAMMAS` | true | true | true | 이미 활성; escaped gamma 보존 |
| `WRITE_EMISSIONABSORPTION_SPEC_AT_END` | true | true | true | 이미 활성; 종료 시 packet 기반 스펙트럼 계산 |
| `KEEP_ALL_RESTART_FILES` | false | true | true | I/O만 변경 |
| `DETAILED_BF_ESTIMATORS_ON` | true | true | true | 이미 물리에 사용 중 |
| `DETAILED_BF_ESTIMATORS_USEFROMTIMESTEP` | 4 | 4 | 4 | ts4 이후 detailed BF rate 사용 |
| `DETAILED_LINE_ESTIMATORS_ON` | false | **false** | **true** | true이면 Fe 선 radiative rate 변경 |
| `VPKT_ON` / `VPKT_WRITE_CONTRIBS` | false/false | false/false | false/false | 별도 virtual-packet lane에서만 사용 |

### Primary comparison lane

```diff
--- artisoptions.h
+++ artisoptions.primary.h
@@
-constexpr bool KEEP_ALL_RESTART_FILES = false;
+constexpr bool KEEP_ALL_RESTART_FILES = true;
```

제안본 SHA-256:

```text
f39683a54d83df8b94f440f9cd73466d0f49dc05f882f64bf4d4be1ca325116d
```

`KEEP_ALL_RESTART_FILES`는 새 checkpoint를 쓴 뒤 이전 checkpoint를 삭제하는 코드만 차단합니다: [sn3d.cc](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/sn3d.cc:585). transport 물리를 바꾸지 않습니다.

### §5 detailed-estimator diagnostic lane

```diff
--- artisoptions.h
+++ artisoptions.diagnostic.h
@@
-constexpr bool DETAILED_LINE_ESTIMATORS_ON = false;
+constexpr bool DETAILED_LINE_ESTIMATORS_ON = true;
@@
-constexpr bool KEEP_ALL_RESTART_FILES = false;
+constexpr bool KEEP_ALL_RESTART_FILES = true;
```

제안본 SHA-256:

```text
17c3cbc6b9ef60dbfe00e1dfaedd184389e1c40343e4e6cc9bb598a6f1afff1c
```

이 lane은 기존 bk 런과 같은 물리가 아닙니다. `DETAILED_LINE_ESTIMATORS_ON`은 Fe의 선택된 선—현재 코드에서는 `lowerlevel <= 15`, `A_ul > 0`—에 대해 `J_blue` estimator를 만듭니다: [radfield.cc](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/radfield.cc:494). 이후 macro-atom radiative rate가 일반 binned radiation field 대신 그 estimator를 직접 사용합니다: [macroatom.cc](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/macroatom.cc:588). 따라서 “추가 출력만 켠 기존-물리 비교”로 표시하면 안 됩니다.

`DETAILED_BF_ESTIMATORS_ON=true`도 이름과 달리 출력 전용이 아닙니다. ts4부터 photoionization coefficient 계산에 estimator가 사용됩니다: [ratecoeff.cc](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/ratecoeff.cc:689). 다만 이것은 기존 bk 런에도 이미 활성화되어 있으므로 그대로 유지해야 합니다.

§5 원문 근거는 [CODEX_ARTIS_OUTPUT_INVENTORY.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_ARTIS_OUTPUT_INVENTORY.md:235)와 재런 권고 [동 문서](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/CODEX_ARTIS_OUTPUT_INVENTORY.md:256)입니다.

## 3. 실행 자원·시간 근거

기존 job은 [sbatch_toy06_nlte_bk.sh](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk/sbatch_toy06_nlte_bk.sh:1)에 따라:

- 1 node
- 8 MPI ranks
- rank당 8 OpenMP threads
- Slurm 요청상 64 CPUs
- `OMP_PLACES=cores`, `OMP_PROC_BIND=close`
- `mpirun --bind-to none -np 8 ./sn3d_nlte`
- 종료 후 단일 `./exspec_nlte`
- GCC 14.2.0, OpenMPI 5.0.7
- CPU-only; GPU는 사용하지 않음

실행 로그 [slurm output](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk/slurm_toy06_nlte_bk_175613.out:1)는 실제로 `8 ranks × 8 threads`, host `syn01`, partition `a10`을 기록합니다. 단 `nproc` 출력은 8 cores visible이므로, 재제출 때 64 thread가 실제로 분리된 CPU set을 받는지 affinity 확인이 필요합니다.

시간 근거:

- Slurm start→sn3d 종료: `14:17:58 → 15:09:50`, 51분 52초
- exspec 포함 종료: `15:09:58`, 총 약 52분
- ARTIS 내부 final-packet 시각: `tstart + 3104.7 s`, 51분 44.7초: [output_0-0.txt](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/tests/toy06_nlte_bk/output_0-0.txt:3385)
- 기존 ts27의 일반 처리 완료 시각은 시작 후 약 42분 18초

따라서 물리 옵션을 유지한 primary lane에서 ts27 종료·최종 출력까지의 직접 근거 기반 중심값은 약 43분입니다. 다만 새 `tmax`, final emission/absorption 처리, `KEEP_ALL_RESTART_FILES` I/O 때문에 정확한 walltime은 기존 기록만으로 확정할 수 없습니다.

`DETAILED_LINE_ESTIMATORS_ON=true` lane은 모든 선택 line×cell에 대한 estimator 갱신·MPI reduction을 추가합니다: [radfield.cc](/home/kjhan/BACKUP/Eunha.A1/Claude/artis-ref/radfield.cc:942). 기존 측정 런이 없으므로 숫자 배율을 제시할 근거가 없습니다.

저장량은 주의가 필요합니다.

- 기존 rundir 실측: `667,253,236 B`
- checkpoint 한 세트 실측:
  - packet binary: `8 × 24,000,008 B`
  - grid: `5,044,220 B`
  - 합계: `197,044,284 B`
- ts0–27의 28개 세트를 모두 유지할 경우 동일 크기 가정 산술값: 약 `5.52 GB`
- 기존형 기타 출력까지 포함하면 약 `6.0 GB` 수준

diagnostic lane은 detailed-line restart state까지 들어가므로 grid checkpoint가 더 커집니다. 정확한 증가량은 실제 선택 line 수와 생성 파일 없이는 산정할 수 없습니다.

## 4. Gate 0 manifest 초안

```yaml
gate: 0
run_id: toy06_nlte_epoch1948_primary
status: PROPOSED_NOT_MATERIALIZED

source:
  repo: ../artis-ref
  branch: develop
  commit: 36f86476d870cec55bcbe9ab80c1b24ada692eb4
  commit_evidence: ".git/HEAD + refs/heads/develop"
  existing_binary_embedded_version: 36f8647
  worktree_clean: UNVERIFIED_GIT_COMMAND_FORBIDDEN
  existing_build_status_note: >
    version.h records only untracked paths at the prior build.
    Do not claim a clean build without a submission-time check.

epoch:
  method: LOGARITHMIC
  ntimesteps: 30
  target_timestep: 27
  timestep_start: 0
  timestep_finish_exclusive: 28
  tmin_days: 2.0
  tmax_days: 23.958407567406027
  expected_ts27_start_days: 18.6902518421
  expected_ts27_mid_days: 19.48
  expected_ts27_width_days: 1.61286676782
  expected_ts27_end_days: 20.3031186099

monte_carlo:
  seed: 23111963
  mpkts_per_rank: 100000
  mpi_ranks: 8
  omp_threads_per_rank: 8
  nprocs_exspec: 8
  fresh_run: true

options_primary:
  DETAILED_LINE_ESTIMATORS_ON: false
  DETAILED_BF_ESTIMATORS_ON: true
  DETAILED_BF_ESTIMATORS_USEFROMTIMESTEP: 4
  KEEP_ESCAPED_GAMMAS: true
  WRITE_EMISSIONABSORPTION_SPEC_AT_END: true
  KEEP_ALL_RESTART_FILES: true
  VPKT_ON: false
  VPKT_WRITE_CONTRIBS: false

sha256:
  proposed_input_txt: cf59b7156666efeaa633057fc1b08b4568fb1cec27635655470182fbaf860b68
  proposed_artisoptions_primary: f39683a54d83df8b94f440f9cd73466d0f49dc05f882f64bf4d4be1ca325116d
  baseline_input_newrun: a728ee23dc8ce815885e794db1effd8b2af9f89255a91cebfa1fd63ca761a081
  baseline_artisoptions: d60b1b2b2cd2775314c1cf8099326241dd0afb9770dcd923f7628cf0e54cd196
  model_txt: 7a2ed18e6ad2637b43d701af63bda9ca5272d55eaa6451116bf9ec7c2a22cbf9
  abundances_txt: 9845d4008a47f3ec221958e88814f76f015b3b328b5e9d774435cfbe8aeadb1e
  compositiondata_txt: a357111ad92f482a6c0a440b1bca624a4e585e93ffe0710fce9c7df22de5a1ca
  adata_txt: 3de8367ec134271379483918d0f9a7d4c8d15cb917719b36c8f24148b125a679
  phixsdata_txt: 1a6d4c94bef73fd1e5ff8ebe33cd1bdcc8b0066cbc24d5ab1678881f5fdc29e6
  transitiondata_txt: 6504858166756828346e484f31e22dd33b9daf007a87b589335a736a243d9110
  sbatch_baseline: d42c43f83cf308953c57a1a2897a64a9a7f0253ce15357f8b8a86bf141a94c0f
  rebuilt_sn3d: TBD_AFTER_BUILD
  rebuilt_exspec: TBD_AFTER_BUILD

comparison_class:
  primary: >
    Epoch-aligned run with baseline transport options, except grid/tmax and
    checkpoint retention. Not bitwise comparable because timestep history and
    pellet normalisation/sampling change with tmax.
  detailed_line_diagnostic: >
    Separate physics lane. DETAILED_LINE_ESTIMATORS_ON changes Fe radiative rates.
```

현재 기존 binary checksum은 참고용으로 `sn3d_nlte = 317684…e2686f`, `exspec_nlte = 46d70f…99aa1`이지만, 옵션 변경 후 반드시 새 binary checksum으로 교체해야 합니다.

## 5. 제출 체크리스트

- [ ] 기존 결과 디렉터리가 아닌 새 isolated rundir 사용. 현재 job script는 기존 checkpoint와 `output_*.txt`를 삭제합니다.
- [ ] fresh deck은 현재 restart 상태 `input.txt`가 아니라 `input-newrun.txt`에서 시작.
- [ ] `timestep_start=000`, `timestep_finish=028`, `simulation_continued_from_saved=0`.
- [ ] `tmin=2.0`, `tmax=23.958407567406027`, `ntimesteps=30`.
- [ ] `nprocs_exspec=8`과 실제 MPI rank 수 8 일치.
- [ ] primary/diagnostic binary를 섞지 않고 별도 이름과 SHA-256 부여.
- [ ] primary 비교에서는 `DETAILED_LINE_ESTIMATORS_ON=false`.
- [ ] diagnostic lane에만 `DETAILED_LINE_ESTIMATORS_ON=true`; 결과 표에 “physics changed” 명시.
- [ ] 제출 전 commit, options, deck, atomic/model 자료, binary checksum을 manifest에 고정.
- [ ] 최소 약 6 GB의 primary 저장 공간 확보; detailed-line lane은 추가 여유 필요.
- [ ] 시작 로그에서 `timestep_start 0 timestep_finish 28`, `starting a new simulation`, BF estimator from ts4 확인.
- [ ] `timesteps.out`의 ts27 행에서 `tmid_days=19.48` 확인.
- [ ] ts27 최종 `packets00_0000..0007.out` 8개 존재 확인.
- [ ] `KEEP_ALL_RESTART_FILES=true`라면 ts0–27 checkpoint 세트 존재 확인. 이 binary checkpoint는 각 timestep propagation 전 상태이고, `packets00_*`가 ts27 propagation 후 최종 text입니다.
- [ ] `spec.out`, `emission.out`, `emissiontrue.out`, `absorption.out`, `gamma_light_curve.out`, `estimators_*`, `nlte_*` 존재와 ts0–27 범위 확인.
- [ ] sn3d와 exspec exit code 0 확인.
- [ ] ARTIS가 실행 중 `input.txt`를 restart 상태로 변경하므로, 비교·checksum에는 보존된 pre-run deck 또는 생성된 `input-newrun.txt`를 사용.
- [ ] 기존 20.2549 d 런과 비교할 때 “epoch alignment + altered timestep history/MC sampling”임을 명시하고, 옵션 효과만을 분리한 비교로 해석하지 않기.

조사 과정에서 파일 수정, ARTIS/exspec 실행, git 명령은 수행하지 않았습니다.