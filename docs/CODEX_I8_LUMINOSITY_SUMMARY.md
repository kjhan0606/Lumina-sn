읽기 전용 규율에 따라 `docs/CODEX_I8_LUMINOSITY.md`는 생성하지 않았습니다. 아래가 해당 보고서의 완전한 내용입니다. 파일 수정·commit·모델 실행·GPU 사용은 하지 않았습니다.

# I8 경계조건 정의 감사

## 판정: 잔류

31.07배는 같은 좌표 비교가 아니어서 사라진다. 그러나 같은 반지름·같은 공이동계 광도로 맞춰도 CMFGEN이 Lumina보다 15.07% 높다.

- Lumina의 실제 내부 경계: 3900 km/s
- CMFGEN \(L(3900)\): \(3.5611\times10^{42}\) erg/s
- Lumina \(L_{\rm inj}\): \(3.094761\times10^{42}\) erg/s
- 잔류: CMFGEN/Lumina = 1.15069, 즉 +15.07%

따라서 “정의 차이로 완전 설명”이 아니며 I8은 잔류다.

## 1. 선행 31.07배의 재현과 좌표 정정

정본은 Lumina \(L_{\rm inj}=3.0948\times10^{42}\) erg/s와 CMFGEN `LSTAR=2.60e7 Lsun`의 비 31.07을 기록하면서 동일 좌표 여부를 미확립으로 남겼다: [OUTSIDE_LOOP_POOL.md](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/OUTSIDE_LOOP_POOL.md:100).

기존 31.0739는 Lumina 설정 요청값 \(3.0927255\times10^{42}\) erg/s와 \(L_\odot=3.828\times10^{33}\) erg/s로 재현된다: [config.json](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/config.json:4).

그러나 좌표 전제는 틀렸다.

- Lumina 경계는 \(v_{\rm inner}=3900\) km/s, \(r_{\rm inner}=6.5639808\times10^{14}\) cm이다: [geometry.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/geometry.csv:2), [config.json](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/config.json:10).
- 4264 km/s는 첫 셸의 속도 중심 \((3900+4628)/2\)이다.
- 실행 로그도 경계를 3900 km/s 및 \(6.563981\times10^{14}\) cm로 읽는다: [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:139).

## 2. 두 광도의 정확한 정의

### Lumina `L_inj`

Lumina 내부 BC는 다음과 같다.

1. \(p<r_{\rm inner}[0]\)인 core ray만 내부 표면에 닿는다: [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:2433).
2. 외부에서 들어오는 세기는 \(I_\nu=0\)이다: [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:2451).
3. 내부에서 나가는 세기는
   \[
   I_\nu=W_{\rm inner}B_\nu(T_{\rm inner})
   \]
   이다: [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:2475), [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:2560).
4. \(B_\nu\)는 코드의 Planck 함수다: [lumina_cmfgen.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_cmfgen.c:68).
5. 생산 설정은 \(W_{\rm inner}=1\)이다: [PARITY59_INSTR.env](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/docs/PARITY59_INSTR.env:68).

따라서

\[
L_{\rm inj}
 =W_{\rm inner}\,4\pi r_{\rm inner}^2\sigma T_{\rm inner}^4.
\]

이는 전 파장 bolometric 내부 경계 backlight만 세며, 전 층 γ-침착은 포함하지 않는다: [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:18232), [lumina_plasma.c](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/src/lumina_plasma.c:18253).

캡처 실측값은

\[
L_{\rm inj}=3.094761\times10^{42}\ {\rm erg\,s^{-1}}
\]

이며 \(r=6.5640\times10^{14}\) cm, \(T=10020\) K이다: [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:38113). 침착은 별도로 \(L_{\rm dep}=7.787639\times10^{42}\) erg/s로 합산된다: [stdout.log](/gpfs/kjhan/lumina_runner2/scratch/instr_capture_188932/stdout.log:38114).

### CMFGEN `LSTAR`

`LSTAR=2.60e7 Lsun`은 CMFGEN 격자의 가장 안쪽 depth \(d=90\)에 부과되는 확산 광도다: [VADAT](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/VADAT:11), [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MODEL:58).

그 좌표는

\[
v_{90}=1024.971\ {\rm km\,s^{-1}},\qquad
r_{90}=1.7251\times10^{14}\ {\rm cm}
\]

이다: [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ:25), [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ:38).

`DIF=T`, `IB_METH=DIFFUSION`으로 설정되어 있다: [MODEL](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/MODEL:123). 코드에서는 `LSTAR`가 \(d=90\)의 온도구배를 정규화하는 데 사용된다:

\[
{\tt DTDR}\propto {LSTAR\over R(90)^2}.
\]

근거: [cmfgen_sub.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:1567).

따라서 `LSTAR`는 전체 구조의 방사성 침착 적분도, 외곽 광도도 아니다. `VADAT` 주석상 “해당 시각에 이 경계보다 안쪽에서 오는 decay power”를 내부 확산 광도로 넣은 값이다.

CMFGEN의 깊이별 공이동계 방사광도는

\[
{L_{\rm CMF}(r)\over L_\odot}
 ={16\pi^2r^2\over L_\odot}\int H_\nu(r)\,d\nu
\]

로 적분된다: [cmfgen_sub.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:2664), [cmfgen_sub.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:2680), [cmfgen_sub.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:2700).

최종 반복에서 실제 \(L(d=90)=2.6000006\times10^7L_\odot\)로 `LSTAR`와 일치한다: [OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/OUTGEN:273).

CMFGEN 코드 자체의 \(L_\odot=3.826\times10^{33}\) erg/s를 적용하면

\[
LSTAR=9.9476\times10^{40}\ {\rm erg\,s^{-1}}.
\]

## 3. 같은 좌표 대조

### 보간 규약

인접 depth 사이에서 속도에 대해 선형 보간했다. 외삽·clamp·대체값은 사용하지 않았다.

\[
w={v-v_{\rm low}\over v_{\rm high}-v_{\rm low}},\qquad
L(v)=L_{\rm low}+w(L_{\rm high}-L_{\rm low}).
\]

4264 km/s의 bracket과 \(w=0.5686338792\)는 기존 구조 대조에도 기록돼 있다: [stage31_summary.json](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/validation/emiss_e10/stage31_summary.json:149).

### 결과

| 좌표 | CMFGEN 공이동계 \(L\) | Lumina 대조값 | 비 |
|---|---:|---:|---:|
| 실제 공통 경계 3900 km/s | \(3.5611\times10^{42}\) erg/s | \(L_{\rm inj}=3.094761\times10^{42}\) | 1.15069 |
| 요청 좌표 4264 km/s | \(4.1501\times10^{42}\) erg/s | 경계값을 그대로 사용할 경우 | 1.34101 |

3900 km/s는 CMFGEN의 3811.3275와 4092.3915 km/s 사이이며, 해당 공이동계 광도는 \(8.9346\times10^8\)와 \(1.0117\times10^9L_\odot\)이다: [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ:35), [OBSFLUX](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/OBSFLUX:37404).

4264 km/s는 4092.3915와 4394.1824 km/s 사이이며, 광도는 각각 \(1.0117\times10^9\), \(1.1401\times10^9L_\odot\)이다. 선형 보간 결과는 \(1.08471\times10^9L_\odot\)이다.

CMFGEN이 별도로 기록한 \(O(v/c)\) 운동 보정 광도를 사용하면 결과는 더 커진다.

- 3900 km/s: \(5.2591\times10^{42}\) erg/s, Lumina의 1.699배
- 4264 km/s: \(6.0665\times10^{42}\) erg/s, Lumina의 1.960배

근거 값은 [OBSFLUX](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/OBSFLUX:37417), 보정 정의는 [cmfgen_sub.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/cmfgen_sub.f:3759)이다. Lumina `L_inj`가 공이동계 Planck 경계이므로 판정의 주 대조에는 CMFGEN의 보정 전 `Luminosity`를 사용했다.

## 4. γ-침착 기여

### 공개 19.48 d 프로파일 적분

공개 프로파일은 속도 중심별 체적 가열률 \(q_i\)를 제공한다: [edep_toy06_cmfgen.txt](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/standart_data1/toy06/edep_toy06_cmfgen.txt:3).

각 속도 중심 사이의 산술중점을 셸 경계로 하고, 셸 안에서는 기록된 \(q_i\)를 그대로 사용했다.

\[
\Delta L_\gamma
=\sum_i q_i{4\pi\over3}t^3
 \left(v_{i,+}^3-v_{i,-}^3\right),\qquad t=19.48\ {\rm d}.
\]

결과:

- CMFGEN 내부 경계 1024.971 → 실제 Lumina 경계 3900 km/s:
  \[
  \Delta L_\gamma=2.87970\times10^{42}\ {\rm erg\,s^{-1}}.
  \]
  이는 \(L_{\rm inj}-LSTAR\)의 96.14%이며,
  \(LSTAR+\Delta L_\gamma=2.97918\times10^{42}\) erg/s로 Lumina보다 3.88% 낮다.

- 1024.971 → 요청 좌표 4264 km/s:
  \[
  \Delta L_\gamma=3.46275\times10^{42}\ {\rm erg\,s^{-1}}.
  \]
  이는 원래 \(L_{\rm inj}-LSTAR\) 차이의 115.61%, 실제 CMFGEN 공이동계 광도 증가분의 85.49%다.

따라서 비영 γ 프로파일을 기준으로 하면 깊은 경계와 침착은 원래 31배 차이의 대부분을 설명한다.

Lumina 첫 셸의 저장 가열률은 \(1.506865\times10^{-3}\) erg/s/cm³이다: [deposition_cmfgen.csv](/home/kjhan/BACKUP/Eunha.A1/Claude/Lumina-sn/data/tardis_reference_toy06_19p48d_sivcaiv/deposition_cmfgen.csv:2). 이를 3900→4264 km/s 체적에 적분하면 \(5.47933\times10^{41}\) erg/s이다. 따라서 4264 km/s까지의 입력 에너지 회계값은

\[
L_{\rm inj}+\Delta L_{\gamma,\rm Lumina}
=3.64269\times10^{42}\ {\rm erg\,s^{-1}}.
\]

CMFGEN 공이동계 값과 비교하면 여전히 1.13930배, 즉 +13.93%가 남는다. 단, 이것은 입력 에너지 회계이며 Lumina가 기록한 깊이별 방사광도는 아니다.

### 실행상 모순: 인과 귀속은 UNRESOLVED

대조 CMFGEN 실행 자체는 방사성 가열을 전 depth에서 정확히 0으로 기록한다: [RVTJ](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/RVTJ:91). 총 방사성 침착 광도도 0이다: [OBSFLUX](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/OBSFLUX:37503).

`OUTGEN`은 decay 갱신 시간간격이 0초였음을 기록하고: [OUTGEN](/gpfs/kjhan/cmfgen_runs/toy06_19.48d_jnu4/OUTGEN:76), 코드도 `DELTA_T=0`이면 침착 계산 분기를 수행하지 않는다: [do_species_decays_v2.f](/gpfs/kjhan/cmfgen_src/cur_cmf/new_main/subs/do_species_decays_v2.f:92).

따라서:

- 공개 프로파일 기반 침착 적분값: 수치 확립
- 그것이 `toy06_19.48d_jnu4` 실행의 광도 증가를 실제로 일으켰다는 귀속: `UNRESOLVED`

결판 요건은 동일한 `LSTAR`·RVTJ 구조를 가진 기존 19.48 d CMFGEN 상태에서 비영 `dE_RAD_DECAY`와 `RAD_DECAY_LUM`이 기록됐다는 provenance, 또는 공개 프로파일이 이 실행에 로드·적용됐다는 직접 기록이다.

## 최종 요약 (-o)

- Lumina \(L_{\rm inj}\): 3900 km/s 경계의 공이동계 bolometric Planck backlight, \(3.094761\times10^{42}\) erg/s. γ-침착 제외.
- CMFGEN `LSTAR`: 1024.971 km/s, \(d=90\) 확산 내부 BC 광도, \(9.9476\times10^{40}\) erg/s. 같은 반지름의 양이 아니다.
- 실제 공통 경계 3900 km/s: CMFGEN \(3.5611\times10^{42}\) erg/s, Lumina보다 15.07% 높음.
- 요청 좌표 4264 km/s: CMFGEN \(4.1501\times10^{42}\) erg/s. Lumina 침착 회계까지 포함해도 13.93% 높음.
- 공개 γ 프로파일은 1025→3900 km/s에서 원래 광도 차이의 96.14%를 설명한다. 다만 지정 CMFGEN 실행은 실제 침착을 0으로 기록하므로 인과 귀속은 `UNRESOLVED`.
- 판정: **잔류**. 31.07배 자체는 좌표 정의 차이로 붕괴하지만, 같은 좌표·공이동계에서도 15.07% 불일치가 남는다.