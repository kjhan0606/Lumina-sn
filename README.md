# LUMINA-SN

**1D Monte Carlo 복사 전달 코드** — Ia형 초신성 스펙트럼 시뮬레이션

LUMINA-SN은 [TARDIS](https://tardis-sn.github.io/tardis/)의 핵심 물리를 C/CUDA로 재구현한 고성능 복사 전달 코드입니다. 균질 팽창(homologous expansion) 상태의 초신성 분출물을 통과하는 광자 패킷을 Monte Carlo 방법으로 추적하여, 관측 가능한 스펙트럼을 계산합니다.

---

## 디렉토리 구조

```
lumina-sn/
├── README.md                 ← 이 문서
├── Makefile                  ← 빌드 시스템
├── .gitignore
│
├── src/                      ← C/CUDA 소스코드 (6개 파일, ~3900줄)
│   ├── lumina.h              ← 구조체, 상수, 함수 프로토타입
│   ├── lumina_main.c         ← CPU 메인 드라이버 + 반복 루프
│   ├── lumina_transport.c    ← CPU 전달 물리 커널
│   ├── lumina_plasma.c       ← 플라즈마 솔버 + 수렴 로직
│   ├── lumina_atomic.c       ← 데이터 로더 (NPY/CSV) + 메모리 관리
│   └── lumina_cuda.cu        ← GPU 전달 커널 (CUDA)
│
├── scripts/                  ← Python 분석/시각화 스크립트 (14개)
│   ├── plot_spectrum_comparison.py   ← 스펙트럼 비교 플롯 (주요)
│   ├── compare_spectra.py           ← TARDIS vs LUMINA 스펙트럼 비교
│   ├── compare_spectra_v2.py        ← 상세 형상 비교
│   ├── diagnose_w.py                ← 희석 인자 W 진단
│   ├── validate_partition.py        ← 분배함수 검증
│   ├── validate_plasma.py           ← 플라즈마 상태 검증
│   ├── validate_tau_detail.py       ← tau_sobolev 상세 분석
│   ├── validate_tau_impact.py       ← 중성 종 tau 영향 분석
│   ├── debug_neutral_tau.py         ← 중성 원자 tau 디버깅
│   ├── compare_tau_c_vs_python.py   ← C vs Python tau 비교
│   ├── check_c_ions.py              ← 이온 밀도 확인
│   ├── check_ion_ordering.py        ← 이온 순서 검증
│   ├── extract_atomic_data.py       ← HDF5에서 원자 데이터 추출
│   └── export_tardis_reference.py   ← TARDIS 기준 데이터 내보내기
│
├── data/                     ← 입력 데이터
│   ├── atomic/               ← 원자 데이터 (HDF5)
│   ├── tardis_reference/     ← TARDIS 수렴 상태 (CSV/NPY)
│   ├── model/                ← 모델 파라미터
│   └── sn2011fe/             ← SN 2011fe 관측 데이터 + 설정
│
└── docs/                     ← 기술 문서
    ├── ARCHITECTURE.md       ← 코드 구조 설명
    ├── PHYSICS.md            ← 물리 모델 설명
    └── HISTORY.md            ← 개발 이력 + 주요 버그 수정
```

---

## 의존성 (Dependencies)

### 필수
| 패키지 | 최소 버전 | 용도 |
|--------|----------|------|
| **GCC** | 8.0+ | C11 컴파일러 |
| **GNU Make** | 3.81+ | 빌드 시스템 |

### GPU 빌드 (선택)
| 패키지 | 최소 버전 | 용도 |
|--------|----------|------|
| **CUDA Toolkit** | 11.0+ | GPU 커널 컴파일 (nvcc) |
| **NVIDIA GPU** | Compute Capability 7.0+ | GPU 실행 |

### Python 스크립트 (선택)
| 패키지 | 용도 |
|--------|------|
| **Python 3** | 스크립트 실행 |
| **numpy** | 수치 계산, NPY 파일 읽기 |
| **matplotlib** | 스펙트럼 플롯 생성 |
| **scipy** | 스펙트럼 스무딩 |
| **h5py** | HDF5 원자 데이터 읽기 (extract_atomic_data.py만) |

### 설치 예시 (RHEL/CentOS/Rocky)
```bash
# C 컴파일러 + Make
sudo dnf install gcc make

# Python 패키지
pip install numpy matplotlib scipy h5py

# CUDA (GPU 사용 시) — https://developer.nvidia.com/cuda-downloads 참조
```

---

## 빌드 (Build)

프로젝트 루트에서 실행합니다.

### CPU 빌드 (기본)
```bash
make
```
성공 시 루트에 `lumina` 바이너리가 생성됩니다.

### CPU + OpenMP 병렬 빌드
```bash
make OMP=1
```
멀티코어 CPU에서 패킷 전달을 병렬화합니다.

### GPU (CUDA) 빌드
```bash
make cuda
```
성공 시 루트에 `lumina_cuda` 바이너리가 생성됩니다.

> **참고**: GPU 빌드는 Makefile의 `-arch=sm_89`를 본인의 GPU에 맞게 수정해야 할 수 있습니다.
> - RTX 3090/A100: `-arch=sm_80`
> - RTX 4090: `-arch=sm_89`
> - V100: `-arch=sm_70`

### 클린 빌드
```bash
make clean   # 바이너리 삭제
make         # 다시 빌드
```

---

## 실행 (Run)

### 기본 실행

```bash
# CPU (기본 설정: 2M 패킷 × 19 + 20M 최종, 20 반복)
./lumina

# GPU (동일한 설정)
./lumina_cuda
```

### 커맨드라인 인자

```
./lumina [ref_dir] [n_packets] [n_iterations]
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `ref_dir` | `data/tardis_reference` | TARDIS 기준 데이터 디렉토리 |
| `n_packets` | 2000000 | Monte Carlo 패킷 수 |
| `n_iterations` | 20 | 수렴 반복 횟수 |

### 실행 예시

```bash
# 빠른 테스트 (1만 패킷, 5 반복 — 수 초)
./lumina data/tardis_reference 10000 5

# 중간 테스트 (20만 패킷, 10 반복)
./lumina data/tardis_reference 200000 10

# 프로덕션 (200만 패킷, 20 반복 — CPU 수 분, GPU 수십 초)
./lumina data/tardis_reference 2000000 20

# Make 단축 명령
make run    # 기본 설정으로 실행
make test   # 빠른 테스트 (1만 패킷, 5 반복)
```

### 예상 출력 (터미널)

```
=== LUMINA-SN Monte Carlo Transport ===
Reference data: data/tardis_reference
  n_shells = 30, n_lines = 137252
  T_inner = 10521.52 K, L_inner = 8.952e+42 erg/s
  ...

--- Iteration 1/20: 2000000 packets ---
  Escaped: 820143 (41.0%), Reabsorbed: 1179857 (59.0%)
  W error: 2.31%, T_rad error: 1.05%
  ...

--- Final Results ---
  T_inner: 10509.3 K (TARDIS: 10521.5 K, error: 0.12%)
  Max W error: 0.67%
  Max T_rad error: 0.28%
```

---

## 출력 파일

시뮬레이션은 프로젝트 루트에 다음 파일을 생성합니다:

| 파일 | 형식 | 설명 |
|------|------|------|
| `lumina_spectrum.csv` | CSV | 최종 스펙트럼 (wavelength_angstrom, L_lambda_cgs) |
| `lumina_spectrum_virtual.csv` | CSV | 가상 패킷 스펙트럼 (GPU만) |
| `lumina_spectrum_rotation.csv` | CSV | 회전 패킷 스펙트럼 |
| `lumina_plasma_state.csv` | CSV | 최종 플라즈마 상태 (shell, W, T_rad, n_e) |

### 스펙트럼 CSV 형식
```csv
wavelength_angstrom,L_lambda_cgs
500.0,1.234e+38
519.5,2.345e+38
...
```
- `wavelength_angstrom`: 파장 (Angstrom 단위, 500 ~ 20000)
- `L_lambda_cgs`: 광도 밀도 (erg/s/cm, 파장당 광도)

### 플라즈마 상태 CSV 형식
```csv
shell,W,T_rad,n_e
0,0.380,12291,2.34e+09
1,0.312,11543,1.98e+09
...
```
- `W`: 복사 희석 인자 (dilution factor)
- `T_rad`: 복사 온도 (K)
- `n_e`: 전자 밀도 (cm^-3)

---

## 시각화 (Visualization)

모든 Python 스크립트는 프로젝트 루트에서 실행합니다.

### 스펙트럼 비교 플롯 (주요)
```bash
python3 scripts/plot_spectrum_comparison.py
```
TARDIS 기준 스펙트럼과 LUMINA 스펙트럼을 비교하는 PNG 파일을 생성합니다:
- `spectrum_comparison.png` — 전체 파장 범위 비교
- `spectrum_comparison_siII.png` — Si II 6355A 흡수 트로프 확대

### 기타 분석 스크립트
```bash
python3 scripts/compare_spectra.py       # 스펙트럼 형상 + Si II 분석
python3 scripts/diagnose_w.py            # W 불일치 진단
python3 scripts/validate_plasma.py       # 플라즈마 상태 전체 검증
```

---

## FAQ / 트러블슈팅

### Q: `make cuda`에서 `nvcc: command not found` 오류
CUDA Toolkit이 설치되어 있지 않거나 PATH에 없습니다.
```bash
# CUDA 경로 확인
which nvcc
# 없으면 PATH에 추가
export PATH=/usr/local/cuda/bin:$PATH
```

### Q: `Failed to load reference data` 오류
`data/tardis_reference/` 디렉토리에 기준 데이터가 없습니다. TARDIS를 설치한 환경에서 먼저 기준 데이터를 생성해야 합니다:
```bash
# TARDIS가 설치된 Python 환경에서
python3 scripts/export_tardis_reference.py
```

### Q: GPU 빌드 시 `unsupported gpu architecture 'compute_89'` 오류
Makefile의 `-arch=sm_89`를 본인의 GPU에 맞게 수정하세요:
```bash
# GPU compute capability 확인
nvidia-smi
# 또는
nvcc --list-gpu-arch
```

### Q: 결과가 TARDIS와 다릅니다
- 패킷 수가 너무 적으면 통계적 노이즈가 큽니다. 최소 20만 패킷을 사용하세요.
- 20 반복을 사용해야 W/T_rad가 충분히 수렴합니다.
- 200만 패킷 × 20 반복에서 W 오차 < 1%, T_rad 오차 < 0.5%가 기대됩니다.

### Q: HDF5 경고 메시지
```
HDF5-DIAG: Error detected in HDF5 ...
```
원자 데이터 로드 시 나오는 비치명적(non-fatal) 경고입니다. 무시해도 됩니다.

### Q: `make clean`이 출력 파일을 삭제합니다
`make clean`은 바이너리만 삭제합니다. 출력 CSV/PNG는 `.gitignore`에 등록되어 있지만, 중요한 결과는 별도로 백업하세요.

---

## 기술 문서

자세한 기술 정보는 `docs/` 디렉토리를 참조하세요:

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — 코드 구조, 데이터 흐름, CPU vs GPU 차이점
- **[docs/PHYSICS.md](docs/PHYSICS.md)** — Monte Carlo 복사 전달, Sobolev 근사, 매크로 원자 모델
- **[docs/HISTORY.md](docs/HISTORY.md)** — 개발 이력, 주요 버그 수정 기록

---

## 참고 문헌

- Kerzendorf & Sim (2014), *TARDIS: A Monte Carlo radiative-transfer spectral synthesis code*, MNRAS 440, 387
- Lucy (2002), *Monte Carlo transition probabilities*, A&A 384, 725
- Lucy (2003), *Monte Carlo transition probabilities. II.*, A&A 403, 261
- Mazzali & Lucy (1993), *The application of Monte Carlo methods to the synthesis of early-time supernovae spectra*, A&A 279, 447
