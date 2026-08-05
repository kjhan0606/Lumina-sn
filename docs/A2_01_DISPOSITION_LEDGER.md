# A2-01 소유권 disposition 원장

- 행 수: 157
- 미분류: 0
- 이 표는 측량 결과이며 A2-01에서 공급원을 교체하지 않는다.

| 파일:행 | 심볼 | 현재 공급원 | 물리 의미 | 새 공급원 | 이행 단계 | 최종 상태 |
|---|---|---|---|---|---|---|
| src/lumina_plasma.c:4556 | W | local alias of plasma->W[s] | [rate] bound-bound dilute Planck pump | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4556 | T_rad | local alias of plasma->T_rad[s] | [rate] bound-bound Planck color | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4596 | W | local alias of plasma->W[s] | [rate] LTE comparison field amplitude | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4596 | T_rad | local alias of plasma->T_rad[s] | [rate] LTE comparison field color | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4701 | W | local alias of plasma->W[s] | [rate] line upward radiative rate | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4701 | T_rad | local alias of plasma->T_rad[s] | [rate] line upward radiative rate | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4879 | T_rad | local alias of plasma->T_rad[s] | [rate] Boltzmann fallback exponent in line rate | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4880 | W | local alias of plasma->W[s] | [rate] metastable dilution in line rate | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:9160 | T_rad | bf_rate_pop argument from plasma->T_rad | [rate] bound-free population exponent | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:9162 | W | bf_rate_pop argument from plasma->W | [rate] bound-free population dilution | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:11908 | W | local alias of plasma->W[s] | [rate] line source fallback | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:11908 | T_rad | local alias of plasma->T_rad[s] | [rate] line source fallback | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:11915 | W | local alias of plasma->W[s] | [rate] bin field construction | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:11915 | T_rad | local alias of plasma->T_rad[s] | [rate] bin field construction | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:11943 | W | local alias of plasma->W[s] | [rate] bound-free rate population call | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:11943 | T_rad | local alias of plasma->T_rad[s] | [rate] bound-free rate population call | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:11976 | W | local alias of plasma->W[s] | [rate] dilute photoheating integral | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:12034 | T_rad | local alias of plasma->T_rad[s] | [rate] Planck comparison in rate integral | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:12093 | W | local alias of plasma->W[s] | [rate] lower-level radiative weight | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:12100 | W | local alias of plasma->W[s] | [rate] upper-level radiative weight | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:13672 | W | local alias of plasma->W[s] | [rate] coupled bound-free rate call | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:13672 | T_rad | local alias of plasma->T_rad[s] | [rate] coupled bound-free rate call | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:13739 | W | local alias of plasma->W[s] | [rate] coupled lower-level weight | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:13743 | W | local alias of plasma->W[s] | [rate] coupled upper-level weight | RadiationField.J_nu | A2-05 | REPLACE_SCALAR_RATE_READ |
| src/lumina_main.c:747 | plasma.W[i] | plasma.W | [comparator] CPU reference W comparator | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_main.c:748 | plasma.T_rad[i] | plasma.T_rad | [comparator] CPU reference T_rad comparator | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_main.c:750 | plasma.W[i] | plasma.W | [comparator] CPU W comparison report | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_main.c:751 | plasma.T_rad[i] | plasma.T_rad | [comparator] CPU T_rad comparison report | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_main.c:757 | plasma.W[i] | plasma.W | [comparator] CPU W mean error | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_main.c:758 | plasma.T_rad[i] | plasma.T_rad | [comparator] CPU T_rad mean error | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_main.c:850 | plasma.W[i] | plasma.W | [comparator] CPU scalar comparison CSV | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_main.c:850 | plasma.T_rad[i] | plasma.T_rad | [comparator] CPU scalar comparison CSV | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_cuda.cu:10879 | plasma.W[i] | plasma.W | [comparator] CUDA-host reference W comparator | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_cuda.cu:10880 | plasma.T_rad[i] | plasma.T_rad | [comparator] CUDA-host reference T_rad comparator | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_cuda.cu:10882 | plasma.W[i] | plasma.W | [comparator] CUDA-host W comparison report | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_cuda.cu:10883 | plasma.T_rad[i] | plasma.T_rad | [comparator] CUDA-host T_rad comparison report | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_cuda.cu:10888 | plasma.W[i] | plasma.W | [comparator] CUDA-host W mean error | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_cuda.cu:10889 | plasma.T_rad[i] | plasma.T_rad | [comparator] CUDA-host T_rad mean error | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_bf_gemm.cu:82 | T_rad[s] | GPU BF-kernel T_rad parameter | [GPU_opacity_rate] GPU bound-free Boltzmann factor | RadiationField.J_nu | A2-14 | REPLACE_GPU_SCALAR_OPACITY_RATE_READ |
| src/lumina_bf_gemm.cu:83 | W[s] | GPU BF-kernel W parameter | [GPU_opacity_rate] GPU bound-free dilution factor | RadiationField.J_nu | A2-14 | REPLACE_GPU_SCALAR_OPACITY_RATE_READ |
| src/lumina_bf_gemm.cu:208 | plasma->T_rad | plasma.T_rad upload source | [GPU_opacity_rate] GPU BF rate state upload | RadiationField.J_nu | A2-14 | REPLACE_GPU_SCALAR_OPACITY_RATE_READ |
| src/lumina_bf_gemm.cu:210 | plasma->W | plasma.W upload source | [GPU_opacity_rate] GPU BF rate state upload | RadiationField.J_nu | A2-14 | REPLACE_GPU_SCALAR_OPACITY_RATE_READ |
| src/lumina_bf_gemm.cu:225 | g_bf_gemm.d_T_rad | GPU BF T_rad buffer | [GPU_opacity_rate] GPU BF kernel argument | RadiationField.J_nu | A2-14 | REPLACE_GPU_SCALAR_OPACITY_RATE_READ |
| src/lumina_bf_gemm.cu:225 | g_bf_gemm.d_W | GPU BF W buffer | [GPU_opacity_rate] GPU BF kernel argument | RadiationField.J_nu | A2-14 | REPLACE_GPU_SCALAR_OPACITY_RATE_READ |
| src/lumina_bf_gemm.cu:296 | plasma->T_rad | plasma.T_rad refresh source | [GPU_opacity_rate] GPU BF iteration refresh | RadiationField.J_nu | A2-14 | REPLACE_GPU_SCALAR_OPACITY_RATE_READ |
| src/lumina_bf_gemm.cu:297 | plasma->W | plasma.W refresh source | [GPU_opacity_rate] GPU BF iteration refresh | RadiationField.J_nu | A2-14 | REPLACE_GPU_SCALAR_OPACITY_RATE_READ |
| src/lumina_bf_gemm.cu:303 | g_bf_gemm.d_T_rad | GPU BF T_rad buffer | [GPU_opacity_rate] GPU BF refreshed kernel argument | RadiationField.J_nu | A2-14 | REPLACE_GPU_SCALAR_OPACITY_RATE_READ |
| src/lumina_bf_gemm.cu:304 | g_bf_gemm.d_W | GPU BF W buffer | [GPU_opacity_rate] GPU BF refreshed kernel argument | RadiationField.J_nu | A2-14 | REPLACE_GPU_SCALAR_OPACITY_RATE_READ |
| src/lumina_nlte_assemble.cu:169 | d_W[s] | GPU NLTE assembly W parameter | [GPU_opacity_rate] GPU bound-bound Planck fallback | RadiationField.J_nu | A2-14 | REPLACE_GPU_SCALAR_OPACITY_RATE_READ |
| src/lumina_nlte_assemble.cu:413 | plasma->W | plasma.W upload source | [GPU_opacity_rate] GPU NLTE assembly upload | RadiationField.J_nu | A2-14 | REPLACE_GPU_SCALAR_OPACITY_RATE_READ |
| src/lumina_nlte_assemble.cu:428 | plasma->T_rad[0] | plasma.T_rad | [GPU_opacity_rate] GPU NLTE dilute temperature fallback | RadiationField.J_nu | A2-14 | REPLACE_GPU_SCALAR_OPACITY_RATE_READ |
| src/lumina_cuda.cu:3760 | d_T_rad[shell_id] | GPU transport T_rad array | [GPU_transport] GPU BF re-emission temperature read | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_SCALAR_TRANSPORT_STATE |
| src/lumina_cuda.cu:3793 | d_T_rad[shell_id] | GPU transport T_rad array | [GPU_transport] GPU band re-emission temperature read | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_SCALAR_TRANSPORT_STATE |
| src/lumina_cuda.cu:5978 | d_T_rad | GPU transport T_rad pointer | [GPU_transport] transport kernel scalar-field argument | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_SCALAR_TRANSPORT_STATE |
| src/lumina_cuda.cu:6242 | d_T_rad | GPU transport T_rad pointer | [GPU_transport] transport interaction call | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_SCALAR_TRANSPORT_STATE |
| src/lumina_cuda.cu:6552 | d_T_rad | GPU transport T_rad pointer | [GPU_transport] legacy BF re-emission call | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_SCALAR_TRANSPORT_STATE |
| src/lumina_cuda.cu:8834 | dev.d_T_rad | GPU device T_rad owner | [GPU_transport] main transport launch argument | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_SCALAR_TRANSPORT_STATE |
| src/lumina_cuda.cu:10248 | dev.d_T_rad | GPU device T_rad owner | [GPU_transport] final transport launch argument | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_SCALAR_TRANSPORT_STATE |
| src/lumina_cuda.cu:8549 | plasma.W[s] | plasma.W | [GPU_transport] GPU-host packet source tier | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_SCALAR_TRANSPORT_STATE |
| src/lumina_cuda.cu:8550 | plasma.W | plasma.W owner pointer | [GPU_transport] GPU-host packet source validity gate | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_SCALAR_TRANSPORT_STATE |
| src/lumina_cuda.cu:10794 | plasma.T_rad[i] | plasma.T_rad | [GPU_transport] GPU-host transport temperature ratio | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_SCALAR_TRANSPORT_STATE |
| src/lumina_cuda.cu:10794 | plasma.T_rad[i] | plasma.T_rad | [GPU_transport] GPU-host transport temperature ratio denominator | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_SCALAR_TRANSPORT_STATE |
| src/lumina_plasma.c:2435 | plasma->T_rad[s] | plasma.T_rad | [opacity_rate] nebular ionization opacity/rate temperature | RadiationField.J_nu | A2-08 | REPLACE_SCALAR_OPACITY_RATE_READ |
| src/lumina_plasma.c:2437 | plasma->W[s] | plasma.W | [opacity_rate] nebular ionization opacity/rate dilution | RadiationField.J_nu | A2-08 | REPLACE_SCALAR_OPACITY_RATE_READ |
| src/lumina_plasma.c:2498 | T_rad | local alias of plasma->T_rad[s] | [opacity_rate] zeta interpolation temperature | RadiationField.J_nu | A2-08 | REPLACE_SCALAR_OPACITY_RATE_READ |
| src/lumina_plasma.c:2499 | T_rad | local alias of plasma->T_rad[s] | [opacity_rate] electron-to-radiation temperature ratio | RadiationField.J_nu | A2-08 | REPLACE_SCALAR_OPACITY_RATE_READ |
| src/lumina_plasma.c:2500 | W | local alias of plasma->W[s] | [opacity_rate] nebular rate dilution | RadiationField.J_nu | A2-08 | REPLACE_SCALAR_OPACITY_RATE_READ |
| src/lumina_plasma.c:2501 | T_rad | local alias of plasma->T_rad[s] | [opacity_rate] nebular rate temperature ratio | RadiationField.J_nu | A2-08 | REPLACE_SCALAR_OPACITY_RATE_READ |
| src/lumina_plasma.c:2502 | W | local alias of plasma->W[s] | [opacity_rate] non-metastable dilution term | RadiationField.J_nu | A2-08 | REPLACE_SCALAR_OPACITY_RATE_READ |
| src/lumina_plasma.c:2503 | T_rad | local alias of plasma->T_rad[s] | [opacity_rate] ML correction temperature | RadiationField.J_nu | A2-08 | REPLACE_SCALAR_OPACITY_RATE_READ |
| src/lumina_plasma.c:2504 | W | local alias of plasma->W[s] | [opacity_rate] two-component rate lock threshold | RadiationField.J_nu | A2-08 | REPLACE_SCALAR_OPACITY_RATE_READ |
| src/lumina_plasma.c:18369 | plasma->W[shell_mid] | plasma.W | [formal_transfer] observer continuum source | RadiationField.J_nu | A2-11 | REPLACE_FORMAL_TRANSFER_SCALAR_READ |
| src/lumina_plasma.c:18370 | plasma->T_rad[shell_mid] | plasma.T_rad | [formal_transfer] observer continuum source | RadiationField.J_nu | A2-11 | REPLACE_FORMAL_TRANSFER_SCALAR_READ |
| src/lumina_plasma.c:18389 | plasma->W[shell] | plasma.W | [formal_transfer] observer line fallback source | RadiationField.J_nu | A2-11 | REPLACE_FORMAL_TRANSFER_SCALAR_READ |
| src/lumina_plasma.c:18389 | plasma->T_rad[shell] | plasma.T_rad | [formal_transfer] observer line fallback source | RadiationField.J_nu | A2-11 | REPLACE_FORMAL_TRANSFER_SCALAR_READ |
| src/lumina_plasma.c:18416 | plasma->T_rad[shell] | plasma.T_rad | [formal_transfer] formal-transfer thermal width | RadiationField.J_nu | A2-11 | REPLACE_FORMAL_TRANSFER_SCALAR_READ |
| src/lumina_plasma.c:18416 | plasma->W[shell] | plasma.W | [formal_transfer] formal-transfer dilution | RadiationField.J_nu | A2-11 | REPLACE_FORMAL_TRANSFER_SCALAR_READ |
| src/lumina_plasma.c:18472 | plasma->W[shell_mid] | plasma.W | [formal_transfer] red-side continuum source | RadiationField.J_nu | A2-11 | REPLACE_FORMAL_TRANSFER_SCALAR_READ |
| src/lumina_plasma.c:18473 | plasma->T_rad[shell_mid] | plasma.T_rad | [formal_transfer] red-side continuum source | RadiationField.J_nu | A2-11 | REPLACE_FORMAL_TRANSFER_SCALAR_READ |
| src/lumina_plasma.c:18722 | plasma->W[shell] | plasma.W | [formal_transfer] electron-scattering source fallback | RadiationField.J_nu | A2-11 | REPLACE_FORMAL_TRANSFER_SCALAR_READ |
| src/lumina_bf_gemm.cu:140 | g_bf_gemm.d_T_rad | GPU BF T_rad allocation | [GPU_lifecycle] allocate GPU scalar owner | RadiationField generation lifecycle | A2-12 | REMOVE_GPU_SCALAR_LIFECYCLE |
| src/lumina_bf_gemm.cu:141 | g_bf_gemm.d_W | GPU BF W allocation | [GPU_lifecycle] allocate GPU scalar owner | RadiationField generation lifecycle | A2-12 | REMOVE_GPU_SCALAR_LIFECYCLE |
| src/lumina_bf_gemm.cu:390 | g_bf_gemm.d_T_rad | GPU BF T_rad allocation | [GPU_lifecycle] free GPU scalar owner | RadiationField generation lifecycle | A2-12 | REMOVE_GPU_SCALAR_LIFECYCLE |
| src/lumina_bf_gemm.cu:391 | g_bf_gemm.d_W | GPU BF W allocation | [GPU_lifecycle] free GPU scalar owner | RadiationField generation lifecycle | A2-12 | REMOVE_GPU_SCALAR_LIFECYCLE |
| src/lumina_cuda.cu:273 | dev->d_T_rad | GPU transport T_rad allocation | [GPU_lifecycle] allocate GPU scalar owner | RadiationField generation lifecycle | A2-12 | REMOVE_GPU_SCALAR_LIFECYCLE |
| src/lumina_cuda.cu:341 | dev->d_T_rad | GPU transport T_rad allocation | [GPU_lifecycle] test GPU scalar allocation | RadiationField generation lifecycle | A2-12 | REMOVE_GPU_SCALAR_LIFECYCLE |
| src/lumina_cuda.cu:342 | dev->d_T_rad | GPU transport T_rad allocation | [GPU_lifecycle] lazy allocate GPU scalar owner | RadiationField generation lifecycle | A2-12 | REMOVE_GPU_SCALAR_LIFECYCLE |
| src/lumina_cuda.cu:3286 | dev->d_T_rad | GPU transport T_rad allocation | [GPU_lifecycle] free GPU scalar owner | RadiationField generation lifecycle | A2-12 | REMOVE_GPU_SCALAR_LIFECYCLE |
| src/lumina_cuda.cu:1467 | plasma->T_rad[s] | plasma.T_rad | [GPU_rate] GPU-host NLTE Boltzmann fallback | RadiationField.J_nu | A2-13 | REPLACE_GPU_SCALAR_RATE_READ |
| src/lumina_cuda.cu:1621 | plasma->T_rad[s] | plasma.T_rad | [GPU_rate] GPU-host lower-ion fallback | RadiationField.J_nu | A2-13 | REPLACE_GPU_SCALAR_RATE_READ |
| src/lumina_cuda.cu:1652 | plasma->T_rad[s] | plasma.T_rad | [GPU_rate] GPU-host upper-ion fallback | RadiationField.J_nu | A2-13 | REPLACE_GPU_SCALAR_RATE_READ |
| src/lumina_cuda.cu:1682 | plasma->T_rad[s] | plasma.T_rad | [GPU_rate] GPU-host top-stage fallback | RadiationField.J_nu | A2-13 | REPLACE_GPU_SCALAR_RATE_READ |
| src/lumina_cuda.cu:2019 | plasma->T_rad[s] | plasma.T_rad | [GPU_rate] GPU-host rate dump electron seed | RadiationField.J_nu | A2-13 | REPLACE_GPU_SCALAR_RATE_READ |
| src/lumina_cuda.cu:2020 | plasma->T_rad[s] | plasma.T_rad | [GPU_rate] GPU-host rate dump radiation temperature | RadiationField.J_nu | A2-13 | REPLACE_GPU_SCALAR_RATE_READ |
| src/lumina_cuda.cu:2021 | plasma->W[s] | plasma.W | [GPU_rate] GPU-host rate dump dilution | RadiationField.J_nu | A2-13 | REPLACE_GPU_SCALAR_RATE_READ |
| src/lumina_cuda.cu:2068 | plasma->T_rad[s] | plasma.T_rad | [GPU_rate] GPU-host rate fallback seed | RadiationField.J_nu | A2-13 | REPLACE_GPU_SCALAR_RATE_READ |
| src/lumina_atomic.c:573 | W | plasma-state W input array | [owner_validation] validate owner presence | RadiationField commit API | A2-04 | VALIDATE_CANONICAL_FIELD_INSTEAD |
| src/lumina_atomic.c:573 | T_rad | plasma-state T_rad input array | [owner_validation] validate owner presence | RadiationField commit API | A2-04 | VALIDATE_CANONICAL_FIELD_INSTEAD |
| src/lumina_atomic.c:594 | W[s] | plasma-state W input array | [owner_validation] validate finite physical dilution | RadiationField commit API | A2-04 | VALIDATE_CANONICAL_FIELD_INSTEAD |
| src/lumina_atomic.c:595 | T_rad[s] | plasma-state T_rad input array | [owner_validation] validate finite positive color temperature | RadiationField commit API | A2-04 | VALIDATE_CANONICAL_FIELD_INSTEAD |
| src/lumina_atomic.c:599 | T_rad[s] | plasma-state T_rad input array | [owner_validation] validate color invariant | RadiationField commit API | A2-04 | VALIDATE_CANONICAL_FIELD_INSTEAD |
| src/lumina_atomic.c:599 | W[s] | plasma-state W input array | [owner_validation] validate color invariant | RadiationField commit API | A2-04 | VALIDATE_CANONICAL_FIELD_INSTEAD |
| src/lumina_cmfgen.c:663 | plasma->T_rad | plasma.T_rad owner pointer | [owner_validation] CMF solver owner-presence validation | RadiationField commit API | A2-04 | VALIDATE_CANONICAL_FIELD_INSTEAD |
| src/lumina_atomic.c:780 | plasma->T_rad[i2] | plasma.T_rad | [owner_update] fixed-color profile overwrite | RadiationField commit API | A2-04 | REMOVE_SCALAR_OWNER_UPDATE |
| src/lumina_plasma.c:917 | plasma->T_rad[i] | plasma.T_rad | [owner_update] fixed radiation profile update | RadiationField commit API | A2-04 | REMOVE_SCALAR_OWNER_UPDATE |
| src/lumina_plasma.c:918 | plasma->W[i] | plasma.W | [owner_update] fixed radiation profile update | RadiationField commit API | A2-04 | REMOVE_SCALAR_OWNER_UPDATE |
| src/lumina_plasma.c:949 | plasma->T_rad[i] | plasma.T_rad | [owner_update] damped T_rad owner update | RadiationField commit API | A2-04 | REMOVE_SCALAR_OWNER_UPDATE |
| src/lumina_plasma.c:950 | plasma->T_rad[i] | plasma.T_rad | [owner_update] damped T_rad prior generation read | RadiationField commit API | A2-04 | REMOVE_SCALAR_OWNER_UPDATE |
| src/lumina_plasma.c:951 | plasma->W[i] | plasma.W | [owner_update] damped W owner update | RadiationField commit API | A2-04 | REMOVE_SCALAR_OWNER_UPDATE |
| src/lumina_plasma.c:952 | plasma->W[i] | plasma.W | [owner_update] damped W prior generation read | RadiationField commit API | A2-04 | REMOVE_SCALAR_OWNER_UPDATE |
| src/lumina_plasma.c:2999 | plasma->T_rad[s] | plasma.T_rad | [seed_radeq] electron-temperature seed | RadiationField.J_nu | A2-16 | LIMIT_TO_GENERATION_ZERO_SEED |
| src/lumina_plasma.c:3038 | plasma->T_rad[s] | plasma.T_rad | [seed_radeq] analytic RADEQ radiation seed | RadiationField.J_nu | A2-16 | LIMIT_TO_GENERATION_ZERO_SEED |
| src/lumina_plasma.c:3039 | plasma->W[s] | plasma.W | [seed_radeq] analytic RADEQ energy-density seed | RadiationField.J_nu | A2-16 | LIMIT_TO_GENERATION_ZERO_SEED |
| src/lumina_plasma.c:3042 | T_rad | local alias of plasma->T_rad[s] | [seed_radeq] invalid-cell electron-temperature seed | RadiationField.J_nu | A2-16 | LIMIT_TO_GENERATION_ZERO_SEED |
| src/lumina_plasma.c:11631 | plasma->T_rad[s] | plasma.T_rad | [seed_radeq] RADEQ disabled-path seed | RadiationField.J_nu | A2-16 | LIMIT_TO_GENERATION_ZERO_SEED |
| src/lumina_plasma.c:11845 | T_rad | local alias of plasma->T_rad[s] | [seed_radeq] RADEQ invalid-cell seed | RadiationField.J_nu | A2-16 | LIMIT_TO_GENERATION_ZERO_SEED |
| src/lumina_atomic.c:761 | plasma->W | plasma.W owner pointer | [input_owner] load W column as runtime owner | RadiationField.J_nu | A2-16 | MOVE_TO_OFFLINE_LEGACY_CONVERTER |
| src/lumina_atomic.c:762 | plasma->T_rad | plasma.T_rad owner pointer | [input_owner] load T_rad column as runtime owner | RadiationField.J_nu | A2-16 | MOVE_TO_OFFLINE_LEGACY_CONVERTER |
| src/lumina_atomic.c:765 | plasma->W | plasma.W owner pointer | [input_owner] pass W owner into cross-field validation | RadiationField.J_nu | A2-16 | MOVE_TO_OFFLINE_LEGACY_CONVERTER |
| src/lumina_atomic.c:765 | plasma->T_rad | plasma.T_rad owner pointer | [input_owner] pass T_rad owner into cross-field validation | RadiationField.J_nu | A2-16 | MOVE_TO_OFFLINE_LEGACY_CONVERTER |
| src/lumina_atomic.c:785 | plasma->W[0] | plasma.W | [input_owner] loaded owner summary | RadiationField.J_nu | A2-16 | MOVE_TO_OFFLINE_LEGACY_CONVERTER |
| src/lumina_plasma.c:976 | plasma->T_rad[i] | plasma.T_rad | [diagnostic] binned-field fit diagnostic | RadiationField generation-bound diagnostic | A2-11 | KEEP_OUTPUT_ONLY_DIAGNOSTIC |
| src/lumina_cmfgen.c:970 | plasma->T_rad[s] | plasma.T_rad | [diagnostic] CMF frozen-state diagnostic | RadiationField generation-bound diagnostic | A2-11 | KEEP_OUTPUT_ONLY_DIAGNOSTIC |
| src/lumina_cmfgen.c:1612 | plasma->T_rad | plasma.T_rad owner array | [diagnostic] CMF state checksum diagnostic | RadiationField generation-bound diagnostic | A2-11 | KEEP_OUTPUT_ONLY_DIAGNOSTIC |
| src/lumina_element_wide.c:2329 | plasma->W[shell] | plasma.W | [diagnostic] element-wide provenance diagnostic | RadiationField generation-bound diagnostic | A2-11 | KEEP_OUTPUT_ONLY_DIAGNOSTIC |
| src/lumina_cuda.cu:5446 | d_T_rad | GPU transport T_rad pointer | [GPU_emissivity] GPU macro-atom Planck re-emission | RadiationField.J_nu | A2-15 | REPLACE_GPU_PLANCK_EMISSIVITY_READ |
| src/lumina_cuda.cu:5453 | d_T_rad | GPU transport T_rad pointer | [GPU_emissivity] GPU UV thermalization | RadiationField.J_nu | A2-15 | REPLACE_GPU_PLANCK_EMISSIVITY_READ |
| src/lumina_cuda.cu:5471 | d_T_rad | GPU transport T_rad pointer | [GPU_emissivity] GPU IR thermalization | RadiationField.J_nu | A2-15 | REPLACE_GPU_PLANCK_EMISSIVITY_READ |
| src/lumina_cuda.cu:5733 | d_T_rad | GPU transport T_rad pointer | [GPU_emissivity] GPU packet source re-emission | RadiationField.J_nu | A2-15 | REPLACE_GPU_PLANCK_EMISSIVITY_READ |
| src/lumina_plasma.c:13920 | W | local alias of plasma->W[s] | [rate_diagnostic] coupled-rate luminosity diagnostic | RadiationField generation-bound diagnostic | A2-06 | DERIVE_RATE_DIAGNOSTIC_FROM_CANONICAL_FIELD |
| src/lumina_plasma.c:13940 | W | local alias of plasma->W[s] | [rate_diagnostic] coupled-rate floor diagnostic | RadiationField generation-bound diagnostic | A2-06 | DERIVE_RATE_DIAGNOSTIC_FROM_CANONICAL_FIELD |
| src/lumina_plasma.c:14080 | T_rad | local alias of plasma->T_rad[s] | [rate_diagnostic] coupled-rate residual diagnostic | RadiationField generation-bound diagnostic | A2-06 | DERIVE_RATE_DIAGNOSTIC_FROM_CANONICAL_FIELD |
| src/lumina_cuda.cu:530 | plasma->T_rad | plasma.T_rad upload source | [GPU_transfer] transport scalar upload | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_TRANSFER_SCALAR_READ |
| src/lumina_cuda.cu:10008 | plasma.W[i] | plasma.W | [GPU_transfer] GPU transfer-state CSV | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_TRANSFER_SCALAR_READ |
| src/lumina_cuda.cu:10008 | plasma.T_rad[i] | plasma.T_rad | [GPU_transfer] GPU transfer-state CSV | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_TRANSFER_SCALAR_READ |
| src/lumina_cmfgen.c:908 | plasma->T_rad[s] | plasma.T_rad | [opacity] CMF emissivity/opacity regime split | RadiationField.J_nu | A2-08 | REPLACE_OPACITY_SCALAR_READ |
| src/lumina_cmfgen.c:2144 | plasma->T_rad[s] | plasma.T_rad | [opacity] CMF hot-regime opacity split | RadiationField.J_nu | A2-08 | REPLACE_OPACITY_SCALAR_READ |
| src/lumina_plasma.c:18010 | T_rad | local alias of plasma->T_rad[s] | [opacity] formal opacity thermal width | RadiationField.J_nu | A2-08 | REPLACE_OPACITY_SCALAR_READ |
| src/lumina_plasma.c:14987 | plasma->T_rad[shell] | plasma.T_rad | [seed_rate] NLTE rate seed temperature | RadiationField.J_nu | A2-16 | LIMIT_RATE_SEED_TO_GENERATION_ZERO |
| src/lumina_plasma.c:15179 | plasma->W[shell] | plasma.W | [seed_rate] dilute GPU-assembly seed field | RadiationField.J_nu | A2-16 | LIMIT_RATE_SEED_TO_GENERATION_ZERO |
| src/lumina_plasma.c:15181 | plasma->T_rad[0] | plasma.T_rad | [seed_rate] dilute GPU-assembly seed color | RadiationField.J_nu | A2-16 | LIMIT_RATE_SEED_TO_GENERATION_ZERO |
| src/lumina_atomic.c:1008 | ps->W | plasma.W allocation | [lifecycle] free scalar owner | RadiationField generation lifecycle | A2-17 | REMOVE_SCALAR_LIFECYCLE |
| src/lumina_atomic.c:1009 | ps->T_rad | plasma.T_rad allocation | [lifecycle] free scalar owner | RadiationField generation lifecycle | A2-17 | REMOVE_SCALAR_LIFECYCLE |
| src/lumina_main.c:334 | plasma.T_rad[i] | plasma.T_rad | [output] CPU plasma-state owner output | RadiationField generation-bound diagnostic | A2-17 | REMOVE_SCALAR_OWNER_OUTPUT |
| src/lumina_cuda.cu:11020 | plasma.W[i] | plasma.W | [output] CUDA plasma-state owner output | RadiationField generation-bound diagnostic | A2-17 | REMOVE_SCALAR_OWNER_OUTPUT |
| src/lumina_plasma.c:2081 | plasma->T_rad[s] | plasma.T_rad | [Boltzmann_partition] partition-function temperature | plasma->T_e | A2-07 | USE_MATTER_TEMPERATURE |
| src/lumina_plasma.c:2082 | plasma->W[s] | plasma.W | [Boltzmann_partition] non-metastable partition dilution | plasma->T_e | A2-07 | USE_MATTER_TEMPERATURE |
| src/lumina_plasma.c:2826 | plasma->T_rad[s] | plasma.T_rad | [transition_probability] macro-atom transition population temperature | Jbar[RadiationField.J_nu] | A2-09 | DERIVE_TRANSITION_PROBABILITY_FROM_JBAR |
| src/lumina_plasma.c:2827 | plasma->W[s] | plasma.W | [transition_probability] macro-atom transition population dilution | Jbar[RadiationField.J_nu] | A2-09 | DERIVE_TRANSITION_PROBABILITY_FROM_JBAR |
| src/lumina_plasma.c:7402 | plasma->T_rad[s] | plasma.T_rad | [rate_Boltzmann] Boltzmann rate temperature | plasma->T_e | A2-07 | USE_MATTER_TEMPERATURE_FOR_BOLTZMANN_RATE |
| src/lumina_plasma.c:7403 | plasma->W[s] | plasma.W | [rate_Boltzmann] Boltzmann rate dilution | plasma->T_e | A2-07 | USE_MATTER_TEMPERATURE_FOR_BOLTZMANN_RATE |
| src/lumina_plasma.c:12379 | plasma->T_rad[s] | plasma.T_rad | [rate_radeq] RADEQ rate temperature | RadiationField.J_nu | A2-10 | USE_CANONICAL_FIELD_IN_RADEQ |
| src/lumina_plasma.c:12380 | plasma->W[s] | plasma.W | [rate_radeq] RADEQ rate dilution | RadiationField.J_nu | A2-10 | USE_CANONICAL_FIELD_IN_RADEQ |
| src/lumina_plasma.c:17832 | plasma->T_rad[s] | plasma.T_rad | [Boltzmann_diagnostic] level-population Boltzmann diagnostic | plasma->T_e | A2-07 | DIAGNOSE_BOLTZMANN_WITH_MATTER_TEMPERATURE |
| src/lumina_plasma.c:17833 | plasma->W[s] | plasma.W | [Boltzmann_diagnostic] level-population dilution diagnostic | plasma->T_e | A2-07 | DIAGNOSE_BOLTZMANN_WITH_MATTER_TEMPERATURE |
| src/lumina_atomic.c:826 | plasma->T_rad[i] | plasma.T_rad | [seed] initial electron-temperature seed | RadiationField.J_nu | A2-16 | LIMIT_SCALAR_SEED_TO_GENERATION_ZERO |
| src/lumina_plasma.c:7897 | plasma->T_rad[pkt->current_shell_id] | plasma.T_rad | [emissivity] CPU BF Planck re-emission | RadiationField.J_nu | A2-09 | REPLACE_PLANCK_REEMISSION_SOURCE |
