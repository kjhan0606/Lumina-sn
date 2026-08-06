# A2-01 소유권 disposition 원장

- 행 수: 157
- 미분류: 0
- 이 표는 측량 결과이며 A2-01에서 공급원을 교체하지 않는다.

| 파일:행 | 심볼 | 현재 공급원 | 물리 의미 | 새 공급원 | 이행 단계 | 최종 상태 |
|---|---|---|---|---|---|---|
| src/lumina_plasma.c:4632 | W | local alias of plasma->W[s] | [rate] bound-bound dilute Planck pump | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4632 | T_rad | local alias of plasma->T_rad[s] | [rate] bound-bound Planck color | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4672 | W | local alias of plasma->W[s] | [rate] LTE comparison field amplitude | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4672 | T_rad | local alias of plasma->T_rad[s] | [rate] LTE comparison field color | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4777 | W | local alias of plasma->W[s] | [rate] line upward radiative rate | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:4778 | T_rad | local alias of plasma->T_rad[s] | [rate] line upward radiative rate | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | T_rad | local alias of plasma->T_rad[s] | [rate] Boltzmann fallback exponent in line rate | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | W | local alias of plasma->W[s] | [rate] metastable dilution in line rate | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | T_rad | bf_rate_pop argument from plasma->T_rad | [rate] bound-free population exponent | RadiationField.J_nu | A2-07 | REPLACE_SCALAR_RATE_READ |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | W | bf_rate_pop argument from plasma->W | [rate] bound-free population dilution | RadiationField.J_nu | A2-07 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:12066 | W | local alias of plasma->W[s] | [rate] line source fallback | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:12066 | T_rad | local alias of plasma->T_rad[s] | [rate] line source fallback | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:12073 | W | local alias of plasma->W[s] | [rate] bin field construction | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:12073 | T_rad | local alias of plasma->T_rad[s] | [rate] bin field construction | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | W | local alias of plasma->W[s] | [rate] bound-free rate population call | RadiationField.J_nu | A2-07 | REPLACE_SCALAR_RATE_READ |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | T_rad | local alias of plasma->T_rad[s] | [rate] bound-free rate population call | RadiationField.J_nu | A2-07 | REPLACE_SCALAR_RATE_READ |
| src/lumina_plasma.c:12134 | W | local alias of plasma->W[s] | [rate] dilute photoheating integral | (진단 유지) | 진단 | KEEP_DIAGNOSTIC_READ |
| src/lumina_plasma.c:12192 | T_rad | local alias of plasma->T_rad[s] | [rate] Planck comparison in rate integral | (진단 유지) | 진단 | KEEP_DIAGNOSTIC_READ |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | W | local alias of plasma->W[s] | [rate] lower-level radiative weight | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | W | local alias of plasma->W[s] | [rate] upper-level radiative weight | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | W | local alias of plasma->W[s] | [rate] coupled bound-free rate call | RadiationField.J_nu | A2-07 | REPLACE_SCALAR_RATE_READ |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | T_rad | local alias of plasma->T_rad[s] | [rate] coupled bound-free rate call | RadiationField.J_nu | A2-07 | REPLACE_SCALAR_RATE_READ |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | W | local alias of plasma->W[s] | [rate] coupled lower-level weight | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | W | local alias of plasma->W[s] | [rate] coupled upper-level weight | RadiationField.J_nu | A2-06 | REPLACE_SCALAR_RATE_READ |
| src/lumina_main.c:828 | plasma.W[i] | plasma.W | [comparator] CPU reference W comparator | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_main.c:829 | plasma.T_rad[i] | plasma.T_rad | [comparator] CPU reference T_rad comparator | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_main.c:831 | plasma.W[i] | plasma.W | [comparator] CPU W comparison report | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_main.c:832 | plasma.T_rad[i] | plasma.T_rad | [comparator] CPU T_rad comparison report | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_main.c:838 | plasma.W[i] | plasma.W | [comparator] CPU W mean error | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_main.c:839 | plasma.T_rad[i] | plasma.T_rad | [comparator] CPU T_rad mean error | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_main.c:931 | plasma.W[i] | plasma.W | [comparator] CPU scalar comparison CSV | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_main.c:931 | plasma.T_rad[i] | plasma.T_rad | [comparator] CPU scalar comparison CSV | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_cuda.cu:10899 | plasma.W[i] | plasma.W | [comparator] CUDA-host reference W comparator | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_cuda.cu:10900 | plasma.T_rad[i] | plasma.T_rad | [comparator] CUDA-host reference T_rad comparator | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_cuda.cu:10902 | plasma.W[i] | plasma.W | [comparator] CUDA-host W comparison report | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_cuda.cu:10903 | plasma.T_rad[i] | plasma.T_rad | [comparator] CUDA-host T_rad comparison report | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_cuda.cu:10908 | plasma.W[i] | plasma.W | [comparator] CUDA-host W mean error | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
| src/lumina_cuda.cu:10909 | plasma.T_rad[i] | plasma.T_rad | [comparator] CUDA-host T_rad mean error | RadiationField generation-bound diagnostic | A2-11 | KEEP_DIAGNOSTIC_ONLY |
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
| src/lumina_cuda.cu:8842 | dev.d_T_rad | GPU device T_rad owner | [GPU_transport] main transport launch argument | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_SCALAR_TRANSPORT_STATE |
| src/lumina_cuda.cu:10256 | dev.d_T_rad | GPU device T_rad owner | [GPU_transport] final transport launch argument | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_SCALAR_TRANSPORT_STATE |
| src/lumina_cuda.cu:8557 | plasma.W[s] | plasma.W | [GPU_transport] GPU-host packet source tier | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_SCALAR_TRANSPORT_STATE |
| src/lumina_cuda.cu:8558 | plasma.W | plasma.W owner pointer | [GPU_transport] GPU-host packet source validity gate | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_SCALAR_TRANSPORT_STATE |
| src/lumina_cuda.cu:10814 | plasma.T_rad[i] | plasma.T_rad | [GPU_transport] GPU-host transport temperature ratio | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_SCALAR_TRANSPORT_STATE |
| src/lumina_cuda.cu:10814 | plasma.T_rad[i] | plasma.T_rad | [GPU_transport] GPU-host transport temperature ratio denominator | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_SCALAR_TRANSPORT_STATE |
| src/lumina_plasma.c:2624 | plasma->T_rad[s] | plasma.T_rad | [opacity_rate] nebular ionization opacity/rate temperature | RadiationField.J_nu | A2-08 | REPLACE_SCALAR_OPACITY_RATE_READ |
| src/lumina_plasma.c:2626 | plasma->W[s] | plasma.W | [opacity_rate] nebular ionization opacity/rate dilution | RadiationField.J_nu | A2-08 | REPLACE_SCALAR_OPACITY_RATE_READ |
| src/lumina_plasma.c:2695 | T_rad | local alias of plasma->T_rad[s] | [opacity_rate] zeta interpolation temperature | RadiationField.J_nu | A2-08 | REPLACE_SCALAR_OPACITY_RATE_READ |
| src/lumina_plasma.c:2696 | T_rad | local alias of plasma->T_rad[s] | [opacity_rate] electron-to-radiation temperature ratio | RadiationField.J_nu | A2-08 | REPLACE_SCALAR_OPACITY_RATE_READ |
| src/lumina_plasma.c:2697 | W | local alias of plasma->W[s] | [opacity_rate] nebular rate dilution | RadiationField.J_nu | A2-08 | REPLACE_SCALAR_OPACITY_RATE_READ |
| src/lumina_plasma.c:2698 | T_rad | local alias of plasma->T_rad[s] | [opacity_rate] nebular rate temperature ratio | RadiationField.J_nu | A2-08 | REPLACE_SCALAR_OPACITY_RATE_READ |
| src/lumina_plasma.c:2699 | W | local alias of plasma->W[s] | [opacity_rate] non-metastable dilution term | RadiationField.J_nu | A2-08 | REPLACE_SCALAR_OPACITY_RATE_READ |
| src/lumina_plasma.c:2700 | T_rad | local alias of plasma->T_rad[s] | [opacity_rate] ML correction temperature | RadiationField.J_nu | A2-08 | REPLACE_SCALAR_OPACITY_RATE_READ |
| src/lumina_plasma.c:2701 | W | local alias of plasma->W[s] | [opacity_rate] two-component rate lock threshold | RadiationField.J_nu | A2-08 | REPLACE_SCALAR_OPACITY_RATE_READ |
| src/lumina_plasma.c:18673 | plasma->W[shell_mid] | plasma.W | [formal_transfer] observer continuum source | RadiationField.J_nu | A2-11 | REPLACE_FORMAL_TRANSFER_SCALAR_READ |
| src/lumina_plasma.c:18674 | plasma->T_rad[shell_mid] | plasma.T_rad | [formal_transfer] observer continuum source | RadiationField.J_nu | A2-11 | REPLACE_FORMAL_TRANSFER_SCALAR_READ |
| src/lumina_plasma.c:18693 | plasma->W[shell] | plasma.W | [formal_transfer] observer line fallback source | RadiationField.J_nu | A2-11 | REPLACE_FORMAL_TRANSFER_SCALAR_READ |
| src/lumina_plasma.c:18693 | plasma->T_rad[shell] | plasma.T_rad | [formal_transfer] observer line fallback source | RadiationField.J_nu | A2-11 | REPLACE_FORMAL_TRANSFER_SCALAR_READ |
| src/lumina_plasma.c:18720 | plasma->T_rad[shell] | plasma.T_rad | [formal_transfer] formal-transfer thermal width | RadiationField.J_nu | A2-11 | REPLACE_FORMAL_TRANSFER_SCALAR_READ |
| src/lumina_plasma.c:18720 | plasma->W[shell] | plasma.W | [formal_transfer] formal-transfer dilution | RadiationField.J_nu | A2-11 | REPLACE_FORMAL_TRANSFER_SCALAR_READ |
| src/lumina_plasma.c:18776 | plasma->W[shell_mid] | plasma.W | [formal_transfer] red-side continuum source | RadiationField.J_nu | A2-11 | REPLACE_FORMAL_TRANSFER_SCALAR_READ |
| src/lumina_plasma.c:18777 | plasma->T_rad[shell_mid] | plasma.T_rad | [formal_transfer] red-side continuum source | RadiationField.J_nu | A2-11 | REPLACE_FORMAL_TRANSFER_SCALAR_READ |
| src/lumina_plasma.c:19026 | plasma->W[shell] | plasma.W | [formal_transfer] electron-scattering source fallback | RadiationField.J_nu | A2-11 | REPLACE_FORMAL_TRANSFER_SCALAR_READ |
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
| src/lumina_atomic.c:662 | W | plasma-state W input array | [owner_validation] validate owner presence | RadiationField commit API | A2-04 | VALIDATE_CANONICAL_FIELD_INSTEAD |
| src/lumina_atomic.c:662 | T_rad | plasma-state T_rad input array | [owner_validation] validate owner presence | RadiationField commit API | A2-04 | VALIDATE_CANONICAL_FIELD_INSTEAD |
| src/lumina_atomic.c:683 | W[s] | plasma-state W input array | [owner_validation] validate finite physical dilution | RadiationField commit API | A2-04 | VALIDATE_CANONICAL_FIELD_INSTEAD |
| src/lumina_atomic.c:684 | T_rad[s] | plasma-state T_rad input array | [owner_validation] validate finite positive color temperature | RadiationField commit API | A2-04 | VALIDATE_CANONICAL_FIELD_INSTEAD |
| src/lumina_atomic.c:688 | T_rad[s] | plasma-state T_rad input array | [owner_validation] validate color invariant | RadiationField commit API | A2-04 | VALIDATE_CANONICAL_FIELD_INSTEAD |
| src/lumina_atomic.c:688 | W[s] | plasma-state W input array | [owner_validation] validate color invariant | RadiationField commit API | A2-04 | VALIDATE_CANONICAL_FIELD_INSTEAD |
| src/lumina_cmfgen.c:663 | plasma->T_rad | plasma.T_rad owner pointer | [owner_validation] CMF solver owner-presence validation | RadiationField commit API | A2-04 | VALIDATE_CANONICAL_FIELD_INSTEAD |
| src/lumina_atomic.c:869 | plasma->T_rad[i2] | plasma.T_rad | [owner_update] fixed-color profile overwrite | RadiationField commit API | A2-04 | REMOVE_SCALAR_OWNER_UPDATE |
| src/lumina_plasma.c:1123 | plasma->T_rad[i] | plasma.T_rad | [owner_update] fixed radiation profile update | RadiationField commit API | A2-04 | REMOVE_SCALAR_OWNER_UPDATE |
| src/lumina_plasma.c:1124 | plasma->W[i] | plasma.W | [owner_update] fixed radiation profile update | RadiationField commit API | A2-04 | REMOVE_SCALAR_OWNER_UPDATE |
| src/lumina_plasma.c:1155 | plasma->T_rad[i] | plasma.T_rad | [owner_update] damped T_rad owner update | RadiationField commit API | A2-04 | REMOVE_SCALAR_OWNER_UPDATE |
| src/lumina_plasma.c:1156 | plasma->T_rad[i] | plasma.T_rad | [owner_update] damped T_rad prior generation read | RadiationField commit API | A2-04 | REMOVE_SCALAR_OWNER_UPDATE |
| src/lumina_plasma.c:1157 | plasma->W[i] | plasma.W | [owner_update] damped W owner update | RadiationField commit API | A2-04 | REMOVE_SCALAR_OWNER_UPDATE |
| src/lumina_plasma.c:1158 | plasma->W[i] | plasma.W | [owner_update] damped W prior generation read | RadiationField commit API | A2-04 | REMOVE_SCALAR_OWNER_UPDATE |
| src/lumina_plasma.c:3120 | plasma->T_rad[s] | plasma.T_rad | [seed_radeq] electron-temperature seed | RadiationField.J_nu | A2-16 | LIMIT_TO_GENERATION_ZERO_SEED |
| src/lumina_plasma.c:3159 | plasma->T_rad[s] | plasma.T_rad | [seed_radeq] analytic RADEQ radiation seed | RadiationField.J_nu | A2-16 | LIMIT_TO_GENERATION_ZERO_SEED |
| src/lumina_plasma.c:3160 | plasma->W[s] | plasma.W | [seed_radeq] analytic RADEQ energy-density seed | RadiationField.J_nu | A2-16 | LIMIT_TO_GENERATION_ZERO_SEED |
| src/lumina_plasma.c:3163 | T_rad | local alias of plasma->T_rad[s] | [seed_radeq] invalid-cell electron-temperature seed | RadiationField.J_nu | A2-16 | LIMIT_TO_GENERATION_ZERO_SEED |
| src/lumina_plasma.c:11789 | plasma->T_rad[s] | plasma.T_rad | [seed_radeq] RADEQ disabled-path seed | RadiationField.J_nu | A2-16 | LIMIT_TO_GENERATION_ZERO_SEED |
| src/lumina_plasma.c:12003 | T_rad | local alias of plasma->T_rad[s] | [seed_radeq] RADEQ invalid-cell seed | RadiationField.J_nu | A2-16 | LIMIT_TO_GENERATION_ZERO_SEED |
| src/lumina_atomic.c:850 | plasma->W | plasma.W owner pointer | [input_owner] load W column as runtime owner | RadiationField.J_nu | A2-16 | MOVE_TO_OFFLINE_LEGACY_CONVERTER |
| src/lumina_atomic.c:851 | plasma->T_rad | plasma.T_rad owner pointer | [input_owner] load T_rad column as runtime owner | RadiationField.J_nu | A2-16 | MOVE_TO_OFFLINE_LEGACY_CONVERTER |
| src/lumina_atomic.c:854 | plasma->W | plasma.W owner pointer | [input_owner] pass W owner into cross-field validation | RadiationField.J_nu | A2-16 | MOVE_TO_OFFLINE_LEGACY_CONVERTER |
| src/lumina_atomic.c:854 | plasma->T_rad | plasma.T_rad owner pointer | [input_owner] pass T_rad owner into cross-field validation | RadiationField.J_nu | A2-16 | MOVE_TO_OFFLINE_LEGACY_CONVERTER |
| src/lumina_atomic.c:874 | plasma->W[0] | plasma.W | [input_owner] loaded owner summary | RadiationField.J_nu | A2-16 | MOVE_TO_OFFLINE_LEGACY_CONVERTER |
| src/lumina_plasma.c:1182 | plasma->T_rad[i] | plasma.T_rad | [diagnostic] binned-field fit diagnostic | RadiationField generation-bound diagnostic | A2-11 | KEEP_OUTPUT_ONLY_DIAGNOSTIC |
| src/lumina_cmfgen.c:970 | plasma->T_rad[s] | plasma.T_rad | [diagnostic] CMF frozen-state diagnostic | RadiationField generation-bound diagnostic | A2-11 | KEEP_OUTPUT_ONLY_DIAGNOSTIC |
| src/lumina_cmfgen.c:1612 | plasma->T_rad | plasma.T_rad owner array | [diagnostic] CMF state checksum diagnostic | RadiationField generation-bound diagnostic | A2-11 | KEEP_OUTPUT_ONLY_DIAGNOSTIC |
| src/lumina_element_wide.c:2325 | plasma->W[shell] | plasma.W | [diagnostic] element-wide provenance diagnostic | RadiationField generation-bound diagnostic | A2-11 | KEEP_OUTPUT_ONLY_DIAGNOSTIC |
| src/lumina_cuda.cu:5446 | d_T_rad | GPU transport T_rad pointer | [GPU_emissivity] GPU macro-atom Planck re-emission | RadiationField.J_nu | A2-15 | REPLACE_GPU_PLANCK_EMISSIVITY_READ |
| src/lumina_cuda.cu:5453 | d_T_rad | GPU transport T_rad pointer | [GPU_emissivity] GPU UV thermalization | RadiationField.J_nu | A2-15 | REPLACE_GPU_PLANCK_EMISSIVITY_READ |
| src/lumina_cuda.cu:5471 | d_T_rad | GPU transport T_rad pointer | [GPU_emissivity] GPU IR thermalization | RadiationField.J_nu | A2-15 | REPLACE_GPU_PLANCK_EMISSIVITY_READ |
| src/lumina_cuda.cu:5733 | d_T_rad | GPU transport T_rad pointer | [GPU_emissivity] GPU packet source re-emission | RadiationField.J_nu | A2-15 | REPLACE_GPU_PLANCK_EMISSIVITY_READ |
| src/lumina_plasma.c:14127 | W | local alias of plasma->W[s] | [rate_diagnostic] coupled-rate luminosity diagnostic | RadiationField generation-bound diagnostic | A2-06 | DERIVE_RATE_DIAGNOSTIC_FROM_CANONICAL_FIELD |
| src/lumina_plasma.c:14147 | W | local alias of plasma->W[s] | [rate_diagnostic] coupled-rate floor diagnostic | RadiationField generation-bound diagnostic | A2-06 | DERIVE_RATE_DIAGNOSTIC_FROM_CANONICAL_FIELD |
| src/lumina_plasma.c:14287 | T_rad | local alias of plasma->T_rad[s] | [rate_diagnostic] coupled-rate residual diagnostic | RadiationField generation-bound diagnostic | A2-06 | DERIVE_RATE_DIAGNOSTIC_FROM_CANONICAL_FIELD |
| src/lumina_cuda.cu:530 | plasma->T_rad | plasma.T_rad upload source | [GPU_transfer] transport scalar upload | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_TRANSFER_SCALAR_READ |
| src/lumina_cuda.cu:10016 | plasma.W[i] | plasma.W | [GPU_transfer] GPU transfer-state CSV | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_TRANSFER_SCALAR_READ |
| src/lumina_cuda.cu:10016 | plasma.T_rad[i] | plasma.T_rad | [GPU_transfer] GPU transfer-state CSV | RadiationField generation lifecycle | A2-12 | REPLACE_GPU_TRANSFER_SCALAR_READ |
| src/lumina_cmfgen.c:908 | plasma->T_rad[s] | plasma.T_rad | [opacity] CMF emissivity/opacity regime split | RadiationField.J_nu | A2-08 | REPLACE_OPACITY_SCALAR_READ |
| src/lumina_cmfgen.c:2144 | plasma->T_rad[s] | plasma.T_rad | [opacity] CMF hot-regime opacity split | RadiationField.J_nu | A2-08 | REPLACE_OPACITY_SCALAR_READ |
| src/lumina_plasma.c:18314 | T_rad | local alias of plasma->T_rad[s] | [opacity] formal opacity thermal width | RadiationField.J_nu | A2-08 | REPLACE_OPACITY_SCALAR_READ |
| src/lumina_plasma.c:15230 | plasma->T_rad[shell] | plasma.T_rad | [seed_rate] NLTE rate seed temperature | RadiationField.J_nu | A2-16 | LIMIT_RATE_SEED_TO_GENERATION_ZERO |
| src/lumina_plasma.c:15422 | plasma->W[shell] | plasma.W | [seed_rate] dilute GPU-assembly seed field | RadiationField.J_nu | A2-16 | LIMIT_RATE_SEED_TO_GENERATION_ZERO |
| src/lumina_plasma.c:15424 | plasma->T_rad[0] | plasma.T_rad | [seed_rate] dilute GPU-assembly seed color | RadiationField.J_nu | A2-16 | LIMIT_RATE_SEED_TO_GENERATION_ZERO |
| src/lumina_atomic.c:1097 | ps->W | plasma.W allocation | [lifecycle] free scalar owner | RadiationField generation lifecycle | A2-17 | REMOVE_SCALAR_LIFECYCLE |
| src/lumina_atomic.c:1098 | ps->T_rad | plasma.T_rad allocation | [lifecycle] free scalar owner | RadiationField generation lifecycle | A2-17 | REMOVE_SCALAR_LIFECYCLE |
| src/lumina_main.c:357 | plasma.T_rad[i] | plasma.T_rad | [output] CPU plasma-state owner output | RadiationField generation-bound diagnostic | A2-17 | REMOVE_SCALAR_OWNER_OUTPUT |
| src/lumina_cuda.cu:11040 | plasma.W[i] | plasma.W | [output] CUDA plasma-state owner output | RadiationField generation-bound diagnostic | A2-17 | REMOVE_SCALAR_OWNER_OUTPUT |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | plasma->T_rad[s] | plasma.T_rad | [Boltzmann_partition] partition-function temperature | plasma->T_e | A2-07 | USE_MATTER_TEMPERATURE |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | plasma->W[s] | plasma.W | [Boltzmann_partition] non-metastable partition dilution | plasma->T_e | A2-07 | USE_MATTER_TEMPERATURE |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | plasma->T_rad[s] | plasma.T_rad | [transition_probability] macro-atom transition population temperature | Jbar[RadiationField.J_nu] | A2-09 | DERIVE_TRANSITION_PROBABILITY_FROM_JBAR |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | plasma->W[s] | plasma.W | [transition_probability] macro-atom transition population dilution | Jbar[RadiationField.J_nu] | A2-09 | DERIVE_TRANSITION_PROBABILITY_FROM_JBAR |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | plasma->T_rad[s] | plasma.T_rad | [rate_Boltzmann] Boltzmann rate temperature | plasma->T_e | A2-07 | USE_MATTER_TEMPERATURE_FOR_BOLTZMANN_RATE |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | plasma->W[s] | plasma.W | [rate_Boltzmann] Boltzmann rate dilution | plasma->T_e | A2-07 | USE_MATTER_TEMPERATURE_FOR_BOLTZMANN_RATE |
| src/lumina_plasma.c:12561 | plasma->T_rad[s] | plasma.T_rad | [rate_radeq] RADEQ rate temperature | RadiationField.J_nu | A2-10 | USE_CANONICAL_FIELD_IN_RADEQ |
| src/lumina_plasma.c:12562 | plasma->W[s] | plasma.W | [rate_radeq] RADEQ rate dilution | RadiationField.J_nu | A2-10 | USE_CANONICAL_FIELD_IN_RADEQ |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | plasma->T_rad[s] | plasma.T_rad | [Boltzmann_diagnostic] level-population Boltzmann diagnostic | plasma->T_e | A2-07 | DIAGNOSE_BOLTZMANN_WITH_MATTER_TEMPERATURE |
| 이관 완료(A2-07·3ddd95c0de20abea3284ca326ce41b7968d4b26d) | plasma->W[s] | plasma.W | [Boltzmann_diagnostic] level-population dilution diagnostic | plasma->T_e | A2-07 | DIAGNOSE_BOLTZMANN_WITH_MATTER_TEMPERATURE |
| src/lumina_atomic.c:915 | plasma->T_rad[i] | plasma.T_rad | [seed] initial electron-temperature seed | RadiationField.J_nu | A2-16 | LIMIT_SCALAR_SEED_TO_GENERATION_ZERO |
| src/lumina_plasma.c:8043 | plasma->T_rad[pkt->current_shell_id] | plasma.T_rad | [emissivity] CPU BF Planck re-emission | RadiationField.J_nu | A2-09 | REPLACE_PLANCK_REEMISSION_SOURCE |

## ADDENDUM (A2-05 폐합, 2026-08-06) — R3 재배치 + bf_rate_estimator 소비 실측 이관분

재배치: 기존 A2-05 표기 24행 중 population 경로 6행 → A2-07, BB/line W·T_rad 16행 → A2-06,
진단 2행 → 유지. A2-05 의 실 이관 대상은 아래 bf_rate_estimator CPU 소비 7지점(스펙 6 + 실측 발견 1)이며
전부 canonical view 적분(`nlte_bf_gamma_canonical` → `bf_rate_gamma_legacy_grid`)으로 교체 완료.

| 파일:행(이관 전 HEAD=bafd2bb 기준) | 심볼 | 물리 의미 | 처분 |
|---|---|---|---|
| src/lumina_plasma.c:2277-2284 | bf_rate_estimator 스캔 | parity field-built 판정 | view validity 행 스캔으로 교체 (완료) |
| src/lumina_plasma.c:2342-2344 | σ·Γ_bf [C2]+pref·J [C1] | parity_gamma_phot 이온균형 Γ | canonical view 적분 (완료) |
| src/lumina_plasma.c:5132-5139 | σ·Γ_bf [C2]+pref·J [C1] | MA iup INTERNALUPHIGHER R_ph | canonical view 적분 (완료; Seaton σ_edge 추출 루프만 잔존) |
| src/lumina_plasma.c:16137-16150 | σ·Γ_bf [C2]+pref·J [C1] | NLTE 행렬 R_bf (source 0/1) | canonical view 적분 (완료; JEQB source 2 는 falsifier 장치로 유지) |
| src/lumina_plasma.c:16451-16457 | σ·Γ_bf [C2] | 스테이지-IV 들뜬준위 R_bf_hl | **실측 발견(스펙 목록 밖)** — canonical view 적분 (완료) |
| src/lumina_element_wide.c:601-610 | estimator+pref·J | EW capture rad_ion | canonical view 적분 (완료) |
| src/lumina_element_wide.c:1121-1136 | estimator+pref·J | EW boundary-mix rad_ion | canonical view 적분 (완료) |

잔존 bf_rate_estimator 참조(전수·허용 목록, R4): 생산자 정규화 lumina_plasma.c:1538-1542 ·
C2 덤프(출력 전용) :1563-1564 · 오라클 계수 출력 :147-159 · field-source NULL 검사 :419 ·
할당/해제 :14712/:14731 · GPU 경로 lumina_cuda.cu (A2-12/13). CPU 생산 물리 소비자 = 0.

## ADDENDUM (A2-06 폐합, 2026-08-06) — V4 §3 census 처분 정정 및 실측 이관분

행번호는 A2-06 구현 후 작업트리에서 재실측했다. `nlte_bb_jbar_canonical`은
`line_view_status`와 `line_jbar_lookup`을 함께 검사하며, 비-OK view·MISS·UNSAMPLED는
복사 항만 무기여로 만들고 원인별 `bb_view_*` 카운터를 증가시킨다. 정상 경로의 분리식은
`R_lu=B_lu*Jbar`, `R_ul^stim=B_ul*Jbar`, `R_ul^sp=A_ul`이다.

### A2-06 census 6행 — 이관 완료

| V4 census 행(bafd2bb) | 심볼 | 구현 후 생산 소비 | 처분 |
|---|---|---|---|
| src/lumina_plasma.c:4556 | W | src/lumina_plasma.c:4832-4836 | `line_view` lookup의 `B_lu*Jbar`로 이관 완료 |
| src/lumina_plasma.c:4556 | T_rad | src/lumina_plasma.c:4832-4836 | `line_view` lookup의 `B_lu*Jbar`로 이관 완료 |
| src/lumina_plasma.c:4596 | W | src/lumina_plasma.c:4832-4836 | LTE 비교장 선택을 생산율에서 제거, 이관 완료 |
| src/lumina_plasma.c:4596 | T_rad | src/lumina_plasma.c:4832-4836 | LTE 비교장 선택을 생산율에서 제거, 이관 완료 |
| src/lumina_plasma.c:4701 | W | src/lumina_plasma.c:4832-4836 | 선 상향률을 `B_lu*Jbar`로 이관 완료 |
| src/lumina_plasma.c:4701 | T_rad | src/lumina_plasma.c:4832-4836 | 선 상향률을 `B_lu*Jbar`로 이관 완료 |

### census-밖 A2-06 9지점 — 신설행 및 이관 완료

| V4 현행 표식 | 구현 후 위치 | 기존 의미 | 처분 |
|---|---|---|---|
| 4633 그룹 | src/lumina_plasma.c:4698-4710 → 4832-4836 | `jbar_line_det`/`jbar_line`/coarse 선택 사다리 | 생산 소비를 `line_view` lookup으로 이관 완료; 사다리는 진단 shadow |
| 4661 | src/lumina_plasma.c:4726-4783 → 4832-4836 | `jblue_line` 선택·fallback | 생산 소비를 `line_view` lookup으로 이관 완료; falsifier/진단 보존 |
| 4731 | src/lumina_plasma.c:4796-4815 → 4832-4836 | `(B_lu-B_ul*n_u/n_l)*beta*J_blue` | 생산 상향률을 `B_lu*Jbar`로 이관 완료; legacy 산식은 진단 shadow |
| 10827 | src/lumina_plasma.c:10914-10925 | simul ETLA 선율 | `B_lu*Jbar`, `A_ul+B_ul*Jbar`로 이관 완료 |
| 12182 | src/lumina_plasma.c:12269-12280 | RADEQ ETLA 선율 | `B_lu*Jbar`, `A_ul+B_ul*Jbar`로 이관 완료 |
| 13823 | src/lumina_plasma.c:13912-13923 | coupled ETLA 선율 | `B_lu*Jbar`, `A_ul+B_ul*Jbar`로 이관 완료 |
| 15238 | src/lumina_plasma.c:15348-15351 → 15566-15573 | dilute W/T source | 생산 소비를 `line_view` 분리율로 이관 완료; legacy 선택은 진단 shadow |
| 15292 | src/lumina_plasma.c:15398-15411 → 15566-15573 | 직접 `jbar_line`/coarse 선택 | 생산 소비를 `line_view` 분리율로 이관 완료; falsifier/진단 보존 |
| 15361 | src/lumina_plasma.c:15463-15556 → 15566-15573 | mode별 bb rate 분기 | 최종 생산 분리율을 canonical 값으로 덮어써 이관 완료 |

### V4 §3 재배치 diff — A2-07 6행, A2-08 4행

| V4 census 행(bafd2bb) | 심볼 | 구현 후 위치 | 정정 처분 |
|---|---|---|---|
| src/lumina_plasma.c:4879 | T_rad | src/lumina_plasma.c:5009-5013 | A2-07 — population fallback 지수 |
| src/lumina_plasma.c:4880 | W | src/lumina_plasma.c:5011-5013 | A2-07 — population fallback 희석 |
| src/lumina_plasma.c:12093 | W | src/lumina_plasma.c:12224 | A2-07 — lower-level population fallback |
| src/lumina_plasma.c:12100 | W | src/lumina_plasma.c:12231 | A2-07 — upper-level population fallback |
| src/lumina_plasma.c:13739 | W | src/lumina_plasma.c:13872 | A2-07 — coupled lower population fallback |
| src/lumina_plasma.c:13743 | W | src/lumina_plasma.c:13876 | A2-07 — coupled upper population fallback |
| src/lumina_plasma.c:11908 | W | src/lumina_plasma.c:12039 | A2-08 — line-source fallback |
| src/lumina_plasma.c:11908 | T_rad | src/lumina_plasma.c:12039 | A2-08 — line-source fallback |
| src/lumina_plasma.c:11915 | W | src/lumina_plasma.c:12046 | A2-08 — blanketed-heating 빈 장 |
| src/lumina_plasma.c:11915 | T_rad | src/lumina_plasma.c:12046 | A2-08 — blanketed-heating 빈 장 |

### A2-06 잔류 허용목록

| 구현 후 위치 | 처분 |
|---|---|
| src/lumina_plasma.c:4651-4829 | 매크로 원자 source-selection/falsifier 진단 shadow 유지 |
| src/lumina_plasma.c:15398-15556 | NLTE 행렬 falsifier와 legacy mode 진단 shadow 유지 |
| src/lumina_plasma.c:15581, 15714-15715 | raw `jbar_line` 출력·오라클 진단 — `KEEP_DIAGNOSTIC_READ` |
| src/lumina_plasma.c:14055, 14075, 14215 | V5 정정 진단 3행 — `KEEP_DIAGNOSTIC_READ` |
| src/lumina_cmfgen.c:3153, 3159 | A2-08 재배치 |
| GPU bb/rate 경로 | A2-12/A2-13 재배치 |

## ADDENDUM (A2-07 구현, 2026-08-06) — population 온도·rate 소유권 18행 종결

기준 HEAD는 `ece5aef8e192e2166b647ee00aae5fdd1f935a1c`이며, 아래 줄번호는
A2-07 구현 작업트리에서 재측정했다. 18개 이관 ID는
`scripts/a2_07_population_census.py`가 누락·중복 없이 고정한다. 생산 population의
partition/LTE reference는 단일 `population_lte_level_fraction`/`population_partition_build`
경로를 쓰며 legacy 식은 진단 shadow로만 남는다.

| # | 고정 이관 ID | 구현 후 위치 | 1:1 처분 |
|---:|---|---|---|
| 1 | `A2-05:old9160:T_rad` | `src/lumina_plasma.c:9279-9292` | `bf_rate_pop`을 shell `T_e`와 단일 partition accessor로 이관 완료 |
| 2 | `A2-05:old9162:W` | `src/lumina_plasma.c:9279-9292` | 함수 인자와 식에서 dilution 제거 완료 |
| 3 | `A2-05:old11943:W` | `src/lumina_plasma.c:12082` | RADEQ 호출은 `Te_lag`만 전달; `W` 전달 제거 완료 |
| 4 | `A2-05:old11943:T_rad` | `src/lumina_plasma.c:12082` | RADEQ population 공급을 `LTE@T_e` accessor로 이관 완료 |
| 5 | `A2-05:old13672:W` | `src/lumina_plasma.c:13835-13836` | coupled 호출의 dilution 전달 제거 완료 |
| 6 | `A2-05:old13672:T_rad` | `src/lumina_plasma.c:13835-13836` | coupled population 공급을 `LTE@T_e` accessor로 이관 완료 |
| 7 | `A2-06:old4879:T_rad` | `src/lumina_plasma.c:4938-4949` | macro/k-packet reference를 단일 `LTE@T_e` accessor로 이관 완료 |
| 8 | `A2-06:old4880:W` | `src/lumina_plasma.c:4938-4949` | metastable/dilution 분기 제거 완료 |
| 9 | `A2-06:old12093:W` | `src/lumina_plasma.c:12230-12240` | RADEQ lower population의 dilution 제거 완료 |
| 10 | `A2-06:old12100:W` | `src/lumina_plasma.c:12247-12257` | RADEQ upper population의 dilution 제거 완료 |
| 11 | `A2-06:old13739:W` | `src/lumina_plasma.c:13903-13913` | coupled lower population의 dilution 제거 완료 |
| 12 | `A2-06:old13743:W` | `src/lumina_plasma.c:13919-13929` | coupled upper population의 dilution 제거 완료 |
| 13 | `BASE:old2081:T_rad` | `src/lumina_plasma.c:2272-2284` | 단일 `population_partition_build(atomic,T_e)` 정본으로 이관 완료 |
| 14 | `BASE:old2082:W` | `src/lumina_plasma.c:2272-2284` | partition `W` 항과 병렬 partition 저장소 제거 완료 |
| 15 | `BASE:old7402:T_rad` | `src/lumina_plasma.c:7575-7585,7683-7693` | BF 부수 population을 solved 또는 `LTE@T_e`로만 공급 |
| 16 | `BASE:old7403:W` | `src/lumina_plasma.c:7575-7585,7683-7693` | lower/upper dilution 제거 완료 |
| 17 | `BASE:old17832:T_rad` | `src/lumina_plasma.c:18100-18108` | dump 온도를 유효한 `T_e` 하나와 population generation으로 기록 |
| 18 | `BASE:old17833:W` | `src/lumina_plasma.c:18085,18100-18108` | dump의 `W/T_rad` 열·읽기 제거 완료 |

원장 밖 소비자는 17군으로 별도 census하였다. 결과와 개별 처분은
`validation/a2_07/A2_07_STATIC_CENSUS.json` 및 `docs/CODEX_IMPL_A2_07.md`에 있으며,
생산 CPU population call graph의 금지 읽기 0, 별도 `partition_functions_Te` 저장소 0을
검사한다. A2-10 온도 해법, A2-13 GPU, 명시적 A2-06 진단 shadow만 좁은 allowlist다.
