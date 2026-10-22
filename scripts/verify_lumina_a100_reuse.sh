#!/bin/bash
# 공통 준비기의 compile.sh 자리에 두는 신원 확인. 재컴파일하지 않는다.
# 실제 빌드 재현은 K57YkA/scripts/build_lumina_total_cap.sh를 따른다.
set -euo pipefail
run_root=$1
test "$(sha256sum "$run_root/history_cuda.so" | cut -d ' ' -f 1)" = b288dd5c4d4d31f494448dd9de3cda939b8551338f29d58f862f09d2e9da18ee
test "$(sha256sum "$run_root/driver.so" | cut -d ' ' -f 1)" = 6f82306bff6170037a40f09303f33e1cb00d930ddbdeb0cbe38d7b8b61855be4
