#!/bin/bash
# 이미 장치13사례를 통과한 동일 바이너리를 재컴파일하지 않는다.
set -euo pipefail
cd "$1"
printf '%s\n' \
  'f6696f6e9d83a082a7730c9f11f0195ace949b41087ed9b16993b18fabba61c3  history_cuda.so' \
  '6f82306bff6170037a40f09303f33e1cb00d930ddbdeb0cbe38d7b8b61855be4  driver.so' |
  sha256sum --check
