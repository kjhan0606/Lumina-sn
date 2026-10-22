#!/bin/bash
# 재컴파일 없이 기존 장치 검증 바이너리의 정체성을 확인한다.
set -euo pipefail
cd "$1"
printf '%s\n' \
  'ae49788865afeee46ba36dfb82dc7279f9a70989df583e4066b4b02438e39335  history_cuda.so' \
  '6f82306bff6170037a40f09303f33e1cb00d930ddbdeb0cbe38d7b8b61855be4  driver.so' |
  sha256sum --check
