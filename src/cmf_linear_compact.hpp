#ifndef CMF_LINEAR_COMPACT_HPP
#define CMF_LINEAR_COMPACT_HPP
#include "cmf_linear_native.h"
namespace lumina_linear_compact {
/* 총오차는 원래 적분에서 먼저 완성한다. 합성/수용의 모든 소비 필드를 그대로 보존한다. */
struct Map {
    CMFLinearPositive transmission,source,source_error;
    double tau_error;
};
static_assert(sizeof(Map)==56,"compact affine ABI");
}
#endif
