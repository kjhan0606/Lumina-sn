#ifndef LUMINA_CMF_DECAY_H
#define LUMINA_CMF_DECAY_H
#ifdef __cplusplus
extern "C" {
#endif
/* 연속 두 단계 붕괴 parent->daughter->stable. 입력/출력은 같은 고정
 * Lagrange 단위의 핵수이며 팽창 희석을 여기서 적용하지 않는다.
 * 상수는 호출자가 실제 nuclear 입력에서 제공한다. 실패시 출력 보존.
 * decays에는 구간 내 parent/daughter의 붕괴 핵수를 돌려준다. */
int cmf_decay_chain_step(double lambda_parent,double lambda_daughter,double dt,
    const double previous[3],double current[3],double decays[2]);
#ifdef __cplusplus
}
#endif
#endif
