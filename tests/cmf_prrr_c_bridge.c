/* 실제 생산 BF 적분을 호출하는 시험 전용 벡터 어댑터. */
#include "bf_channels.h"
int bf_vector(int n, const double *edges, double th, const double *sigma,
              double T, double ne, double gl, double gu, const double *J,
              double *up, double *down) {
    for(int b=0;b<n;b++) {
        BfChannelBin c;
        if(bf_channel_bin(edges[b],edges[b+1],th,sigma[b],T,ne,gl,gu,
                          1,1,J[b],0,&c))return b+1;
        up[b]=c.ionization;down[b]=c.recombination;
    }
    return 0;
}
