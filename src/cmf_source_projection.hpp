#ifndef CMF_SOURCE_PROJECTION_HPP
#define CMF_SOURCE_PROJECTION_HPP
#include "cmf_source_measure.hpp"
#include <map>
#include <stdexcept>
#include <vector>

namespace lumina_source_projection {
constexpr int N=lumina_source_measure::degree;
struct Matrix {int width;std::vector<double> weights;};

/* B_k^N(a+(b-a)t)의 Bernstein 계수는 N-l개의 Bernoulli(a)와
 * l개의 Bernoulli(b)를 더한 분포의 k번째 확률이다. 차감/음의 계수가 없다.
 * 각 원래 bin 안에서만 선형 η를 정확히 적분한다. */
inline Matrix generate(int width){
    if(width<1||width>256)throw std::invalid_argument("projection width");
    std::vector<long double> weights((size_t)(width+1)*(N+1),0.L);
    constexpr long double denominator=(N+1.L)*(N+2.L);
    for(int bin=0;bin<width;bin++){
        long double a=(long double)bin/width,b=(long double)(bin+1)/width;
        for(int l=0;l<=N;l++){
            long double probability[N+1]={1.L};
            for(int trial=0;trial<N;trial++){
                long double p=trial<l?b:a;
                for(int k=trial+1;k>=1;k--)probability[k]=(1.L-p)*probability[k]+p*probability[k-1];
                probability[0]*=1.L-p;
            }
            for(int k=0;k<=N;k++){
                weights[(size_t)bin*(N+1)+k]+=probability[k]*(N-l+1.L)/(denominator*width);
                weights[(size_t)(bin+1)*(N+1)+k]+=probability[k]*(l+1.L)/(denominator*width);
            }
        }
    }
    Matrix result={width,std::vector<double>(weights.size())};
    for(size_t i=0;i<weights.size();i++){
        double v=(double)weights[i];
        if(!lumina_ordered::finite(v)||v<=0)throw std::runtime_error("nonpositive projection coefficient");
        result.weights[i]=v;
    }
    return result;
}
inline const Matrix &matrix(int width){
    // 물질·J·endpoint를 저장하지 않는다. 순수한 좌표 변환만 스레드별 재사용한다.
    static thread_local std::map<int,Matrix> matrices;
    auto it=matrices.find(width);
    if(it==matrices.end())it=matrices.emplace(width,generate(width)).first;
    return it->second;
}

#ifndef __CUDACC__
template<class Reader> inline bool block(const Reader &reader,int nf,int first,int last,
    lumina_source_measure::Node &out){
    using lumina_ordered::Positive;
    using lumina_ordered::sum;
    using lumina_characteristic::scaled;
    int width=last-first;
    if(first<0||last>=nf||width<1||width>256)return false;
    const Matrix &m=matrix(width);
    auto n=lumina_source_measure::empty();n.width=width;n.first=first;n.last=last;
    // 이전 순차 width-bin 합성의 깊이 상계를 보존한다. 오차 문턱을 조이지 않는다.
    n.depth=width-1;
    for(int j=0;j<=width;j++){
        Positive eta[2];
        for(int c=0;c<2;c++){
            double chi,value;
            if(!reader.at(2*c,first+j,chi,value)||!Positive::from_double(value,eta[c]))return false;
        }
        for(int k=0;k<=N;k++)for(int c=0;c<2;c++){
            Positive term;
            if(!scaled(eta[c],m.weights[(size_t)j*(N+1)+k],term)||!sum(n.m[c][k],term,n.m[c][k]))return false;
        }
    }
    out=n;return true;
}
#endif
}
#endif
