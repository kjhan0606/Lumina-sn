#include "population_contract.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define POP_EV_TO_ERG 1.602176634e-12
#define POP_K_BOLTZMANN 1.380649e-16
/* R0: catalog 는 cm^-1 로 온다 (CMFGEN osc · Cloudy Stout 공통 단위) */
#define POP_CM1_TO_ERG 1.98644586e-16

typedef struct {
    uint32_t h[8];
    uint64_t bits;
    unsigned char block[64];
    size_t used;
} PopSha256;

static uint32_t pop_rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32U - n)); }
static void pop_sha_block(PopSha256 *s, const unsigned char block[64]) {
    static const uint32_t k[64] = {
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
    };
    uint32_t w[64];
    for (int i=0;i<16;i++) w[i]=((uint32_t)block[4*i]<<24)|((uint32_t)block[4*i+1]<<16)|((uint32_t)block[4*i+2]<<8)|block[4*i+3];
    for (int i=16;i<64;i++) {
        uint32_t a=w[i-15],b=w[i-2];
        uint32_t s0=pop_rotr(a,7)^pop_rotr(a,18)^(a>>3);
        uint32_t s1=pop_rotr(b,17)^pop_rotr(b,19)^(b>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    uint32_t a=s->h[0],b=s->h[1],c=s->h[2],d=s->h[3],e=s->h[4],f=s->h[5],g=s->h[6],h=s->h[7];
    for(int i=0;i<64;i++){
        uint32_t S1=pop_rotr(e,6)^pop_rotr(e,11)^pop_rotr(e,25),ch=(e&f)^(~e&g);
        uint32_t t1=h+S1+ch+k[i]+w[i],S0=pop_rotr(a,2)^pop_rotr(a,13)^pop_rotr(a,22);
        uint32_t maj=(a&b)^(a&c)^(b&c),t2=S0+maj;
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    s->h[0]+=a;s->h[1]+=b;s->h[2]+=c;s->h[3]+=d;s->h[4]+=e;s->h[5]+=f;s->h[6]+=g;s->h[7]+=h;
}
static void pop_sha_init(PopSha256 *s) {
    static const uint32_t h[8]={0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};
    memcpy(s->h,h,sizeof(h));s->bits=0;s->used=0;
}
static void pop_sha_update(PopSha256 *s,const void *data,size_t n){
    const unsigned char *p=(const unsigned char*)data;s->bits+=(uint64_t)n*8U;
    while(n){size_t room=64-s->used,t=n<room?n:room;memcpy(s->block+s->used,p,t);s->used+=t;p+=t;n-=t;if(s->used==64){pop_sha_block(s,s->block);s->used=0;}}
}
static void pop_sha_final(PopSha256 *s,char out[65]){
    uint64_t bits=s->bits;unsigned char one=0x80,zero=0,len[8],digest[32];
    pop_sha_update(s,&one,1);while(s->used!=56)pop_sha_update(s,&zero,1);
    for (int i = 0; i < 8; i++)
        len[7-i] = (unsigned char)(bits >> (8*i));
    pop_sha_update(s, len, 8);
    for(int i=0;i<8;i++){digest[4*i]=(unsigned char)(s->h[i]>>24);digest[4*i+1]=(unsigned char)(s->h[i]>>16);digest[4*i+2]=(unsigned char)(s->h[i]>>8);digest[4*i+3]=(unsigned char)s->h[i];}
    static const char hex[]="0123456789abcdef";for(int i=0;i<32;i++){out[2*i]=hex[digest[i]>>4];out[2*i+1]=hex[digest[i]&15];}out[64]='\0';
}
static void pop_hash_u64(PopSha256 *s,uint64_t x){unsigned char b[8];for(int i=0;i<8;i++)b[7-i]=(unsigned char)(x>>(8*i));pop_sha_update(s,b,8);}
static void pop_hash_i32(PopSha256 *s,int x){uint32_t u=(uint32_t)x;unsigned char b[4]={(unsigned char)(u>>24),(unsigned char)(u>>16),(unsigned char)(u>>8),(unsigned char)u};pop_sha_update(s,b,4);}
static void pop_hash_f64(PopSha256 *s,double x){uint64_t u;memcpy(&u,&x,8);pop_hash_u64(s,u);}

const char *population_status_name(PopulationStatus s){
    static const char *n[]={"POP_OK","POP_EXACT_ZERO","POP_INVALID_TE","POP_INVALID_PARTITION","POP_STALE_DERIVED_TEMPERATURE","POP_BF_STALE","POP_BF_UNSAMPLED","POP_BF_OOG","POP_BF_MISS","POP_BB_STALE","POP_BB_UNSAMPLED","POP_BB_OOG","POP_BB_MISS","POP_PROFILE_MISMATCH","POP_QUERY_HASH_MISMATCH","POP_ATOMIC_MISSING","POP_RANK_INCOMPLETE","POP_NE_NOT_CONVERGED","POP_SOLVE_FAILED","POP_NONFINITE","POP_FORBIDDEN_FALLBACK"};
    return (s>=POP_OK&&s<=POP_FORBIDDEN_FALLBACK)?n[s]:"POP_UNKNOWN";
}

PopulationStatus population_te_manifest_sha256(const double *te,size_t n,char out[65]){
    if (!te || !out || n == 0) return POP_INVALID_TE;
    PopSha256 s;
    pop_sha_init(&s);
    const char d[]="A2-07:T_e:K:IEEE754:shell-order:v1";
    pop_sha_update(&s,d,sizeof(d)-1);pop_hash_u64(&s,(uint64_t)n);
    for(size_t i=0;i<n;i++){if(!isfinite(te[i])||te[i]<=0.0)return POP_INVALID_TE;pop_hash_u64(&s,(uint64_t)i);pop_hash_f64(&s,te[i]);}pop_sha_final(&s,out);return POP_OK;
}
PopulationStatus population_atomic_model_sha256(const PopulationAtomicView *a,char out[65]){
    if(!a||!out||!a->level_offset||!a->energy_eV||!a->g||a->n_ions==0||a->n_levels==0)return POP_ATOMIC_MISSING;
    if(a->topion_n&&(!a->topion_ion_index||!a->topion_E_cm||!a->topion_g))
        return POP_ATOMIC_MISSING;
    PopSha256 s;pop_sha_init(&s);const char d[]="A2-07:atomic-partition-membership:v2:topion-bound";pop_sha_update(&s,d,sizeof(d)-1);pop_hash_u64(&s,a->n_ions);pop_hash_u64(&s,a->n_levels);
    for(size_t i=0;i<=a->n_ions;i++)pop_hash_i32(&s,a->level_offset[i]);
    for(size_t i=0;i<a->n_levels;i++){pop_hash_f64(&s,a->energy_eV[i]);pop_hash_i32(&s,a->g[i]);pop_hash_i32(&s,a->runtime_membership?a->runtime_membership[i]:0);pop_hash_i32(&s,a->level_Z?a->level_Z[i]:0);pop_hash_i32(&s,a->level_ion?a->level_ion[i]:0);}
    /* The level-less top-ion catalog participates in Z(T_e), hence in n_e,
     * populations, tau, opacity, emissivity, and A210.  Omitting it let a
     * catalog mutation retain the same atomic_model_sha256.  Bind the exact
     * thermodynamic membership values in canonical loader order. */
    pop_hash_u64(&s,a->topion_n);
    for(size_t i=0;i<a->topion_n;i++){
        if(a->topion_ion_index[i]<0||(size_t)a->topion_ion_index[i]>=a->n_ions||
           !isfinite(a->topion_E_cm[i])||a->topion_E_cm[i]<0.0||
           !isfinite(a->topion_g[i])||a->topion_g[i]<=0.0)
            return POP_INVALID_PARTITION;
        pop_hash_i32(&s,a->topion_ion_index[i]);
        pop_hash_f64(&s,a->topion_E_cm[i]);
        pop_hash_f64(&s,a->topion_g[i]);
    }
    pop_sha_final(&s,out);return POP_OK;
}
PopulationStatus population_partition_ion(const PopulationAtomicView *a,size_t ion,double te,double *out){
    if(!out||!a||!a->level_offset||!a->energy_eV||!a->g||ion>=a->n_ions)
        return POP_ATOMIC_MISSING;
    if(!isfinite(te)||te<=0.0)return POP_INVALID_TE;
    int lo=a->level_offset[ion],hi=a->level_offset[ion+1];
    if(lo<0||hi<lo||(size_t)hi>a->n_levels)return POP_ATOMIC_MISSING;
    /* ★2026-08-07 T0: hi==lo 는 **손상이 아니라 정상**이다.  로더가 전리에너지 n 개에
     * 대해 population n+1 개를 만들므로 원소마다 최상단 population 은 속박준위가 없다
     * (실측 15/74, 전부 원소 최상단; 덱 3종 동일).  구 조건 `hi<=lo` 가 이 정상을
     * POP_ATOMIC_MISSING 으로 거부해 compute_plasma_state 가 전혀 성공할 수 없었다.
     * ★기준 배선(ARTIS)은 최상단 이온에 준위를 **0 개가 아니라 1 개** 준다:
     *   input.cc:1226 "optionally limit the top ion to one level and no transitions"
     *   input.cc:153  "in case the top ion has nlevelsmax = 1"
     * 그러면 Z = g_ground 이지 1 이 아니다.  우리 로더가 그 한 준위를 떨어뜨렸다.
     * 그 이온의 바닥 g 는 덱에도 구조체에도 없으므로 여기서 지어낼 수 없다 ⟹
     * Z=1 은 **g=1(무구조·맨핵)에서만 정확한 임시 대입**이며, 미지의 g 를 가정하지 않고
     * 소비자 쪽에서 **상한으로 감싸 검사**한다(lumina_plasma.c reservoir 게이트).
     * 정본 수리 = 최상단 15 이온의 바닥 g 를 외부 앵커로 도입해 준위 1 개를 실제로 주는 것.
     * 대장 기재: docs/CLASSIC_DEBT_CENSUS.md (검증불가 고아 금지 — 방치하지 않는다). */
    if(hi==lo){
        /* ★R0(2026-08-07): 준위 없는 최상단 이온.  **Z=1 임시 대입을 폐기**한다 —
         * 실측으로 최대 80배 틀렸다(V II: Z(10kK)=80.88 vs 1).
         * 분배함수 전용 catalog(thermodynamic membership)에서 Z(T)=Sum g_i exp(-E_i/kT) 를
         * 다른 이온과 **같은 식**으로 계산한다.  catalog 가 없으면 지어내지 않고 거부한다. */
        if(!a->topion_n||!a->topion_ion_index||!a->topion_E_cm||!a->topion_g)
            return POP_ATOMIC_MISSING;
        /* catalog 는 **ion-pop 인덱스**로 키가 걸린다(level_offset 과 같은 키). */
        double sum=0.0,e0=INFINITY;size_t cnt=0;
        for(size_t k=0;k<a->topion_n;k++){
            if((size_t)a->topion_ion_index[k]!=ion) continue;
            if(a->topion_g[k]>0.0&&isfinite(a->topion_E_cm[k])&&a->topion_E_cm[k]<e0)
                e0=a->topion_E_cm[k];
            cnt++;
        }
        if(!cnt) return POP_ATOMIC_MISSING;
        if(!isfinite(e0)) return POP_INVALID_PARTITION;
        for(size_t k=0;k<a->topion_n;k++){
            if((size_t)a->topion_ion_index[k]!=ion) continue;
            if(!(a->topion_g[k]>0.0)||!isfinite(a->topion_E_cm[k])) continue;
            double x=(a->topion_E_cm[k]-e0)*POP_CM1_TO_ERG/(POP_K_BOLTZMANN*te);
            if(x<745.0) sum+=a->topion_g[k]*exp(-x);
        }
        if(!(sum>0.0)||!isfinite(sum)) return POP_INVALID_PARTITION;
        *out=sum; return POP_OK;
    }
    double e0=INFINITY;
    for(int l=lo;l<hi;l++)
        if((!a->runtime_membership||a->runtime_membership[l]>=0)&&isfinite(a->energy_eV[l])&&a->g[l]>0&&a->energy_eV[l]<e0)
            e0=a->energy_eV[l];
    if(!isfinite(e0))return POP_INVALID_PARTITION;
    double sum=0.0,c=0.0;for(int l=lo;l<hi;l++){if((a->runtime_membership&&a->runtime_membership[l]<0)||!isfinite(a->energy_eV[l])||a->g[l]<=0)continue;double x=(a->energy_eV[l]-e0)*POP_EV_TO_ERG/(POP_K_BOLTZMANN*te);double term=(x<745.0)?(double)a->g[l]*exp(-x):0.0;double t=sum+term;if(fabs(sum)>=fabs(term))c+=(sum-t)+term;else c+=(term-t)+sum;sum=t;}sum+=c;if(!isfinite(sum)||sum<=0.0)return POP_INVALID_PARTITION;*out=sum;return POP_OK;
}
PopulationStatus population_partition_build(const PopulationAtomicView *a,const double *te,size_t ns,uint64_t req,uint64_t teg,double *pub,PopulationDerivedStamp *stamp){
    if(!pub||!stamp||req==0||teg==0)
        return POP_STALE_DERIVED_TEMPERATURE;
    char th[65],ah[65];PopulationStatus st=population_te_manifest_sha256(te,ns,th);if(st!=POP_OK)return st;st=population_atomic_model_sha256(a,ah);if(st!=POP_OK)return st;size_t n=a->n_ions*ns;double *work=(double*)malloc(n*sizeof(double));if(!work)return POP_SOLVE_FAILED;
    for(size_t i=0;i<a->n_ions;i++)for(size_t s=0;s<ns;s++){st=population_partition_ion(a,i,te[s],&work[i*ns+s]);if(st!=POP_OK){free(work);return st;}}
    memcpy(pub,work,n*sizeof(double));free(work);memset(stamp,0,sizeof(*stamp));stamp->required_population_generation=req;stamp->computed_population_generation=req;stamp->te_generation=teg;memcpy(stamp->te_manifest_sha256,th,65);memcpy(stamp->atomic_model_sha256,ah,65);stamp->n_shells=ns;stamp->n_items=a->n_ions;stamp->status=POP_OK;return POP_OK;
}
PopulationStatus population_partition_view_check(const PopulationDerivedStamp *st,const PopulationAtomicView *a,const double *te,size_t ns,uint64_t req,uint64_t teg){
    if(!st||!a||st->status!=POP_OK||st->required_population_generation!=req||st->computed_population_generation!=req||st->te_generation!=teg||st->n_shells!=ns||st->n_items!=a->n_ions)
        return POP_STALE_DERIVED_TEMPERATURE;
    char th[65],ah[65];PopulationStatus rc=population_te_manifest_sha256(te,ns,th);if(rc!=POP_OK)return rc;rc=population_atomic_model_sha256(a,ah);if(rc!=POP_OK)return rc;return (!strcmp(th,st->te_manifest_sha256)&&!strcmp(ah,st->atomic_model_sha256))?POP_OK:POP_STALE_DERIVED_TEMPERATURE;
}

PopulationStatus population_within_superlevel_build(
        const PopulationDerivedStamp *partition_stamp,
        const PopulationAtomicView *atomic,
        const double *te,
        size_t n_shells,
        uint64_t required_population_generation,
        uint64_t te_generation,
        size_t n_full_levels,
        int super_mode,
        size_t n_superlevels,
        const int *nlte_to_global_level,
        const int *full_to_superlevel,
        const int *super_anchor_global_level,
        double *fractions,
        PopulationDerivedStamp *fraction_stamp) {
    if (!fractions || !fraction_stamp || !atomic || n_full_levels == 0 ||
        n_shells == 0 || n_full_levels > SIZE_MAX / n_shells ||
        n_full_levels * n_shells > SIZE_MAX / sizeof(double))
        return POP_ATOMIC_MISSING;
    PopulationStatus status = population_partition_view_check(
        partition_stamp, atomic, te, n_shells,
        required_population_generation, te_generation);
    if (status != POP_OK) return status;

    size_t count = n_full_levels * n_shells;
    double *work = (double *)malloc(count * sizeof(*work));
    if (!work) return POP_SOLVE_FAILED;
    if (!super_mode) {
        for (size_t i = 0; i < count; ++i) work[i] = 1.0;
    } else {
        if (n_superlevels == 0 ||
            n_superlevels > SIZE_MAX / sizeof(double) ||
            !nlte_to_global_level ||
            !full_to_superlevel || !super_anchor_global_level) {
            free(work);
            return POP_ATOMIC_MISSING;
        }
        double *super_partition = (double *)malloc(
            n_superlevels * sizeof(*super_partition));
        if (!super_partition) {
            free(work);
            return POP_SOLVE_FAILED;
        }
        for (size_t shell = 0; shell < n_shells; ++shell) {
            double temperature = te[shell];
            if (!isfinite(temperature) || temperature <= 0.0) {
                status = POP_INVALID_TE;
                break;
            }
            for (size_t sl = 0; sl < n_superlevels; ++sl)
                super_partition[sl] = 0.0;
            for (size_t level = 0; level < n_full_levels; ++level) {
                int global_level = nlte_to_global_level[level];
                int superlevel = full_to_superlevel[level];
                if (global_level < 0 ||
                    (size_t)global_level >= atomic->n_levels ||
                    superlevel < 0 ||
                    (size_t)superlevel >= n_superlevels) {
                    status = POP_ATOMIC_MISSING;
                    break;
                }
                int anchor = super_anchor_global_level[superlevel];
                if (anchor < 0 || (size_t)anchor >= atomic->n_levels ||
                    atomic->g[global_level] <= 0 ||
                    !isfinite(atomic->energy_eV[global_level]) ||
                    !isfinite(atomic->energy_eV[anchor])) {
                    status = POP_ATOMIC_MISSING;
                    break;
                }
                double relative_energy =
                    (atomic->energy_eV[global_level] -
                     atomic->energy_eV[anchor]) * POP_EV_TO_ERG;
                if (!isfinite(relative_energy) || relative_energy < 0.0) {
                    status = POP_INVALID_PARTITION;
                    break;
                }
                double weight = (double)atomic->g[global_level] *
                    exp(-relative_energy /
                        (POP_K_BOLTZMANN * temperature));
                if (!isfinite(weight) || weight < 0.0) {
                    status = POP_NONFINITE;
                    break;
                }
                work[level * n_shells + shell] = weight;
                super_partition[superlevel] += weight;
            }
            if (status != POP_OK) break;
            for (size_t level = 0; level < n_full_levels; ++level) {
                int superlevel = full_to_superlevel[level];
                double z = super_partition[superlevel];
                if (!isfinite(z) || z <= 0.0) {
                    status = POP_INVALID_PARTITION;
                    break;
                }
                work[level * n_shells + shell] /= z;
            }
            if (status != POP_OK) break;
        }
        free(super_partition);
        if (status != POP_OK) {
            free(work);
            return status;
        }
    }

    PopulationDerivedStamp completed_stamp = *partition_stamp;
    completed_stamp.n_items = n_full_levels;
    memcpy(fractions, work, count * sizeof(*fractions));
    *fraction_stamp = completed_stamp;
    free(work);
    return POP_OK;
}

PopulationStatus population_lte_level_fraction(const PopulationAtomicView *a,size_t ion,size_t level,double te,double z,double *f){
    if(!f||!a||ion>=a->n_ions||level>=a->n_levels)
        return POP_ATOMIC_MISSING;
    if(!isfinite(te)||te<=0.0)return POP_INVALID_TE;
    if(!isfinite(z)||z<=0.0)return POP_INVALID_PARTITION;
    int lo=a->level_offset[ion],hi=a->level_offset[ion+1];if((int)level<lo||(int)level>=hi||a->g[level]<=0||!isfinite(a->energy_eV[level])||(a->runtime_membership&&a->runtime_membership[level]<0))return POP_ATOMIC_MISSING;double e0=INFINITY;
    for(int l=lo;l<hi;l++)
        if((!a->runtime_membership||a->runtime_membership[l]>=0)&&isfinite(a->energy_eV[l])&&a->g[l]>0&&a->energy_eV[l]<e0)
            e0=a->energy_eV[l];
    if(!isfinite(e0))return POP_INVALID_PARTITION;
    double x=(a->energy_eV[level]-e0)*POP_EV_TO_ERG/(POP_K_BOLTZMANN*te);
    *f=(x<745.0)?(double)a->g[level]*exp(-x)/z:0.0;
    if(!isfinite(*f)||*f<0.0)return POP_NONFINITE;
    return *f==0.0?POP_EXACT_ZERO:POP_OK;
}
PopulationStatus population_line_level_number_density(
        PopulationLineView view,const PopulationAtomicView *a,size_t ion,
        size_t level,double te,double z,double nion,double nnlte,double *nlevel){
    if(!nlevel)return POP_ATOMIC_MISSING;
    *nlevel=NAN;
    if(view==POP_LINE_VIEW_NLTE_COMMITTED){
        if(!isfinite(nnlte)||nnlte<0.0)return POP_NONFINITE;
        *nlevel=nnlte;
        return nnlte==0.0?POP_EXACT_ZERO:POP_OK;
    }
    if(view!=POP_LINE_VIEW_LTE_TE)return POP_FORBIDDEN_FALLBACK;
    if(!isfinite(nion)||nion<0.0)return POP_NONFINITE;
    double fraction=NAN;
    PopulationStatus status=population_lte_level_fraction(
        a,ion,level,te,z,&fraction);
    if(status!=POP_OK&&status!=POP_EXACT_ZERO)return status;
    double value=nion*fraction;
    if(!isfinite(value)||value<0.0)return POP_NONFINITE;
    *nlevel=value;
    return value==0.0?POP_EXACT_ZERO:POP_OK;
}
PopulationStatus population_rate_views_check(
        PopulationStatus bf_status, uint64_t bf_generation,
        PopulationStatus bb_status, uint64_t bb_generation,
        uint64_t required_rate_generation) {
    if (bf_status != POP_OK && bf_status != POP_EXACT_ZERO)
        return bf_status;
    if (bb_status != POP_OK && bb_status != POP_EXACT_ZERO)
        return bb_status;
    if (required_rate_generation == 0 ||
        bf_generation != required_rate_generation ||
        bb_generation != required_rate_generation)
        return POP_STALE_DERIVED_TEMPERATURE;
    return POP_OK;
}
PopulationStatus population_dense_rank_check(const double *matrix, size_t n,
                                              double relative_tolerance) {
    if (!matrix || n == 0) return POP_RANK_INCOMPLETE;
    if (!isfinite(relative_tolerance) || relative_tolerance <= 0.0)
        relative_tolerance = 1.0e-14;
    double *work = (double *)malloc(n * n * sizeof(double));
    if (!work) return POP_SOLVE_FAILED;
    double scale = 0.0;
    for (size_t i = 0; i < n * n; i++) {
        if (!isfinite(matrix[i])) { free(work); return POP_NONFINITE; }
        work[i] = matrix[i];
        if (fabs(work[i]) > scale) scale = fabs(work[i]);
    }
    if (scale == 0.0) { free(work); return POP_RANK_INCOMPLETE; }
    size_t rank = 0;
    for (size_t col = 0; col < n && rank < n; col++) {
        size_t pivot = rank;
        for (size_t row = rank + 1; row < n; row++)
            if (fabs(work[row * n + col]) > fabs(work[pivot * n + col]))
                pivot = row;
        if (fabs(work[pivot * n + col]) <= relative_tolerance * scale)
            continue;
        if (pivot != rank)
            for (size_t j = col; j < n; j++) {
                double tmp = work[rank * n + j];
                work[rank * n + j] = work[pivot * n + j];
                work[pivot * n + j] = tmp;
            }
        for (size_t row = rank + 1; row < n; row++) {
            double factor = work[row * n + col] / work[rank * n + col];
            for (size_t j = col; j < n; j++)
                work[row * n + j] -= factor * work[rank * n + j];
        }
        rank++;
    }
    free(work);
    return rank == n ? POP_OK : POP_RANK_INCOMPLETE;
}

#define POP_CM(A,n,i,j) ((A)[(j) * (n) + (i)])

static double pop_stable_norm2_row(const double *matrix, size_t n, size_t row)
{
    double scale = 0.0, sumsq = 1.0;
    for (size_t col = 0; col < n; ++col) {
        double value = fabs(POP_CM(matrix, n, row, col));
        if (value == 0.0) continue;
        if (scale < value) {
            double ratio = scale / value;
            sumsq = 1.0 + sumsq * ratio * ratio;
            scale = value;
        } else {
            double ratio = value / scale;
            sumsq += ratio * ratio;
        }
    }
    return scale == 0.0 ? 0.0 : scale * sqrt(sumsq);
}

static double pop_stable_norm2_column(
    const double *matrix, size_t n, size_t col)
{
    double scale = 0.0, sumsq = 1.0;
    for (size_t row = 0; row < n; ++row) {
        double value = fabs(POP_CM(matrix, n, row, col));
        if (value == 0.0) continue;
        if (scale < value) {
            double ratio = scale / value;
            sumsq = 1.0 + sumsq * ratio * ratio;
            scale = value;
        } else {
            double ratio = value / scale;
            sumsq += ratio * ratio;
        }
    }
    return scale == 0.0 ? 0.0 : scale * sqrt(sumsq);
}

/* ARTIS nltepop.cc row/column balancing: A' = R A C, b' = R b and
 * x = C y.  Each sequential index update uses
 * f=sqrt(||column_i||_2/||row_i||_2), R_i*=f and C_i/=f. */
static int pop_dense_equilibrate(
    double *matrix, double *rhs, size_t n,
    double *row_scale, double *column_scale)
{
    for (size_t i = 0; i < n; ++i)
        row_scale[i] = column_scale[i] = 1.0;
    int iterations = 0;
    for (int iteration = 0; iteration < 10; ++iteration) {
        int changed = 0;
        for (size_t i = 0; i < n; ++i) {
            double row_norm = pop_stable_norm2_row(matrix, n, i);
            double column_norm = pop_stable_norm2_column(matrix, n, i);
            if (!(row_norm > 0.0) || !(column_norm > 0.0) ||
                !isfinite(row_norm) || !isfinite(column_norm))
                return -1;
            double exponent = 0.5 * (log(column_norm) - log(row_norm));
            double factor = exp(exponent);
            if (!(factor > 0.0) || !isfinite(factor)) return -1;
            if (fabs(factor - 1.0) <= 1.0e-3) continue;
            changed = 1;
            for (size_t col = 0; col < n; ++col)
                POP_CM(matrix, n, i, col) *= factor;
            rhs[i] *= factor;
            row_scale[i] *= factor;
            for (size_t row = 0; row < n; ++row)
                POP_CM(matrix, n, row, i) /= factor;
            column_scale[i] /= factor;
            if (!isfinite(rhs[i]) || !isfinite(row_scale[i]) ||
                !isfinite(column_scale[i]))
                return -1;
        }
        iterations = iteration + 1;
        if (!changed) break;
    }
    for (size_t i = 0; i < n * n; ++i)
        if (!isfinite(matrix[i])) return -1;
    return iterations;
}

static PopulationStatus pop_dense_lu_factor(
    double *matrix, size_t n, size_t *pivots,
    size_t *rank, double *pivot_growth)
{
    double matrix_max = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        double value = fabs(matrix[i]);
        if (!isfinite(value)) return POP_NONFINITE;
        if (value > matrix_max) matrix_max = value;
    }
    if (!(matrix_max > 0.0)) return POP_RANK_INCOMPLETE;
    double tolerance = DBL_EPSILON * (double)n * matrix_max;
    *rank = 0;
    for (size_t k = 0; k < n; ++k) {
        size_t pivot = k;
        double best = fabs(POP_CM(matrix, n, k, k));
        for (size_t row = k + 1; row < n; ++row) {
            double value = fabs(POP_CM(matrix, n, row, k));
            if (value > best) { best = value; pivot = row; }
        }
        pivots[k] = pivot;
        if (!(best > tolerance) || !isfinite(best))
            return POP_RANK_INCOMPLETE;
        (*rank)++;
        if (pivot != k) {
            for (size_t col = 0; col < n; ++col) {
                double value = POP_CM(matrix, n, k, col);
                POP_CM(matrix, n, k, col) =
                    POP_CM(matrix, n, pivot, col);
                POP_CM(matrix, n, pivot, col) = value;
            }
        }
        double diagonal = POP_CM(matrix, n, k, k);
        for (size_t row = k + 1; row < n; ++row) {
            POP_CM(matrix, n, row, k) /= diagonal;
            double multiplier = POP_CM(matrix, n, row, k);
            for (size_t col = k + 1; col < n; ++col)
                POP_CM(matrix, n, row, col) -=
                    multiplier * POP_CM(matrix, n, k, col);
        }
    }
    double upper_max = 0.0;
    for (size_t row = 0; row < n; ++row)
        for (size_t col = row; col < n; ++col) {
            double value = fabs(POP_CM(matrix, n, row, col));
            if (!isfinite(value)) return POP_NONFINITE;
            if (value > upper_max) upper_max = value;
        }
    *pivot_growth = upper_max / matrix_max;
    return isfinite(*pivot_growth) ? POP_OK : POP_NONFINITE;
}

static PopulationStatus pop_dense_lu_solve(
    const double *lu, size_t n, const size_t *pivots,
    const double *rhs, double *solution)
{
    memcpy(solution, rhs, n * sizeof(*solution));
    for (size_t k = 0; k < n; ++k) {
        size_t pivot = pivots[k];
        if (pivot != k) {
            double value = solution[k];
            solution[k] = solution[pivot];
            solution[pivot] = value;
        }
    }
    for (size_t row = 0; row < n; ++row)
        for (size_t col = 0; col < row; ++col)
            solution[row] -= POP_CM(lu, n, row, col) * solution[col];
    for (size_t back = n; back-- > 0;) {
        for (size_t col = back + 1; col < n; ++col)
            solution[back] -= POP_CM(lu, n, back, col) * solution[col];
        double diagonal = POP_CM(lu, n, back, back);
        if (diagonal == 0.0 || !isfinite(diagonal)) return POP_SOLVE_FAILED;
        solution[back] /= diagonal;
        if (!isfinite(solution[back])) return POP_NONFINITE;
    }
    return POP_OK;
}

/* Componentwise backward error of the original, unscaled Ax=b.  The residual
 * accumulation is long double so refinement gains information beyond the
 * double-precision LU without changing the physical coefficients. */
static double pop_dense_backward_error(
    const double *matrix, const double *rhs, const double *solution,
    size_t n, double *residual)
{
    long double worst = 0.0L;
    for (size_t row = 0; row < n; ++row) {
        long double ax = 0.0L, denominator = fabsl((long double)rhs[row]);
        for (size_t col = 0; col < n; ++col) {
            long double a = (long double)POP_CM(matrix, n, row, col);
            long double x = (long double)solution[col];
            ax += a * x;
            denominator += fabsl(a) * fabsl(x);
        }
        long double value = (long double)rhs[row] - ax;
        residual[row] = (double)value;
        long double relative = denominator > 0.0L
            ? fabsl(value) / denominator : fabsl(value);
        if (relative > worst) worst = relative;
    }
    return (double)worst;
}

PopulationStatus population_dense_solve_equilibrated(
    const double *matrix, const double *rhs, size_t n, double *solution,
    PopulationLinearSolveDiagnostic *diagnostic)
{
    PopulationLinearSolveDiagnostic local;
    if (!diagnostic) diagnostic = &local;
    memset(diagnostic, 0, sizeof(*diagnostic));
    diagnostic->pivot_growth = INFINITY;
    diagnostic->initial_backward_error = INFINITY;
    diagnostic->final_backward_error = INFINITY;
    if (!matrix || !rhs || !solution || n == 0 || n > SIZE_MAX / n ||
        n * n > SIZE_MAX / sizeof(double) || n > SIZE_MAX / sizeof(double) ||
        n > SIZE_MAX / sizeof(size_t))
        return POP_SOLVE_FAILED;
    for (size_t i = 0; i < n * n; ++i)
        if (!isfinite(matrix[i])) return POP_NONFINITE;
    for (size_t i = 0; i < n; ++i)
        if (!isfinite(rhs[i])) return POP_NONFINITE;

    double *equilibrated = (double *)malloc(n * n * sizeof(double));
    double *scaled_rhs = (double *)malloc(n * sizeof(double));
    double *lu = (double *)malloc(n * n * sizeof(double));
    double *row_scale = (double *)malloc(n * sizeof(double));
    double *column_scale = (double *)malloc(n * sizeof(double));
    double *scaled_solution = (double *)malloc(n * sizeof(double));
    double *residual = (double *)malloc(n * sizeof(double));
    double *scaled_residual = (double *)malloc(n * sizeof(double));
    double *correction = (double *)malloc(n * sizeof(double));
    double *work_solution = (double *)malloc(n * sizeof(double));
    size_t *pivots = (size_t *)malloc(n * sizeof(size_t));
    if (!equilibrated || !scaled_rhs || !lu || !row_scale ||
        !column_scale || !scaled_solution || !residual || !scaled_residual ||
        !correction || !work_solution || !pivots) {
        free(equilibrated); free(scaled_rhs); free(lu); free(row_scale);
        free(column_scale); free(scaled_solution); free(residual);
        free(scaled_residual); free(correction); free(work_solution);
        free(pivots);
        return POP_SOLVE_FAILED;
    }
    memcpy(equilibrated, matrix, n * n * sizeof(double));
    memcpy(scaled_rhs, rhs, n * sizeof(double));
    diagnostic->equilibration_iterations = pop_dense_equilibrate(
        equilibrated, scaled_rhs, n, row_scale, column_scale);
    PopulationStatus status = POP_OK;
    if (diagnostic->equilibration_iterations < 0) {
        status = POP_RANK_INCOMPLETE;
        goto done;
    }
    memcpy(lu, equilibrated, n * n * sizeof(double));
    status = pop_dense_lu_factor(
        lu, n, pivots, &diagnostic->rank, &diagnostic->pivot_growth);
    if (status != POP_OK) goto done;
    status = pop_dense_lu_solve(
        lu, n, pivots, scaled_rhs, scaled_solution);
    if (status != POP_OK) goto done;
    for (size_t i = 0; i < n; ++i) {
        work_solution[i] = column_scale[i] * scaled_solution[i];
        if (!isfinite(work_solution[i])) { status = POP_NONFINITE; goto done; }
    }

    for (int iteration = 0; iteration <= 10; ++iteration) {
        double error = pop_dense_backward_error(
            matrix, rhs, work_solution, n, residual);
        if (iteration == 0) diagnostic->initial_backward_error = error;
        diagnostic->final_backward_error = error;
        diagnostic->refinement_iterations = iteration;
        if (!isfinite(error)) { status = POP_NONFINITE; goto done; }
        if ((iteration >= 2 && error <= 1.0e-15) || iteration == 10) break;
        for (size_t i = 0; i < n; ++i)
            scaled_residual[i] = row_scale[i] * residual[i];
        status = pop_dense_lu_solve(
            lu, n, pivots, scaled_residual, correction);
        if (status != POP_OK) goto done;
        for (size_t i = 0; i < n; ++i) {
            work_solution[i] += column_scale[i] * correction[i];
            if (!isfinite(work_solution[i])) {
                status = POP_NONFINITE;
                goto done;
            }
        }
    }
    if (!(diagnostic->final_backward_error <=
          POP_DENSE_BACKWARD_ERROR_LIMIT)) {
        status = POP_SOLVE_FAILED;
        goto done;
    }
    memcpy(solution, work_solution, n * sizeof(double));

done:
    free(equilibrated); free(scaled_rhs); free(lu); free(row_scale);
    free(column_scale); free(scaled_solution); free(residual);
    free(scaled_residual); free(correction); free(work_solution); free(pivots);
    return status;
}

PopulationStatus population_generator_stationary_gth(
    const double *generator, size_t n, double total_population,
    double *solution, PopulationGeneratorSolveDiagnostic *diagnostic)
{
    PopulationGeneratorSolveDiagnostic local;
    if (!diagnostic) diagnostic = &local;
    memset(diagnostic, 0, sizeof(*diagnostic));
    diagnostic->input_column_relative_error = INFINITY;
    diagnostic->exact_generator_componentwise_residual = INFINITY;
    diagnostic->minimum_population = NAN;
    diagnostic->maximum_population = NAN;
    if (!generator || !solution || n == 0 || !isfinite(total_population) ||
        !(total_population > 0.0) || n > SIZE_MAX / n ||
        n * n > SIZE_MAX / sizeof(long double) ||
        n > SIZE_MAX / sizeof(long double) ||
        n > SIZE_MAX / sizeof(double))
        return POP_SOLVE_FAILED;

    /* Recognize a rounded physical generator without repairing it in place.
     * The diagonal is checked as provenance, then excluded from the solve. */
    long double worst_column_error = 0.0L;
    for (size_t source = 0; source < n; ++source) {
        long double outflow = 0.0L;
        double diagonal = POP_CM(generator, n, source, source);
        if (!isfinite(diagonal) || diagonal > 0.0)
            return isfinite(diagonal) ? POP_SOLVE_FAILED : POP_NONFINITE;
        for (size_t dest = 0; dest < n; ++dest) {
            if (dest == source) continue;
            double rate = POP_CM(generator, n, dest, source);
            if (!isfinite(rate)) return POP_NONFINITE;
            if (rate < 0.0) return POP_SOLVE_FAILED;
            outflow += (long double)rate;
        }
        long double denominator = fmaxl(fabsl((long double)diagonal), outflow);
        long double error = denominator > 0.0L
            ? fabsl((long double)diagonal + outflow) / denominator
            : fabsl((long double)diagonal + outflow);
        if (error > worst_column_error) worst_column_error = error;
    }
    diagnostic->input_column_relative_error = (double)worst_column_error;
    if (!(worst_column_error <= POP_GENERATOR_COLUMN_ERROR_LIMIT))
        return POP_SOLVE_FAILED;
    diagnostic->generator_recognized = 1;

    long double *work = (long double *)calloc(n * n, sizeof(*work));
    long double *denominator = (long double *)calloc(n, sizeof(*denominator));
    long double *probability = (long double *)calloc(n, sizeof(*probability));
    double *projected = (double *)malloc(n * sizeof(*projected));
    if (!work || !denominator || !probability || !projected) {
        free(work); free(denominator); free(probability); free(projected);
        return POP_SOLVE_FAILED;
    }

    /* Row-generator view q(source,dest).  In column-major A[dest,source]
     * this has the same linear byte index, but not the same matrix semantics. */
    for (size_t source = 0; source < n; ++source)
        for (size_t dest = 0; dest < n; ++dest)
            if (dest != source)
                work[source * n + dest] =
                    (long double)POP_CM(generator, n, dest, source);

    PopulationStatus status = POP_OK;
    /* Continuous-time Grassmann-Taksar-Heyman state reduction. */
    for (size_t reduced = n; reduced-- > 1;) {
        long double scale = 0.0L;
        for (size_t dest = 0; dest < reduced; ++dest)
            scale += work[reduced * n + dest];
        if (!(scale > 0.0L) || !isfinite(scale)) {
            status = isfinite(scale) ? POP_RANK_INCOMPLETE : POP_NONFINITE;
            goto done;
        }
        denominator[reduced] = scale;
        for (size_t source = 0; source < reduced; ++source) {
            long double factor = work[source * n + reduced] / scale;
            if (!isfinite(factor)) { status = POP_NONFINITE; goto done; }
            for (size_t dest = 0; dest < reduced; ++dest) {
                work[source * n + dest] +=
                    factor * work[reduced * n + dest];
                if (!isfinite(work[source * n + dest])) {
                    status = POP_NONFINITE;
                    goto done;
                }
            }
        }
    }

    probability[0] = 1.0L;
    for (size_t state = 1; state < n; ++state) {
        long double incoming = 0.0L;
        for (size_t source = 0; source < state; ++source)
            incoming += probability[source] * work[source * n + state];
        if (!(incoming > 0.0L) || !(denominator[state] > 0.0L) ||
            !isfinite(incoming)) {
            status = isfinite(incoming) ? POP_RANK_INCOMPLETE : POP_NONFINITE;
            goto done;
        }
        probability[state] = incoming / denominator[state];
        if (!(probability[state] > 0.0L) || !isfinite(probability[state])) {
            status = POP_NONFINITE;
            goto done;
        }
    }
    {
        long double probability_sum = 0.0L;
        for (size_t state = 0; state < n; ++state)
            probability_sum += probability[state];
        if (!(probability_sum > 0.0L) || !isfinite(probability_sum)) {
            status = POP_NONFINITE;
            goto done;
        }
        long double normalization =
            (long double)total_population / probability_sum;
        double minimum = INFINITY, maximum = -INFINITY;
        for (size_t state = 0; state < n; ++state) {
            long double value = probability[state] * normalization;
            projected[state] = (double)value;
            if (!(projected[state] > 0.0) || !isfinite(projected[state])) {
                status = POP_NONFINITE;
                goto done;
            }
            if (projected[state] < minimum) minimum = projected[state];
            if (projected[state] > maximum) maximum = projected[state];
        }
        diagnostic->minimum_population = minimum;
        diagnostic->maximum_population = maximum;
    }

    /* Residual against the exact generator defined by the imported
     * off-diagonals.  Its diagonal is an extended-precision outflow sum, not
     * the cancellation-contaminated float64 diagonal from assembly. */
    {
        long double worst = 0.0L;
        for (size_t state = 0; state < n; ++state) {
            long double balance = 0.0L, scale = 0.0L, outflow = 0.0L;
            for (size_t other = 0; other < n; ++other) {
                if (other == state) continue;
                long double incoming_rate =
                    (long double)POP_CM(generator, n, state, other);
                long double outgoing_rate =
                    (long double)POP_CM(generator, n, other, state);
                long double incoming =
                    incoming_rate * (long double)projected[other];
                outflow += outgoing_rate;
                balance += incoming;
                scale += fabsl(incoming);
            }
            long double outgoing =
                outflow * (long double)projected[state];
            balance -= outgoing;
            scale += fabsl(outgoing);
            long double relative = scale > 0.0L
                ? fabsl(balance) / scale : fabsl(balance);
            if (relative > worst) worst = relative;
        }
        diagnostic->exact_generator_componentwise_residual = (double)worst;
        if (!(worst <= POP_GENERATOR_RESIDUAL_LIMIT)) {
            status = POP_SOLVE_FAILED;
            goto done;
        }
    }
    memcpy(solution, projected, n * sizeof(*solution));

done:
    free(work); free(denominator); free(probability); free(projected);
    return status;
}

#undef POP_CM

PopulationStatus population_superlevel_aggregate(
        const double *level_population, const int *membership, size_t n_levels,
        size_t n_superlevels, double *super_population) {
    if (!level_population || !membership || !super_population ||
        n_levels == 0 || n_superlevels == 0)
        return POP_ATOMIC_MISSING;
    memset(super_population, 0, n_superlevels * sizeof(double));
    for (size_t i = 0; i < n_levels; i++) {
        if (membership[i] < 0 || (size_t)membership[i] >= n_superlevels)
            return POP_ATOMIC_MISSING;
        if (!isfinite(level_population[i]) || level_population[i] < 0.0)
            return POP_NONFINITE;
        super_population[membership[i]] += level_population[i];
        if (!isfinite(super_population[membership[i]]))
            return POP_NONFINITE;
    }
    return POP_OK;
}
static double *pop_copy(const double *p,size_t n){if(!p||n==0)return NULL;double *q=(double*)malloc(n*sizeof(double));if(q)memcpy(q,p,n*sizeof(double));return q;}
int population_transaction_begin(PopulationTransaction *t,double *i,size_t ni,double *l,size_t nl,double *ne,size_t nn,double *p,size_t np,uint64_t req,uint64_t *com){
    if(!t||req==0||!com)return -1;
    memset(t,0,sizeof(*t));t->public_ion=i;t->public_level=l;t->public_ne=ne;t->public_partition=p;t->n_ion_values=ni;t->n_level_values=nl;t->n_ne_values=nn;t->n_partition_values=np;t->required_generation=req;t->committed_generation=com;t->status=POP_OK;t->work_ion=pop_copy(i,ni);t->work_level=pop_copy(l,nl);t->work_ne=pop_copy(ne,nn);t->work_partition=pop_copy(p,np);if((ni&&!t->work_ion)||(nl&&!t->work_level)||(nn&&!t->work_ne)||(np&&!t->work_partition)){population_transaction_abort(t,POP_SOLVE_FAILED);return -1;}return 0;
}
void population_transaction_abort(PopulationTransaction *t,PopulationStatus s){if(!t)return;free(t->work_ion);free(t->work_level);free(t->work_ne);free(t->work_partition);t->work_ion=t->work_level=t->work_ne=t->work_partition=NULL;t->status=s;}
static int pop_valid(const double *p,size_t n){for(size_t i=0;i<n;i++)if(!isfinite(p[i])||p[i]<0.0)return 0;return 1;}
PopulationStatus population_transaction_commit(PopulationTransaction *t){if(!t||t->status!=POP_OK)return t?t->status:POP_SOLVE_FAILED;if(!pop_valid(t->work_ion,t->n_ion_values)||!pop_valid(t->work_level,t->n_level_values)||!pop_valid(t->work_ne,t->n_ne_values)||!pop_valid(t->work_partition,t->n_partition_values)){population_transaction_abort(t,POP_NONFINITE);return POP_NONFINITE;}if(t->n_ion_values)memcpy(t->public_ion,t->work_ion,t->n_ion_values*sizeof(double));if(t->n_level_values)memcpy(t->public_level,t->work_level,t->n_level_values*sizeof(double));if(t->n_ne_values)memcpy(t->public_ne,t->work_ne,t->n_ne_values*sizeof(double));if(t->n_partition_values)memcpy(t->public_partition,t->work_partition,t->n_partition_values*sizeof(double));*t->committed_generation=t->required_generation;population_transaction_abort(t,POP_OK);return POP_OK;}
void population_counter_note(PopulationCounters *c,PopulationStatus s){if(!c)return;switch(s){case POP_EXACT_ZERO:c->pop_exact_zero_terms++;break;case POP_INVALID_TE:c->pop_blocked_te++;break;case POP_INVALID_PARTITION:c->pop_blocked_partition++;break;case POP_STALE_DERIVED_TEMPERATURE:c->pop_generation_mismatch++;break;case POP_BF_STALE:case POP_BB_STALE:c->pop_blocked_stale++;break;case POP_BF_UNSAMPLED:case POP_BB_UNSAMPLED:c->pop_blocked_unsampled++;break;case POP_BF_OOG:case POP_BB_OOG:c->pop_blocked_oog++;break;case POP_BF_MISS:case POP_BB_MISS:c->pop_blocked_miss++;break;case POP_PROFILE_MISMATCH:c->pop_blocked_profile++;break;case POP_QUERY_HASH_MISMATCH:c->pop_blocked_qhash++;break;case POP_RANK_INCOMPLETE:c->pop_rank_incomplete++;break;case POP_NE_NOT_CONVERGED:c->pop_ne_not_converged++;break;case POP_SOLVE_FAILED:c->pop_solve_failed++;break;case POP_NONFINITE:c->pop_nonfinite++;break;case POP_FORBIDDEN_FALLBACK:c->pop_fallback_attempts++;break;default:break;}}
void population_counters_print(FILE *f,const PopulationCounters *c){if(!f||!c)return;fprintf(f,"[A2-07][POP-VIEW] pop_generation_required=%llu pop_generation_committed=%llu pop_shells_attempted=%llu pop_shells_published=%llu pop_bf_terms=%llu pop_bb_terms=%llu pop_exact_zero_terms=%llu pop_blocked_stale=%llu pop_blocked_unsampled=%llu pop_blocked_oog=%llu pop_blocked_miss=%llu pop_blocked_profile=%llu pop_blocked_qhash=%llu pop_blocked_te=%llu pop_blocked_partition=%llu pop_rank_incomplete=%llu pop_ne_not_converged=%llu pop_solve_failed=%llu pop_nonfinite=%llu pop_generation_mismatch=%llu pop_fallback_attempts=%llu pop_partial_publish_attempts=%llu\n",(unsigned long long)c->pop_generation_required,(unsigned long long)c->pop_generation_committed,(unsigned long long)c->pop_shells_attempted,(unsigned long long)c->pop_shells_published,(unsigned long long)c->pop_bf_terms,(unsigned long long)c->pop_bb_terms,(unsigned long long)c->pop_exact_zero_terms,(unsigned long long)c->pop_blocked_stale,(unsigned long long)c->pop_blocked_unsampled,(unsigned long long)c->pop_blocked_oog,(unsigned long long)c->pop_blocked_miss,(unsigned long long)c->pop_blocked_profile,(unsigned long long)c->pop_blocked_qhash,(unsigned long long)c->pop_blocked_te,(unsigned long long)c->pop_blocked_partition,(unsigned long long)c->pop_rank_incomplete,(unsigned long long)c->pop_ne_not_converged,(unsigned long long)c->pop_solve_failed,(unsigned long long)c->pop_nonfinite,(unsigned long long)c->pop_generation_mismatch,(unsigned long long)c->pop_fallback_attempts,(unsigned long long)c->pop_partial_publish_attempts);}
