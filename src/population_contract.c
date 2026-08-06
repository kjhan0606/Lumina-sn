#include "population_contract.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define POP_EV_TO_ERG 1.602176634e-12
#define POP_K_BOLTZMANN 1.380649e-16

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
    PopSha256 s;pop_sha_init(&s);const char d[]="A2-07:atomic-partition-membership:v1";pop_sha_update(&s,d,sizeof(d)-1);pop_hash_u64(&s,a->n_ions);pop_hash_u64(&s,a->n_levels);
    for(size_t i=0;i<=a->n_ions;i++)pop_hash_i32(&s,a->level_offset[i]);
    for(size_t i=0;i<a->n_levels;i++){pop_hash_f64(&s,a->energy_eV[i]);pop_hash_i32(&s,a->g[i]);pop_hash_i32(&s,a->runtime_membership?a->runtime_membership[i]:0);pop_hash_i32(&s,a->level_Z?a->level_Z[i]:0);pop_hash_i32(&s,a->level_ion?a->level_ion[i]:0);}pop_sha_final(&s,out);return POP_OK;
}
PopulationStatus population_partition_ion(const PopulationAtomicView *a,size_t ion,double te,double *out){
    if(!out||!a||!a->level_offset||!a->energy_eV||!a->g||ion>=a->n_ions)
        return POP_ATOMIC_MISSING;
    if(!isfinite(te)||te<=0.0)return POP_INVALID_TE;
    int lo=a->level_offset[ion],hi=a->level_offset[ion+1];if(lo<0||hi<=lo||(size_t)hi>a->n_levels)return POP_ATOMIC_MISSING;double e0=INFINITY;
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
