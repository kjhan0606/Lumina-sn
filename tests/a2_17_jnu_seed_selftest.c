#include "jnu_seed.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(x,n) do{if(!(x)){fprintf(stderr,"A2_17_JNU_SEED_FAIL %s line=%d\n",n,__LINE__);failures++;}}while(0)

static int make_seed(const char *path, int poison_edge, int poison_s44)
{
    const size_t ns=50, nb=LUMINA_RADFIELD_N_BINS, cells=ns*nb;
    JnuSeedDiskHeader h; memset(&h,0,sizeof(h));
    memcpy(h.magic,LUMINA_JNU_SEED_MAGIC,16); h.version=1; h.endian_tag=0x01020304U;
    h.n_shells=ns; h.n_bins=nb; h.units=RADIATION_FIELD_UNITS_ERG_S_NEG1_CM_NEG2_HZ_NEG1_SR_NEG1;
    h.frame=RADIATION_FIELD_FRAME_SHELL_COMOVING; h.provenance=RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY; h.epoch=1683072.0;
    memcpy(h.shape_sha256,LUMINA_JNU_SEED_SHAPE_SHA256,65);
    memcpy(h.edge_sha256,poison_edge?"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff":LUMINA_RADFIELD_EDGE_SHA256,65);
    memset(h.source_payload_sha256,'1',64); memset(h.source_geometry_sha256,'2',64);
    uint64_t *ids=malloc(ns*sizeof(*ids)); double *se=malloc((ns+1)*sizeof(*se));
    double *fe=malloc((nb+1)*sizeof(*fe)); unsigned char *v=malloc(cells); double *j=malloc(cells*sizeof(*j));
    if(!ids||!se||!fe||!v||!j)return -1;
    for(size_t s=0;s<ns;s++){ids[s]=s;se[s]=1.0e8*(double)(s+1);}se[ns]=1.0e8*(double)(ns+1);
    double dl=log(LUMINA_RADFIELD_NU_MAX_HZ/LUMINA_RADFIELD_NU_MIN_HZ)/(double)nb;
    for(size_t b=0;b<=nb;b++) fe[b]=LUMINA_RADFIELD_NU_MIN_HZ*exp((double)b*dl);
    fe[0]=LUMINA_RADFIELD_NU_MIN_HZ;fe[nb]=LUMINA_RADFIELD_NU_MAX_HZ;
    for(size_t q=0;q<cells;q++){v[q]=RADIATION_FIELD_VALID;j[q]=1.0e-12*(double)(q+1);}
    if(poison_s44)v[44*nb+17]=RADIATION_FIELD_UNSAMPLED;
    FILE*f=fopen(path,"wb");int bad=!f||fwrite(&h,sizeof(h),1,f)!=1||fwrite(ids,sizeof(*ids),ns,f)!=ns||fwrite(se,sizeof(*se),ns+1,f)!=ns+1||fwrite(fe,sizeof(*fe),nb+1,f)!=nb+1||fwrite(v,1,cells,f)!=cells||fwrite(j,sizeof(*j),cells,f)!=cells;if(f&&fclose(f))bad=1;
    free(ids);free(se);free(fe);free(v);free(j);return bad?-1:0;
}

static JnuSeedStatus load(const char *path, RadiationFieldOwner *owner, SeedCapability *cap, JnuSeedCounters *ct)
{
    double se[51]; uint64_t ids[50]; char hash[65];
    for(size_t s=0;s<50;s++){ids[s]=s;se[s]=1.0e8*(double)(s+1);}se[50]=5.1e9;
    return jnu_seed_load_native(path,se,ids,50,1683072.0,owner,cap,ct,hash);
}

int main(void)
{
    const char *good="/tmp/a2_17_seed_good.bin",*edge="/tmp/a2_17_seed_edge.bin",*cov="/tmp/a2_17_seed_cov.bin";
    CHECK(make_seed(good,0,0)==0,"fixture-good");CHECK(make_seed(edge,1,0)==0,"fixture-edge");CHECK(make_seed(cov,0,1)==0,"fixture-coverage");
    RadiationFieldOwner o;SeedCapability c={0};JnuSeedCounters n={0};CHECK(radiation_field_owner_init(&o,50)==0,"owner");CHECK(load(good,&o,&c,&n)==JNU_SEED_OK,"native-load");CHECK(o.field.provenance.kind==RADIATION_FIELD_PROVENANCE_CMFGEN_REPLAY,"provenance");CHECK(o.field.J_nu.values[17]>0.0,"actual-array");radiation_field_owner_free(&o);
    memset(&c,0,sizeof(c));memset(&n,0,sizeof(n));CHECK(radiation_field_owner_init(&o,50)==0,"owner-edge");CHECK(load(edge,&o,&c,&n)==JNU_SEED_BLOCKED_INCOMPLETE_COVERAGE,"edge-blocked");CHECK(o.field.provenance.kind==RADIATION_FIELD_PROVENANCE_NONE,"edge-no-partial-publish");CHECK(n.edge_hash_failures==1,"edge-counter");radiation_field_owner_free(&o);
    memset(&c,0,sizeof(c));memset(&n,0,sizeof(n));CHECK(radiation_field_owner_init(&o,50)==0,"owner-cov");CHECK(load(cov,&o,&c,&n)==JNU_SEED_BLOCKED_INCOMPLETE_COVERAGE,"coverage-blocked");CHECK(n.coverage_failures_s44_s49==1,"coverage-counter");CHECK(n.hold_attempts==0&&n.extrapolation_attempts==0&&n.neighbor_copy_attempts==0&&n.zero_fill_attempts==0&&n.seed_fallback_attempts==0&&n.partial_seed_publish_attempts==0,"no-fallbacks");radiation_field_owner_free(&o);
    remove(good);remove(edge);remove(cov);
    if(failures)return 1;
    printf("A2_17_JNU_SEED_SELFTEST PASS native=1 edge_poison=blocked s44_poison=blocked fallbacks=0\n");
    return 0;
}
