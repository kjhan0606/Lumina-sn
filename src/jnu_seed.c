#include "jnu_seed.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t h[8]; uint64_t bits; unsigned char b[64]; size_t n; } SeedSha;
static uint32_t rr(uint32_t x,unsigned n){return(x>>n)|(x<<(32U-n));}
static void sb(SeedSha*s,const unsigned char*p){
 static const uint32_t k[64]={0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};
 uint32_t w[64];for(int i=0;i<16;i++)w[i]=((uint32_t)p[4*i]<<24)|((uint32_t)p[4*i+1]<<16)|((uint32_t)p[4*i+2]<<8)|p[4*i+3];for(int i=16;i<64;i++){uint32_t a=w[i-15],b=w[i-2];w[i]=w[i-16]+(rr(a,7)^rr(a,18)^(a>>3))+w[i-7]+(rr(b,17)^rr(b,19)^(b>>10));}uint32_t a=s->h[0],b=s->h[1],c=s->h[2],d=s->h[3],e=s->h[4],f=s->h[5],g=s->h[6],h=s->h[7];for(int i=0;i<64;i++){uint32_t t1=h+(rr(e,6)^rr(e,11)^rr(e,25))+((e&f)^(~e&g))+k[i]+w[i],t2=(rr(a,2)^rr(a,13)^rr(a,22))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}s->h[0]+=a;s->h[1]+=b;s->h[2]+=c;s->h[3]+=d;s->h[4]+=e;s->h[5]+=f;s->h[6]+=g;s->h[7]+=h;
}
static void si(SeedSha*s){static const uint32_t h[8]={0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};memcpy(s->h,h,sizeof(h));s->bits=0;s->n=0;}
static void su(SeedSha*s,const void*v,size_t n){const unsigned char*p=v;s->bits+=(uint64_t)n*8U;while(n){size_t z=64-s->n,t=n<z?n:z;memcpy(s->b+s->n,p,t);s->n+=t;p+=t;n-=t;if(s->n==64){sb(s,s->b);s->n=0;}}}
static void sf(SeedSha*s,char out[65]){uint64_t bits=s->bits;unsigned char one=128,zero=0,len[8],d[32];su(s,&one,1);while(s->n!=56)su(s,&zero,1);for(int i=0;i<8;i++)len[7-i]=(unsigned char)(bits>>(8*i));su(s,len,8);static const char x[]="0123456789abcdef";for(int i=0;i<8;i++){d[4*i]=s->h[i]>>24;d[4*i+1]=s->h[i]>>16;d[4*i+2]=s->h[i]>>8;d[4*i+3]=s->h[i];}for(int i=0;i<32;i++){out[2*i]=x[d[i]>>4];out[2*i+1]=x[d[i]&15];}out[64]=0;}

int jnu_seed_sha256_file(const char *path,char out[65]){FILE*f=fopen(path,"rb");if(!f)return-1;SeedSha s;si(&s);unsigned char b[65536];size_t n;while((n=fread(b,1,sizeof(b),f))!=0)su(&s,b,n);int bad=ferror(f)||fclose(f)!=0;if(bad)return-1;sf(&s,out);return 0;}
static int rd(FILE*f,void*p,size_t z,size_t n){return fread(p,z,n,f)==n?0:-1;}

JnuSeedStatus jnu_seed_load_native(const char *path,const double *want_edges,
 const uint64_t *want_ids,size_t ns,double epoch,RadiationFieldOwner *owner,
 SeedCapability *cap,JnuSeedCounters *ct,char manifest[65]){
 if(!path||!want_edges||!want_ids||!ns||!owner||!owner->enabled||!cap||!ct||!manifest)return JNU_SEED_IO_OR_SCHEMA;
 if(owner->field.generation.computed_generation!=0||owner->field.provenance.kind!=RADIATION_FIELD_PROVENANCE_NONE){ct->hold_attempts++;return JNU_SEED_FORBIDDEN_FALLBACK;}
 if(jnu_seed_sha256_file(path,manifest)!=0)return JNU_SEED_IO_OR_SCHEMA;
 FILE*f=fopen(path,"rb");if(!f)return JNU_SEED_IO_OR_SCHEMA;ct->seed_files_opened++;
 JnuSeedDiskHeader h;memset(&h,0,sizeof(h));if(rd(f,&h,sizeof(h),1)||memcmp(h.magic,LUMINA_JNU_SEED_MAGIC,16)||h.version!=LUMINA_JNU_SEED_VERSION||h.endian_tag!=0x01020304U||h.n_shells!=ns||h.n_bins!=LUMINA_RADFIELD_N_BINS||h.units!=RADIATION_FIELD_UNITS_ERG_S_NEG1_CM_NEG2_HZ_NEG1_SR_NEG1||h.frame!=RADIATION_FIELD_FRAME_SHELL_COMOVING||!(h.epoch==epoch)){fclose(f);return JNU_SEED_IO_OR_SCHEMA;}
 size_t cells=ns*(size_t)h.n_bins;uint64_t*ids=malloc(ns*sizeof(*ids));double*se=malloc((ns+1)*sizeof(*se));double*fe=malloc(((size_t)h.n_bins+1)*sizeof(*fe));unsigned char*va=malloc(cells);double*j=malloc(cells*sizeof(*j));if(!ids||!se||!fe||!va||!j){free(ids);free(se);free(fe);free(va);free(j);fclose(f);return JNU_SEED_IO_OR_SCHEMA;}
 int io=rd(f,ids,sizeof(*ids),ns)||rd(f,se,sizeof(*se),ns+1)||rd(f,fe,sizeof(*fe),(size_t)h.n_bins+1)||rd(f,va,1,cells)||rd(f,j,sizeof(*j),cells);int tail=fgetc(f);fclose(f);if(io||tail!=EOF){free(ids);free(se);free(fe);free(va);free(j);return JNU_SEED_IO_OR_SCHEMA;}
 int shape=strcmp(h.shape_sha256,LUMINA_JNU_SEED_SHAPE_SHA256)!=0;int edge=strcmp(h.edge_sha256,LUMINA_RADFIELD_EDGE_SHA256)!=0;for(size_t b=0;b<=h.n_bins&&!edge;b++)if(fe[b]!=owner->field.frequency_bin_edges.values[b])edge=1;if(shape)ct->shape_hash_failures++;if(edge)ct->edge_hash_failures++;
 size_t invalid=0;for(size_t s=0;s<ns;s++){int shell=ids[s]!=want_ids[s]||se[s]!=want_edges[s]||se[s+1]!=want_edges[s+1];if(shell)ct->shell_identity_failures++;for(size_t b=0;b<h.n_bins;b++){size_t q=s*h.n_bins+b;int ok=!shape&&!edge&&!shell&&va[q]==RADIATION_FIELD_VALID&&isfinite(j[q])&&j[q]>=0.0;if(ns>=50&&s>=44&&s<=49&&!ok)ct->coverage_failures_s44_s49++;if(!ok)invalid++;}}
 ct->seed_cells_loaded+=cells;ct->seed_invalid_cells+=invalid;
 if(invalid){free(ids);free(se);free(fe);free(va);free(j);return JNU_SEED_BLOCKED_INCOMPLETE_COVERAGE;}
 if(seed_capability_open(cap,h.epoch,manifest)!=SEED_OK||seed_capability_check_read(cap,0,0,h.epoch,manifest)!=SEED_OK){free(ids);free(se);free(fe);free(va);free(j);return JNU_SEED_FORBIDDEN_FALLBACK;}
 RadiationField*rf=&owner->field;memcpy(rf->shell_boundaries.values,se,(ns+1)*sizeof(*se));memcpy(rf->J_nu.values,j,cells*sizeof(*j));for(size_t q=0;q<cells;q++)rf->validity.values[q]=(RadiationFieldValidityState)va[q];rf->epoch=h.epoch;rf->generation.required_generation=0;rf->generation.computed_generation=0;rf->provenance.kind=(RadiationFieldProvenanceKind)h.provenance;rf->provenance.producer="A2_16_NATIVE_JNU_SEED";rf->provenance.raw_ledger_sha256=cap->manifest_sha256;owner->seed_capability=cap;
 free(ids);free(se);free(fe);free(va);free(j);return JNU_SEED_OK;
}
void jnu_seed_report(const JnuSeedCounters*c){if(!c)return;printf("[A2-17][JNU-SEED] files=%llu cells=%llu invalid=%llu shape_hash_failures=%llu edge_hash_failures=%llu shell_identity_failures=%llu coverage_s44_s49=%llu hold=%llu extrapolation=%llu neighbor_copy=%llu zero_fill=%llu fallback=%llu partial_publish=%llu\n",(unsigned long long)c->seed_files_opened,(unsigned long long)c->seed_cells_loaded,(unsigned long long)c->seed_invalid_cells,(unsigned long long)c->shape_hash_failures,(unsigned long long)c->edge_hash_failures,(unsigned long long)c->shell_identity_failures,(unsigned long long)c->coverage_failures_s44_s49,(unsigned long long)c->hold_attempts,(unsigned long long)c->extrapolation_attempts,(unsigned long long)c->neighbor_copy_attempts,(unsigned long long)c->zero_fill_attempts,(unsigned long long)c->seed_fallback_attempts,(unsigned long long)c->partial_seed_publish_attempts);}
