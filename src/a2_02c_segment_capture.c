/* A2-02C gated raw MC segment capture (CPU transport lane).
 *
 * Binary schema is deliberately fixed-width and endian-tagged.  Header:
 * 128 bytes, followed by n_shells 16-byte shell entries, followed by 88-byte
 * segment records.  The offline reader rejects an incomplete header. */

#include "a2_02c_segment_capture.h"

#include <errno.h>
#include <stdatomic.h>

#define A2C_HEADER_BYTES 128U
#define A2C_SHELL_BYTES 16U
#define A2C_RECORD_BYTES 88U
#define A2C_ENDIAN_TAG 0x01020304U

static FILE *a2c_stream;
static atomic_flag a2c_lock = ATOMIC_FLAG_INIT;
static _Atomic unsigned long long a2c_records;
static int a2c_gate = -1;
static int a2c_active;
static _Atomic int a2c_failed;
static unsigned long long a2c_generation;
static int a2c_n_shells;
static const double *a2c_volume;
static double a2c_delta_t;

static void a2c_put_u32(unsigned char *p, unsigned int value) {
    p[0]=(unsigned char)value; p[1]=(unsigned char)(value>>8);
    p[2]=(unsigned char)(value>>16); p[3]=(unsigned char)(value>>24);
}
static void a2c_put_u64(unsigned char *p, unsigned long long value) {
    for (int i=0;i<8;i++) p[i]=(unsigned char)(value>>(8*i));
}
static void a2c_put_f64(unsigned char *p, double value) {
    unsigned long long bits=0; memcpy(&bits,&value,sizeof(bits)); a2c_put_u64(p,bits);
}
static void a2c_acquire(void) {
    while (atomic_flag_test_and_set_explicit(&a2c_lock,memory_order_acquire)) { }
}
static void a2c_release(void) {
    atomic_flag_clear_explicit(&a2c_lock,memory_order_release);
}
static int a2c_enabled(void) {
    if (a2c_gate < 0) {
        const char *value=getenv("LUMINA_A2_02C_SEGMENT_CAPTURE");
        a2c_gate=(value && strcmp(value,"1")==0) ? 1 : 0;
    }
    return a2c_gate;
}
static unsigned long long a2c_target_generation(void) {
    const char *value=getenv("LUMINA_A2_02C_CAPTURE_GENERATION");
    char *end=NULL; unsigned long long result;
    if (!value || !*value || value[0]=='-') return 0;
    errno=0; result=strtoull(value,&end,10);
    if (errno || !end || *end || result==0) return 0;
    return result;
}

void a2_02c_capture_begin(unsigned long long generation,
                          unsigned long long production_packet_count,
                          const Geometry *geo, const double *shell_volume,
                          double delta_t) {
    unsigned char header[A2C_HEADER_BYTES]={0};
    const char *path;
    unsigned long long target;
    if (!a2c_enabled()) return;
    target=a2c_target_generation();
    if (!target || generation != target || a2c_active || a2c_stream) return;
    path=getenv("LUMINA_A2_02C_CAPTURE_PATH");
    if (!path || !*path || !geo || !shell_volume || geo->n_shells<=0 ||
        production_packet_count==0 || !(delta_t>0.0) || !isfinite(delta_t)) {
        a2c_failed=1; return;
    }
    /* x refuses replacement: neither an old BLOCKED artifact nor a previous
     * A2-02C capture can be silently overwritten. */
    a2c_stream=fopen(path,"wbx");
    if (!a2c_stream) { a2c_failed=1; return; }
    memcpy(header,"LA2SGC1\0",8);
    a2c_put_u32(header+8,A2C_ENDIAN_TAG);
    a2c_put_u32(header+12,1U);
    a2c_put_u32(header+16,A2C_HEADER_BYTES);
    a2c_put_u32(header+20,A2C_RECORD_BYTES);
    a2c_put_u32(header+24,(unsigned int)geo->n_shells);
    a2c_put_u32(header+28,0U); /* complete flag, set only by end */
    a2c_put_u64(header+32,production_packet_count);
    a2c_put_u64(header+40,generation);
    a2c_put_f64(header+48,geo->time_explosion);
    a2c_put_f64(header+56,delta_t);
    a2c_put_u64(header+64,0ULL); /* final segment count */
    a2c_put_u64(header+72,(unsigned long long)geo->n_shells*A2C_SHELL_BYTES);
    if (fwrite(header,1,sizeof(header),a2c_stream)!=sizeof(header)) a2c_failed=1;
    for (int shell=0; !a2c_failed && shell<geo->n_shells; ++shell) {
        unsigned char item[A2C_SHELL_BYTES]={0};
        a2c_put_u32(item,(unsigned int)shell);
        a2c_put_f64(item+8,shell_volume[shell]);
        if (!(shell_volume[shell]>0.0) || !isfinite(shell_volume[shell]) ||
            fwrite(item,1,sizeof(item),a2c_stream)!=sizeof(item)) a2c_failed=1;
    }
    if (a2c_failed) { fclose(a2c_stream); a2c_stream=NULL; return; }
    atomic_store_explicit(&a2c_records,0ULL,memory_order_relaxed);
    a2c_generation=generation; a2c_n_shells=geo->n_shells;
    a2c_volume=shell_volume; a2c_delta_t=delta_t; a2c_active=1;
}

void a2_02c_capture_segment(const RPacket *pkt,
                            unsigned long long segment_id,
                            double path_length, double time_explosion) {
    unsigned char item[A2C_RECORD_BYTES]={0};
    double r1,mu1,d0,d1,nu0,nu1,e0,e1;
    int shell;
    if (!a2c_active || a2c_failed) return;
    if (!pkt || !(path_length>0.0) || !isfinite(path_length) ||
        !(time_explosion>0.0) || !isfinite(time_explosion)) { a2c_failed=1; return; }
    shell=pkt->current_shell_id;
    if (shell<0 || shell>=a2c_n_shells) { a2c_failed=1; return; }
    d0=get_doppler_factor(pkt->r,pkt->mu,time_explosion);
    r1=sqrt(pkt->r*pkt->r + path_length*path_length +
            2.0*pkt->r*path_length*pkt->mu);
    mu1=(pkt->mu*pkt->r+path_length)/r1;
    d1=get_doppler_factor(r1,mu1,time_explosion);
    nu0=pkt->nu*d0; nu1=pkt->nu*d1;
    e0=pkt->energy*d0; e1=pkt->energy*d1;
    if (!(nu0>0.0) || !(nu1>0.0) || !(e0>=0.0) || !(e1>=0.0) ||
        !isfinite(nu0) || !isfinite(nu1) || !isfinite(e0) || !isfinite(e1)) {
        a2c_failed=1; return;
    }
    a2c_put_u64(item,(unsigned long long)pkt->index);
    a2c_put_u64(item+8,segment_id);
    a2c_put_u64(item+16,a2c_generation);
    a2c_put_u32(item+24,(unsigned int)shell);
    a2c_put_u32(item+28,1U); /* comoving endpoints; linear homologous trajectory */
    a2c_put_f64(item+32,nu0); a2c_put_f64(item+40,nu1);
    a2c_put_f64(item+48,e0); a2c_put_f64(item+56,e1);
    a2c_put_f64(item+64,path_length);
    a2c_put_f64(item+72,a2c_volume[shell]);
    a2c_put_f64(item+80,a2c_delta_t);
    a2c_acquire();
    if (fwrite(item,1,sizeof(item),a2c_stream)!=sizeof(item)) a2c_failed=1;
    else atomic_fetch_add_explicit(&a2c_records,1ULL,memory_order_relaxed);
    a2c_release();
}

void a2_02c_capture_end(void) {
    unsigned char value[8];
    unsigned long long count;
    if (!a2c_active) return;
    a2c_acquire();
    count=atomic_load_explicit(&a2c_records,memory_order_relaxed);
    if (!a2c_failed && fflush(a2c_stream)==0 && fseek(a2c_stream,64L,SEEK_SET)==0) {
        a2c_put_u64(value,count);
        if (fwrite(value,1,8,a2c_stream)!=8) a2c_failed=1;
    } else a2c_failed=1;
    if (!a2c_failed && fseek(a2c_stream,28L,SEEK_SET)==0) {
        unsigned char complete[4]; a2c_put_u32(complete,1U);
        if (fwrite(complete,1,4,a2c_stream)!=4) a2c_failed=1;
    } else a2c_failed=1;
    fflush(a2c_stream); fclose(a2c_stream); a2c_stream=NULL;
    a2c_active=0; a2c_volume=NULL;
    a2c_release();
}
