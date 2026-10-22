#ifndef CMF_GPU_RUNTIME_H
#define CMF_GPU_RUNTIME_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct CMFGpuRuntime CMFGpuRuntime;
typedef struct {
    uint64_t live,peak,reserved,observed_peak,baseline_free,total,kernel_local,stack,threads;
} CMFGpuMemory;
/* baseline_free는 생성 직전 nvidia-smi에서 얻은 바이트다. cap<=16GiB, GPU0만.
 * 문맥/모듈/최대 resident local+stack 예약을 포함해 한도를 지킨다. */
CMFGpuRuntime *cmf_gpu_create(int device,uint64_t cap,uint64_t baseline_free,int *code);
/* 현재 host 스레드의 마지막 생성 예약 계산. 문맥 실패도 숫자를 보존한다.
 * 기존 ABI는 그대로이며 선택적 진단 API를 지원하는 바이너리에서만 쓴다. */
int cmf_gpu_last_preflight(CMFGpuMemory *out,uint64_t *available,uint64_t *cap,int *reason);
void cmf_gpu_destroy(CMFGpuRuntime *ctx);
int cmf_gpu_allocate(CMFGpuRuntime *ctx,uint64_t bytes,void **out);
int cmf_gpu_release(CMFGpuRuntime *ctx,void *pointer);
int cmf_gpu_upload(CMFGpuRuntime *ctx,void *device,const void *host,uint64_t bytes);
int cmf_gpu_download(CMFGpuRuntime *ctx,void *host,const void *device,uint64_t bytes);
int cmf_gpu_zero(CMFGpuRuntime *ctx,void *device,uint64_t bytes);
int cmf_gpu_sync(CMFGpuRuntime *ctx);
int cmf_gpu_memory(CMFGpuRuntime *ctx,CMFGpuMemory *out);
int *cmf_gpu_status(CMFGpuRuntime *ctx);
#ifdef __cplusplus
}
#endif
#endif
