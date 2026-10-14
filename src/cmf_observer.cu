#include "cmf_observer_ray.h"
#include <cuda_runtime.h>
#include <stdlib.h>

static __global__ void observer_kernel(CMFObserverInput in, const double *p,
                                       double *v, size_t n) {
    const int nr=in.ncore+in.nouter;
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;
        i+=(size_t)gridDim.x*blockDim.x)
        v[i]=cmf_observer_ray(&in,p[i%nr],in.nu_obs[i/nr]);
}

extern "C" int cmf_observer_gpu(const CMFObserverInput *in, double *Lnu) {
    if(!Lnu || cmf_observer_validate(in))return -1;
    const int nr=in->ncore+in->nouter;
    const size_t cells=(size_t)in->ns*in->nf, n=(size_t)in->nobs*nr;
    CMFObserverInput d=*in;
    double *buffers[9]={0};
    const double *host[9]={in->ri,in->ro,in->chi,in->es,in->source,in->J,
                          in->nu_obs,NULL,NULL};
    size_t counts[9]={(size_t)in->ns,(size_t)in->ns,cells,cells,cells,cells,
                     (size_t)in->nobs,(size_t)nr,n};
    double *p=(double*)malloc((size_t)nr*sizeof(double));
    double *w=(double*)malloc((size_t)nr*sizeof(double));
    double *v=(double*)malloc(n*sizeof(double));
    int rc=-1;
    if(!p || !w || !v || cmf_observer_quadrature(in,p,w))goto done;
    host[7]=p;
    for(int j=0;j<9;++j) {
        if(cudaMalloc((void**)&buffers[j],counts[j]*sizeof(double))!=cudaSuccess)goto done;
        if(host[j] && cudaMemcpy(buffers[j],host[j],counts[j]*sizeof(double),
                                cudaMemcpyHostToDevice)!=cudaSuccess)goto done;
    }
    d.ri=buffers[0];d.ro=buffers[1];d.chi=buffers[2];d.es=buffers[3];
    d.source=buffers[4];d.J=buffers[5];d.nu_obs=buffers[6];
    observer_kernel<<<256,128>>>(d,buffers[7],buffers[8],n);
    if(cudaGetLastError()!=cudaSuccess || cudaDeviceSynchronize()!=cudaSuccess ||
       cudaMemcpy(v,buffers[8],n*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)goto done;
    rc=cmf_observer_reduce(in,v,w,Lnu);
done:
    for(int j=0;j<9;++j)if(buffers[j])cudaFree(buffers[j]);
    free(p);free(w);free(v);return rc;
}
