#include "cmf_warp_device.hpp"
extern "C" int warp_device_probe(int major,int minor,int sms,int threads_per_sm,
    int block_threads,int warp_size){
    return lumina_warp_device::compatible(major,minor,sms,threads_per_sm,block_threads,warp_size)?1:0;
}
extern "C" int warp_device_budget_probe(){
    return lumina_warp_device::maximum_sms*lumina_warp_device::maximum_blocks_per_sm*4*64*24;
}
