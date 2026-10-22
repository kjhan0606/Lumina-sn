#ifndef CMF_WARP_DEVICE_HPP
#define CMF_WARP_DEVICE_HPP
namespace lumina_warp_device {
constexpr int maximum_sms=108;
constexpr int maximum_blocks_per_sm=32;
constexpr int block_threads=128;
// A100과 Ada만 허용한다. 기존 동시 공유상태 상한은 확대하지 않는다.
inline bool compatible(int major,int minor,int sms,int threads_per_sm,
                       int maximum_block_threads,int warp_size){
    return major==8&&(minor==0||minor==9)&&sms>0&&sms<=maximum_sms&&
        threads_per_sm>=block_threads&&
        threads_per_sm<=maximum_blocks_per_sm*block_threads&&
        maximum_block_threads>=block_threads&&warp_size==32;
}
}
#endif
