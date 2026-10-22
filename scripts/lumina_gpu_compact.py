"""양의 원천 측도의 축약 맵/독립 χ·η 트리 후보. 기존 후보를 자동 교체하지 않는다."""
import ctypes as ct

from lumina_gpu_driver import CACHE_LIMIT
from lumina_gpu_linear import Runtime as LinearRuntime,LinearDriver,LOCAL_BYTES
from lumina_gpu_measure import NODE_BYTES,PROJECTION_BYTES,SOURCE_NODE_BYTES
from lumina_linear_native import CPositive

CHI_BLOCK=2
SOURCE_BLOCK=32


class CompactMap(ct.Structure):
    _fields_=[(name,CPositive) for name in ('transmission','source','source_error')]+[('tau_error',ct.c_double)]


MAP_BYTES=ct.sizeof(CompactMap)


class Runtime(LinearRuntime):
    def __init__(self,library,baseline_free):
        super().__init__(library,baseline_free,_node_bytes=NODE_BYTES,_map_bytes=MAP_BYTES)


def memory_layout(nf,width,*,endpoint_cache=False):
    if type(nf) is not int or type(width) is not int or not 8<=nf<=2000000 or not 1<=width<=65536:
        raise ValueError('축약 GPU 격자/폭 오류')
    if type(endpoint_cache) is not bool:raise ValueError('끝점 캐시 선택 형식 오류')
    arena=max(192*1024**2,3*nf*MAP_BYTES);arena=(arena+255)//256*256
    def leaves(block):
        blocks=(nf-2)//block+1
        return 1<<(blocks-1).bit_length()
    chi_tree=2*leaves(CHI_BLOCK)*272;source_tree=2*leaves(SOURCE_BLOCK)*SOURCE_NODE_BYTES
    results=max(72*1024**2,nf*LOCAL_BYTES);slots=min(4096,results//(width*LOCAL_BYTES))
    slabs=(nf+width-1)//width
    weighted=nf*8+((nf+4095)//4096)*32
    aux=3*nf*8+nf*16+4096*8*2+slabs*8+8+weighted
    endpoint=4*nf*8 if endpoint_cache else 0
    workspace=arena+chi_tree+source_tree+endpoint+results+aux+PROJECTION_BYTES
    if slots<1 or workspace>CACHE_LIMIT:raise MemoryError('축약 활성 적분 작업공간1GiB 초과')
    return dict(arena=arena,tree=chi_tree+source_tree+endpoint,chi_tree=chi_tree,source_tree=source_tree,
                results=results,slots=slots,workspace=workspace,endpoint_bytes=endpoint)


class CompactDriver(LinearDriver):
    map_bytes=MAP_BYTES
    node_bytes=NODE_BYTES
    projection_bytes=PROJECTION_BYTES
    chi_block=CHI_BLOCK
    source_block=SOURCE_BLOCK
    engine='SPACE_LINEAR_COMPACT_SOURCE_MEASURE_GPU'

    def layout_for(self,nf,width):
        return memory_layout(nf,width,endpoint_cache=self.endpoint_cache)

    def tree_bytes(self,piece):
        return 2*piece.leaves*272+2*piece.source_leaves*SOURCE_NODE_BYTES+(4*piece.nf*8 if self.endpoint_cache else 0)


class EndpointCompactDriver(CompactDriver):
    endpoint_cache=True
    engine='SPACE_LINEAR_COMPACT_EXACT_ENDPOINT_CACHE_GPU'
