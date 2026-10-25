"""실제 경로 끝점 계수의 GPU 맵/순서 있는 축약. 기존 메모리 runtime을 재사용한다."""
import ctypes as ct
from pathlib import Path
import time
import uuid

import numpy as np

from lumina_fullsr_sc import SCALED
from lumina_linear_native import DenseView
from lumina_gpu_driver import check, check_host

LOCATION_DTYPE = np.dtype([('mode', 'i4'), ('first', 'i4'), ('last', 'i4'), ('weight', 'f8')], align=True)


class DeviceInput:
    """기존 reader의 동일 dense 세대를 한 번 업로드한다. 물리 계수는 변경하지 않는다."""
    def __init__(self, runtime, provider):
        self.blocks = []; self.runtime = runtime; started = time.monotonic()
        view = provider.borrow_dense_view(); self.view = DenseView.from_buffer_copy(view)
        n = view.ns*view.nf
        try:
            for name, count in (('canonical', 3*n), ('outer', view.nf*(2*view.ne+1)),
                                ('j', n if view.jmode else 0), ('radius', view.ns), ('external_radius', view.ne)):
                if not count: continue
                buffer = runtime.allocate(count*8); self.blocks.append(buffer)
                runtime.upload_pointer(buffer, getattr(view, name), count*8)
                setattr(self.view, name, ct.cast(buffer.address, ct.POINTER(ct.c_double)))
            runtime.sync(); self.upload_seconds = time.monotonic()-started
        except Exception:
            self.close(); raise

    def close(self):
        for block in reversed(self.blocks): block.close()
        self.blocks = []

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


class ChainGPU:
    def __init__(self, library):
        self.lib = ct.CDLL(str(Path(library).resolve()))
        specs = {'build': [ct.c_int]*2+[ct.c_void_p]*6,
                 'provider_build': [ct.c_void_p]+[ct.c_int]*2+[ct.c_void_p]*7,
                 'reduce': [ct.c_int]*2+[ct.c_void_p]*3,
                 'apply': [ct.c_int]+[ct.c_void_p]*4,
                 'device_uuid': [ct.c_void_p],
                 'workspace': [ct.c_void_p]}
        for name, args in specs.items():
            fun = getattr(self.lib, 'lumina_fullsr_chain_'+name); fun.argtypes = args; fun.restype = ct.c_int

    def workspace(self):
        out = ct.c_size_t(); check(self.lib.lumina_fullsr_chain_workspace(ct.byref(out))); return out.value

    def device_uuid(self):
        out = ct.create_string_buffer(16); check(self.lib.lumina_fullsr_chain_device_uuid(out))
        return 'GPU-'+str(uuid.UUID(bytes=out.raw))

    def execute(self, runtime, lengths, deltas, chi, eta, initial, *, device_input=None,
                locations=None, phases=None, columns=None):
        lengths, deltas, chi, eta = [np.ascontiguousarray(a, dtype='f8') for a in (lengths, deltas, chi, eta)]
        steps = len(lengths); rays = len(initial)
        if (not 1 <= steps <= 65536 or not 1 <= rays <= 512 or steps*rays > 2000000 or
                lengths.shape != (steps,) or deltas.shape != (steps,) or
                chi.shape != (steps+1, rays) or eta.shape != chi.shape or
                initial.shape != (rays,) or initial.dtype != SCALED or not initial.flags.c_contiguous):
            raise ValueError('GPU chain 크기/dtype 오류')
        arrays = (lengths, deltas, chi, eta, initial)
        if device_input is not None:
            locations = np.ascontiguousarray(locations, dtype=LOCATION_DTYPE)
            phases = np.ascontiguousarray(phases, dtype='f8'); columns = np.ascontiguousarray(columns, dtype='i4')
            if locations.shape != (steps+1,) or phases.shape != (steps+1,) or columns.shape != (rays,):
                raise ValueError('GPU source 위치/주파수 크기 오류')
            arrays = (lengths, deltas, locations, phases, columns, initial)
        check_host(); buffers = []; started = time.monotonic()
        def allocate(size):
            out = runtime.allocate(int(size)); buffers.append(out); return out
        try:
            device = []
            for array in arrays:
                out = allocate(array.nbytes); runtime.upload(out, array); device.append(out)
            maps = allocate(steps*rays*32); spare = allocate(((steps+1)//2)*rays*32)
            result = allocate(initial.nbytes); uploaded = time.monotonic()
            if device_input is None:
                check(self.lib.lumina_fullsr_chain_build(steps, rays, *[b.address for b in device[:4]], maps.address, runtime.status))
            else:
                check(self.lib.lumina_fullsr_chain_provider_build(ct.byref(device_input.view), steps, rays,
                    *[b.address for b in device[:5]], maps.address, runtime.status))
            runtime.sync(); built = time.monotonic(); rows = steps; launches = 1
            while rows > 1:
                check(self.lib.lumina_fullsr_chain_reduce(rows, rays, maps.address, spare.address, runtime.status))
                maps, spare = spare, maps; rows = (rows+1)//2; launches += 1
            check(self.lib.lumina_fullsr_chain_apply(rays, maps.address, device[-1].address, result.address, runtime.status))
            runtime.sync(); reduced = time.monotonic()
            values = runtime.download(result, rays, dtype=SCALED)
            return values, dict(steps=steps, outputs=rays, endpoint_maps=steps*rays,
                kernel_launches=launches+1, upload_allocate_seconds=uploaded-started,
                build_sync_seconds=built-uploaded, reduce_apply_sync_seconds=reduced-built,
                download_seconds=time.monotonic()-reduced, total_seconds=time.monotonic()-started,
                explicit_device_bytes=sum(b.size for b in buffers), accepted_physical_field=False)
        finally:
            for buffer in reversed(buffers): buffer.close()
