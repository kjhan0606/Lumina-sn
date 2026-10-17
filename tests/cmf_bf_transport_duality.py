"""보존된 동일 J에서 BF 수송과 SE의 광자 수 표현 차이를 실제 율행렬에 대조.

시험 전용: 생산 물리율/복사장/저장점은 바꾸지 않는다. 수송 계수에 맞춘
대체율도 정확한 연속 물리의 해라고 간주하지 않는다.
"""
import csv
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

base = Path('/gpfs/kjhan/lumina/cmfgen_map/trace13_20261010/results')
origin = base/'continuum-origin-20261017.3YwNB2'
prior = base/'toy-material-425642'
response = base/'saved-field-response-20261017.vNeZQn'
bridge = base/'prrr-comparison-20261017.1y1nJD'
output = Path(sys.argv[1])
assert not output.exists()
identities = {}


def read(path):
    data = path.read_bytes()
    identities[str(path)] = hashlib.sha256(data).hexdigest()
    return data


def stationary(a):
    m = np.array(a, dtype=np.longdouble, copy=True)
    np.fill_diagonal(m, 0)
    assert np.all(np.isfinite(m)) and np.all(m >= 0)
    for q in range(len(m)-1, 0, -1):
        denominator = m[:q, q].sum()
        assert denominator > 0
        m[q, :q] /= denominator
        m[:q, :q] += m[:q, q, None]*m[q, None, :q]
    x = np.ones(len(m), dtype=np.longdouble)
    for q in range(1, len(m)):
        x[q] = m[q, :q]@x[:q]
    x /= x.sum()
    m = np.array(a, dtype=np.longdouble, copy=True)
    np.fill_diagonal(m, 0)
    np.fill_diagonal(m, -m.sum(axis=0))
    residual = float(np.max(abs(m@x)/(abs(m)@x)))
    assert residual < 1e-10
    return x, residual


fine = read(origin/'fine_shell39.bin')
nf = struct.unpack_from('<Q', fine, 8)[0]
nu, chi, es, eta, J = np.frombuffer(fine, '<f8', 5*nf, 16).reshape(5, nf)
raw = read(origin/'origin_shell39.bin')
nb, nf2 = struct.unpack_from('<QQ', raw, 8)
blo, dl, T, ne, z2ni = struct.unpack_from('<5d', raw, 24)
assert nf2 == nf
coarse_actual = np.frombuffer(raw, '<f8', 14*nb, 64).reshape(14, nb)
pack = read(prior/'input/bf_channels.bin')
assert identities[str(prior/'input/bf_channels.bin')] == '656329de676b540a08beb1e6652e839785f799428a40b31fd2774ce44d8af0cf'
magic, version, nl, nr, nb2, blo2, bhi = struct.unpack_from('<IIiiidd', pack)
assert magic == 0x43424643 and version == 2 and nb2 == nb and blo2 == blo
pos = 36
offset = np.frombuffer(pack, '<i4', nl+1, pos); pos += 4*(nl+1)
upper = np.frombuffer(pack, '<i4', nr, pos); pos += 4*nr
threshold = np.frombuffer(pack, '<f8', nr, pos); pos += 8*nr
sigma = np.frombuffer(pack, '<f8', nr*nb, pos).reshape(nr, nb); pos += 8*nr*nb
terminal_g = np.frombuffer(pack, '<f8', nr, pos)
atom = list(csv.DictReader(read(prior/'input/levels.csv').decode().splitlines()))
edges = np.ascontiguousarray(blo*np.exp(np.arange(nb+1)*dl)); edges[-1] = bhi
centers = np.sqrt(edges[:-1]*edges[1:])
fdl = math.log(nu[1]/nu[0])
fine_edges = np.r_[nu*np.exp(-.5*fdl), nu[-1]*np.exp(.5*fdl)]
union = np.unique(np.r_[edges, fine_edges[(fine_edges > blo) & (fine_edges < bhi)]])
left, right = union[:-1], union[1:]
mid = .5*(left+right)
cb = np.searchsorted(edges, mid, side='right')-1
fb = np.searchsorted(fine_edges, mid, side='right')-1
Jcoarse = np.ascontiguousarray(np.bincount(cb, weights=J[fb]*(right-left), minlength=nb)/np.diff(edges))

# 실제 producer의 log-nu 중심 보간의 전치: 각 fine bin은 수송에서 일정하다.
ix = np.searchsorted(centers, nu, side='right')-1
ix = np.clip(ix, 0, nb-2)
w = np.clip(np.log(nu/centers[ix])/np.log(centers[ix+1]/centers[ix]), 0, 1)
h, k, c, me = 6.62607015e-27, 1.380649e-16, 2.99792458e10, 9.1093837015e-28
valid = nu >= blo
photon = 4*math.pi/h*fdl*valid


def transpose(v):
    return np.bincount(ix, weights=(1-w)*v, minlength=nb)+np.bincount(ix+1, weights=w*v, minlength=nb)


WJ, W1 = transpose(photon*J), transpose(photon)
library = bridge/'comparison.so'
read(library)
assert identities[str(library)] == (bridge/'binary.sha256').read_text().split()[0]
lib = ct.CDLL(str(library)); ptr = ct.POINTER(ct.c_double)
lib.bf_vector.argtypes = [ct.c_int, ptr, ct.c_double, ptr]+[ct.c_double]*4+[ptr]*3
lib.bf_vector.restype = ct.c_int


def p(v):
    assert v.dtype == np.float64 and v.flags.c_contiguous
    return v.ctypes.data_as(ptr)


# 원래 물질로 채널별 계수를 조립해 기존 C 캡처와 재현 대조한다.
checkpoint = np.memmap(prior/'iterate_input.bin', dtype='u1', mode='r')
digest = hashlib.sha256()
for begin in range(0, len(checkpoint), 1024*1024):
    digest.update(checkpoint[begin:begin+1024*1024])
identities[str(prior/'iterate_input.bin')] = digest.hexdigest()
ns, ni, nlev, nelem, nf3, gen, iteration = struct.unpack_from('<7Q', checkpoint, 72)
assert ns == 113 and nf3 == nf and nlev == nl and iteration == 0
mapping = np.frombuffer(checkpoint, '<i4', nl+2*ni, 152)
start = 152+4*(nl+2*ni)
saved_ne = struct.unpack_from('<d', checkpoint, start+8*((4+nelem)*ns+39))[0]
assert saved_ne == ne
checkpoint_J = np.frombuffer(checkpoint, '<f8', ns*nf, start+8*ns*(5+nelem+ni+nl)).reshape(ns, nf)[39]
assert np.array_equal(checkpoint_J, J)
ion = np.frombuffer(checkpoint, '<f8', ni*ns, start+8*ns*(5+nelem)).reshape(ni, ns)[:, 39]
pop = np.frombuffer(checkpoint, '<f8', nl*ns, start+8*ns*(5+nelem+ni)).reshape(nl, ns)[:, 39]
global_pop = np.empty(nl); global_pop[mapping[:nl]] = pop
ionmap = {(int(mapping[nl+i]), int(mapping[nl+ni+i])): i for i in range(ni)}
gx, gw = np.polynomial.legendre.leggauss(8); gw *= .5
out = dict(scope='동일 수정본 첫 J/T/ne, 모든 Si/S BF율의 수송 표현 대조; 새 수송/공동해 아님',
           shell=39, T=T, ne=ne, ions={}, checkpoint_identity=bytes(checkpoint[8:72]).decode())
feedback = '--feedback' in sys.argv[2:]
nonlinear = '--nonlinear' in sys.argv[2:]
band_response = '--band-response' in sys.argv[2:]
assert not (nonlinear or band_response) or feedback
bf_response_channels = []
blocks = []
for z, component in ((14, 0), (16, 1)):
    matrix = read(response/f'matrix_Z{z}.bin')
    n = struct.unpack_from('<Q', matrix, 8)[0]
    total = struct.unpack_from('<d', matrix, 16)[0]
    A = np.frombuffer(matrix, '<f8', n*n, 24).reshape(n, n, order='F')
    saved = np.frombuffer(matrix, '<f8', n, 24+8*n*n)/total
    meta = {int(r['global_level']): r for r in csv.DictReader(read(response/f'levels_Z{z}.csv').decode().splitlines())}
    stages = np.full(n, -1)
    for row in meta.values():
        stages[int(row['matrix_index'])] = int(row['charge'])
    assert np.count_nonzero(stages < 0) == 1 and stages[-1] < 0
    stages[-1] = stages.max()+1
    ratio = lambda x: float(x[stages == 2].sum()/x[stages == 1].sum())
    x, residual = stationary(A)
    assert abs(ratio(x)/ratio(saved)-1) < 1e-10
    delta = np.zeros((n, n), dtype=np.longdouble)
    Dchi = np.zeros((nb, n)); Deta = np.zeros((nb, n)); Rbf = np.zeros((n, nb))
    x0 = np.zeros(n)
    for gl, row in meta.items():
        x0[int(row['matrix_index'])] += global_pop[gl]/total
    x0[-1] = ion[ionmap[z, int(stages[-1])]]/total
    assert abs(x0.sum()-1) < 1e-10
    reconstructed = np.zeros((2, nb)); flows = np.zeros((2, 2), dtype=np.longdouble)
    native_flows = np.zeros(2, dtype=np.longdouble)
    channel_changes = []; count = 0
    for gl, row in meta.items():
        l = int(row['matrix_index']); wl = float(row['within'])
        for r in range(offset[gl], offset[gl+1]):
            gu = int(upper[r])
            u = int(meta[gu]['matrix_index']) if gu >= 0 else n-1
            wu = float(meta[gu]['within']) if gu >= 0 else 1.
            g_upper = float(atom[gu]['g']) if gu >= 0 else terminal_g[r]
            su = np.ascontiguousarray(sigma[r]); up, down = np.empty(nb), np.empty(nb)
            assert lib.bf_vector(nb, p(edges), threshold[r], p(su), T, ne,
                                 float(atom[gl]['g']), g_upper, p(Jcoarse), p(up), p(down)) == 0
            old = np.array([up.sum(), down.sum()])
            active = np.flatnonzero(su)
            lo = np.maximum(edges[active], threshold[r]); hi = edges[active+1]
            assert np.all(hi > lo)
            fq = .5*(hi+lo)[:, None]+.5*(hi-lo)[:, None]*gx
            inv = ne*(h*h/(2*math.pi*me*k*T))**1.5*float(atom[gl]['g'])/(2*g_upper)*np.exp(-h*(fq-threshold[r])/(k*T))
            stim = su[active]*np.sum(inv*gw, axis=1)
            emiss = su[active]*np.sum(inv*(2*h*fq**3/c**2)*gw, axis=1)
            new = np.array([su@WJ, stim@WJ[active]+emiss@W1[active]])
            Dchi[active, l] += su[active]*wl*total
            Dchi[active, u] -= stim*wu*total
            Deta[active, u] += emiss*wu*total
            photo_kernel = 4*math.pi/h*su[active]*np.diff(edges)[active]*np.sum(gw/fq, axis=1)*wl
            stim_kernel = 4*math.pi/h*su[active]*np.diff(edges)[active]*np.sum(gw*inv/fq, axis=1)*wu
            if nonlinear or band_response:
                bf_response_channels.append((z, l, u, active.copy(), photo_kernel.copy(), stim_kernel.copy()))
            net_kernel = photo_kernel*x0[l]-stim_kernel*x0[u]
            Rbf[u, active] += net_kernel
            Rbf[l, active] -= net_kernel
            delta[u, l] += (new[0]-old[0])*wl
            delta[l, u] += (new[1]-old[1])*wu
            nu_pop = global_pop[gu] if gu >= 0 else ion[ionmap[z, int(atom[gl]['ion_number'])+1]]
            native_flows += new*np.array([global_pop[gl], nu_pop])
            reconstructed[0, active] += global_pop[gl]*su[active]-nu_pop*stim
            reconstructed[1, active] += nu_pop*emiss
            if stages[l] == 1:
                assert stages[u] == 2
                population = np.array([float(x[l])*wl, float(x[u])*wu])*total
                flows[0] += old*population; flows[1] += new*population
                channel_changes.append(dict(lower=gl, upper=gu, configuration=atom[gl]['configuration'],
                                            net_change=float((new-old)@ (population*np.array([1, -1])))))
            count += 1
    actual = coarse_actual[2*component:2*component+2]
    errors = []
    for i in range(2):
        used = actual[i] != 0
        errors.append(float(np.max(abs(reconstructed[i, used]/actual[i, used]-1))))
    assert max(errors) < 1e-10, errors
    interpolated_chi = np.interp(np.log(nu), np.log(centers), actual[0])*valid
    interpolated_eta = np.interp(np.log(nu), np.log(centers), actual[1])*valid
    direct_net = np.sum(photon*(interpolated_chi*J-interpolated_eta), dtype=np.longdouble)
    dual_error = float(abs(native_flows[0]-native_flows[1]-direct_net)/native_flows.sum())
    assert dual_error < 1e-12
    samples = []
    for amplitude in (-1., -.5, .5, 1.):
        candidate, res = stationary(A+amplitude*delta)
        samples.append(dict(amplitude=amplitude, III_II=ratio(candidate),
                            relative_change=ratio(candidate)/ratio(x)-1, residual=res))
    out['ions'][z] = dict(channels=count, baseline_III_II=ratio(x), baseline_residual=residual,
                         source_reconstruction_relative_error=errors,
                         native_transport_photon_net=float(direct_net),
                         native_rate_transport_identity_error_over_gross=dual_error,
                         II_III_gross_flow_SE_and_transport=flows.astype(float).tolist(),
                         II_III_net_flow_SE_and_transport=(flows[:, 0]-flows[:, 1]).astype(float).tolist(),
                         replacements=samples,
                         top_net_flow_changes=sorted(channel_changes, key=lambda r: -abs(r['net_change']))[:8])
    blocks.append(dict(z=z, A=A, delta=delta, x0=x0, stages=stages, total=total,
                       Dchi=Dchi, Deta=Deta, Rbf=Rbf, meta=meta))

if feedback:
    from scipy import sparse
    A = sparse.block_diag([b['A'] for b in blocks]).toarray()
    perturbation = sparse.block_diag([b['delta'] for b in blocks]).toarray()
    np.fill_diagonal(perturbation, 0)
    np.fill_diagonal(perturbation, -perturbation.sum(axis=0))
    np.fill_diagonal(A, 0); np.fill_diagonal(A, -A.sum(axis=0))
    x0 = np.concatenate([b['x0'] for b in blocks]); N = len(x0)
    Dcb = np.hstack([b['Dchi'] for b in blocks]); Deb = np.hstack([b['Deta'] for b in blocks])
    Rbf = np.vstack([b['Rbf'] for b in blocks])
    groups = []; location = 0
    pr, pc, pv, cr, cc, cv, er, ec, ee = ([] for _ in range(9))
    lower, upper_sl, blu, bul = ([] for _ in range(4))
    line_error = 0.; line_count = 0
    for b in blocks:
        z, n, total = b['z'], len(b['x0']), b['total']
        groups.append((z, location, location+n, b['stages']))
        for row in csv.DictReader(read(response/f'lines_Z{z}.csv').decode().splitlines()):
            f, jl, al, bl, bu = (float(row[key]) for key in ('nu', 'Jbar', 'A_ul', 'B_lu', 'B_ul'))
            wl, wu = float(row['w_lower']), float(row['w_upper'])
            l, u = location+int(row['lower_matrix']), location+int(row['upper_matrix'])
            width = f*5e6/c
            idx = np.arange(np.searchsorted(nu, f-4*width), np.searchsorted(nu, f+4*width, side='right'))
            assert len(idx) > 0
            profile = np.exp(-((nu[idx]-f)/width)**2)/(math.sqrt(math.pi)*width)
            weights = profile*nu[idx]*fdl; weights /= weights.sum()
            measured = weights@J[idx]
            if jl > 0:
                line_error = max(line_error, abs(measured/jl-1))
            else:
                assert measured == 0
            pr.extend([line_count]*len(idx)); pc.extend(idx); pv.extend(weights)
            for col, value in ((l, h*f/(4*math.pi)*bl*wl*total),
                               (u, -h*f/(4*math.pi)*bu*wu*total)):
                cr.extend(idx); cc.extend([col]*len(idx)); cv.extend(value*profile)
            er.extend(idx); ec.extend([u]*len(idx)); ee.extend(h*f/(4*math.pi)*al*wu*total*profile)
            lower.append(l); upper_sl.append(u); blu.append(bl*wl); bul.append(bu*wu)
            line_count += 1
        location += n
    assert line_error < 1e-10, line_error
    P = sparse.csr_matrix((pv, (pr, pc)), shape=(line_count, nf))
    dc = sparse.csr_matrix((cv, (cr, cc)), shape=(nf, N))
    de = sparse.csr_matrix((ee, (er, ec)), shape=(nf, N))
    incidence = sparse.csr_matrix((np.r_[-np.ones(line_count), np.ones(line_count)],
                                 (np.r_[lower, upper_sl], np.tile(np.arange(line_count), 2))), shape=(N, line_count))
    R = incidence@sparse.diags(np.array(blu)*x0[lower]-np.array(bul)*x0[upper_sl])@P
    W = sparse.csr_matrix(((right-left)/np.diff(edges)[cb], (cb, fb)), shape=(nb, nf))
    I = sparse.csr_matrix((np.r_[(1-w)*valid, w*valid],
                           (np.tile(np.arange(nf), 2), np.r_[ix, ix+1])), shape=(nf, nb))
    # ne/T 고정에서도 Si/S의 전하제곱 합은 물질 변화에 반응한다.
    charge_weight = np.concatenate([b['stages']**2*b['total'] for b in blocks])
    ff = 3.6926e8/math.sqrt(T)*ne/centers**3*(-np.expm1(-h*centers/(k*T)))
    B = 2*h*centers**3/c**2*np.exp(-h*centers/(k*T))/(-np.expm1(-h*centers/(k*T)))
    Dcb += ff[:, None]*charge_weight
    Deb += (ff*B)[:, None]*charge_weight
    # 생산 common 경로는 BF 하한 아래에서 FF를 주파수별로 직접 계산한다.
    red_ff = 3.6926e8/math.sqrt(T)*ne/nu**3*(-np.expm1(-h*nu/(k*T)))*(~valid)
    red_B = 2*h*nu**3/c**2*np.exp(-h*nu/(k*T))/(-np.expm1(-h*nu/(k*T)))

    def constrained(matrix, rhs):
        m = np.array(matrix, dtype=np.longdouble, copy=True)
        v = np.array(rhs, dtype=np.longdouble, copy=True)
        for z, lo, hi, stages in groups:
            pivot = lo+int(np.argmax(x0[lo:hi]))
            m[pivot] = 0; m[pivot, lo:hi] = 1; v[pivot] = 0
        scale = np.max(abs(m), axis=1); m /= scale[:, None]; v /= scale
        original_m, original_v = m.copy(), v.copy()
        for col in range(N-1):
            pivot = col+int(np.argmax(abs(m[col:, col])))
            assert m[pivot, col] != 0
            m[[col, pivot]] = m[[pivot, col]]; v[[col, pivot]] = v[[pivot, col]]
            factor = m[col+1:, col]/m[col, col]
            m[col+1:, col+1:] -= factor[:, None]*m[col, col+1:]
            v[col+1:] -= factor*v[col]; m[col+1:, col] = 0
        result = np.empty(N, dtype=np.longdouble)
        for col in range(N-1, -1, -1):
            result[col] = (v[col]-m[col, col+1:]@result[col+1:])/m[col, col]
        residual = float(max(abs(original_m@result-original_v)))
        assert residual < 1e-12
        return result, residual

    def dln(dx):
        return {str(z): float(dx[lo:hi][stages == 2].sum()/x0[lo:hi][stages == 2].sum()
                             -dx[lo:hi][stages == 1].sum()/x0[lo:hi][stages == 1].sum())
                for z, lo, hi, stages in groups}

    forcing = perturbation@x0
    fixed, res = constrained(A, -forcing)
    local = dict(scope='초기 LTE 물질 주위의 기존 국소 Jacobian 근사, BF율 표현 차이의 1차 강제항; 새 공간 수송 아님',
                 line_J_reproduction_error=line_error, lines=line_count,
                 without_feedback_dln=dln(fixed), without_feedback_residual=res, cases=[])
    chi_abs = chi-es
    depth = chi_abs*min(864000000000., c*172800*fdl)
    error = abs((eta+es*J)/(chi*J)-1)
    for mindepth, maxerror in ((1e4, 1e-5), (1e2, 1e-3)):
        mask = (depth > mindepth) & (error < maxerror) & np.isfinite(error)
        inv = np.zeros(nf); inv[mask] = 1/chi_abs[mask]
        D = sparse.diags(inv)@(de-sparse.diags(J)@dc)
        L0, LJ = sparse.diags(inv), sparse.diags(inv*J)
        C = (R@D).toarray()+Rbf@(W@D).toarray()
        C += ((R@L0@I).toarray()+Rbf@(W@L0@I).toarray())@Deb
        C -= ((R@LJ@I).toarray()+Rbf@(W@LJ@I).toarray())@Dcb
        red_response = inv*red_ff*(red_B-J)
        C += np.outer(R@red_response+Rbf@(W@red_response), charge_weight)
        direction = x0*np.random.default_rng(20261017).normal(size=N)
        for z, lo, hi, stages in groups:
            direction[lo:hi] -= x0[lo:hi]*direction[lo:hi].sum()
        change_chi = dc@direction+I@(Dcb@direction)
        change_eta = de@direction+I@(Deb@direction)
        change_chi += red_ff*(charge_weight@direction)
        change_eta += red_ff*red_B*(charge_weight@direction)
        differences = []
        for step in (-1e-4, 1e-4):
            changed_J = np.zeros(nf)
            denominator = chi_abs+step*change_chi
            assert np.all(denominator[mask] > 0)
            changed_J[mask] = step*(change_eta[mask]-J[mask]*change_chi[mask])/denominator[mask]
            differences.append(R@changed_J+Rbf@(W@changed_J))
        numerical = (differences[1]-differences[0])/2e-4
        derivative_error = float(max(abs(numerical-C@direction))/max(abs(C@direction)))
        assert derivative_error < 1e-6
        extra, res = constrained(A+C, -forcing)
        original_step, res0 = constrained(A+C, -A@x0)
        local['cases'].append(dict(minimum_depth=mindepth, maximum_source_error=maxerror,
                                   bins=int(mask.sum()), incremental_dln=dln(extra),
                                   derivative_relative_error=derivative_error,
                                   baseline_dln=dln(original_step), residual=res,
                                   baseline_residual=res0,
                                   minimum_candidate_population=float(min(x0+original_step+extra))))
    out['local_feedback'] = local
    if nonlinear:
        from cmf_nonlinear_local_feedback import investigate
        out['nonlinear_local_feedback'] = investigate(globals())
    if band_response:
        from cmf_frozen_band_response import investigate
        out['frozen_band_response'] = investigate(globals())
out['sha256'] = identities
out['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
with output.open('x') as f:
    json.dump(out, f, indent=2, ensure_ascii=False, allow_nan=False)
print(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False))
