"""외부대기 폐쇄관계의 원소스 대조. 수송/결합 완료 시험은 별도다."""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import lumina_outer_atmosphere as O

PTR = ct.POINTER(ct.c_double)


@pytest.fixture(scope='module')
def oracle(tmp_path_factory):
    path = Path('/gpfs/kjhan/cmfgen_runs/lumina_mom_capture_build_20261019-RbUeST/'
                'formal_primary_v11/cmf_formal_rel_v4.f')
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == O.REFERENCE_SOURCE_SHA256
    source = raw.decode()
    start = source.index('\t  DO I=1,ND\n\t    R_EXT(ND_ADD+I)=R(I)')
    end = source.index('\n', source.index('LOG_R_EXT(1:ND_EXT)=LOG(R_EXT(1:ND_EXT))', start))
    grid = source[start:end]
    start = source.index('\t  IF(ND_ADD .NE. 0)THEN', end)
    end = source.index('\n\t  END IF', start)+len('\n\t  END IF')
    indices = source[start:end]
    start = source.index('\tIF(ND_ADD .NE. 0)THEN', end)
    end = source.index('! This must be after', start)
    coefficients = source[start:end]
    # 원래 fixed-form 블록을 그대로 사용한다. 전체 native/수송 실행은 아니다.
    body = '''      subroutine outer_oracle(nd,nf,r,v,sigma,rext_fac,
     1 ichi,ies,ieta,r_ext,v_ext,sigma_ext,ochi,oeta,idx) bind(C)
      use iso_c_binding
      implicit none
      integer,parameter::ldp=c_double
      integer(c_int),value::nd,nf
      real(ldp),value::rext_fac
      real(ldp)::r(nd),v(nd),sigma(nd)
      real(ldp)::ichi(nd,nf),ies(nd,nf),ieta(nd,nf)
      real(ldp)::r_ext(nd+24),v_ext(nd+24),sigma_ext(nd+24)
      real(ldp)::ochi(nd+24,nf),oeta(nd+24,nf)
      integer(c_int)::idx
      real(ldp)::chi(nd),esec(nd),eta(nd),vdop_vec(nd)
      real(ldp)::chi_ext(nd+24),eta_ext(nd+24)
      real(ldp)::vdop_vec_ext(nd+24),log_r_ext(nd+24)
      real(ldp)::rmax,alpha,del_r_fac,beta,vinf,c_kms,t1,t2
      real(ldp)::esec_pow,speed_of_light
      integer::i,k,nd_add,nd_ext,idmin,idmax,luer
      logical::thick_ob
      nd_add=24
      nd_ext=nd+nd_add
      thick_ob=.true.
      vdop_vec=1
      luer=6
'''+grid+'\n'+indices+'''
      idx=idmax
      do k=1,nf
       chi=ichi(:,k)
       esec=ies(:,k)
       eta=ieta(:,k)
'''+coefficients+'''
       ochi(:,k)=chi_ext
       oeta(:,k)=eta_ext
      end do
      end subroutine
      double precision function speed_of_light()
      speed_of_light=2.99792458d10
      end function
'''
    out = tmp_path_factory.mktemp('outer-oracle')
    (out/'oracle.f').write_text(body)
    subprocess.run(['gfortran','-O2','-fPIC','-shared','-ffixed-line-length-none',
        'oracle.f','-o','oracle.so'], cwd=out, check=True, capture_output=True, timeout=45)
    dll = ct.CDLL(str(out/'oracle.so'))
    dll.outer_oracle.argtypes = [ct.c_int, ct.c_int]+[PTR]*3+[ct.c_double]+[PTR]*8+[
        ct.POINTER(ct.c_int)]
    dll.outer_oracle.restype = None
    return dll.outer_oracle


def sample():
    r = np.geomspace(1e15, 2e14, 24)
    v = .06*O.C*r/r[0]
    g = O.geometry(r, v, np.zeros_like(r))
    scale = (r[0]/r)[:, None]
    chi = np.tile(1e-14*scale**4, (1, 8))
    es = np.tile(2e-15*scale**3, (1, 8))
    eta = np.tile(1e-18*scale**6, (1, 8))
    j = np.full_like(chi, 1e-4)
    return g, chi, es, eta, j


@pytest.mark.parametrize('case',['normal','special','lower_powers','legacy_eta','explicit_radius','interior_signed','outer_gain','outer_zero','second_gain'])
def test_exact_native_geometry_and_extrapolation_blocks(oracle, case):
    g, chi, es, eta, j = sample()
    if case == 'special':
        chi[:, :2] = .99*es[:, :2]
    elif case == 'lower_powers':
        ratio = g.native_radius[0]/g.native_radius
        chi[:, :2] = 1e-14*ratio[:, None]
        es[:, :2] = 2e-15*ratio[:, None]
        eta[:, :2] = 1e-18*ratio[:, None]
    elif case == 'legacy_eta':
        eta[:, :2] = 1e-300; j[:, :2] = 0
    elif case == 'explicit_radius':
        g = O.geometry(g.native_radius, g.velocity[24:], g.sigma[24:], 1.6)
    elif case == 'interior_signed':
        chi[10] *= -1  # 외부 기준점이 아닌 내부의 실제 gain은 그대로 보존한다.
    elif case == 'outer_gain':
        chi[0,:2] = -1e-16
    elif case == 'outer_zero':
        chi[0,:2] = 0
    elif case == 'second_gain':
        chi[g.reference_indices[1],:2] = -1e-16
    expected = O.coefficients(g, chi, es, eta, j)
    n, nf = chi.shape
    # 원소스는 R:10^10cm, V:km/s, χ/η: native 길이^-1이다.
    arrays = [np.asfortranarray(a) for a in (g.native_radius/1e10, g.velocity[24:]/1e5,
        g.sigma[24:], chi*1e10, es*1e10, (eta+es*j)*1e10)]
    outputs = [np.empty(n+24) for _ in range(3)]+[
        np.empty((n+24,nf), order='F') for _ in range(2)]
    idx = ct.c_int()
    def ptr(a): return a.ctypes.data_as(PTR)
    oracle(n,nf,*[ptr(a) for a in arrays[:3]],1.6 if case=='explicit_radius' else 0.,
        *[ptr(a) for a in arrays[3:]],*[ptr(a) for a in outputs],ct.byref(idx))
    outputs[0] *= 1e10
    outputs[3] *= 1e-10
    outputs[4] *= 1e-10
    assert idx.value == g.reference_indices[1]+1
    for actual, reference in zip((g.radius,g.velocity/1e5,g.sigma,
                                 expected['chi'],expected['eta_total']),outputs):
        np.testing.assert_allclose(actual, reference, rtol=2e-13, atol=0)
    if case == 'legacy_eta':
        assert np.all(expected['eta_total'][:24,:2] == O.LEGACY_ETA_MINIMUM)


def test_total_eta_recomputed_from_both_own_node_unknowns():
    g, chi, es, eta, j = sample()
    eta[:] = 0
    j[:] = (g.native_radius[0]/g.native_radius[:,None])**3*1e-3
    before = O.coefficients(g,chi,es,eta,j)
    changed = j.copy(); changed[g.reference_indices[1]] *= 1.1
    after = O.coefficients(g,chi,es,eta,changed)
    assert np.all(after['eta_power'] > before['eta_power'])
    assert np.all(after['eta_total'][:24] < before['eta_total'][:24])
    changed = j.copy(); changed[0] *= 1.1
    other = O.coefficients(g,chi,es,eta,changed)
    assert np.all(other['eta_total'][:24] > before['eta_total'][:24])
    assert not other['scattering_added_locally_in_extension']


def test_zero_source_limit_and_red_upwind_are_not_hidden():
    g, chi, es, eta, j = sample()
    eta[0] = 0; j[0] = 0
    result = O.coefficients(g,chi,es,eta,j)
    assert np.all(result['eta_total'][:24] == O.LEGACY_ETA_MINIMUM)
    assert np.all(result['eta_total'][24] == 0)
    eta[:] = 0; j[:] = 0
    both_zero = O.coefficients(g,chi,es,eta,j)
    eta[:] = 1e-300
    limit = O.coefficients(g,chi,es,eta,j)
    np.testing.assert_array_equal(both_zero['eta_total'][:24],limit['eta_total'][:24])
    eta[:] = 0; eta[0] = 1e-18
    other = O.coefficients(g,chi,es,eta,j)
    np.testing.assert_array_equal(other['eta_power'],np.full(8,3.5))
    expected = eta[0]*(g.native_radius[0]/g.radius[:24,None])**3.5
    np.testing.assert_allclose(other['eta_total'][:24],expected,rtol=2e-13,atol=0)
    with pytest.raises(ValueError):
        O.segment_geometry(g,0.,g.radius[0],-g.radius[0])
    with pytest.raises(ValueError):
        O.velocity_at(g,np.nextafter(g.radius[0],np.inf))


def test_original_radius_witness_avoids_roundtrip_extrapolation_without_clipping():
    g,*_ = sample()
    p,r = 1.25e13,g.radius[0]
    z = -np.sqrt((r-p)*(r+p))
    d = O.doppler(g,p,z,radius=r)
    assert np.isfinite(d)
    with pytest.raises(ValueError,match='불일치'):
        O.doppler(g,p,np.nextafter(z,0),radius=r)
