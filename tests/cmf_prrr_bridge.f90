! 원본 PRRR_SL_V6를 수정하지 않고 호출한다. BF만 검사하며 FF 호출은 거부한다.
module set_kind_module
  use iso_c_binding
  implicit none
  integer, parameter :: ldp=c_double
end module

subroutine prrr_bridge(n,w,hn,hnst,di,ld,jp,jr,pr,rr) bind(C)
  use iso_c_binding
  implicit none
  integer(c_int),value :: n
  real(c_double) :: w(n),hn(n),hnst(n),di(2,n),ld(2,n),jp(n),jr(n),pr(n),rr(n)
  real(c_double) :: wc(n),bf(n),ff(n),ed(n),t(n),zero(n)
  wc=0; bf=0; ff=0; ed=1; t=1; zero=0
  call prrr_sl_v6(pr,rr,bf,ff,w,wc,hn,hnst,1,1.0_c_double, &
       di,ld,2,1,2,ed,t,jr,jp,zero,zero,zero,1.0_c_double,1.0_c_double,.true.,n)
end subroutine

function plancks_constant() result(h)
  use set_kind_module
  implicit none
  real(ldp) :: h
  h=6.62607015e-27_ldp
end function

subroutine gff_vec(g,nu,t,z,n)
  use set_kind_module
  implicit none
  integer :: n
  real(ldp) :: g(n),nu,t(n),z
  error stop 'FF is outside this comparison'
end subroutine

subroutine do_h0_ff_cool(ff,g,ed,t,b,j,nu,n)
  use set_kind_module
  implicit none
  integer :: n
  real(ldp) :: ff(n),g(n),ed(n),t(n),b(n),j(n),nu
  error stop 'H0 FF is outside this comparison'
end subroutine
