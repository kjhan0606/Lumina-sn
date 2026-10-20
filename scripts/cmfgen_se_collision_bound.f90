! 실제 native 충돌 계수의 양의 subset만 사용한다. 물리 상태를 수정하지 않는다.
! raw 구간은 이미 평가된 local SE 값의 합만 감싼다. native 내부 산술 전체의
! 실수방정식 잔차, NC/charge/RE 또는 최종 수렴을 인증하는 모듈이 아니다.
module lumina_se_collision_bound
use iso_fortran_env, only: real64
use ieee_arithmetic
implicit none
private
public :: collision_lower, lower_add, raw_interval_add, residual_upper
contains

logical function supported_arithmetic()
logical::gradual
call ieee_get_underflow_mode(gradual)
supported_arithmetic=gradual.and.ieee_support_denormal(0.0_real64)
end function

function lower_add(a,b,ok) result(c)
real(real64),intent(in)::a,b
logical,intent(inout)::ok
real(real64)::c
c=0
if (.not.ok) return
if (.not.ieee_is_finite(a).or..not.ieee_is_finite(b).or.a<0.or.b<0) then
  ok=.false.; return
end if
if (a==0) then
  c=b; return
else if (b==0) then
  c=a; return
end if
c=a+b
if (.not.ieee_is_finite(c)) then
  ok=.false.; c=0; return
end if
c=ieee_next_after(c,0.0_real64)
end function

function lower_product(a,b,ok) result(c)
real(real64),intent(in)::a,b
logical,intent(inout)::ok
real(real64)::c
c=0
if (.not.ok) return
if (.not.ieee_is_finite(a).or..not.ieee_is_finite(b).or.a<0.or.b<0) then
  ok=.false.; return
end if
c=a*b
if (.not.ieee_is_finite(c)) then
  ok=.false.; c=0; return
end if
! 물리량에 하한을 적용하지 않고 양의 연산의 하측 경계만 반환한다.
if (c>0) c=ieee_next_after(c,0.0_real64)
end function

function rate_lower(density,coefficient,ne,ok) result(c)
real(real64),intent(in)::density,coefficient,ne
logical,intent(inout)::ok
real(real64)::c,t
t=lower_product(density,coefficient,ok)
c=lower_product(t,ne,ok)
end function

subroutine collision_lower(n,cnm,hn,hnst,ne,incoming,outgoing,ok)
integer,intent(in)::n
real(real64),intent(in)::cnm(n,n),hn(n),hnst(n),ne
real(real64),intent(out)::incoming(n+1),outgoing(n+1)
logical,intent(out)::ok
real(real64)::ionization,recombination,term
integer::j,l
incoming=0; outgoing=0
ok=n>0.and.ieee_is_finite(ne).and.ne>0.and.supported_arithmetic()
if (.not.ok) return
if (.not.all(ieee_is_finite(cnm)).or..not.all(ieee_is_finite(hn)).or. &
    .not.all(ieee_is_finite(hnst)).or.any(cnm<0).or.any(hn<0).or.any(hnst<0)) then
  ok=.false.; return
end if
do j=1,n
  do l=1,n
    if (l==j) cycle
    term=rate_lower(hn(l),cnm(l,j),ne,ok)
    incoming(j)=lower_add(incoming(j),term,ok)
    term=rate_lower(hn(j),cnm(j,l),ne,ok)
    outgoing(j)=lower_add(outgoing(j),term,ok)
  end do
  ionization=rate_lower(hn(j),cnm(j,j),ne,ok)
  recombination=rate_lower(hnst(j),cnm(j,j),ne,ok)
  incoming(j)=lower_add(incoming(j),recombination,ok)
  outgoing(j)=lower_add(outgoing(j),ionization,ok)
  incoming(n+1)=lower_add(incoming(n+1),ionization,ok)
  outgoing(n+1)=lower_add(outgoing(n+1),recombination,ok)
end do
end subroutine

subroutine raw_interval_add(lo,hi,term,ok)
real(real64),intent(inout)::lo,hi
real(real64),intent(in)::term
logical,intent(inout)::ok
real(real64)::a,b
if (.not.ok) return
if (.not.supported_arithmetic()) then
  ok=.false.;return
end if
if (.not.ieee_is_finite(lo).or..not.ieee_is_finite(hi).or. &
    .not.ieee_is_finite(term).or.lo>hi) then
  ok=.false.; return
end if
if (term==0) return
a=lo+term; b=hi+term
if (.not.ieee_is_finite(a).or..not.ieee_is_finite(b)) then
  ok=.false.; return
end if
lo=ieee_next_after(a,ieee_value(a,ieee_negative_inf))
hi=ieee_next_after(b,ieee_value(b,ieee_positive_inf))
if (.not.ieee_is_finite(lo).or..not.ieee_is_finite(hi)) ok=.false.
end subroutine

subroutine residual_upper(lo,hi,inc,out,covered,valid,tolerance,upper,proven)
real(real64),intent(in)::lo,hi,inc,out,tolerance
logical,intent(in)::covered,valid
real(real64),intent(out)::upper
logical,intent(out)::proven
real(real64)::denominator,numerator
proven=.false.; upper=ieee_value(0.0_real64,ieee_quiet_nan)
if (.not.covered.or..not.valid) return
if (.not.supported_arithmetic()) return
if (.not.ieee_is_finite(lo).or..not.ieee_is_finite(hi).or. &
    .not.ieee_is_finite(inc).or..not.ieee_is_finite(out).or. &
    .not.ieee_is_finite(tolerance)) return
if (lo>hi.or.inc<0.or.out<0.or.tolerance<=0) return
denominator=max(inc,out)
if (denominator<=0) return
numerator=max(abs(lo),abs(hi))
if (numerator==0) then
  upper=0
else
  upper=numerator/denominator
  if (.not.ieee_is_finite(upper)) return
  ! 거대한 유한값의 다음 표현값도 상향 경계로 포함한다.
  upper=ieee_next_after(upper,ieee_value(upper,ieee_positive_inf))
end if
proven=ieee_is_finite(upper).and.upper<=tolerance
end subroutine
end module
