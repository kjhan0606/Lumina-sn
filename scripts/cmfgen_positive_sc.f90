! 양의 endpoint 사이의 연속 quadratic Bezier source 적분 후보.
! control point만 명시적으로 제한한다. intensity 또는 고정점 해를 clamp하지 않는다.
module lumina_positive_sc
use set_kind_module
implicit none
private
public::sc_bernstein_weights,sc_control,sc_bezier_source
integer sc_index
real(ldp),parameter::fac(0:12)=[1._ldp,1._ldp,2._ldp,6._ldp,24._ldp,120._ldp,720._ldp,5040._ldp, &
 40320._ldp,362880._ldp,3628800._ldp,39916800._ldp,479001600._ldp]
real(ldp),parameter::c0(0:12)=[(1._ldp/(fac(sc_index)*(sc_index+3)),sc_index=0,12)]
real(ldp),parameter::c1(0:12)=[(2._ldp/(fac(sc_index)*(sc_index+2)*(sc_index+3)),sc_index=0,12)]
real(ldp),parameter::c2(0:12)=[(2._ldp/(fac(sc_index)*(sc_index+1)*(sc_index+2)*(sc_index+3)),sc_index=0,12)]
contains
subroutine sc_bernstein_weights(tau,w)
real(ldp),intent(in)::tau
real(ldp),intent(out)::w(3)
real(ldp) inverse,ee
integer i
w=0
if(tau<=0)return
if(tau<=.5_ldp)then
 w=[c0(12),c1(12),c2(12)]
 do i=11,0,-1
  w=-tau*w+[c0(i),c1(i),c2(i)]
 end do
 w=tau*w
else
 inverse=1/tau;ee=exp(-tau)
 w(1)=2*inverse*inverse-ee*(1+2*inverse+2*inverse*inverse)
 w(2)=2*inverse-4*inverse*inverse+ee*(2*inverse+4*inverse*inverse)
 w(3)=1-2*inverse+2*inverse*inverse-2*ee*inverse*inverse
end if
end subroutine

real(ldp) function sc_control(sup,target,far,tau,neighbor)result(control)
real(ldp),intent(in)::sup,target,far,tau,neighbor
real(ldp) w,base,term,lo,hi,ratio,distance
if(neighbor<=0.or.tau<=0)then
 control=.5_ldp*sup+.5_ldp*target;return
end if
if(tau>=neighbor)then
 w=1/(1+neighbor/tau)
else
 ratio=tau/neighbor;w=ratio/(1+ratio)
end if
base=.5_ldp*(1-w)*sup+.5_ldp*(1+w)*target
term=.5_ldp*w*(far-target);lo=min(sup,target);hi=max(sup,target)
if(term==0)then
 control=min(hi,max(lo,base));return
end if
if(term>0)then
 distance=max(0._ldp,base-lo)
else
 distance=max(0._ldp,hi-base)
end if
if(tau>neighbor)then
 ratio=neighbor/tau
 if(abs(term)>=distance*ratio)then
  distance=huge(1._ldp)
 else
  distance=abs(term)/ratio
 end if
else
 distance=abs(term)*(tau/neighbor)
end if
if(term>0)then
 if(distance>=base-lo)then
  control=lo
 else
  control=base-distance
 end if
else
 if(distance>=hi-base)then
  control=hi
 else
  control=base+distance
 end if
end if
end function

real(ldp) function sc_bezier_source(sup,target,far,tau,neighbor)result(integral)
real(ldp),intent(in)::sup,target,far,tau,neighbor
real(ldp) w(3),control
call sc_bernstein_weights(tau,w)
control=sc_control(sup,target,far,tau,neighbor)
integral=w(1)*sup+w(2)*control+w(3)*target
end function
end module
