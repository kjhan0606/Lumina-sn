! 명시된 s-선형 물질의 양의 formal 적분. 생산 SC에는 아직 연결하지 않았다.
module lumina_material_cell
use set_kind_module
use lumina_positive_sc,only:sc_bernstein_weights
use,intrinsic::ieee_arithmetic
implicit none
private
public::material_cell_transfer
contains

real(ldp) function attenuation_mass(x)result(value)
real(ldp),intent(in)::x
real(ldp) weights(3)
call sc_bernstein_weights(x,weights)
value=sum(weights)
end function

subroutine material_cell_transfer(iup,chi,eta,b,blue,length,alpha,rtol,iend,error,evaluations,status)
real(ldp),intent(in)::iup,chi(2),eta(2),b(2),blue(2),length,alpha,rtol
real(ldp),intent(out)::iend,error
integer,intent(out)::evaluations,status
real(ldp) effective(2),total,tau,ymax,a,up_fraction,source_bound,incident,seed,seed_error,tolerance
real(ldp) left,right,width,part,estimate,mass,tail_bound,position,tail_value
real(ldp) lag8,lag16,tail_correction,lag_scale
integer j
logical space_mode
real(ldp),parameter::lx8(8)=[.1702796323051009308_ldp,.9037017767993800188_ldp, &
 2.251086629866131172_ldp,4.266700170287658800_ldp,7.045905402393465522_ldp, &
 10.75851601018099579_ldp,15.74067864127800398_ldp,22.86313173688926526_ldp]
real(ldp),parameter::lw8(8)=[.3691885893416349540_ldp,.4187867808143446724_ldp, &
 .1757949866371725511_ldp,.03334349226121579440_ldp,.002794536235225683358_ldp, &
 .00009076508773358138871_ldp,.0000008485746716272570517_ldp,.000000001048001174871515341_ldp]
real(ldp),parameter::lx16(16)=[.08764941047892775561_ldp,.4626963289150803904_ldp, &
 1.141057774831226501_ldp,2.129283645098380529_ldp,3.437086633893206677_ldp, &
 5.078018614549767662_ldp,7.070338535048233730_ldp,9.438314336391938397_ldp, &
 12.21422336886615945_ldp,15.44152736878161747_ldp,19.18015685675313620_ldp, &
 23.51590569399190755_ldp,28.57872974288213896_ldp,34.58339870228662249_ldp, &
 41.94045264768833192_ldp,51.70116033954332124_ldp]
real(ldp),parameter::lw16(16)=[.2061517149578049046_ldp,.3310578549508783053_ldp, &
 .2657957776442144149_ldp,.1362969342963787356_ldp,.04732892869412563125_ldp, &
 .01129990008033959770_ldp,.001849070943526327131_ldp,.0002042719153082808931_ldp, &
 .00001484458687398150224_ldp,.0000006828319330871330663_ldp,.00000001881024841079722241_ldp, &
 .0000000002862350242973896881_ldp,.000000000002127079033224121367_ldp, &
 6.297967002517879738e-15_ldp,5.050473700035608235e-18_ldp,4.161462370372850995e-22_ldp]
real(ldp),parameter::x4(4)=[.069431844202973712388_ldp,.33000947820757186760_ldp, &
 .66999052179242813240_ldp,.93056815579702628761_ldp]
real(ldp),parameter::w4(4)=[.17392742256872692869_ldp,.32607257743127307131_ldp, &
 .32607257743127307131_ldp,.17392742256872692869_ldp]
real(ldp),parameter::x8(8)=[.019855071751231884158_ldp,.10166676129318663020_ldp, &
 .23723379504183550709_ldp,.40828267875217509753_ldp,.59171732124782490247_ldp, &
 .76276620495816449291_ldp,.89833323870681336980_ldp,.98014492824876811584_ldp]
real(ldp),parameter::w8(8)=[.050614268145188129576_ldp,.11119051722668723527_ldp, &
 .15685332293894364367_ldp,.18134189168918099148_ldp,.18134189168918099148_ldp, &
 .15685332293894364367_ldp,.11119051722668723527_ldp,.050614268145188129576_ldp]
evaluations=0;status=1;iend=0;error=huge(1._ldp)
if(.not.all(ieee_is_finite([iup,chi,eta,b,blue,length,alpha,rtol])))return
if(min(iup,minval(eta),minval(b),minval(blue),alpha)<0.or.minval(chi)<=0.or.length<=0.or.rtol<=0.or.rtol>=1)return
effective=chi+(alpha+3)*b;total=sum(effective);tau=.5_ldp*length*total
if(.not.all(ieee_is_finite([effective,total,tau])).or.tau<=0)return
a=effective(2)/total;up_fraction=effective(1)/total
! 두 linear/linear 비율의 최댓값은 끝점에서 얻는다. 약한 방출을 버리지 않는다.
source_bound=maxval(eta/effective)+maxval((alpha*b)/effective)*maxval(blue)
if(.not.ieee_is_finite(source_bound))return
incident=0
if(iup>0)incident=exp(log(iup)-tau)
if(source_bound==0)then
 iend=incident;error=0;status=0;return
end if
ymax=attenuation_mass(tau)
space_mode=tau<=50
if(tau>lx16(16))then
 ! 무한 적분에서는 tau 밖 source를 upstream 값으로 정의하고 그 tail을 정확히 뺀다.
 ! 모든 quadrature node는 실제 cell 안에 있다. 경계 이후의 영향 상한도 별도 검사한다.
 lag8=0;lag16=0
 do j=1,8
  lag8=lag8+lw8(j)*source_at(lx8(j))
 end do
 do j=1,16
  lag16=lag16+lw16(j)*source_at(lx16(j))
 end do
 lag8=lag8/sum(lw8);lag16=lag16/sum(lw16)
 tail_correction=eta(1)/effective(1)+(alpha*b(1)/effective(1))*blue(1)
 if(tail_correction>0)tail_correction=exp(log(tail_correction)-tau)
 tail_bound=exp(log(source_bound)-tau)
 lag_scale=max(incident,lag16)
 if(lag16>=tail_correction.and. &
    abs(lag16-lag8)+2*tail_bound<=rtol*lag_scale/4)then
  iend=incident+(lag16-tail_correction)
  error=abs(lag16-lag8)+2*tail_bound;status=0;return
 end if
end if
! 첫 optical-depth 구간으로 크기를 잡는다. 물리적 absolute floor는 없다.
if(space_mode)then
 call gauss(0._ldp,1._ldp,seed,seed_error)
else
 call gauss(0._ldp,min(tau,1._ldp),seed,seed_error)
end if
tolerance=rtol*max(incident,seed)
if(tolerance<=0.or..not.ieee_is_finite(tolerance))then
 status=4;return
end if
iend=incident;error=0;status=0;left=0;width=min(tau,1._ldp)
if(space_mode)then
 call adaptive(0._ldp,1._ldp,.75_ldp*tolerance,0,part,estimate)
 if(status/=0)return
 iend=iend+part;error=estimate;return
end if
do while(left<tau)
 mass=attenuation_mass(tau-left)
 tail_bound=exp(log(source_bound)-left+log(mass))
 if(tail_bound<=tolerance/8)then
  ! 남은 구간 전체를 양의 exponential 가중치로 포함한다. 절단/flux 맞춤이 아니다.
  position=left+min(1._ldp,.5_ldp*(tau-left))
  tail_value=source_at(position)
  if(tail_value>0)iend=iend+exp(log(tail_value)-left+log(mass))
  error=error+tail_bound;exit
 end if
 right=min(tau,left+width)
 if(right<=left)then
  status=4;return
 end if
 mass=exp(-left)*attenuation_mass(right-left)/ymax
 call adaptive(left,right,.75_ldp*tolerance*mass,0,part,estimate)
 if(status/=0)return
 iend=iend+part;error=error+estimate
 left=right;width=2*width
end do
if(.not.all(ieee_is_finite([iend,error])).or.iend<0)status=5
contains

real(ldp) function source_at(x)result(value)
real(ldp),intent(in)::x
real(ldp) q,distance
q=x/tau
! downstream에서 위쪽으로 거슬러 가는 optical depth의 안정적 역함수.
distance=q/(a+sqrt(a*a*(1-q)+up_fraction*up_fraction*q))
value=source_from_distance(distance)
end function

real(ldp) function source_from_distance(distance)result(value)
real(ldp),intent(in)::distance
real(ldp) cc,ee,bb,previous
! 양의 convex 합으로 endpoint 0과 매우 작은 내부 방출도 보존한다.
cc=distance*chi(1)+(1-distance)*chi(2);ee=distance*eta(1)+(1-distance)*eta(2)
bb=distance*b(1)+(1-distance)*b(2);previous=distance*blue(1)+(1-distance)*blue(2)
value=ee/(cc+(alpha+3)*bb)+(alpha*bb/(cc+(alpha+3)*bb))*previous
evaluations=evaluations+1
end function

real(ldp) function space_integrand(distance)result(value)
real(ldp),intent(in)::distance
real(ldp) s,opacity,remaining
s=source_from_distance(distance);value=0
opacity=distance*effective(1)+(1-distance)*effective(2)
remaining=length*distance*(effective(2)+.5_ldp*(effective(1)-effective(2))*distance)
if(s>0)value=exp(log(length)+log(opacity)+log(s)-remaining)
end function

subroutine gauss(lo,hi,value,estimate)
real(ldp),intent(in)::lo,hi
real(ldp),intent(out)::value,estimate
real(ldp) low,d,norm8,norm4,weight,mass
integer k
value=0;low=0;d=hi-lo;norm8=0;norm4=0
if(space_mode)then
 do k=1,8
  value=value+w8(k)*space_integrand(lo+d*x8(k))
 end do
 do k=1,4
  low=low+w4(k)*space_integrand(lo+d*x4(k))
 end do
 value=d*value;estimate=abs(value-d*low);return
end if
do k=1,8
 weight=w8(k)*exp(-d*x8(k));norm8=norm8+weight
 value=value+weight*source_at(lo+d*x8(k))
end do
do k=1,4
 weight=w4(k)*exp(-d*x4(k));norm4=norm4+weight
 low=low+weight*source_at(lo+d*x4(k))
end do
! 알려진 exponential mass로 양의 가중치를 정규화한다. constant source는 정확하다.
mass=attenuation_mass(d)
if(value>0)value=exp(log(value/norm8)-lo+log(mass))
if(low>0)low=exp(log(low/norm4)-lo+log(mass))
estimate=abs(value-low)
end subroutine

recursive subroutine adaptive(lo,hi,allowed,depth,value,estimate)
real(ldp),intent(in)::lo,hi,allowed
integer,intent(in)::depth
real(ldp),intent(out)::value,estimate
real(ldp) midpoint,vl,vr,el,er
call gauss(lo,hi,value,estimate)
if(estimate<=allowed)return
if(depth>=24.or.allowed<=0)then
 status=3;return
end if
midpoint=.5_ldp*(lo+hi)
if(midpoint<=lo.or.midpoint>=hi)then
 status=3;return
end if
call adaptive(lo,midpoint,.5_ldp*allowed,depth+1,vl,el)
if(status/=0)return
call adaptive(midpoint,hi,.5_ldp*allowed,depth+1,vr,er)
value=vl+vr;estimate=el+er
end subroutine
end subroutine
end module
