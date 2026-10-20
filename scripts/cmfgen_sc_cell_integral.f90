! 같은 SC source의 셀 내부 강도/공간 적분. 아직 whole 관측기에 연결하지 않았다.
! 실제 angular blue/물리 eta/chi/팽창 항의 적분을 대체하지 않는 기본 함수다.
module lumina_sc_cell_integral
use set_kind_module
use lumina_positive_sc,only:sc_bernstein_weights
use,intrinsic::ieee_arithmetic
implicit none
private
public::sc_cell_intensity_at,sc_cell_integrated_intensity,sc_physical_cell_budget
contains

subroutine checked_cell(iup,sup,control,target,tau)
real(ldp),intent(in)::iup,sup,control,target,tau
if(.not.all(ieee_is_finite([iup,sup,control,target,tau])))error stop 'nonfinite SC cell'
if(min(iup,sup,control,target,tau)<0)error stop 'negative SC cell'
end subroutine

real(ldp) function sc_cell_intensity_at(iup,sup,control,target,tau,t)result(value)
real(ldp),intent(in)::iup,sup,control,target,tau,t
real(ldp) u,a,b,s,w(3)
call checked_cell(iup,sup,control,target,tau)
if(.not.ieee_is_finite(t).or.t<0.or.t>tau)error stop 'SC position outside cell'
if(tau==0)then
 value=iup;return
end if
u=t/tau
! De Casteljau로 [0,t]의 source control point를 제한 없이 정확히 얻는다.
a=(1-u)*sup+u*control
b=(1-u)*control+u*target
s=(1-u)*a+u*b
call sc_bernstein_weights(t,w)
value=exp(-t)*iup+w(1)*sup+w(2)*a+w(3)*s
end function

real(ldp) function sc_cell_integrated_intensity(iup,sup,control,target,tau)result(value)
real(ldp),intent(in)::iup,sup,control,target,tau
real(ldp) w(3),area(3),coef(3,13),factorial
integer k
call checked_cell(iup,sup,control,target,tau)
call sc_bernstein_weights(tau,w)
if(tau<=.5_ldp)then
 ! int I dtau의 source 가중치는 tau/3-w다. 얇은 셀에서는 그 차감을 하지 않는다.
 factorial=1
 do k=1,13
  factorial=factorial*k
  coef(:,k)=[1._ldp/(k+3),2._ldp/((k+2)*(k+3)), &
   2._ldp/((k+1)*(k+2)*(k+3))]/factorial
 end do
 area=coef(:,13)
 do k=12,1,-1
  area=-tau*area+coef(:,k)
 end do
 area=tau*(tau*area)
else
 area=tau/3-w
end if
! 모든 가중치는 양수다. 큰 방출/흡수나 끝점 flux를 빼서 답을 만들지 않는다.
value=iup*sum(w)+sup*area(1)+control*area(2)+target*area(3)
end function

subroutine sc_physical_cell_budget(iup,sup,control,target,chi,eta,b,blue,doppler,length,alpha,order,budget)
! 반환: 물리 eta-chi*I, b*I, alpha*b*(Iblue-I), 기하 보간 결함,
!       SC source와 원래 eta/chi/blue 보간의 차이. 마지막 두 항은 물리 source가 아니다.
! p dp/2 측도와 주파수 가중치는 호출자가 적용한다.
real(ldp),intent(in)::iup,sup,control,target,chi(2),eta(2),b(2),blue(2),doppler(2),length,alpha
integer,intent(in)::order
real(ldp),intent(out)::budget(5)
real(ldp),parameter::x4(4)=[.06943184420297371_ldp,.33000947820757187_ldp,.6699905217924281_ldp,.9305681557970262_ldp]
real(ldp),parameter::w4(4)=[.17392742256872693_ldp,.32607257743127305_ldp,.32607257743127305_ldp,.17392742256872693_ldp]
real(ldp),parameter::x8(8)=[.019855071751231884_ldp,.10166676129318664_ldp,.2372337950418355_ldp,.4082826787521751_ldp, &
 .5917173212478248_ldp,.7627662049581645_ldp,.8983332387068134_ldp,.9801449282487681_ldp]
real(ldp),parameter::w8(8)=[.05061426814518813_ldp,.11119051722668717_ldp,.15685332293894366_ldp,.18134189168918088_ldp, &
 .18134189168918088_ldp,.15685332293894366_ldp,.11119051722668717_ldp,.05061426814518813_ldp]
real(ldp)::x(8),weight(8),effective(2),tau,u,t,scp,intensity,bb,previous,cc,ee,dd,measure,geometric
real(ldp)::linear,quadratic,amplitude,e0,ww(3),y,q,a,denominator,dsdt
integer k
call checked_cell(iup,sup,control,target,0._ldp)
if(.not.all(ieee_is_finite([chi,eta,b,blue,doppler,length,alpha])))error stop 'nonfinite physical cell'
if(minval(chi)<=0.or.minval(eta)<0.or.minval(b)<0.or.minval(blue)<0.or.minval(doppler)<=0.or.length<=0.or.alpha<0) &
 error stop 'unsupported physical cell'
effective=chi+(alpha+3)*b;tau=.5_ldp*length*sum(effective)
if(.not.ieee_is_finite(tau).or.tau<=0)error stop 'invalid effective depth'
if(order==4)then
 x(1:4)=x4;weight(1:4)=w4
else if(order==8)then
 x=x8;weight=w8
else
 error stop 'physical cell quadrature order'
end if
budget=0
linear=0;quadratic=0;amplitude=0
if(tau>1)then
 linear=2*(control-sup)/tau
 quadratic=((sup-control)+(target-control))/tau/tau
 amplitude=(iup-sup)+linear-2*quadratic
end if
do k=1,order
 u=x(k);t=length*u*(effective(1)+.5_ldp*u*(effective(2)-effective(1)))
 q=t/tau
 scp=(1-q)*((1-q)*sup+q*control)+q*((1-q)*control+q*target)
 if(tau<=1)then
  intensity=sc_cell_intensity_at(iup,sup,control,target,tau,t)
 else
  ! 지수 경계층은 아래에서 별도 적분한다. 균일 공간 Gauss로 놓치지 않는다.
  intensity=scp-linear-2*quadratic*t+2*quadratic
 end if
 cc=(1-u)*chi(1)+u*chi(2);ee=(1-u)*eta(1)+u*eta(2)
 bb=(1-u)*b(1)+u*b(2);previous=(1-u)*blue(1)+u*blue(2)
 dd=(1-u)*doppler(1)+u*doppler(2)
 measure=1/dd**3
 geometric=-3*(doppler(2)-doppler(1))/length/dd**4-3*bb*measure
 budget(1)=budget(1)+weight(k)*length*measure*(ee-cc*intensity)
 budget(2)=budget(2)+weight(k)*length*measure*bb*intensity
 budget(3)=budget(3)+weight(k)*length*measure*alpha*bb*(previous-intensity)
 budget(4)=budget(4)+weight(k)*length*geometric*intensity
 budget(5)=budget(5)+weight(k)*length*measure*((cc+(alpha+3)*bb)*scp-ee-alpha*bb*previous)
end do
if(tau<=1)return
! y=1-exp(-t) 변환은 경계 지수의 전체 지지를 적분한다. 임의 tau 절단은 없다.
call sc_bernstein_weights(tau,ww);e0=sum(ww)
a=effective(1)/sum(effective)
do k=1,order
 y=e0*x(k);t=-log(1-y);q=t/tau
 denominator=a+sqrt(a*a+(effective(2)-effective(1))/sum(effective)*q)
 u=q/denominator
 cc=(1-u)*chi(1)+u*chi(2);bb=(1-u)*b(1)+u*b(2)
 dd=(1-u)*doppler(1)+u*doppler(2);measure=1/dd**3
 dsdt=1/((1-u)*effective(1)+u*effective(2))
 geometric=-3*(doppler(2)-doppler(1))/length/dd**4-3*bb*measure
 budget(1)=budget(1)-weight(k)*e0*dsdt*measure*cc*amplitude
 budget(2)=budget(2)+weight(k)*e0*dsdt*measure*bb*amplitude
 budget(3)=budget(3)-weight(k)*e0*dsdt*measure*alpha*bb*amplitude
 budget(4)=budget(4)+weight(k)*e0*dsdt*geometric*amplitude
end do
end subroutine
end module
