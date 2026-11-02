! 공통 J의 실제 적분 RE 행만 평가하는 private 실행파일 전용이다.
! 출력 뒤 반드시 종료하므로 물질/T 갱신이나 새 수송을 수행하지 않는다.
subroutine lumina_fixedj_heat_guard(fixedj,impurity,coherent,ehb,check_ehb,ddt,flux, &
                                  no_cont_velocity,formal_primary,line_switch,newmod,ndietot,gamma_mode)
implicit none
logical,intent(in)::fixedj,impurity,coherent,ehb,check_ehb,ddt,flux
logical,intent(in)::no_cont_velocity,formal_primary
logical,intent(in)::newmod
integer,intent(in)::ndietot
character(*),intent(in)::line_switch,gamma_mode
character(8)::setting
integer::ios
call get_environment_variable('LUMINA_FIXEDJ_HEAT_ONLY',setting,status=ios)
if(ios/=0.or.trim(setting)/='1') stop 94
! 고정 J의 적분 RE와 전자 EHB/비일관 산란/다른 수송 실행을 혼동하지 않는다.
if(.not.fixedj.or.impurity.or..not.coherent.or.ehb.or.check_ehb.or.ddt.or.flux.or. &
   no_cont_velocity.or.formal_primary.or.line_switch/='BLANK'.or.newmod.or. &
   ndietot/=0.or.gamma_mode=='RAD_TRANS') stop 94
end subroutine

subroutine lumina_fixedj_heat_finish(pops,r,v,sigma,t,ed,atoms,clump,heat,charge, &
                                   nt,nd,ncf,counter,weak_lines)
use iso_fortran_env,only:int32
use ieee_arithmetic,only:ieee_is_finite
use set_kind_module,only:ldp
implicit none
integer,intent(in)::nt,nd,ncf,counter,weak_lines
real(ldp),intent(in)::pops(nt,nd),r(nd),v(nd),sigma(nd),t(nd),ed(nd)
real(ldp),intent(in)::atoms(nd),clump(nd),heat(nd),charge(nd)
integer::lu,ios
if(storage_size(0.0_ldp)/=64.or.nt<3.or.nd<2.or.ncf<1.or.counter<1.or.weak_lines<0) stop 94
if(any(.not.ieee_is_finite(pops)).or.any(pops<0).or.any(.not.ieee_is_finite(r)).or. &
   any(.not.ieee_is_finite(v)).or.any(.not.ieee_is_finite(sigma)).or. &
   any(.not.ieee_is_finite(atoms)).or.any(.not.ieee_is_finite(clump)).or. &
   any(.not.ieee_is_finite(heat)).or.any(.not.ieee_is_finite(charge))) stop 94
if(any(r<=0).or.any(r(2:nd)>=r(1:nd-1)).or.any(t<=0).or.any(ed<=0).or. &
   any(atoms<=0).or.any(clump<=0).or.any(t/=pops(nt,:)).or.any(ed/=pops(nt-1,:))) stop 94
open(newunit=lu,file='NATIVE_HEAT_ROW.bin',status='new',access='stream', &
     form='unformatted',iostat=ios)
if(ios/=0) stop 94
! T=10^4 K, R=10^10 cm. STEQ_T는 native 단위 그대로이며 H/광도는 저장하지 않는다.
! counter는 다음 갱신 번호일 뿐이다. 갱신은 실행되지 않으며 상태는 파일로 직접 대조한다.
write(lu,iostat=ios)'NATIVEHEATROWV1 ', &
    int([1,64,nt,nd,ncf,counter,9,weak_lines],int32), &
    r,v,sigma,t,ed,atoms,clump,heat,charge,pops
close(lu)
if(ios/=0) stop 94
write(6,'(A)')'FIXED_J_INTEGRAL_RE_CAPTURED_BEFORE_POP_UPDATE_NOT_EHB_OR_CONVERGENCE'
stop 96
end subroutine
