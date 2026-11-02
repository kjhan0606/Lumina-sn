! 전용 fixed-J 열행 평가의 실제 입력 복원만 보존한다. SE/T/J는 변경하지 않는다.
subroutine lumina_fixedj_full_capture(pops,r,nt,nd,counter)
use iso_fortran_env,only:int32
use ieee_arithmetic,only:ieee_is_finite
use set_kind_module,only:ldp
use mod_cmfgen,only:atm,num_ions,at_no,species_lnk,ion_id
implicit none
integer,intent(in)::nt,nd,counter
real(ldp),intent(in)::pops(nt,nd),r(nd)
integer::id,i,nfull,npres,lu(3),meta,state,phase,ios
character(32)::names(3)
if(storage_size(0.0_ldp)/=64.or.nt<3.or.nd<2.or.counter<1)stop 94
nfull=0;npres=0
do id=1,num_ions
 if(.not.atm(id)%xzv_pres)cycle
 if(atm(id)%nxzv_f<1.or.atm(id)%nxzv<1)stop 94
 if(.not.allocated(atm(id)%xzv_f).or..not.allocated(atm(id)%xzvlte_f_on_s).or. &
    .not.allocated(atm(id)%w_xzv_f).or..not.allocated(atm(id)%f_to_s_xzv).or. &
    .not.allocated(atm(id)%edgexzv_f).or..not.allocated(atm(id)%gxzv_f))stop 94
 if(any(shape(atm(id)%xzv_f)/=[atm(id)%nxzv_f,nd]).or. &
    any(shape(atm(id)%xzvlte_f_on_s)/=[atm(id)%nxzv_f,nd]).or. &
    any(shape(atm(id)%w_xzv_f)/=[atm(id)%nxzv_f,nd]).or. &
    size(atm(id)%f_to_s_xzv)/=atm(id)%nxzv_f.or. &
    size(atm(id)%edgexzv_f)/=atm(id)%nxzv_f.or. &
    size(atm(id)%gxzv_f)/=atm(id)%nxzv_f)stop 94
 if(any(.not.ieee_is_finite(atm(id)%xzv_f)).or.any(atm(id)%xzv_f<0).or. &
    any(.not.ieee_is_finite(atm(id)%xzvlte_f_on_s)).or.any(atm(id)%xzvlte_f_on_s<0).or. &
    any(.not.ieee_is_finite(atm(id)%w_xzv_f)).or.any(atm(id)%w_xzv_f<0).or. &
    any(.not.ieee_is_finite(atm(id)%edgexzv_f)).or. &
    any(.not.ieee_is_finite(atm(id)%gxzv_f)).or.any(atm(id)%gxzv_f<=0))stop 94
 if(any(atm(id)%f_to_s_xzv<1).or.any(atm(id)%f_to_s_xzv>atm(id)%nxzv))stop 94
 if(atm(id)%eqxzv<1.or.atm(id)%eqxzv+atm(id)%nxzv-1>nt-2)stop 94
 nfull=nfull+atm(id)%nxzv_f;npres=npres+1
end do
if(nfull<1)stop 94
names=[character(32)::'NATIVE_FULL_POP.bin','NATIVE_LTE_FRACTION.bin','NATIVE_OCCUPATION.bin']
do i=1,3
 open(newunit=lu(i),file=trim(names(i)),status='new',access='stream',form='unformatted',iostat=ios)
 if(ios/=0)stop 94
 ! 기존 C full matrix 형식을 재사용한다. 각 파일의 물리량은 이름과 index로 구별한다.
 write(lu(i),iostat=ios)'LMATFL01',int([nfull,nd],int32)
 if(ios/=0)stop 94
end do
open(newunit=meta,file='NATIVE_FULL_INDEX.csv',status='new',iostat=ios)
if(ios/=0)stop 94
write(meta,'(A)')'native_ion,Z,full_level,superlevel,eq_sl,ion_label,edge_native,g'
do id=1,num_ions
 if(.not.atm(id)%xzv_pres)cycle
 write(lu(1),iostat=ios)transpose(atm(id)%xzv_f)
 if(ios/=0)stop 94
 write(lu(2),iostat=ios)transpose(atm(id)%xzvlte_f_on_s)
 if(ios/=0)stop 94
 write(lu(3),iostat=ios)transpose(atm(id)%w_xzv_f)
 if(ios/=0)stop 94
 do i=1,atm(id)%nxzv_f
  write(meta,'(I0,4(",",I0),",",A,2(",",ES25.16E3))')id,int(at_no(species_lnk(id))), &
    i,atm(id)%f_to_s_xzv(i),atm(id)%eqxzv+atm(id)%f_to_s_xzv(i)-1,trim(ion_id(id)), &
    atm(id)%edgexzv_f(i),atm(id)%gxzv_f(i)
 end do
end do
do i=1,3
 close(lu(i))
end do
close(meta)
open(newunit=state,file='NATIVE_FULL_INPUT.bin',status='new',access='stream',form='unformatted',iostat=ios)
if(ios/=0)stop 94
write(state,iostat=ios)'NATIVEHEATINPV1 ',int([1,64,nt,nd],int32),r,pops
close(state)
if(ios/=0)stop 94
open(newunit=phase,file='NATIVE_FULL_PHASE.txt',status='new',iostat=ios)
if(ios/=0)stop 94
write(phase,'(A)')'AFTER_CURRENT_STATE_EVAL_LTE_BEFORE_RATE_AND_HEAT_EVALUATION'
write(phase,*)counter,nt,nd,nfull,npres
close(phase)
end subroutine
