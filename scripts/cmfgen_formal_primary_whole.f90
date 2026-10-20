! 같은 공통장의 주파수 coverage, 실제 소비자 신원, 최초 실패만 보존한다.
module lumina_formal_primary_whole
use lumina_formal_primary
use lumina_formal_diag
implicit none
integer,save::pw_count=0,pw_expected=0,pw_csv=0,pw_field=0,pw_n=0,pw_calls=0
logical,save::pw_open=.false.,pw_input_ready=.false.
real(ldp),save::pw_previous_freq=huge(1._ldp)
type(lumina_formal_args),save::pw_input
contains
logical function pw_enabled()
character(len=16) value
call get_environment_variable('LUMINA_FORMAL_PRIMARY_WHOLE',value)
if(value/=' '.and.value/='1')stop 85
pw_enabled=value=='1'
if(pw_enabled.and..not.fp_enabled())stop 85
end function

subroutine pw_remember(__NAMES__)
integer nc,np,nd
__DECLARATIONS__
if(.not.pw_enabled())return
call formal_args_take(pw_input,.false.,__NAMES__)
pw_input_ready=.true.
end subroutine

subroutine pw_failure()
type(lumina_ray_state) state
type(lumina_formal_args) a
integer lu
if(.not.pw_enabled())return
if(pw_open)then
 flush(pw_csv);flush(pw_field)
end if
if(.not.pw_input_ready)return
if(pw_input%freq/=fp_freq)return
! 이는 실패 당시 raw 문맥이다. formal 진입 전 raw state로 표시하지 않는다.
open(newunit=lu,file='FORMAL_PRIMARY_CONTEXT.txt',status='new')
write(lu,'(A)')'ACTUAL_SOURCE_AND_BLUE_WITH_FAILURE_TIME_RAW_CONTEXT_NOT_ENTRY_STATE'
write(lu,*)lumina_outer,lumina_freq_index,fp_count,pw_count
close(lu)
if(.not.allocated(r_store).or..not.allocated(ray(1)%i_p))return
call formal_ray_take(state)
call formal_state_io('FORMAL_PRIMARY_CONTEXT.bin',pw_input,state,.false.)
a=pw_input;a%new_freq=.false.
call formal_rays_io('FORMAL_PRIMARY_BLUE.bin',a,.false.,.false.)
call formal_rays_io('FORMAL_PRIMARY_RAYS.bin',a,.false.,.true.)
call formal_geometry('FORMAL_PRIMARY_GEOMETRY.bin',a)
end subroutine

subroutine pw_frequency_end(index,expected,freq,weight,r,v,j,h,k,r2h,n)
integer,intent(in)::index,expected,n
real(ldp),intent(in)::freq,weight,r(n),v(n),j(n),h(n),k(n),r2h(n)
integer(kind=8) bytes
character(len=16) tag
if(.not.pw_enabled())return
if(.not.fp_ready.or.fp_in_call.or.fp_freq/=freq.or.n/=nd_store)call fp_fail('whole field missing actual formal result')
if(index/=pw_count+1.or.index/=lumina_freq_index.or.lumina_outer/=60)call fp_fail('whole frequency coverage mismatch')
if(freq>=pw_previous_freq.or..not.ieee_is_finite(weight).or.weight<=0)call fp_fail('whole frequency/weight order mismatch')
if(any(j/=jnu_store).or.any(r/=r_store))call fp_fail('whole J consumer differs from angular producer')
if(any(h/=hnu_store).or.any(k/=knu_store).or.any(r2h/=r*r*hnu_store))call fp_fail('whole H/K/flux consumer differs from angular producer')
if(max(fp_j_residual,fp_source_residual,fp_h_residual)>1.e-8_ldp.or.fp_calls>20)call fp_fail('whole unconverged source consumed')
if(.not.pw_input_ready)call fp_fail('whole source identity missing')
if(pw_input%freq/=freq.or.size(pw_input%chi)/=n)call fp_fail('whole source identity missing')
if(.not.pw_open)then
 pw_expected=expected;pw_n=n
 open(newunit=pw_csv,file='FORMAL_PRIMARY_SWEEP.csv',status='new')
 write(pw_csv,'(A)')'index,formal_calls,freq_native,frequency_weight,J_residual,source_residual,H_change_over_J,F_change,angular_support_min'
 open(newunit=pw_field,file='FORMAL_PRIMARY_FIELDS.bin',status='new',access='stream',form='unformatted')
 tag='FORMALPRIMARYV1'
 ! 각 record: index,calls,freq,weight,J,H,K,N,chi,eta_true,chi_scat,J_source.
 write(pw_field)tag,[ldp,storage_size(0._ldp),storage_size(0),n,expected,8],r,v,pw_input%sigma,gam_rel_store
 pw_open=.true.
end if
if(n/=pw_n.or.expected/=pw_expected)call fp_fail('whole field shape changed')
write(pw_csv,'(I8,",",I3,7(",",ES18.10E3))')index,fp_calls,freq,weight,fp_j_residual,fp_source_residual,fp_h_residual,fp_f_change,fp_support_min
write(pw_field)index,fp_calls,freq,weight,jnu_store,hnu_store,knu_store,nnu_store,pw_input%chi,fp_eta_true,fp_scattering,fp_source_j
pw_count=pw_count+1;pw_previous_freq=freq;pw_calls=pw_calls+fp_calls
inquire(unit=pw_csv,size=bytes)
if(bytes>67108864_8)call fp_fail('whole summary exceeds 64MiB')
inquire(unit=pw_field,size=bytes)
if(bytes>2147483648_8)call fp_fail('whole field exceeds 2GiB')
if(mod(index,4096)==0)then
 flush(pw_csv);flush(pw_field)
end if
end subroutine

subroutine pw_finish()
integer lu
if(.not.pw_enabled())return
if(.not.pw_open.or.pw_count/=pw_expected.or.pw_count==0)call fp_fail('whole incomplete pre-pop sweep')
close(pw_csv);close(pw_field);pw_open=.false.
open(newunit=lu,file='FORMAL_PRIMARY_COMPLETE.txt',status='new')
write(lu,*)60,pw_count,pw_expected,pw_calls
write(lu,'(A)')'FORMAL_ANGULAR_DIAGNOSTIC_COMPLETE_NOT_CLOSURE_OR_PHYSICS_PASS'
write(lu,'(A)')'MOM_EQUATION_RESIDUAL_NOT_APPLICABLE_NOT_ZERO'
close(lu)
stop 97
end subroutine
end module

subroutine fp_primary_failure()
use lumina_formal_primary_whole
call pw_failure()
end subroutine
subroutine fp_whole_frequency_end(index,expected,freq,weight,r,v,j,h,k,r2h,n)
use lumina_formal_primary_whole
integer index,expected,n
real(ldp)freq,weight,r(n),v(n),j(n),h(n),k(n),r2h(n)
call pw_frequency_end(index,expected,freq,weight,r,v,j,h,k,r2h,n)
end subroutine
subroutine fp_whole_finish()
use lumina_formal_primary_whole
call pw_finish()
end subroutine
