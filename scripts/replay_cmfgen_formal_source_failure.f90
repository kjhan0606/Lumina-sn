! 마지막 실제 formal 호출을 먼저 재현하고 같은 산란식을 제한된 예산으로 푼다.
program replay_formal_source_failure
use lumina_formal_primary
use lumina_formal_diag
use lumina_mom_diag
implicit none
type(lumina_formal_args) a,b
type(lumina_ray_state) raw
character(len=4096) root,mode,initial_path,message
integer n,lu,trace,states,history,count,saved_count,ninitial
real(ldp) frequency,ignored(5)
real(ldp),allocatable::guess(:),eta(:),scat(:),j(:),lambda(:,:)
logical inaccurate,exact_source
call get_command_argument(1,root);call get_command_argument(2,mode)
call get_command_argument(3,initial_path)
if(.not.fp_enabled().and.mode/='original')stop 85
call formal_state_io(trim(root)//'/FORMAL_PRIMARY_CONTEXT.bin',a,raw,.true.)
n=a%nd;allocate(guess(n),eta(n),scat(n),j(n))
open(newunit=lu,file=trim(root)//'/FORMAL_PRIMARY_FAILURE.txt',status='old')
read(lu,'(A)')message;read(lu,*)frequency,saved_count,ignored;read(lu,*)guess;close(lu)
if(frequency/=a%freq.or.saved_count/=19.or.a%initialize.or.a%new_freq)stop 85
! 원래 true eta는 별도 저장되지 않았다. 이 차감값을 정확한 원본이라고 부르지 않는다.
scat=a%esec;eta=a%eta-scat*guess
inquire(file=trim(root)//'/FORMAL_PRIMARY_SOURCE.bin',exist=exact_source)
if(exact_source)then
 open(newunit=lu,file=trim(root)//'/FORMAL_PRIMARY_SOURCE.bin',status='old',access='stream',form='unformatted')
 block
  character(len=16) source_tag
  integer source_n,source_count,source_calls
  real(ldp) source_freq
  read(lu)source_tag,source_n,source_count,source_calls,source_freq
  if(source_tag/='FPRIMARYSOURCE1'.or.source_n/=n.or.source_freq/=a%freq)stop 86
  read(lu)guess,eta,scat
 end block
 close(lu)
end if
if(any(eta<0))stop 85
j=guess
if(trim(initial_path)/='')then
 open(newunit=lu,file=trim(initial_path),status='old',access='stream',form='unformatted')
 read(lu)ninitial
 if(ninitial/=n)stop 85
 read(lu)j;close(lu)
end if
b=a
call cmf_formal_rel_v4(b%eta,b%chi,b%esec,b%v,b%sigma,b%r,b%p,b%jnu,b%fedd, &
 b%ret_hnu_at_ib,b%ret_hnu_at_ob,b%iplus,b%freq,b%dlog_nu,b%b_planck,b%dbb,b%inner_bnd_meth, &
 b%thick_ob,b%vdop_vec,b%vdop_frac,b%rext_fac,b%method,.true.,.true.,b%nc,b%np,b%nd)
call formal_rays_io(trim(root)//'/FORMAL_PRIMARY_BLUE.bin',a,.true.,.false.)
call lumina_ray_restore(raw)
call cmf_formal_rel_v4(a%eta,a%chi,a%esec,a%v,a%sigma,a%r,a%p,a%jnu,a%fedd, &
 a%ret_hnu_at_ib,a%ret_hnu_at_ob,a%iplus,a%freq,a%dlog_nu,a%b_planck,a%dbb,a%inner_bnd_meth, &
 a%thick_ob,a%vdop_vec,a%vdop_frac,a%rext_fac,a%method,.false.,.false.,a%nc,a%np,a%nd)
call formal_rays_io('ACTUAL_LAST_CALL_RAYS.bin',a,.false.,.true.)
call formal_geometry('ACTUAL_LAST_CALL_GEOMETRY.bin',a)
if(mode=='original')stop
open(newunit=lu,file='DERIVED_SOURCE_INPUT.bin',status='new',access='stream',form='unformatted')
write(lu)n,eta,scat,j;close(lu)
open(newunit=trace,file='SOURCE_REPLAY.csv',status='new')
write(trace,'(A)')'trial,J_residual,source_residual,H_change_over_J,F_change,angular_support_min'
open(newunit=states,file='SOURCE_STATES.bin',status='new',access='stream',form='unformatted')
write(states)n
open(newunit=history,file='SOURCE_HISTORY.bin',status='new',access='stream',form='unformatted')
do count=0,19
 call fp_start(a%freq,count,n,j,eta,scat,.true.,.true.,.true.)
 a%eta=eta+scat*j
 call cmf_formal_rel_v4(a%eta,a%chi,a%esec,a%v,a%sigma,a%r,a%p,a%jnu,a%fedd, &
  a%ret_hnu_at_ib,a%ret_hnu_at_ob,a%iplus,a%freq,a%dlog_nu,a%b_planck,a%dbb,a%inner_bnd_meth, &
  a%thick_ob,a%vdop_vec,a%vdop_frac,a%rext_fac,a%method,.false.,.false.,a%nc,a%np,a%nd)
 call fp_record(n,a%np,a%jnu,a%fedd,a%thick_ob)
 write(states)count,fp_source_j,a%jnu;flush(states)
 if(count==0)then
  lambda=sum(fp_lambda_columns,dim=3)
  open(newunit=lu,file='APPROXIMATE_LAMBDA.bin',status='new',access='stream',form='unformatted')
  write(lu)n,lambda;close(lu)
 end if
 write(trace,'(I3,5(",",ES24.16E3))')count,fp_j_residual,fp_source_residual,fp_h_residual,fp_f_change,fp_support_min
 flush(trace)
 inaccurate=.false.;call fp_end_iteration(count+1,fp_f_change,1.e-4_ldp,inaccurate)
 if(.not.inaccurate)exit
 j=a%jnu
 if(mode=='ali')fp_history_count=0
 if(mode/='picard')call fp_next_source(n,a%jnu,j)
 write(history)count,fp_history_count
 if(fp_history_count>0)write(history)fp_x_history(:,1:fp_history_count),fp_p_history(:,1:fp_history_count)
 flush(history)
end do
close(trace);close(states);close(history)
if(exact_source)then
 write(6,'(A,I5)')'EXACT_SOURCE_CAUSAL_REPLAY_NOT_ANCHOR ',fp_calls
else
 write(6,'(A,I5)')'DERIVED_SOURCE_CAUSAL_REPLAY_NOT_EXACT_SOURCE_OR_ANCHOR ',fp_calls
end if
end program
