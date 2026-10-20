! 실제 사건의 formal/MOM을 복원한 뒤 원래 한 주파수 F-loop만 진행한다.
! LUMINA_WHOLE_MODE는 비워 PSD를 중단 게이트로 쓰지 않는다. CAPTURE_FIRST는 native 대체를 여전히 중단한다.
program event_floop_replay
use lumina_whole_monitor
use lumina_formal_diag
implicit none
type(lumina_arg_state) ma
type(lumina_formal_args) fa,fb
type(lumina_jrel_state) js
type(lumina_ray_state) rs
character(len=4096) root_dir
character(len=16) tag
character(len=20) capture_mode
character(len=80) filename
integer lu,ios,meta(5),loop_nd,loop_count,initial_count,log_unit,bytes
logical compute_f,coherent,inaccurate,native_cap
real(ldp) tolerance,change,bc(7)
logical bc_ready
real(ldp),allocatable::fold_local(:),eta_local(:),scat_local(:),rjes_local(:)
call get_command_argument(1,root_dir)
if(root_dir==' '.or.whole_enabled())stop 85
call get_environment_variable('LUMINA_MOM_CAPTURE_MODE',capture_mode)
if(capture_mode/='CAPTURE_FIRST')stop 85
call formal_state_io(trim(root_dir)//'/FORMAL_EVENT_in.bin',fa,rs,.true.)
fb=fa
call cmf_formal_rel_v4(fb%eta,fb%chi,fb%esec,fb%v,fb%sigma,fb%r,fb%p,fb%jnu,fb%fedd, &
 fb%ret_hnu_at_ib,fb%ret_hnu_at_ob,fb%iplus,fb%freq,fb%dlog_nu,fb%b_planck,fb%dbb,fb%inner_bnd_meth, &
 fb%thick_ob,fb%vdop_vec,fb%vdop_frac,fb%rext_fac,fb%method,.true.,.true.,fb%nc,fb%np,fb%nd)
call formal_geometry('replay_geometry.bin',fa)
call formal_rays_io(trim(root_dir)//'/FORMAL_EVENT_rays_in.bin',fa,.true.,.false.)
call lumina_ray_restore(rs)
call cmf_formal_rel_v4(fa%eta,fa%chi,fa%esec,fa%v,fa%sigma,fa%r,fa%p,fa%jnu,fa%fedd, &
 fa%ret_hnu_at_ib,fa%ret_hnu_at_ob,fa%iplus,fa%freq,fa%dlog_nu,fa%b_planck,fa%dbb,fa%inner_bnd_meth, &
 fa%thick_ob,fa%vdop_vec,fa%vdop_frac,fa%rext_fac,fa%method,fa%initialize,fa%new_freq,fa%nc,fa%np,fa%nd)
call formal_rays_io('first_replay_rays_out.bin',fa,.false.,.true.)
call formal_ray_take(rs)
call formal_state_io('first_replay_formal_out.bin',fa,rs,.false.)
call lumina_snapshot_io(trim(root_dir)//'/MOM_CAPTURE.bin',ma,js,rs,.true.)
call lumina_jrel_restore(js)
call lumina_ray_restore(rs)
open(newunit=lu,file=trim(root_dir)//'/AFFINE_INPUT.bin',status='old',access='stream',form='unformatted')
inquire(unit=lu,size=bytes)
if(bytes/=112)stop 86
read(lu)tag,meta,bc,bc_ready
if(trim(tag)/='LUMINAAFFINEIN1')stop 86
read(lu)tag
if(trim(tag)/='LUMINAAFFINEND1')stop 86
close(lu)
affine_hinc=bc(1);affine_ninc=bc(2);affine_hinc_prev=bc(3);affine_ninc_prev=bc(4)
affine_freq_save=bc(5);affine_jmin=bc(6);affine_jplus=bc(7);affine_ready=bc_ready
open(newunit=lu,file=trim(root_dir)//'/FLOOP_STATE.bin',status='old',access='stream',form='unformatted')
read(lu)tag
if(trim(tag)/='LUMINAFLOOP1')stop 86
read(lu)meta,loop_nd,loop_count,compute_f,coherent,tolerance
if(loop_nd/=ma%nd_sm.or.loop_nd/=fa%nd.or.loop_count/=lumina_jit.or.(coherent.neqv.ma%coherent))stop 86
inquire(unit=lu,size=bytes)
if(bytes/=76+32*loop_nd)stop 86
if(any(meta/=[lumina_outer,lumina_freq_index,lumina_jit,merge(1,0,lumina_first),merge(1,0,lumina_new)]))stop 86
allocate(fold_local(loop_nd),eta_local(loop_nd),scat_local(loop_nd),rjes_local(loop_nd))
read(lu)fold_local,eta_local,scat_local,rjes_local
read(lu)tag
if(trim(tag)/='LUMINAFLOOPEND1')stop 86
read(lu,iostat=ios)tag
if(ios/=-1)stop 86
close(lu)
initial_count=loop_count
if(lu_mom>0)open(unit=lu_mom,file='MOM_J_ERRORS',status='new',action='write')
open(newunit=log_unit,file='FLOOP_REPLAY.csv',status='new')
write(log_unit,'(A)')'jit,F_change,native_F_cap'
do
 call mom_jrel_v9(ma%eta_sm,ma%chi_sm,ma%esec_sm,ma%v_sm,ma%sigma_sm,ma%r_sm, &
  ma%jnu_sm,ma%rsqhnu_sm,ma%hflux_at_ib,ma%hflux_at_ob,ma%vdop_vec,ma%vdop_frac,ma%freq,ma%dlog_nu,ma%dbb, &
  ma%xm_chk_option,ma%j_chk_option,ma%h_chk_option,ma%ib_stab_factor,ma%inner_bnd_meth,ma%outer_bnd_meth, &
  ma%method,ma%coherent,ma%n_type,ma%incl_advec_terms,ma%incl_rel_terms,ma%init,ma%nd_sm)
 call lumina_jrel_take(js)
 call lumina_ray_take(rs)
 write(filename,'(A,I2.2,A)')'JIT_',loop_count,'_MOM.bin'
 call lumina_snapshot_io(trim(filename),ma,js,rs,.false.)
 write(filename,'(A,I2.2,A)')'JIT_',loop_count,'_FORMAL.bin'
 call formal_ray_take(rs)
 call formal_state_io(trim(filename),fa,rs,.false.)
 ! 다음은 원래 COMP_J_BLANK의 MOM 다음 종료 순서 그대로다.
 fa%new_freq=.false.
 loop_count=loop_count+1
 inaccurate=.false.
 if(loop_count<20.or.compute_f)then
  change=maxval(abs(fold_local-fa%fedd))
  fold_local=fa%fedd
  if(change>tolerance)inaccurate=.true.
 end if
 native_cap=loop_count>20
 if(native_cap)inaccurate=.false.
 write(log_unit,'(I3,",",ES24.16E3,",",I1)')loop_count-1,change,merge(1,0,native_cap)
 flush(log_unit)
 if(.not.inaccurate)exit
 if(loop_count>21.or.loop_count<initial_count)stop 88
 if(coherent)then
  fa%eta=eta_local+scat_local*ma%jnu_sm
 else
  fa%eta=eta_local+scat_local*rjes_local
 end if
 call lumina_mom_set_phase(meta(2),loop_count,fa%initialize,.false.)
 call cmf_formal_rel_v4(fa%eta,fa%chi,fa%esec,fa%v,fa%sigma,fa%r,fa%p,fa%jnu,fa%fedd, &
  fa%ret_hnu_at_ib,fa%ret_hnu_at_ob,fa%iplus,fa%freq,fa%dlog_nu,fa%b_planck,fa%dbb,fa%inner_bnd_meth, &
  fa%thick_ob,fa%vdop_vec,fa%vdop_frac,fa%rext_fac,fa%method,fa%initialize,fa%new_freq,fa%nc,fa%np,fa%nd)
 if(coherent)then
  ma%eta_sm=eta_local
 else
  ma%eta_sm=eta_local+scat_local*rjes_local
 end if
end do
close(log_unit)
write(6,'(A)')'ORIGINAL_F_LOOP_TERMINATED_NOT_PHYSICS_PASS'
if(native_cap)stop 94
end program
