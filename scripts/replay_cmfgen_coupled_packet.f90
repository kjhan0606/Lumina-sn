! 한 주파수의 source-J trial만 바꾸고 실제 두 blue 이력을 복원한다.
! MOM 첫 행렬을 보존하고 풀이 전 종료한다. 생산 H 제한을 해제하지 않는다.
program coupled_packet
use lumina_whole_monitor
use lumina_formal_diag
implicit none
type(lumina_arg_state) ma
type(lumina_jrel_state) js
type(lumina_ray_state) input_raw,fresh_raw,full_raw
type(lumina_formal_args) fa,fb
character(len=4096) root_dir,trial_file
character(len=16) tag
integer lu,bytes,meta(5),n,loop_count,ios
logical ready,compute_f,coherent
real(ldp) bc(7),tolerance
real(ldp),allocatable::j_trial(:),eta_true(:),scattering(:),discard(:)
call get_command_argument(1,root_dir);call get_command_argument(2,trial_file)
if(root_dir==' '.or.whole_enabled())stop 85
call formal_state_io(trim(root_dir)//'/FORMAL_EVENT_in.bin',fa,input_raw,.true.)
call lumina_snapshot_io(trim(root_dir)//'/MOM_CAPTURE.bin',ma,js,full_raw,.true.)
if(ma%init.or.fa%initialize.or..not.fa%new_freq.or.fa%nd/=ma%nd_sm)stop 85
n=fa%nd
allocate(j_trial(n),eta_true(n),scattering(n),discard(n))
j_trial=ma%jnu_sm
open(newunit=lu,file=trim(root_dir)//'/FLOOP_STATE.bin',status='old',access='stream',form='unformatted')
inquire(unit=lu,size=bytes)
if(bytes/=76+32*n)stop 86
read(lu)tag,meta,bytes,loop_count,compute_f,coherent,tolerance
if(tag/='LUMINAFLOOP1'.or.bytes/=n.or..not.coherent.or..not.ma%coherent)stop 86
read(lu)discard,eta_true,scattering,discard
read(lu)tag
if(tag/='LUMINAFLOOPEND1')stop 86
close(lu)
if(any(eta_true/=ma%eta_sm).or.any(scattering/=ma%esec_sm))stop 86
if(trial_file/=' ')then
 open(newunit=lu,file=trial_file,status='old',access='stream',form='unformatted')
 inquire(unit=lu,size=bytes)
 if(bytes/=8*n)stop 86
 read(lu)j_trial;close(lu)
 if(.not.all(ieee_is_finite(j_trial)).or.any(j_trial<=0))stop 86
 fa%eta=eta_true+scattering*j_trial
 ma%jnu_sm=j_trial
end if
! 기하는 원래 routine으로 만들고, 실제 blue 광선/경계 이력을 다시 복원한다.
fb=fa
call cmf_formal_rel_v4(fb%eta,fb%chi,fb%esec,fb%v,fb%sigma,fb%r,fb%p,fb%jnu,fb%fedd, &
 fb%ret_hnu_at_ib,fb%ret_hnu_at_ob,fb%iplus,fb%freq,fb%dlog_nu,fb%b_planck,fb%dbb,fb%inner_bnd_meth, &
 fb%thick_ob,fb%vdop_vec,fb%vdop_frac,fb%rext_fac,fb%method,.true.,.true.,fb%nc,fb%np,fb%nd)
call formal_geometry('TRIAL_GEOMETRY.bin',fa)
call formal_rays_io(trim(root_dir)//'/FORMAL_EVENT_rays_in.bin',fa,.true.,.false.)
call lumina_ray_restore(input_raw)
open(newunit=lu,file=trim(root_dir)//'/AFFINE_INPUT.bin',status='old',access='stream',form='unformatted')
inquire(unit=lu,size=bytes)
if(bytes/=112)stop 86
read(lu)tag,meta,bc,ready
if(tag/='LUMINAAFFINEIN1')stop 86
read(lu)tag;close(lu)
if(tag/='LUMINAAFFINEND1')stop 86
affine_hinc=bc(1);affine_ninc=bc(2);affine_hinc_prev=bc(3);affine_ninc_prev=bc(4)
affine_freq_save=bc(5);affine_jmin=bc(6);affine_jplus=bc(7);affine_ready=ready
call cmf_formal_rel_v4(fa%eta,fa%chi,fa%esec,fa%v,fa%sigma,fa%r,fa%p,fa%jnu,fa%fedd, &
 fa%ret_hnu_at_ib,fa%ret_hnu_at_ob,fa%iplus,fa%freq,fa%dlog_nu,fa%b_planck,fa%dbb,fa%inner_bnd_meth, &
 fa%thick_ob,fa%vdop_vec,fa%vdop_frac,fa%rext_fac,fa%method,fa%initialize,fa%new_freq,fa%nc,fa%np,fa%nd)
call formal_rays_io('TRIAL_RAYS_OUT.bin',fa,.false.,.true.)
fb=fa;fb%new_freq=.false.
call formal_rays_io('TRIAL_BLUE_AFTER.bin',fb,.false.,.false.)
call formal_ray_take(fresh_raw)
call formal_state_io('TRIAL_FORMAL.bin',fa,fresh_raw,.false.)
open(newunit=lu,file='TRIAL_BOUNDARY.bin',status='new',access='stream',form='unformatted')
tag='LUMINAAFFINEIN1'
write(lu)tag,meta,[affine_hinc,affine_ninc,affine_hinc_prev,affine_ninc_prev,affine_freq_save,affine_jmin,affine_jplus],affine_ready
tag='LUMINAAFFINEND1';write(lu)tag
close(lu)
! 새 raw formal만 덮고 원래 MOM의 SAVE/PREV closure는 보존한다.
__OVERLAY_RAW__
call lumina_jrel_restore(js)
call lumina_ray_restore(full_raw)
if(lu_mom>0)open(unit=lu_mom,file='MOM_J_ERRORS',status='new',action='write')
call mom_jrel_v9(ma%eta_sm,ma%chi_sm,ma%esec_sm,ma%v_sm,ma%sigma_sm,ma%r_sm, &
 ma%jnu_sm,ma%rsqhnu_sm,ma%hflux_at_ib,ma%hflux_at_ob,ma%vdop_vec,ma%vdop_frac,ma%freq,ma%dlog_nu,ma%dbb, &
 ma%xm_chk_option,ma%j_chk_option,ma%h_chk_option,ma%ib_stab_factor,ma%inner_bnd_meth,ma%outer_bnd_meth, &
 ma%method,ma%coherent,ma%n_type,ma%incl_advec_terms,ma%incl_rel_terms,ma%init,ma%nd_sm)
stop 89
end program
