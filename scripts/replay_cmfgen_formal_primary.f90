! 같은 angular RTE와 실제 blue packet으로 primary source 고정점을 푼다.
program replay_formal_primary
use lumina_formal_primary
use lumina_formal_diag
use lumina_mom_diag
implicit none
type(lumina_formal_args) fa,fb
type(lumina_arg_state) ma
type(lumina_jrel_state) js
type(lumina_ray_state) raw,full
character(len=4096) root_dir,mode
character(len=16) tag
integer lu,n,meta(5),stored_n,count,bytes,log_unit
logical compute_f,coherent,inaccurate
real(ldp) tolerance
real(ldp),allocatable::j(:),eta(:),scat(:),discard(:),rsqh(:)
call get_command_argument(1,root_dir);call get_command_argument(2,mode)
if(.not.fp_enabled())stop 85
call formal_state_io(trim(root_dir)//'/FORMAL_EVENT_in.bin',fa,raw,.true.)
call lumina_snapshot_io(trim(root_dir)//'/MOM_CAPTURE.bin',ma,js,full,.true.)
if(fa%initialize.or..not.fa%new_freq)stop 85
n=fa%nd;allocate(j(n),eta(n),scat(n),discard(n),rsqh(n));j=ma%jnu_sm
open(newunit=lu,file=trim(root_dir)//'/FLOOP_STATE.bin',status='old',access='stream',form='unformatted')
inquire(unit=lu,size=bytes)
if(bytes/=76+32*n)stop 86
read(lu)tag,meta,stored_n,count,compute_f,coherent,tolerance
if(tag/='LUMINAFLOOP1'.or.stored_n/=n.or.count/=0.or..not.coherent)stop 86
read(lu)discard,eta,scat,discard;read(lu)tag;close(lu)
if(tag/='LUMINAFLOOPEND1')stop 86
fb=fa
call cmf_formal_rel_v4(fb%eta,fb%chi,fb%esec,fb%v,fb%sigma,fb%r,fb%p,fb%jnu,fb%fedd, &
 fb%ret_hnu_at_ib,fb%ret_hnu_at_ob,fb%iplus,fb%freq,fb%dlog_nu,fb%b_planck,fb%dbb,fb%inner_bnd_meth, &
 fb%thick_ob,fb%vdop_vec,fb%vdop_frac,fb%rext_fac,fb%method,.true.,.true.,fb%nc,fb%np,fb%nd)
call formal_geometry('PRIMARY_GEOMETRY.bin',fa)
call formal_rays_io(trim(root_dir)//'/FORMAL_EVENT_rays_in.bin',fa,.true.,.false.)
call lumina_ray_restore(raw)
open(newunit=log_unit,file='PRIMARY_REPLAY.csv',status='new')
write(log_unit,'(A)')'trial,J_residual,source_residual,H_change_over_J,F_change,angular_support_min'
do count=0,19
 call fp_start(fa%freq,count,n,j,eta,scat,.true.,.true.,.true.)
 if(count==0.or.mode/='--wrong-source')fa%eta=eta+scat*j
 if(mode=='--wrong-blue'.and.count>0)fa%new_freq=.true.
 call cmf_formal_rel_v4(fa%eta,fa%chi,fa%esec,fa%v,fa%sigma,fa%r,fa%p,fa%jnu,fa%fedd, &
  fa%ret_hnu_at_ib,fa%ret_hnu_at_ob,fa%iplus,fa%freq,fa%dlog_nu,fa%b_planck,fa%dbb,fa%inner_bnd_meth, &
  fa%thick_ob,fa%vdop_vec,fa%vdop_frac,fa%rext_fac,fa%method,fa%initialize,fa%new_freq,fa%nc,fa%np,fa%nd)
 if(count==0)call formal_rays_io('PRIMARY_FIRST_RAYS.bin',fa,.false.,.true.)
 call fp_record(n,fa%np,fa%jnu,fa%fedd,fa%thick_ob)
 j=fa%jnu
 call get_rsqh_rel(rsqh,fa%r,fa%v,fa%freq,n)
 write(log_unit,'(I3,5(",",ES24.16E3))')count,fp_j_residual,fp_source_residual,fp_h_residual,fp_f_change,fp_support_min
 flush(log_unit)
 inaccurate=.false.
 call fp_end_iteration(count+1,fp_f_change,tolerance,inaccurate)
 if(.not.inaccurate)exit
 call fp_next_source(n,fa%jnu,j)
 fa%new_freq=.false.
end do
close(log_unit)
fb=fa;fb%new_freq=.false.
call formal_rays_io('PRIMARY_BLUE_FINAL.bin',fb,.false.,.false.)
call formal_rays_io('PRIMARY_RAYS_FINAL.bin',fa,.false.,.true.)
call formal_ray_take(raw)
call formal_state_io('PRIMARY_FORMAL_FINAL.bin',fa,raw,.false.)
open(newunit=lu,file='PRIMARY_H_INTERFACE.bin',status='new',access='stream',form='unformatted')
write(lu)rsqh(1:n-1);close(lu)
write(6,'(A,I4)')'PRIMARY_PACKET_SOURCE_FIXED_POINT_NOT_ANCHOR_PASS ',fp_calls
end program
