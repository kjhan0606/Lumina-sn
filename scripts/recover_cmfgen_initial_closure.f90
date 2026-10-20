! INIT에서 수송 계수를 지우기 전의 물리 closure를 같은 raw 모멘트로 재현한다.
program recover_initial_closure
use lumina_mom_diag
use mod_jrel_v9
use mod_ray_mom_store
implicit none
type(lumina_arg_state) a
type(lumina_jrel_state) js
type(lumina_ray_state) rs
character(len=4096) input
call get_command_argument(1,input)
if(input==' ')stop 85
call lumina_snapshot_io(trim(input),a,js,rs,.true.)
if(.not.a%init)stop 89
call lumina_jrel_restore(js)
call lumina_ray_restore(rs)
call get_moms_rel(r,v,a%freq,a%n_type,nd)
call lumina_jrel_take(js)
call lumina_ray_take(rs)
call lumina_snapshot_io('INITIAL_PHYSICAL_CLOSURE.bin',a,js,rs,.false.)
end program
