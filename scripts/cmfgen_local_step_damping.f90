! 분기 실험 전용 공통 배율. 기본 경로의 배율과 연산은 바꾸지 않는다.
subroutine lumina_local_step_damping(scale, depth, lu)
  use set_kind_module
  implicit none
  real(kind=ldp), intent(inout) :: scale
  integer, intent(in) :: depth, lu
  character(len=32) :: value
  integer :: length, status
  real(kind=ldp) :: original

  call get_environment_variable('LUMINA_LOCAL_STEP_OMEGA', value, length, status)
  if (status == 1 .or. (status == 0 .and. length == 0)) return
  if (status /= 0) then
    write(lu,*) 'LUMINA_LOCAL_STEP_OMEGA_INVALID'
    stop 118
  end if
  select case (trim(value))
  case ('1')
    return
  case ('0.5')
    original = scale
    scale = 0.5_ldp * scale
    write(lu,'(A,1X,I5,3(1X,ES24.16))') &
      'LUMINA_LOCAL_DAMPING_V1', depth, original, 0.5_ldp, scale
  case default
    write(lu,*) 'LUMINA_LOCAL_STEP_OMEGA_INVALID'
    stop 118
  end select
end subroutine lumina_local_step_damping
