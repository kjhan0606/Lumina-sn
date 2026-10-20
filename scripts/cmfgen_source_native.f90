! 원본 GFF_VEC를 요청 주파수 배열에 적용한다. 새 Gaunt 근사표를 만들지 않는다.
subroutine lumina_native_gaunt(nd,nf,nu_hz,t_kelvin,z,g) bind(C)
  use iso_c_binding
  use set_kind_module
  implicit none
  integer(c_int), value :: nd,nf
  real(c_double), intent(in) :: nu_hz(nf),t_kelvin(nd)
  real(c_double), value :: z
  real(c_double), intent(out) :: g(nd,nf)
  real(kind=LDP) :: t4(nd),freq,z_native
  integer :: b
  t4=t_kelvin*1.0e-4_LDP
  z_native=z
  do b=1,nf
    freq=nu_hz(b)*1.0e-15_LDP
    call gff_vec(g(:,b),freq,t4,z_native,nd)
  end do
end subroutine
