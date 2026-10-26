! 원 native 핵반응 루틴의 격리 시험. 원본/수정본에 같은 초기값을 전달한다.
program native_decay_zero_dt_driver
  use set_kind_module
  use nuc_iso_mod
  implicit none
  integer :: nd=3
  integer :: i,chain
  real(kind=LDP) :: dt
  logical :: instantaneous
  character(len=80) :: arg
  character(len=512) :: population_file

  call get_command_argument(1,arg);read(arg,*)chain
  call get_command_argument(2,arg);read(arg,*)dt
  call get_command_argument(3,arg);read(arg,*)instantaneous
  call get_command_argument(4,population_file)
  if(len_trim(population_file)>0)then
    open(unit=10,file=trim(population_file),access='stream',form='unformatted',status='old')
    read(10)nd
    if(nd<1 .or. nd>90)stop 2
  end if
  if(chain/=1 .and. chain/=2)stop 1
  open(unit=99,file='native.log',status='new')
  num_isotopes=3;num_parents=1;num_decay_paths=chain
  allocate(radioactive_decay_energy(nd),kinetic_decay_energy(nd))
  allocate(par(1)%old_pop(nd),par(1)%old_pop_decay(nd))
  par(1)%decay_chain_available=.true.
  do i=1,3
    allocate(iso(i)%old_pop(nd),iso(i)%old_pop_decay(nd),iso(i)%num_decays(nd),iso(i)%decay_lum(nd))
    iso(i)%lnk_to_par=1;iso(i)%num_decays=0;iso(i)%decay_lum=0
  end do
  iso(1)%species='NICK';iso(1)%mass=55.942132_LDP
  iso(2)%species='COB';iso(2)%mass=55.939839_LDP
  iso(3)%species='IRON';iso(3)%mass=55.934937_LDP
  if(len_trim(population_file)>0)then
    read(10)iso(1)%old_pop,iso(2)%old_pop,iso(3)%old_pop
    close(10)
  else
    iso(1)%old_pop=[1.e6_LDP,2.e6_LDP,0._LDP]
    iso(2)%old_pop=[3.e6_LDP,0._LDP,4.e6_LDP]
    iso(3)%old_pop=[5.e6_LDP,6.e6_LDP,7.e6_LDP]
  end if
  nuc(1)%lnk_to_iso=1;nuc(1)%daughter_lnk_to_iso=2
  nuc(1)%species='NICK';nuc(1)%daughter='COB';nuc(1)%baryon_number=56
  nuc(1)%sequence='E'
  nuc(1)%decay_const=log(2._LDP)/(6.0749_LDP*86400._LDP)
  nuc(1)%energy_per_decay=1.7182758_LDP*1.602176634e-6_LDP
  nuc(1)%kinetic_per_decay=.03_LDP*1.602176634e-6_LDP
  if(chain==2)then
    nuc(1)%sequence='F'
    nuc(2)%lnk_to_iso=2;nuc(2)%daughter_lnk_to_iso=3
    nuc(2)%species='COB';nuc(2)%daughter='IRON';nuc(2)%baryon_number=56
    nuc(2)%sequence='S'
    nuc(2)%decay_const=log(2._LDP)/(77.233_LDP*86400._LDP)
    nuc(2)%energy_per_decay=3.7488932_LDP*1.602176634e-6_LDP
    nuc(2)%kinetic_per_decay=.12_LDP*1.602176634e-6_LDP
  end if
  if(len_trim(population_file)>0)then
    ! 현 원 NUC_DECAY_DATA는 positron kinetic 열이 없는 구형 형식이다.
    nuc(1)%kinetic_per_decay=0._LDP
    if(chain==2)nuc(2)%kinetic_per_decay=0._LDP
  end if
  call do_species_decays_v2(instantaneous,dt,nd)
  open(unit=8,file='result.bin',access='stream',form='unformatted',status='new')
  write(8)radioactive_decay_energy,kinetic_decay_energy
  do i=1,3
    write(8)iso(i)%old_pop,iso(i)%old_pop_decay,iso(i)%num_decays,iso(i)%decay_lum
  end do
  write(8)par(1)%old_pop,par(1)%old_pop_decay
  close(8);close(99)
end program

! 아래 세 stub은 로그/기하 적분 진단만 대체한다. 열·isotope 수식은 원본이다.
integer function error_lu()
  implicit none
  error_lu=99
end function

subroutine get_decay_lum_v1(lum,energy,nd)
  use set_kind_module
  implicit none
  integer,intent(in)::nd
  real(kind=LDP),intent(in)::energy(nd)
  real(kind=LDP),intent(out)::lum
  lum=sum(energy)
end subroutine

subroutine writv_v2(values,n,a,label,unit)
  use set_kind_module
  implicit none
  integer,intent(in)::n,a,unit
  character(len=*),intent(in)::label
  real(kind=LDP),intent(in)::values(n)
end subroutine
