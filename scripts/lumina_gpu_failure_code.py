"""선택 오류 페이로드의 순수 해석. GPU·실행 정책을 변경하지 않는다."""


def decode(code):
    if type(code) is not int:return None
    payload=code-100-100000000
    if payload<0:return None
    column,status=divmod(payload,8)
    if not 0<=column<2000000 or not 1<=status<=7:return None
    return dict(output_column=column,characteristic_status=status,
        scope='첫 관측 실패이며 최소 실패열은 아니다. coarse/두half의 어느 조각인지는 CPU에서 확인한다.')
