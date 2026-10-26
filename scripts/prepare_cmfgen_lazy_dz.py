#!/usr/bin/env python3
"""native dZ의 비관측 컬럼에 대한 행 초기화만 생략한다. 차원·물리식 불변.

새 strong 선의 전체 컬럼 초기화는 유지한다. weak/미사용 컬럼의 이전 값은
관측되지 않으며 다음 strong 선이 사용하기 전에 전체 초기화된다.
"""
import hashlib


SEALED = {
    'new_main/mod_subs/init_line_opac_var_v2.f':
        'b7071275342757fcf2f2bbb06756d8470ce966be997af343f021f953782b046f',
    'new_main/mod_subs/set_line_opac.f':
        '961076e7d17aa1832ef83e2b2c436f3047967f72060774376fea2db39017d290',
    'new_main/subs/update_ba_for_line.f':
        'be4f1cf9de954c91f7f004e0cc1727dacce6639b3bc7de8d2f7a181cb0913a2a',
    'new_main/mod_subs/line_mod.f':
        '6417fe8ac67bb0b775047919b1e81881ded9422b766ea5962be3a5f4657187b3',
    'tools/set_kind_module.f':
        '835cfb88d2421b55f9d5c2ecf50d1d2f79fcec693732cf571e6bb7ef10dbe06f',
}
HELPER_START = '\tSUBROUTINE ZERO_dZ(dZ,J,NM,NX,M)\n'
HELPER_USE = '\tUSE SET_KIND_MODULE\n'
HELPER_MASK = '\tUSE LINE_MOD, ONLY: LINE_STORAGE_USED, WEAK_LINE\n'
OLD_LOOP = ('\tDO L=1,M\n\t  DO K=1,NX\n'
            '\t     dZ(J,K,L)=0.0_LDP\n\t  END DO\n\tEND DO\n')
NEW_LOOP = ('\tDO L=1,M\n'
            '\t  IF(LINE_STORAGE_USED(L) .AND. .NOT. WEAK_LINE(L))THEN\n'
            '\t    DO K=1,NX\n\t       dZ(J,K,L)=0.0_LDP\n'
            '\t    END DO\n\t  END IF\n\tEND DO\n')


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('검토한 dZ 변경 위치가 아님')
    return source.replace(old, new, 1)


def verify(source, name):
    expected = SEALED[name]
    if expected and hashlib.sha256(source).hexdigest() != expected:
        raise ValueError('검토한 native dZ 소스 신원 변경: ' + name)


def helper(source):
    if source.count(HELPER_START) != 1:
        raise ValueError('native ZERO_dZ 정의가 하나가 아님')
    return source[source.index(HELPER_START):]


def initializer(source):
    verify(source, 'new_main/mod_subs/init_line_opac_var_v2.f')
    text = source.decode()
    start = text.index(HELPER_START)
    body = once(text[start:], HELPER_USE, HELPER_USE + HELPER_MASK)
    body = once(body, OLD_LOOP, NEW_LOOP)
    # 실제 새 선의 전체 컬럼 reset은 이 함수의 수정 범위 밖이다.
    return text[:start] + body


def retirement_block(row, corrected=False):
    """실제 main의 행 해제 블록. helper 호출은 WORKSHARE의 barrier 뒤다."""
    if row not in ('NL', 'NUP'):
        raise ValueError('검토한 행 이름만 허용')
    clear = f'\t        dZ({row},:,:,:)=0.0_LDP\t!NM,NUM_BNDS,ND,MAX_SIM\n'
    call = (f'\t        CALL ZERO_dZ(dZ,{row},NM,NUM_BNDS*ND,MAX_SIM)\n'
            if corrected else '')
    return (f'\t      IF(VAR_IN_USE_CNT({row}) .EQ. 0)THEN\n'
            '!$OMP PARALLEL WORKSHARE\n'
            f'\t        TX(:,:,{row})=0.0_LDP\t!ND,ND,NM\n'
            f'\t        TVX(:,:,{row})=0.0_LDP\t!ND-1,ND,NM\n'
            + ('' if corrected else clear) +
            f'\t        VAR_LEV_ID({row})=0\n'
            '!$OMP END PARALLEL WORKSHARE\n' + call +
            '\t      END IF\n')


def main_source(source):
    # actual-nu와 첫 BA 수정 후의 main에도 같은 원 행 해제 블록이 남아 있다.
    for row in ('NL', 'NUP'):
        source = once(source, retirement_block(row), retirement_block(row, True))
    return source
