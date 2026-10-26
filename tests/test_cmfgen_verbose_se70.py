from types import SimpleNamespace

import pytest

from summarize_cmfgen_verbose_se70 import PHASES, parse, printed_value


def test_printed_value_bound():
    assert printed_value('  1.0000E+05') == (100000.0, 5.0)
    assert printed_value(' -1.2500E-03')[0] == -0.00125


def test_parse_two_phases_at_selected_depth(tmp_path):
    path = tmp_path/'fort.19'
    variables = [SimpleNamespace(index=1, ion='Si2', species='Si', charge=1, terminal=False),
                 SimpleNamespace(index=2, ion='Si2', species='Si', charge=1, terminal=False)]
    lines = []
    for phase_offset, phase in enumerate(PHASES):
        lines.append(' '+phase+'\n')
        for block in range(2):
            lines.append('\n')
            for local in (1, 2, 3):
                tag = '** ' if local == 1 else ' * '
                numbers = ''.join(f'{phase_offset*100+block*10+local+k:12.4E}' for k in range(10))
                lines.append(f' {local:4d}{tag}{numbers}\n')
    path.write_text(''.join(lines))
    stages = parse(path, variables, nd=20, node=18)
    assert stages[PHASES[0]][0] == {('Si', 1): 39.0}
    assert stages[PHASES[1]][0] == {('Si', 1): 239.0}


def test_parse_rejects_missing_phase(tmp_path):
    path = tmp_path/'fort.19'
    path.write_text(' '+PHASES[0]+'\n')
    with pytest.raises(ValueError, match='SE 단계 누락'):
        parse(path, [SimpleNamespace(index=1, ion='Si2', species='Si', charge=1, terminal=False)], nd=10, node=0)


def test_target_goes_to_next_ion_and_balance_is_excluded(tmp_path):
    path = tmp_path/'fort.19'
    variables = [SimpleNamespace(index=1, ion='Si2', species='Si', charge=1, terminal=False),
                 SimpleNamespace(index=2, ion='SiIII', species='Si', charge=2, terminal=False),
                 SimpleNamespace(index=3, ion='SiIV', species='Si', charge=3, terminal=True)]
    lines = []
    for phase in PHASES:
        lines.append(' '+phase+'\n')
        for local, tag, value in ((1, '** ', -5), (2, ' * ', 5), (3, ' * ', 100),
                                  (1, '** ', 0), (2, ' * ', 200)):
            lines.append(f' {local:4d}{tag}'+''.join(f'{value:12.4E}' for _ in range(10))+'\n')
    path.write_text(''.join(lines))
    phases = parse(path, variables, nd=10, node=0)
    for phase in PHASES:
        assert phases[phase][0] == {('Si', 1): -5.0, ('Si', 2): 5.0}
