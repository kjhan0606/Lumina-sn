"""이번 실제 원자 입력의 표 밖 충돌 처방을 원자료 헤더에서 확인한다."""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
root=Path(sys.argv[1]);result=[]
with (root/'levels.csv').open() as handle:
    ions={(int(r['atomic_number']),int(r['ion_number'])) for r in csv.DictReader(handle)}
with (root/'coldata_cmfgen_manifest.csv').open() as handle:
    records={(int(r['Z']),int(r['ion0'])):r for r in csv.DictReader(handle)}
for key in sorted(ions):
    row=records[key];path=Path(row['col'])
    assert str(path).startswith('/gpfs/kjhan/cmfgen_21jun23/atomic/')
    data=path.read_bytes();text=data.decode('utf-8')
    values=[line.split('!')[0].strip().replace('D','e').replace('d','e') for line in text.splitlines()
            if re.search(r'Value for OMEGA if f\s*=\s*0',line,re.I)]
    assert len(values)==1 and float(values[0])==.1,(key,values)
    result.append(dict(Z=key[0],charge=key[1],omega_set=float(values[0]),source=str(path),
                       sha256=hashlib.sha256(data).hexdigest()))
with Path('collision_prescriptions.json').open('x') as handle:
    json.dump(result,handle,indent=2);handle.write('\n')
print('COLLISION_PRESCRIPTIONS_CONFIRMED',len(result),'ions OMEGA_SET=0.1')
