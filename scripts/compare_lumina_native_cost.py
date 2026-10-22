"""같은 고정 coarse 표본의 값·반환 오차·관측 비용만 비교한다."""
import argparse
import hashlib
import json
from pathlib import Path

from lumina_linear_native import Positive


def compare(before,after):
    for key in ('baseline_sha256','source_identity','interval','coarse_only','full_frequency','piece','columns'):
        if before.get(key)!=after.get(key):raise ValueError('비교 입력/범위 불일치: '+key)
    if any(x.get('status')!='CPU_SAMPLE_FINISHED_NO_MAP_FAILURES' for x in (before,after)):
        raise ValueError('완료된 실패 없는 두 CPU 표본 필요')
    old={row['column']:row for row in before['rows']};rows=[]
    new={row['column']:row for row in after['rows']}
    if (len(old)!=len(before['columns']) or len(new)!=len(after['rows']) or
        set(old)!=set(new) or set(old)!=set(before['columns'])):raise ValueError('표본 집합/중복 불일치')
    for row in after['rows']:
        previous=old[row['column']]
        p=lambda item,key:Positive(item[key]['m'],item[key]['e'])
        a,b=p(row,'source'),p(previous,'source');error=p(row,'error');old_error=p(previous,'error')
        met=(row['code']==previous['code']==0 and p(row,'t').difference(p(previous,'t'))<=p(previous,'t')*3e-10
             and a.difference(b)<=error+old_error and error<=a*1e-8 and a.difference(b)<=b*3e-10)
        rows.append(dict(column=row['column'],met=met,same_source_bits=row['source']==previous['source'],
                         panels_before=previous['panels'],panels_after=row['panels']))
    return dict(status='CPU_COARSE_COMPARISON_MET_NOT_FIELD' if all(x['met'] for x in rows) else 'CPU_COARSE_NOT_MET',
                rows=rows,query_seconds_before=sum(r['query_seconds'] for r in before['rows']),
                query_seconds_after=sum(r['query_seconds'] for r in after['rows']),
                query_readers_before=sum(r['reader_query'] for r in before['rows']),
                query_readers_after=sum(r['reader_query'] for r in after['rows']),
                limits='공유 CPU 단발 고정 표본 관측이다. GPU/전ν/전체장 ETA·독립 미세참조·CMFGEN 물리 수용이 아니다.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('before',type=Path);parser.add_argument('after',type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    result=compare(json.loads(args.before.read_text()),json.loads(args.after.read_text()))
    result['input_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.before,args.after)}
    with args.output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(result['status'])
    raise SystemExit(0 if result['status']=='CPU_COARSE_COMPARISON_MET_NOT_FIELD' else 1)
