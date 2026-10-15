#ifndef LUMINA_NLTE_LINE_INDEX_H
#define LUMINA_NLTE_LINE_INDEX_H
#include "lumina.h"

/* 불변 선 소유권만 재사용한다. 두 이온의 목록을 전역 line ID 순으로 병합해
 * 기존 전체 순회와 부동소수점 가산 순서를 보존한다. 일반 범위는 기존 경로다. */
typedef struct {
    const NLTEConfig *nlte;
    int lo, hi, count, scan, indexed;
    int pos[2], end[2];
} NLTEPairLineIterator;

static inline NLTEPairLineIterator nlte_pair_line_iterator(
        const NLTEConfig *n,int lo,int hi,int count) {
    NLTEPairLineIterator it={0};
    it.nlte=n;it.lo=lo;it.hi=hi;it.count=count;
    it.indexed=n->nlte_ion_line_offsets && n->nlte_ion_lines &&
        n->nlte_line_index_count==count && lo>=0 && hi>=lo &&
        hi<n->n_nlte_ions && hi-lo<=1;
    if(it.indexed){
        it.pos[0]=n->nlte_ion_line_offsets[lo];
        it.end[0]=n->nlte_ion_line_offsets[lo+1];
        if(hi!=lo){
            it.pos[1]=n->nlte_ion_line_offsets[hi];
            it.end[1]=n->nlte_ion_line_offsets[hi+1];
        }
    }
    return it;
}

static inline int nlte_pair_line_next(NLTEPairLineIterator *it) {
    if(!it->indexed){
        while(it->scan<it->count){
            int line=it->scan++,map=it->nlte->nlte_line_map[line];
            if(map>=it->lo && map<=it->hi)return line;
        }
        return -1;
    }
    int line=it->count,owner=-1;
    for(int k=0;k<2;k++)if(it->pos[k]<it->end[k]){
        int candidate=it->nlte->nlte_ion_lines[it->pos[k]];
        if(candidate<line){line=candidate;owner=k;}
    }
    if(owner<0)return -1;
    it->pos[owner]++;
    return line;
}
#endif
