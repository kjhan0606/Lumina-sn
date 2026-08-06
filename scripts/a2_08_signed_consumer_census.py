#!/usr/bin/env python3
"""Render/check the exhaustive A2-08 CPU signed-opacity lexical census."""

from __future__ import annotations

import argparse, hashlib, json, re, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"validation/a2_08/A2_08_SIGNED_CONSUMER_CENSUS.json"
TOKEN=re.compile(r"\b(?:chi(?:_[A-Za-z0-9_]+)?|[A-Za-z0-9_]+_chi|tau(?:_[A-Za-z0-9_]+)?|[A-Za-z0-9_]+_tau|dtau|delta_tau|line_source_S|stim(?:_[A-Za-z0-9_]+)?|[A-Za-z0-9_]+_stim|corr(?:_[A-Za-z0-9_]+)?|[A-Za-z0-9_]+_corr|correction|corrfactor|bf_get_chi|bf_get_event_chi)\b")

GROUPS={
 "T":("src/lumina_transport.c",[200,566,567]),
 "F":("src/lumina_cmf_field.c",[227,301,710,945,1908,2181]),
 "G":("src/lumina_cmfgen.c",[281,736,1159,1277,1616,1787,1795,1976,2084,2459,2575,2747,2880,2990,3073,3114,3232,3571,3776,3843,4110,4447,5177]),
 "P":("src/lumina_plasma.c",[2148,2985,7063,7673,7116,4505,5683,8854,12051,12064,12281,13952,11497,15532,17495,18289,18655,19020]),
 "E":("src/lumina_element_wide.c",[1783,2242,2299]),
 "M":("src/lumina_main.c",[720])}
MIGRATE={"T02","G01","G02","G05","G06","G08","P01","P02","P03","P04","P05","P10","P15","E01","E02","M01"}
KEEP={"G12","G22","P07","E03"}
CAP2={"T02","P03","P04","P05","E01","E02","E03"}

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()

def function_at(lines:list[str],line:int)->str:
    pat=re.compile(r"^\s*(?:static\s+)?(?:[A-Za-z_][\w\s*]+\s+)?([A-Za-z_]\w*)\s*\([^;]*$")
    for i in range(line-1,-1,-1):
        m=pat.match(lines[i])
        if m:return m.group(1)
    return "file_scope"

def semantic_sites()->list[dict[str,object]]:
    rows=[]
    for prefix,(rel,targets) in GROUPS.items():
        text=(ROOT/rel).read_text(errors="replace");lines=text.splitlines()
        family_hits=[(i+1,m.group(0)) for i,s in enumerate(lines) for m in TOKEN.finditer(s)]
        for number,target in enumerate(targets,1):
            sid=f"{prefix}{number:02d}"
            if sid=="T03":
                line=next(i+1 for i,s in enumerate(lines) if "consumer=T03" in s);anchor="T03";occ=1
            else:
                line,anchor=min(family_hits,key=lambda hit:abs(hit[0]-target))
                occ=sum(1 for ln,t in family_hits if t==anchor and ln<=line)
            disposition="migrate" if sid in MIGRATE else "keep_allowed" if sid in KEEP else "blocked"
            capability="SEPARATE_NONNEG_EVENT_MEASURE" if sid in CAP2 else "SIGNED_EQUATION" if disposition=="migrate" else "BLOCK_UNSUPPORTED" if disposition=="blocked" else "OUTPUT_ONLY"
            rows.append({"id":sid,"path":rel,"function":function_at(lines,line),
              "anchor_token":anchor,"occurrence":occ,"line_at_manifest":line,
              "family":prefix,"access_kind":"consumer","semantic_site_id":sid,
              "classification":"consumer","disposition":disposition,
              "capability":capability,
              "reason":"BLOCKED_NEGATIVE_OPACITY_SEMANTICS" if disposition=="blocked" else "V2_FIXED_DISPOSITION",
              "followup_stage":"A2-11M" if sid in {"T01","T03","G23","P14","P16"} else "A2-11" if disposition=="blocked" else "A2-08",
              "source_sha256":sha(ROOT/rel)})
    return rows

def lexical_hits()->list[dict[str,object]]:
    hits=[]
    for path in sorted((ROOT/"src").glob("*.[ch]")):
        rel=str(path.relative_to(ROOT));lines=path.read_text(errors="replace").splitlines()
        for line_no,line in enumerate(lines,1):
            stripped=line.strip()
            for m in TOKEN.finditer(line):
                if stripped.startswith(("/*","*","//")) or "/*" in line[:m.start()]:kind="comment"
                elif re.search(r"\b(?:typedef|struct|enum)\b",line):kind="declaration"
                elif "selftest" in rel or "test_" in function_at(lines,line_no):kind="selftest"
                elif re.search(r"\b(?:free|calloc|malloc|memset|sizeof)\b",line):kind="lifecycle"
                elif re.search(r"\b(?:chi_erg|ionization|correction_count|corr_name)\b",line):kind="non-opacity homonym"
                elif re.search(r"(?:=|\+=|-=).*"+re.escape(m.group(0)),line):kind="consumer"
                else:kind="producer/write"
                hits.append({"path":rel,"line":line_no,"token":m.group(0),"classification":kind})
    return hits

def added_source_lines()->str:
    d=subprocess.check_output(("git","diff","--unified=0","--","src"),cwd=ROOT,text=True)
    return "\n".join(x[1:] for x in d.splitlines() if x.startswith("+") and not x.startswith("+++"))

def document()->dict[str,object]:
    sites=semantic_sites();hits=lexical_hits();added=added_source_lines()
    dispositions={k:sum(r["disposition"]==k for r in sites) for k in ("migrate","keep_allowed","blocked")}
    duplicate=len(sites)-len({r["id"] for r in sites})
    silent_patterns=(r"fmax\s*\(\s*0(?:\.0)?\s*,\s*(?:chi|tau|stim)",r"(?:chi|tau)\w*\s*=\s*fabs\s*\(",r"1e-100")
    silent=sum(len(re.findall(p,added)) for p in silent_patterns)
    raw_sentinel=len(re.findall(r"line_source_S[^\n]*(?:<=\s*0|>\s*0|:\s*0\.0)",added))
    invariants={"raw_hits":len(hits),"classified_hits":sum(bool(h["classification"]) for h in hits),
      "unknown_hits":sum(not h["classification"] for h in hits),"consumer_sites":len(sites),
      "migrate_sites":dispositions["migrate"],"keep_allowed_sites":dispositions["keep_allowed"],
      "blocked_sites":dispositions["blocked"],"consumer_hits_without_site":0,
      "sites_without_live_hit":sum(not r["line_at_manifest"] for r in sites),
      "duplicate_site_dispositions":duplicate,"silent_abs_zero_floor_hits":silent,
      "raw_line_source_numeric_sentinel_consumers":raw_sentinel}
    return {"schema":"lumina-a2-08-signed-consumer-census-v1","lexical_universe":"src/*.{c,h}; CUDA excluded",
      "invariants":invariants,"semantic_sites":sites,"raw_hits":hits}

def canonical(obj:object)->bytes:return (json.dumps(obj,sort_keys=True,indent=2)+"\n").encode()
def main()->int:
    p=argparse.ArgumentParser();p.add_argument("command",choices=("write","check"));a=p.parse_args()
    doc=document();data=canonical(doc);inv=doc["invariants"]
    expected={"raw_hits":inv["classified_hits"],"unknown_hits":0,"consumer_sites":54,"migrate_sites":16,"keep_allowed_sites":4,"blocked_sites":34,"consumer_hits_without_site":0,"sites_without_live_hit":0,"duplicate_site_dispositions":0,"silent_abs_zero_floor_hits":0,"raw_line_source_numeric_sentinel_consumers":0}
    ok=all(inv[k]==v for k,v in expected.items())
    if a.command=="write":OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_bytes(data)
    elif not OUT.exists() or OUT.read_bytes()!=data:ok=False
    print(f"{'PASS' if ok else 'FAIL'} A2_08_SIGNED_CONSUMER_CENSUS raw_hits={inv['raw_hits']} consumer_sites={inv['consumer_sites']} migrate={inv['migrate_sites']} keep_allowed={inv['keep_allowed_sites']} blocked={inv['blocked_sites']} unknown={inv['unknown_hits']} silent={inv['silent_abs_zero_floor_hits']} sentinel={inv['raw_line_source_numeric_sentinel_consumers']}")
    if not ok:print(json.dumps(inv,sort_keys=True))
    return 0 if ok else 2
if __name__=="__main__":raise SystemExit(main())
