#!/usr/bin/env python3
"""기존 BF 명세/v5 이진 입력에 원자료의 재결합 선택을 함께 봉인한다.

독립 DIE를 지원하지 않는 경로에서 활성 선택을 외부 fit으로 대체하지 않는다.
구형 v4의 수치 배열은 그대로 두고, 선택물리만 명시하는 이관도 지원한다.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import struct

POLICY_MAGIC = 0x43525035
POLICY_VERSION = 1
POLICY_SCHEMA = "cmfgen-recombination-policy-v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def build_policy(model, model_map, link_map, route_sources, ion_keys):
    model = Path(model).resolve(strict=True)
    text = model.read_text(encoding="latin-1")
    records = []
    for Z, stage in sorted(ion_keys):
        if Z not in (14, 16) or stage not in range(2, 6):
            continue
        label = model_map[(Z, stage)]["label"]
        tag = "DIE_" + label
        values = re.findall(r"^\s*([TF])\s*,\s*([TF])\s+\[" + re.escape(tag) + r"\]", text, re.M)
        if len(values) != 1:
            raise ValueError(f"MODEL 독립 DIE 선택 누락/중복: {tag}")
        flags = [int(v == "T") for v in values[0]]
        if any(flags):
            raise ValueError(f"독립 DIE 활성 선택은 아직 지원하지 않음: {tag}={values[0]}")
        source = route_sources.get(f"Z{Z}_stage{stage}", {}).get("sources")
        if not source:
            path = Path(link_map[(Z, stage)]["phot"]).resolve(strict=True)
            source = [dict(path=str(path), sha256=sha(path))]
        for item in source:
            if sha(item["path"]) != item["sha256"]:
                raise ValueError(f"PHOT 원자료 변경: {item['path']}")
        records.append(dict(Z=Z, lower_ion_number=stage-1, upper_ion_number=stage,
            model_ion_label=label, model_key=tag, die_auto=flags[0], die_wi=flags[1],
            external_dr_fit_allowed=0, treatment="PHOT_MILNE_NO_EXTERNAL_FIT",
            phot_sources=source, phot_sources_sha256=hashlib.sha256(
                json.dumps(source, sort_keys=True, separators=(",", ":")).encode()).hexdigest()))
    result = dict(schema=POLICY_SCHEMA, model=dict(path=str(model), sha256=sha(model)), ions=records)
    encode_policy(result)
    return result


def encode_policy(policy):
    if policy.get("schema") != POLICY_SCHEMA or not 1 <= len(policy.get("ions", [])) <= 8:
        raise ValueError("재결합 정책 형식/경계 수 오류")
    model_hash = bytes.fromhex(policy["model"]["sha256"])
    if len(model_hash) != 32 or not any(model_hash):
        raise ValueError("MODEL 정체성 누락")
    result = bytearray(struct.pack("<III32s", POLICY_MAGIC, POLICY_VERSION, len(policy["ions"]), model_hash))
    seen = set()
    for row in policy["ions"]:
        key = (int(row["Z"]), int(row["lower_ion_number"]))
        if key in seen or key[0] not in (14, 16) or key[1] not in range(1, 5):
            raise ValueError("재결합 경계 중복/범위 오류")
        seen.add(key)
        if (int(row["upper_ion_number"]) != key[1]+1 or row["die_auto"] != 0 or
                row["die_wi"] != 0 or row["external_dr_fit_allowed"] != 0 or
                row["treatment"] != "PHOT_MILNE_NO_EXTERNAL_FIT"):
            raise ValueError("지원하지 않는/모순된 재결합 정책")
        sources = row["phot_sources"]
        digest = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        phot_hash = bytes.fromhex(row["phot_sources_sha256"])
        if not sources or digest != row["phot_sources_sha256"] or len(phot_hash) != 32:
            raise ValueError("PHOT 정체성/정책 불일치")
        for source in sources:
            if not source.get("path") or len(bytes.fromhex(source["sha256"])) != 32:
                raise ValueError("PHOT 경로/해시 누락")
        result += struct.pack("<6i32s", key[0], key[1], key[1]+1, 0, 0, 0, phot_hash)
    return bytes(result)


def validate_tail(raw, policy):
    if not isinstance(policy, dict) or raw != encode_policy(policy):
        raise ValueError("BF 이진 입력과 재결합 명세가 다르거나 불완전함")


def stage_view(deck, run, out):
    # 이관도 기존 명세에 기록된 원자료 정체성에서 정책을 만든다.
    from expand_atomic_data_cmfgen import load_cmfgen_model_spec, load_cmfgen_links
    deck, run, out = Path(deck).resolve(), Path(run).resolve(), Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise ValueError("기존 덱/뷰 보존")
    manifest = json.loads((deck / "cmfgen_bf_channels_manifest.json").read_text())
    channel = deck / "cmfgen_bf_channels.bin"
    if manifest["schema"] != "lumina-cmfgen-bf-channels-v4" or sha(channel) != manifest["channel_bin_sha256"]:
        raise ValueError("봉인 v4 BF 입력/명세 불일치")
    stamp = json.loads((deck / "DECK_PROVENANCE.json").read_text())
    if sha(run / "MODEL") != stamp.get("phot_preprocessing", {}).get("model", {}).get("sha256"):
        raise ValueError("덱 PHOT 전처리와 다른 MODEL 선택")
    for filename, key in (("MODEL_SPEC", "cmfgen_model_spec_sha256"), ("atomic_links.txt", "cmfgen_links_sha256")):
        if sha(run / filename) != stamp[key]:
            raise ValueError("원래 덱과 다른 CMFGEN 모델/원자 링크")
    links = load_cmfgen_links(run / "atomic_links.txt")
    models = load_cmfgen_model_spec(run / "MODEL_SPEC", run / "atomic_links.txt")
    keys = set()
    with (deck / "levels.csv").open() as stream:
        for row in csv.DictReader(stream):
            Z, stage = int(row["atomic_number"]), int(row["ion_number"])+1
            if Z in (14, 16) and stage in range(2, 6):
                keys.add((Z, stage))
    policy = build_policy(run / "MODEL", models, links, manifest["ions"], keys)
    raw = bytearray(channel.read_bytes())
    if struct.unpack_from("<II", raw) != (0x434D4643, 4):
        raise ValueError("v4 전용 이관 입력이 아님")
    # 수치 배열의 범위·불변성을 확인한 뒤 버전과 정책 꼬리만 변경한다.
    nl, nf, nr = struct.unpack_from("<iii", raw, 8)
    size = 36 + 4*(nl+1) + nr*(4+8+4+8+8+8) + 8*nr*nf
    if len(raw) != size:
        raise ValueError("v4 입력 길이/차원 불일치")
    payload_hash = hashlib.sha256(raw[8:]).hexdigest()
    out.mkdir()
    for source in deck.iterdir():
        if source.is_file() and source.name not in ("cmfgen_bf_channels.bin", "cmfgen_bf_channels_manifest.json"):
            (out / source.name).symlink_to(source)
    struct.pack_into("<I", raw, 4, 5)
    raw.extend(encode_policy(policy))
    (out / channel.name).write_bytes(raw)
    manifest.update(schema="lumina-cmfgen-bf-channels-v5", recombination_policy=policy,
        channel_bin_sha256=sha(out / channel.name), inherited_v4_payload_sha256=payload_hash,
        policy_builder_sha256=sha(Path(__file__)), source_view=str(deck))
    (out / "cmfgen_bf_channels_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    return dict(view=str(out), policy_ions=len(policy["ions"]), unchanged_v4_payload_sha256=payload_hash,
                channel_sha256=manifest["channel_bin_sha256"], final_goal="NOT_JUDGED")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(stage_view(args.deck, args.run, args.out), indent=2))


if __name__ == "__main__":
    main()
