#!/usr/bin/env python3
"""
oscal.py — flatten the NIST SP 800-53 OSCAL catalog into a control table.

The catalog is the target side of every mapping the system produces. It is not
chunked as text: OSCAL already carries the structure that would otherwise have
to be recovered from a PDF — family, identifier, statement, organization-defined
parameters, related controls. Parsing it as data instead of prose is the whole
reason for choosing the JSON release.

Two details this handles that a naive parse gets wrong:

  Withdrawn controls  Rev.5 retains withdrawn controls in the catalog as
                      tombstones. They must never be offered as mapping
                      targets, so they are flagged and excluded by default.
  Statement prose     Prose is nested arbitrarily deep in `parts`, with
                      parameter placeholders inline. Both are resolved.

Subcommands
-----------
  flatten    catalog JSON -> processed/controls.jsonl  (+ summary)
  families   the family list — the vocabulary for the family router
  show       inspect one control
  params     controls carrying organization-defined parameters

  python oscal.py flatten
  python oscal.py families
  python oscal.py show AC-7
  python oscal.py params --family AC
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

KB = Path(__file__).resolve().parent
DEFAULT_CATALOG = KB / "raw" / "USA-NIST-SP80053-2020.json"
OUT = KB / "processed" / "controls.jsonl"

# {{ insert: param, ac-01_odp.01 }}
PARAM_REF = re.compile(r"\{\{\s*insert:\s*param,\s*([^}\s]+)\s*\}\}")


# --------------------------------------------------------------------------
def _prop(obj: Dict[str, Any], name: str, exclude_class: Optional[str] = None) -> Optional[str]:
    """First matching prop value. Rev.5 carries two `label` props — plain and
    zero-padded — so the padded variant is skipped to keep IDs as printed."""
    for p in obj.get("props") or []:
        if p.get("name") != name:
            continue
        if exclude_class and p.get("class") == exclude_class:
            continue
        return p.get("value")
    return None


def _param_label(param: Dict[str, Any]) -> str:
    """Render an ODP as its label, or as its choice set when it is a selection."""
    if param.get("label"):
        return str(param["label"])
    sel = param.get("select") or {}
    choices = [c if isinstance(c, str) else c.get("choice", "") for c in sel.get("choice") or []]
    choices = [" ".join(str(c).split()) for c in choices if c]
    if choices:
        how = sel.get("how-many", "one")
        return f"[select {how}: " + " | ".join(choices) + "]"
    return param.get("id", "")


# The public OSCAL release merges SP 800-53 controls with SP 800-53A assessment
# procedures in one file. Assessment parts sit as siblings of the statement and
# are several times longer than it — lists of documents to examine and roles to
# interview. They are not the requirement and must never reach the statement.
SKIP_PARTS = {"assessment-objective", "assessment-method", "objects", "assessment"}


def _collect_prose(parts: Optional[Iterable[Dict[str, Any]]], names: set, depth: int = 0) -> List[str]:
    """Depth-first prose collection, keeping the item labels that give the
    statement its a./b./1. structure.

    Descends into a subtree only once it has matched one of `names`, so a part
    of an unrelated kind cannot pull its whole subtree in with it.
    """
    out: List[str] = []
    for part in parts or []:
        name = part.get("name")
        if name in SKIP_PARTS:
            continue
        matched = name in names
        if matched or depth > 0:
            label = _prop(part, "label")
            prose = (part.get("prose") or "").strip()
            if prose:
                out.append(f"{label} {prose}".strip() if label else prose)
            out.extend(_collect_prose(part.get("parts"), names, depth + 1))
        else:
            # still searching for a matching part; do not treat what we pass
            # through as content
            out.extend(_collect_prose(part.get("parts"), names, 0))
    return out


def _resolve_params(text: str, params: Dict[str, str]) -> str:
    """Substitute parameter placeholders, repeatedly.

    A select-type ODP can embed a reference to another parameter inside one of
    its choices, so one pass is not enough. Bounded to avoid looping on any
    circular reference in the source.
    """
    def sub(m):
        pid = m.group(1)
        return f"[{params.get(pid, pid)}]"

    for _ in range(4):
        new = PARAM_REF.sub(sub, text)
        if new == text:
            break
        text = new
    return text


def _related(control: Dict[str, Any]) -> List[str]:
    out = []
    for link in control.get("links") or []:
        if link.get("rel") == "related":
            out.append(str(link.get("href", "")).lstrip("#").upper())
    return out


# --------------------------------------------------------------------------
def walk_control(
    control: Dict[str, Any],
    family_id: str,
    family_title: str,
    parent: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    label = _prop(control, "label", exclude_class="zero-padded") or control.get("id", "").upper()
    status = _prop(control, "status")
    withdrawn = status == "withdrawn"

    raw_params = {p["id"]: _param_label(p) for p in control.get("params") or [] if p.get("id")}
    params = {k: _resolve_params(v, raw_params) for k, v in raw_params.items()}
    prose_parts = _collect_prose(control.get("parts"), {"statement"})
    statement = _resolve_params("\n".join(prose_parts), params)
    guidance = _resolve_params(
        "\n".join(_collect_prose(control.get("parts"), {"guidance"})), params
    )

    rows.append(
        {
            "control_id": label,
            "oscal_id": control.get("id"),
            "family": family_id.upper(),
            "family_title": family_title,
            "title": control.get("title", ""),
            "is_enhancement": parent is not None,
            "parent_id": parent,
            "withdrawn": withdrawn,
            "statement": statement,
            "guidance": guidance,
            "param_ids": list(params.keys()),
            "params": list(params.values()),
            "n_params": len(params),
            "related": _related(control),
        }
    )

    for child in control.get("controls") or []:
        rows.extend(walk_control(child, family_id, family_title, parent=label))
    return rows


def flatten(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found.")
    data = json.loads(path.read_text(encoding="utf-8"))
    catalog = data.get("catalog") or data
    meta = catalog.get("metadata", {})
    print(f"catalog : {meta.get('title','(untitled)')}")
    print(f"version : {meta.get('version','?')}   oscal {meta.get('oscal-version','?')}")
    print(f"modified: {meta.get('last-modified','?')}\n")

    rows: List[Dict[str, Any]] = []
    for group in catalog.get("groups") or []:
        gid = group.get("id", "??")
        gtitle = group.get("title", "")
        for control in group.get("controls") or []:
            rows.extend(walk_control(control, gid, gtitle))
    return rows


# --------------------------------------------------------------------------
def _load_rows(include_withdrawn: bool = False) -> List[Dict[str, Any]]:
    if not OUT.exists():
        sys.exit("ERROR: controls.jsonl not found. Run `python oscal.py flatten` first.")
    rows = [json.loads(l) for l in OUT.open(encoding="utf-8") if l.strip()]
    if not include_withdrawn:
        rows = [r for r in rows if not r["withdrawn"]]
    return rows


def cmd_flatten(args) -> int:
    rows = flatten(Path(args.catalog))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    live = [r for r in rows if not r["withdrawn"]]
    base = [r for r in live if not r["is_enhancement"]]
    enh = [r for r in live if r["is_enhancement"]]
    wd = [r for r in rows if r["withdrawn"]]
    with_params = [r for r in live if r["n_params"]]
    no_stmt = [r for r in live if not r["statement"].strip()]

    print(f"families            : {len({r['family'] for r in rows})}")
    print(f"base controls       : {len(base)}")
    print(f"enhancements        : {len(enh)}")
    print(f"active total        : {len(live)}")
    print(f"withdrawn (excluded): {len(wd)}")
    print(f"with ODPs           : {len(with_params)} "
          f"({100*len(with_params)//max(len(live),1)}% of active)")
    print(f"total ODPs          : {sum(r['n_params'] for r in live)}")
    if no_stmt:
        print(f"WARNING: {len(no_stmt)} active control(s) have no statement prose")
        for r in no_stmt[:5]:
            print(f"         {r['control_id']}")
    print(f"\nwrote {len(rows)} rows -> {OUT.relative_to(KB)}")
    print("\nWithdrawn controls are kept in the file but flagged. They are")
    print("tombstones in Rev.5 and must never be offered as mapping targets.")
    return 0


def cmd_families(args) -> int:
    rows = _load_rows()
    print("The family router selects from this vocabulary.\n")
    print(f"{'':<5} {'family':<6} {'base':>5} {'enh':>5} {'ODPs':>5}  title")
    per = Counter(r["family"] for r in rows)
    for n, fam in enumerate(sorted(per), 1):
        sub = [r for r in rows if r["family"] == fam]
        b = sum(1 for r in sub if not r["is_enhancement"])
        e = len(sub) - b
        p = sum(r["n_params"] for r in sub)
        title = sub[0]["family_title"]
        print(f"{n:>4}. {fam:<6} {b:>5} {e:>5} {p:>5}  {title}")
    print(f"\n{len(per)} families, {len(rows)} active controls")
    return 0


def cmd_show(args) -> int:
    rows = _load_rows(include_withdrawn=True)
    want = args.control_id.upper()
    hits = [r for r in rows if r["control_id"].upper() == want]
    if not hits:
        near = [r["control_id"] for r in rows if want in r["control_id"].upper()][:10]
        print(f"not found: {want}" + (f"   did you mean: {', '.join(near)}" if near else ""))
        return 1
    for r in hits:
        print("=" * 66)
        print(f"{r['control_id']} — {r['title']}")
        print(f"family    : {r['family']} ({r['family_title']})")
        print(f"type      : {'enhancement of ' + r['parent_id'] if r['is_enhancement'] else 'base control'}")
        if r["withdrawn"]:
            print("status    : WITHDRAWN — not a valid mapping target")
        print(f"\nstatement :\n{r['statement'] or '(none)'}")
        if r["params"]:
            print(f"\norganization-defined parameters ({r['n_params']}):")
            for pid, lbl in zip(r["param_ids"], r["params"]):
                print(f"  - {pid}: {lbl}")
            print("\n  Where a Saudi instrument fixes a concrete value for one of")
            print("  these, it is instantiating an open NIST parameter — record it")
            print("  as a parameter-gap finding, not as a plain match.")
        if r["related"]:
            print(f"\nrelated   : {', '.join(r['related'])}")
    return 0


def cmd_params(args) -> int:
    rows = [r for r in _load_rows() if r["n_params"]]
    if args.family:
        rows = [r for r in rows if r["family"] == args.family.upper()]
    rows.sort(key=lambda r: -r["n_params"])
    print(f"{len(rows)} control(s) carry organization-defined parameters\n")
    for r in rows[: args.limit]:
        print(f"{r['control_id']:<12} {r['n_params']:>2} ODP  {r['title'][:52]}")
        for lbl in r["params"][:3]:
            print(f"             · {' '.join(str(lbl).split())[:88]}")
    if len(rows) > args.limit:
        print(f"\n… {len(rows) - args.limit} more (raise --limit)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("flatten", help="catalog JSON -> controls.jsonl")
    p.add_argument("--catalog", default=str(DEFAULT_CATALOG))

    sub.add_parser("families", help="family list for the router")

    p = sub.add_parser("show", help="inspect one control")
    p.add_argument("control_id")

    p = sub.add_parser("params", help="controls with ODPs")
    p.add_argument("--family")
    p.add_argument("--limit", type=int, default=25)

    args = ap.parse_args()
    return {
        "flatten": cmd_flatten,
        "families": cmd_families,
        "show": cmd_show,
        "params": cmd_params,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
