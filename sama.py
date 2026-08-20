#!/usr/bin/env python3
"""
sama.py — parse the SAMA Cyber Security Framework into its official hierarchy.

The chunked text in chunks.jsonl serves retrieval, but it cannot serve mapping:
its boundaries fall at arbitrary character counts and its identifiers are
synthetic. Mapping needs the framework's own units, under the framework's own
identifiers.

Section 2.1 and Figure 1 of the framework document the numbering scheme:

    3.3.12 – 4.a.6.a
    │  │      └── control consideration, up to four levels
    │  └───────── subdomain
    └──────────── domain

That scheme already satisfies the OLIR requirement that every focal document
element carry a unique identifier, so identifiers are extracted, never invented.

Level types alternate strictly — numeric, alpha, numeric, alpha — which is what
makes the hierarchy recoverable at all: the PDF's text layer preserves no
indentation, so depth is inferred from marker type plus sequence continuity.

Three clause types are distinguished, because they are not interchangeable:

    substantive   a requirement in its own right
    stem          introduces a list; meaningless without its children, so it
                  also carries a rolled-up full_text
    referential   points at another document instead of stating a requirement.
                  3.3.12 Payment Systems is entirely of this kind and has no
                  mappable content at all. Counting these as unmatched would
                  understate coverage.

Subcommands
-----------
  parse     text -> processed/sama_clauses.jsonl (+ summary)
  tree      print one subdomain as an indented tree
  show      print one clause by its official id
  params    clauses stating a concrete numeric value or periodicity
  types     clause-type and depth breakdown

  python sama.py parse
  python sama.py tree 3.1.1
  python sama.py show 3.1.1-4.d
  python sama.py params
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

KB = Path(__file__).resolve().parent
SRC = KB / "processed" / "text" / "SAU-SAMA-CSF-2017.txt"
OUT = KB / "processed" / "sama_clauses.jsonl"

PAGE_RE = re.compile(r"<<<PAGE (\d+)>>>\n")

# Running header and footer, stripped before parsing.
NOISE = re.compile(r"^\s*(Version\s+1\.0|Page\s+\d+\s+of\s+\d+)\s*$", re.I)
# The extractor sometimes merges the running header/footer onto the end of a
# content line, so it has to be removed inline too.
NOISE_INLINE = re.compile(r"\s*Version\s+1\.0\s*Page\s+\d+\s+of\s+\d+\s*", re.I)

DOMAIN_RE = re.compile(r"^(3\.\d+)\s+([A-Z][^\n.]{3,80})\s*$")
SUBDOMAIN_RE = re.compile(r"^(3\.\d+\.\d+(?:\.\d+)?)\s+([A-Z(][^\n.]{3,80})\s*$")
SECTION_RE = re.compile(r"^(Principle|Objective|Control\s+[Cc]onsiderations?)\s*$", re.I)

NUM_MARKER = re.compile(r"^(\d{1,2})\.\s+(.*)$")
ALPHA_MARKER = re.compile(r"^([a-z])\.\s+(.*)$")
BULLET = re.compile(r"^[ \t\u2022\uf0b7]+(\S.*)$")

# Concrete values and periodicities. These are the points where SAMA fixes a
# value that NIST leaves as an organization-defined parameter.
PARAM_RE = re.compile(
    r"\b("
    r"quarterly|annual(?:ly)?|monthly|weekly|daily|bi-annual(?:ly)?"
    r"|twice\s+a\s+year|24x7|24/7"
    r"|\d+\s+successive|\d+\s+consecutive"
    r"|at\s+least\s+\d+|\d+\s+days?|\d+\s+months?|\d+\s+years?"
    r")\b",
    re.I,
)

DEONTIC_RE = re.compile(r"\b(should|shall|must|may)\b", re.I)

REFERENTIAL_RE = re.compile(
    r"(please\s+refer\s+to|refer\s+to\s+the|see\s+appendix|as\s+defined\s+in\s+the)", re.I
)


# --------------------------------------------------------------------------
def load_pages() -> List[Tuple[int, str]]:
    if not SRC.exists():
        sys.exit(f"ERROR: {SRC} not found. Run `python fetch.py extract` first.")
    parts = PAGE_RE.split(SRC.read_text(encoding="utf-8"))
    return [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def body_lines() -> List[Tuple[int, str]]:
    """Lines of chapter 3 only, as (page, line). The table of contents repeats
    every heading, so everything before '3 Control domains' is dropped, and
    everything from the appendices on."""
    out: List[Tuple[int, str]] = []
    started = False
    for page, text in load_pages():
        for raw in text.split("\n"):
            line = NOISE_INLINE.sub(" ", raw).rstrip()
            if NOISE.match(line) or not line.strip():
                continue
            if not started:
                if re.match(r"^3\s+Control domains\s*$", line):
                    started = True
                continue
            if re.match(r"^(Appendices|Appendix\s+[A-F])\b", line):
                return out
            out.append((page, line))
    return out


# --------------------------------------------------------------------------
class Numbering:
    """Tracks the open marker sequence at each depth.

    Depth parity is fixed by the framework: 1 and 3 are numeric, 2 and 4 are
    alpha. A marker is first tested as a continuation of an already-open level,
    then as the opening of a new deeper level.
    """

    def __init__(self) -> None:
        self.stack: Dict[int, str] = {}   # depth -> last marker seen
        self.depth = 0

    def reset(self) -> None:
        self.stack.clear()
        self.depth = 0

    @staticmethod
    def _next_num(prev: Optional[str]) -> str:
        return "1" if prev is None else str(int(prev) + 1)

    @staticmethod
    def _next_alpha(prev: Optional[str]) -> str:
        return "a" if prev is None else chr(ord(prev) + 1)

    def place(self, marker: str, numeric: bool) -> int:
        candidates = (3, 1) if numeric else (4, 2)
        nxt = self._next_num if numeric else self._next_alpha

        # continuation of an open level, deepest first
        for d in candidates:
            if d in self.stack and marker == nxt(self.stack[d]):
                self.stack[d] = marker
                for deeper in [k for k in self.stack if k > d]:
                    del self.stack[deeper]
                self.depth = d
                return d

        # opening a new level
        first = marker == ("1" if numeric else "a")
        if first:
            for d in sorted(candidates):
                if d > self.depth:
                    self.stack[d] = marker
                    for deeper in [k for k in self.stack if k > d]:
                        del self.stack[deeper]
                    self.depth = d
                    return d

        # a gap in the sequence: attach to the shallowest legal level
        d = candidates[-1]
        self.stack[d] = marker
        for deeper in [k for k in self.stack if k > d]:
            del self.stack[deeper]
        self.depth = d
        return d

    def path(self, depth: int) -> str:
        return ".".join(self.stack[d] for d in sorted(self.stack) if d <= depth)


# --------------------------------------------------------------------------
def parse() -> List[Dict[str, Any]]:
    clauses: List[Dict[str, Any]] = []
    domain = domain_title = subdomain = subdomain_title = ""
    section = ""
    numbering = Numbering()
    current: Optional[Dict[str, Any]] = None
    prose: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    bullet_n = 0

    def close() -> None:
        nonlocal current
        if current:
            current["text"] = " ".join(current["text"].split())
            clauses.append(current)
            current = None

    for page, line in body_lines():
        if not line.strip():
            continue

        m = SUBDOMAIN_RE.match(line)
        if m:
            close()
            subdomain, subdomain_title = m.group(1), m.group(2).strip()
            section = ""
            numbering.reset()
            bullet_n = 0
            continue

        m = DOMAIN_RE.match(line)
        if m and not SUBDOMAIN_RE.match(line):
            close()
            domain, domain_title = m.group(1), m.group(2).strip()
            subdomain = subdomain_title = section = ""
            numbering.reset()
            continue

        m = SECTION_RE.match(line)
        if m:
            close()
            head = m.group(1).lower()
            section = "control_considerations" if head.startswith("control") else head
            numbering.reset()
            bullet_n = 0
            continue

        if section != "control_considerations":
            if subdomain and section in ("principle", "objective"):
                prose[(subdomain, section)].append(line.strip())
            continue

        mn = NUM_MARKER.match(line)
        ma = ALPHA_MARKER.match(line)

        if mn or ma:
            close()
            marker, text = (mn or ma).group(1), (mn or ma).group(2)
            depth = numbering.place(marker, numeric=bool(mn))
            path = numbering.path(depth)
            parent = ".".join(path.split(".")[:-1])
            current = {
                "clause_id": f"{subdomain}-{path}",
                "domain": domain,
                "domain_title": domain_title,
                "subdomain": subdomain,
                "subdomain_title": subdomain_title,
                "level": depth,
                "marker": marker,
                "path": path,
                "parent_id": f"{subdomain}-{parent}" if parent else None,
                "text": text,
                "page": page,
                "clause_type": "substantive",
            }
            continue

        mb = BULLET.match(line)
        if mb and numbering.depth == 0:
            # A bullet where a numbered item was expected: this subdomain
            # delegates to other documents instead of stating requirements.
            bullet_n += 1
            close()
            current = {
                "clause_id": f"{subdomain}-b{bullet_n}",
                "domain": domain,
                "domain_title": domain_title,
                "subdomain": subdomain,
                "subdomain_title": subdomain_title,
                "level": 1,
                "marker": f"b{bullet_n}",
                "path": f"b{bullet_n}",
                "parent_id": None,
                "text": mb.group(1),
                "page": page,
                "clause_type": "referential",
            }
            continue

        if current:                       # wrapped continuation line
            current["text"] += " " + line.strip()

    close()
    return _enrich(clauses, prose)


def _enrich(clauses: List[Dict[str, Any]], prose) -> List[Dict[str, Any]]:
    by_id = {c["clause_id"]: c for c in clauses}
    children: Dict[str, List[str]] = defaultdict(list)
    for c in clauses:
        if c["parent_id"]:
            children[c["parent_id"]].append(c["clause_id"])

    for c in clauses:
        cid = c["clause_id"]
        kids = children.get(cid, [])
        c["n_children"] = len(kids)
        c["child_ids"] = kids

        if kids and c["text"].rstrip().endswith((":", ";")):
            c["clause_type"] = "stem"
        elif c["clause_type"] != "referential" and REFERENTIAL_RE.search(c["text"]):
            c["clause_type"] = "referential"

        d = DEONTIC_RE.search(c["text"])
        c["deontic"] = d.group(1).lower() if d else None
        c["deontic_own"] = c["deontic"]

        params = sorted({" ".join(m.group(1).split()).lower() for m in PARAM_RE.finditer(c["text"])})
        c["numeric_params"] = params
        c["has_numeric_param"] = bool(params)

        sd = c["subdomain"]
        c["principle"] = " ".join(prose.get((sd, "principle"), []))
        c["objective"] = " ".join(prose.get((sd, "objective"), []))

    # A sub-item carries no verb of its own; its force comes from the stem
    # above it. Without this, most level-2 and level-4 clauses would look
    # non-normative, which they are not.
    for c in sorted(clauses, key=lambda x: x["level"]):
        if c["deontic"] is None and c["parent_id"] in by_id:
            c["deontic"] = by_id[c["parent_id"]]["deontic"]
            c["deontic_inherited"] = True
        else:
            c["deontic_inherited"] = False

    # full_text rolls a stem together with its descendants, because a stem on
    # its own carries no mappable requirement.
    def roll(cid: str, depth: int = 0) -> str:
        c = by_id[cid]
        out = [c["text"]]
        for kid in children.get(cid, []):
            out.append(roll(kid, depth + 1))
        return " ".join(out)

    for c in clauses:
        c["full_text"] = " ".join(roll(c["clause_id"]).split())
    return clauses


# --------------------------------------------------------------------------
def _load() -> List[Dict[str, Any]]:
    if not OUT.exists():
        sys.exit("ERROR: sama_clauses.jsonl not found. Run `python sama.py parse` first.")
    return [json.loads(l) for l in OUT.open(encoding="utf-8") if l.strip()]


def cmd_parse(args) -> int:
    clauses = parse()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for c in clauses:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    subs = {c["subdomain"] for c in clauses}
    lv = Counter(c["level"] for c in clauses)
    ty = Counter(c["clause_type"] for c in clauses)
    mappable = [c for c in clauses if c["clause_type"] != "referential"]
    leaves = [c for c in mappable if not c["n_children"]]

    print(f"domains          : {len({c['domain'] for c in clauses})}")
    print(f"subdomains       : {len(subs)}")
    print(f"clauses total    : {len(clauses)}")
    print(f"\nby level")
    for d in sorted(lv):
        print(f"  level {d}        {lv[d]:>4}")
    print(f"\nby type")
    for t, n in ty.most_common():
        print(f"  {t:<14} {n:>4}")
    print(f"\nmappable units   : {len(mappable)}  (referential excluded)")
    print(f"leaf units       : {len(leaves)}  (the mapping source set)")

    dn = Counter(c["deontic"] for c in clauses)
    print(f"\ndeontic          : " + ", ".join(f"{k}={v}" for k, v in dn.most_common()))
    par = [c for c in clauses if c["has_numeric_param"]]
    print(f"stating a value  : {len(par)} clause(s)")

    ref_subs = sorted({c["subdomain"] for c in clauses if c["clause_type"] == "referential"})
    if ref_subs:
        print(f"\nsubdomains with referential content: {', '.join(ref_subs)}")

    empty = sorted(subs - {c["subdomain"] for c in clauses})
    print(f"\nwrote {len(clauses)} clauses -> {OUT.relative_to(KB)}")
    return 0


def cmd_tree(args) -> int:
    clauses = [c for c in _load() if c["subdomain"] == args.subdomain]
    if not clauses:
        subs = sorted({c["subdomain"] for c in _load()})
        print(f"not found: {args.subdomain}\navailable: {', '.join(subs)}")
        return 1
    head = clauses[0]
    print(f"{head['subdomain']} {head['subdomain_title']}   ({head['domain_title']})")
    if head["principle"]:
        print(f"\nPrinciple : {head['principle'][:200]}")
    if head["objective"]:
        print(f"Objective : {head['objective'][:200]}")
    print()
    for c in clauses:
        pad = "    " * (c["level"] - 1)
        tag = {"stem": "¶", "referential": "→", "substantive": " "}[c["clause_type"]]
        flag = " ⚑" if c["has_numeric_param"] else ""
        body = c["text"] if len(c["text"]) <= 100 else c["text"][:97] + "…"
        print(f"{tag} {pad}{c['marker']}. {body}{flag}")
        print(f"  {pad}   [{c['clause_id']}]")
    return 0


def cmd_show(args) -> int:
    hits = [c for c in _load() if c["clause_id"].lower() == args.clause_id.lower()]
    if not hits:
        print(f"not found: {args.clause_id}")
        return 1
    c = hits[0]
    print("=" * 66)
    print(f"{c['clause_id']}   level {c['level']}   {c['clause_type']}")
    print(f"{c['subdomain']} {c['subdomain_title']}  ({c['domain_title']})   p{c['page']}")
    print(f"\ntext      : {c['text']}")
    if c["n_children"]:
        print(f"\nfull_text : {c['full_text']}")
        print(f"children  : {', '.join(c['child_ids'])}")
    if c["parent_id"]:
        print(f"parent    : {c['parent_id']}")
    print(f"deontic   : {c['deontic']}")
    if c["numeric_params"]:
        print(f"values    : {', '.join(c['numeric_params'])}")
        print("\n  A concrete value here is a candidate parameter-gap finding:")
        print("  check whether the matching NIST control leaves it as an ODP.")
    return 0


def cmd_params(args) -> int:
    rows = [c for c in _load() if c["has_numeric_param"]]
    print(f"{len(rows)} clause(s) state a concrete value or periodicity\n")
    for c in rows:
        print(f"{c['clause_id']:<18} {', '.join(c['numeric_params'])}")
        print(f"{'':<18} {' '.join(c['text'].split())[:96]}")
    print("\nEach is a candidate against a NIST control carrying an open ODP.")
    print("Cross-check with:  python oscal.py params")
    return 0


def cmd_types(args) -> int:
    clauses = _load()
    print(f"{'subdomain':<12} {'title':<42} {'subst':>6} {'stem':>5} {'ref':>4}")
    per: Dict[str, Counter] = defaultdict(Counter)
    titles: Dict[str, str] = {}
    for c in clauses:
        per[c["subdomain"]][c["clause_type"]] += 1
        titles[c["subdomain"]] = c["subdomain_title"]
    for sd in sorted(per, key=lambda s: [int(x) for x in s.split(".")]):
        k = per[sd]
        print(f"{sd:<12} {titles[sd][:42]:<42} {k['substantive']:>6} "
              f"{k['stem']:>5} {k['referential']:>4}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("parse", help="text -> sama_clauses.jsonl")
    p = sub.add_parser("tree", help="print a subdomain as a tree")
    p.add_argument("subdomain")
    p = sub.add_parser("show", help="print one clause")
    p.add_argument("clause_id")
    sub.add_parser("params", help="clauses stating a concrete value")
    sub.add_parser("types", help="clause-type breakdown per subdomain")
    args = ap.parse_args()
    return {"parse": cmd_parse, "tree": cmd_tree, "show": cmd_show,
            "params": cmd_params, "types": cmd_types}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
