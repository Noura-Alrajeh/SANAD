#!/usr/bin/env python3
"""
precision.py — measure how many accepted mappings a reviewer would accept.

Every other number in the evaluation measures recall, coverage, or the shape of
the distribution. None of them says whether an accepted finding is *right*. That
question has only one honest answer: read a sample and label it.

    sample   draw a stratified sample of accepted findings -> review.jsonl
    label    show one case at a time, record a verdict
    score    precision with a Wilson interval, broken down by band

The sample is stratified by confidence band, because precision is not expected
to be uniform across them and reporting one pooled number would hide that. It
is drawn with a fixed seed so the same sample can be re-labelled by a second
reviewer and agreement computed.

Labels:
    correct    a domain expert would accept this mapping as stated
    wrong_type the controls are related but the relationship type is wrong
    wrong      not a real relationship
    unsure     the texts do not settle it

`unsure` is reported separately and excluded from the denominator, with the
count stated. Folding uncertain cases into either side would overstate whichever
side received them.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

KB = Path(__file__).resolve().parent
FINDINGS = KB / "processed" / "findings.jsonl"
FINDINGS_V2 = KB / "processed" / "findings_v2.jsonl"
CLAUSES = KB / "processed" / "sama_clauses.jsonl"
CONTROLS = KB / "processed" / "controls.jsonl"
REVIEW = KB / "processed" / "review.jsonl"

LABELS = {"c": "correct", "t": "wrong_type", "w": "wrong", "u": "unsure"}
BANDS = [("confirmed", 0.70, 1.01), ("provisional", 0.55, 0.70), ("weak", 0.0, 0.55)]


def _jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        sys.exit(f"ERROR: {p.name} not found.")
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def band_of(conf: Optional[float]) -> str:
    c = conf or 0.0
    for name, lo, hi in BANDS:
        if lo <= c < hi:
            return name
    return "weak"


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval. With n around 30 the normal approximation is not
    usable, and reporting a bare proportion would imply more precision than a
    sample this size can carry."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------
def cmd_sample(args) -> int:
    rows = _jsonl(FINDINGS_V2 if args.debated and FINDINGS_V2.exists() else FINDINGS)
    debated = args.debated and FINDINGS_V2.exists()

    pool = []
    if debated:
        for r in rows:
            f = r["final"]
            if f["status"] == "accepted":
                pool.append({
                    "clause_id": r["clause_id"], "control_id": f["control_id"],
                    "relationship": f["relationship"], "confidence": f.get("confidence"),
                    "evidence_sama": f.get("evidence_sama", ""),
                    "evidence_nist": f.get("evidence_nist", ""),
                    "rationale": r.get("ruling", {}).get("reasoning", ""),
                    "source": "debated",
                })
    else:
        for r in rows:
            for f in r["findings"]:
                if f["status"] == "accepted":
                    pool.append({
                        "clause_id": r["clause_id"], "control_id": f["control_id"],
                        "relationship": f["relationship"], "confidence": f.get("confidence"),
                        "evidence_sama": f.get("evidence_sama", ""),
                        "evidence_nist": f.get("evidence_nist", ""),
                        "rationale": f.get("rationale", ""),
                        "source": "single-judge",
                    })

    by_band: Dict[str, List[Dict]] = defaultdict(list)
    for x in pool:
        by_band[band_of(x["confidence"])].append(x)

    rng = random.Random(args.seed)
    picked: List[Dict] = []
    for name, _lo, _hi in BANDS:
        avail = by_band.get(name, [])
        if not avail:
            continue
        # proportional to band size, but never fewer than 5 where the band
        # exists: a band with two reviewed cases supports no claim at all
        want = max(5, round(args.n * len(avail) / max(len(pool), 1)))
        picked += rng.sample(avail, min(want, len(avail)))

    rng.shuffle(picked)          # so the reviewer cannot infer band from order
    for i, x in enumerate(picked, 1):
        x["review_id"] = i
        x["band"] = band_of(x["confidence"])
        x["label"] = ""
        x["note"] = ""

    REVIEW.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in picked),
                      encoding="utf-8")
    print(f"pool: {len(pool)} accepted findings ({'debated' if debated else 'single-judge'})")
    for name, _l, _h in BANDS:
        print(f"  {name:<12} {len(by_band.get(name, [])):>4}")
    print(f"\nsampled {len(picked)} -> {REVIEW.relative_to(KB)}")
    print(f"seed {args.seed}; re-run with the same seed to reproduce the sample,")
    print("or with a different one for a second, independent reviewer.")
    print("\nNext:  python precision.py label")
    return 0


def cmd_label(args) -> int:
    if not REVIEW.exists():
        sys.exit("Run `python precision.py sample` first.")
    items = _jsonl(REVIEW)
    clauses = {c["clause_id"]: c for c in _jsonl(CLAUSES)}
    controls = {c["control_id"]: c for c in _jsonl(CONTROLS)}

    todo = [x for x in items if not x["label"]]
    if not todo:
        print("all labelled. Run `python precision.py score`.")
        return 0
    print(f"{len(todo)} left of {len(items)}. "
          "Labels: [c]orrect  [t] wrong type  [w]rong  [u]nsure  [s]kip  [q]uit\n")

    for x in todo:
        cl = clauses.get(x["clause_id"], {})
        ct = controls.get(x["control_id"], {})
        print("=" * 72)
        print(f"#{x['review_id']}   {x['clause_id']}  ->  {x['control_id']}")
        print(f"claim: {x['relationship']}    (confidence hidden until scoring)")
        print(f"\nSAMA {cl.get('subdomain','')} {cl.get('subdomain_title','')}")
        print(f"  {cl.get('text','')}")
        if cl.get("n_children"):
            print(f"  [with sub-items] {(cl.get('full_text') or '')[:400]}")
        print(f"\nNIST {x['control_id']} — {ct.get('title','')}")
        print(f"  {(ct.get('statement') or '')[:600]}")
        print(f"\nquoted from SAMA: \u201c{x['evidence_sama'][:200]}\u201d")
        print(f"quoted from NIST: \u201c{x['evidence_nist'][:200]}\u201d")
        print(f"stated reason   : {x['rationale'][:220]}")
        print()
        while True:
            k = input("label [c/t/w/u/s/q] > ").strip().lower()
            if k == "q":
                _save(items)
                print("saved.")
                return 0
            if k == "s":
                break
            if k in LABELS:
                x["label"] = LABELS[k]
                x["note"] = input("note (optional) > ").strip()
                _save(items)
                break
            print("  use c, t, w, u, s or q")
        print()

    _save(items)
    print("done. Run `python precision.py score`.")
    return 0


def _save(items) -> None:
    REVIEW.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items),
                      encoding="utf-8")


def cmd_score(args) -> int:
    if not REVIEW.exists():
        sys.exit("Run `python precision.py sample` first.")
    items = [x for x in _jsonl(REVIEW) if x["label"]]
    if not items:
        sys.exit("nothing labelled yet.")

    counts = Counter(x["label"] for x in items)
    unsure = counts["unsure"]
    decided = [x for x in items if x["label"] != "unsure"]
    correct = sum(1 for x in decided if x["label"] == "correct")
    p, lo, hi = wilson(correct, len(decided))

    print("=" * 62)
    print("PRECISION OF ACCEPTED MAPPINGS")
    print("=" * 62)
    print(f"labelled            : {len(items)}")
    print(f"excluded as unsure  : {unsure}")
    print(f"decided             : {len(decided)}")
    print(f"\ncorrect             : {correct}")
    print(f"wrong relationship  : {counts['wrong_type']}")
    print(f"not a relationship  : {counts['wrong']}")
    print(f"\nprecision           : {p:.1%}   95% CI [{lo:.1%}, {hi:.1%}]")

    print("\nby confidence band")
    for name, _l, _h in BANDS:
        sub = [x for x in decided if x["band"] == name]
        if not sub:
            continue
        k = sum(1 for x in sub if x["label"] == "correct")
        bp, blo, bhi = wilson(k, len(sub))
        print(f"  {name:<12} {k:>3}/{len(sub):<3} {bp:>6.1%}  [{blo:.0%}, {bhi:.0%}]")

    print("\nby relationship claimed")
    for rel in sorted({x["relationship"] for x in decided}):
        sub = [x for x in decided if x["relationship"] == rel]
        k = sum(1 for x in sub if x["label"] == "correct")
        print(f"  {rel:<18} {k:>3}/{len(sub)}")

    wrong = [x for x in decided if x["label"] != "correct"]
    if wrong:
        print(f"\nthe {len(wrong)} that failed")
        for x in wrong:
            print(f"  {x['clause_id']:<18} {x['control_id']:<10} "
                  f"{x['relationship']:<16} {x['label']}")
            if x["note"]:
                print(f"      {x['note'][:100]}")

    print(f"\nReport as: precision {p:.0%} on a stratified sample of {len(decided)} "
          f"accepted mappings, 95% CI [{lo:.0%}, {hi:.0%}].")
    print("A sample this size supports a range, not a point estimate — quote the")
    print("interval, and state the sample size beside the figure.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sample", help="draw a stratified sample for review")
    p.add_argument("-n", type=int, default=30)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--debated", action="store_true",
                   help="sample from findings_v2.jsonl (post-debate) instead")

    sub.add_parser("label", help="label the sample interactively")
    sub.add_parser("score", help="precision with a Wilson interval")

    args = ap.parse_args()
    return {"sample": cmd_sample, "label": cmd_label, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
