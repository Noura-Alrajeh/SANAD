#!/usr/bin/env python3
"""
router.py — narrow the NIST target space before candidate generation.

407 leaf clauses against 1014 active controls is 412,000 pairs. No language
model is going to judge that many. The router cuts the target space to a few
families per clause, after which dense retrieval within those families produces
a candidate set small enough to judge.

    all pairs                 412,000
    after family routing       ~60,000
    after retrieval             ~4,000
    after the acceptance gate   accepted findings only

Routing is deliberately model-free. The language-model budget belongs to the
judgment step, where the reasoning actually matters; spending it on a filter
would be a poor trade. Each family is profiled by its title plus the titles of
its base controls, which is a far richer signal than the family name alone
("Access Control" on its own is two words).

Because the router is a filter and not a decision, its only failure mode that
matters is dropping the correct family — so it is scored on recall, not
precision, and `check` reports recall at several values of K against a set of
anchor assignments that no reviewer would dispute (training belongs to AT,
cryptography to SC, incident management to IR, and so on).

Subcommands
-----------
  profiles  build and show the 20 family profiles
  route     route every clause -> processed/routes.jsonl
  show      routing for one clause
  check     recall@K against the anchor assignments

  python router.py profiles
  python router.py check
  python router.py route --k 3
  python router.py show 3.3.5-4.f.1.a
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import index as ix

KB = Path(__file__).resolve().parent
CONTROLS = KB / "processed" / "controls.jsonl"
CLAUSES = KB / "processed" / "sama_clauses.jsonl"
ROUTES = KB / "processed" / "routes.jsonl"
PROF_NPY = KB / "index" / "family_profiles.npy"
PROF_META = KB / "index" / "family_profiles.json"

# Subdomain -> families any competent reviewer would accept. Used only to score
# the router; never used to route. Kept small and uncontroversial on purpose.
ANCHORS: Dict[str, set] = {
    "3.1.5": {"SA", "PL"},      # cyber security in project management
    "3.1.6": {"AT"},            # awareness
    "3.1.7": {"AT"},            # training
    "3.2.1": {"RA", "PM"},      # risk management
    "3.2.5": {"CA", "AU"},      # audits
    "3.3.1": {"PS"},            # human resources
    "3.3.2": {"PE"},            # physical security
    "3.3.3": {"CM"},            # asset management -> CM-8 inventory
    "3.3.5": {"AC", "IA"},      # identity and access management
    "3.3.6": {"SA", "SI"},      # application security
    "3.3.7": {"CM"},            # change management
    "3.3.8": {"SC", "CM"},      # infrastructure security
    "3.3.9": {"SC"},            # cryptography
    "3.3.11": {"MP"},           # secure disposal
    "3.3.14": {"AU", "SI"},     # event management
    "3.3.15": {"IR"},           # incident management
    "3.3.16": {"RA", "SI"},     # threat management
    "3.3.17": {"RA", "SI"},     # vulnerability management
    "3.4.1": {"SA", "SR"},      # contract and vendor management
    "3.4.2": {"SA", "SR"},      # outsourcing
    "3.4.3": {"SA", "SC"},      # cloud computing
}


# --------------------------------------------------------------------------
def load_jsonl(path: Path, what: str) -> List[Dict[str, Any]]:
    if not path.exists():
        sys.exit(f"ERROR: {path.name} not found. {what}")
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def build_profiles() -> List[Dict[str, Any]]:
    """One text blob per family: its title plus the titles of its base controls.

    Enhancement titles are left out. They are numerous and narrow, and they
    swamp the family's centre of gravity.
    """
    controls = load_jsonl(CONTROLS, "Run `python oscal.py flatten` first.")
    live = [c for c in controls if not c["withdrawn"]]
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in live:
        groups[c["family"]].append(c)

    profiles = []
    for fam in sorted(groups):
        rows = groups[fam]
        base = [r for r in rows if not r["is_enhancement"]]
        titles = [r["title"] for r in base]
        profiles.append(
            {
                "family": fam,
                "family_title": rows[0]["family_title"],
                "n_base": len(base),
                "n_total": len(rows),
                "profile": f"{rows[0]['family_title']}. " + "; ".join(titles),
            }
        )
    return profiles


# --------------------------------------------------------------------------
class Router:
    def __init__(self, model_name: str = ix.DEFAULT_MODEL, dense: bool = True):
        self.profiles = build_profiles()
        self.families = [p["family"] for p in self.profiles]
        texts = [p["profile"] for p in self.profiles]

        self.bm25 = ix.BM25([ix.tokenize(t) for t in texts])
        self.emb: Optional[np.ndarray] = None
        self._model = None
        self.model_name = model_name
        if dense:
            self.emb = self._profile_embeddings(texts)

    def _profile_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        PROF_NPY.parent.mkdir(parents=True, exist_ok=True)
        key = {"model": self.model_name, "families": self.families}
        if PROF_NPY.exists() and PROF_META.exists():
            if json.loads(PROF_META.read_text()) == key:
                return np.load(PROF_NPY)
        model = self._get_model()
        emb = model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        np.save(PROF_NPY, emb)
        PROF_META.write_text(json.dumps(key))
        print(f"  built {emb.shape} family profile embeddings")
        return emb

    def _get_model(self):
        if self._model is None:
            self._model = ix._load_model(self.model_name)
        return self._model

    def route(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        lex = self.bm25.top(text, len(self.families))

        den: List[Tuple[int, float]] = []
        if self.emb is not None:
            v = self._get_model().encode(
                [text], normalize_embeddings=True, convert_to_numpy=True
            )[0].astype(np.float32)
            sims = self.emb @ v
            den = [(int(i), float(sims[i])) for i in np.argsort(-sims)]

        fused = ix.rrf(lex, den) if den else lex
        return [(self.families[i], round(s, 6)) for i, s in fused[:k]]

    def route_batch(
        self,
        texts: Sequence[str],
        k: int = 3,
        priors: Optional[Sequence[Sequence[Tuple[int, float]]]] = None,
    ) -> List[List[Tuple[str, float]]]:
        """Encode all clauses in one pass — one model call instead of 407."""
        lex_all = [self.bm25.top(t, len(self.families)) for t in texts]

        den_all: List[List[Tuple[int, float]]] = [[] for _ in texts]
        if self.emb is not None:
            model = self._get_model()
            vecs = model.encode(
                list(texts),
                batch_size=64,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100,
            ).astype(np.float32)
            sims = vecs @ self.emb.T                     # (n_clauses, n_families)
            for r in range(sims.shape[0]):
                order = np.argsort(-sims[r])
                den_all[r] = [(int(i), float(sims[r, i])) for i in order]

        out = []
        for n, (lex, den) in enumerate(zip(lex_all, den_all)):
            lists = [lex] + ([den] if den else [])
            if priors is not None and priors[n]:
                lists.append(priors[n])
            fused = ix.rrf(*lists)
            out.append([(self.families[i], round(s, 6)) for i, s in fused[:k]])
        return out

    def index_of(self, family: str) -> int:
        return self.families.index(family)


# --------------------------------------------------------------------------
def clause_text(c: Dict[str, Any], by_id: Optional[Dict[str, Any]] = None) -> str:
    """Text to route on.

    A level-4 clause can be three words ("confidentiality of passwords;") and
    carries almost no signal alone. Its topic comes from where it sits: the
    subdomain heading, the subdomain principle, and the stems above it. Routing
    on that chain rather than the bare clause is the single largest accuracy
    gain available here, and it costs nothing at inference time.
    """
    parts = [c.get("subdomain_title", ""), c.get("principle", "")]
    if by_id:
        chain, pid = [], c.get("parent_id")
        while pid and pid in by_id:
            chain.append(by_id[pid]["text"])
            pid = by_id[pid].get("parent_id")
        parts.extend(reversed(chain))
    parts.append(c.get("full_text") or c["text"])
    return " ".join(p for p in parts if p)


def subdomain_text(clauses: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """One routing text per subdomain: heading, principle, objective.

    A subdomain is a coherent topic, so it routes more reliably than any single
    clause inside it. Its route is fused with the clause's own.
    """
    out: Dict[str, str] = {}
    for c in clauses:
        sd = c["subdomain"]
        if sd not in out:
            out[sd] = " ".join(
                x for x in (c.get("subdomain_title"), c.get("principle"), c.get("objective")) if x
            )
    return out


def routable(clauses: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Leaf, non-referential clauses. Stems are represented by their children;
    referential clauses state no requirement to map."""
    return [c for c in clauses if c["clause_type"] != "referential" and not c["n_children"]]


def route_clauses(
    r: "Router", clauses: Sequence[Dict[str, Any]], targets: Sequence[Dict[str, Any]], k: int
) -> List[List[Tuple[str, float]]]:
    """Route each clause, fusing its own signal with its subdomain's."""
    by_id = {c["clause_id"]: c for c in clauses}
    sd_text = subdomain_text(clauses)
    sd_keys = sorted(sd_text)
    sd_routes = r.route_batch([sd_text[s] for s in sd_keys], k=len(r.families))
    sd_rank = {
        s: [(r.index_of(f), sc) for f, sc in route] for s, route in zip(sd_keys, sd_routes)
    }
    priors = [sd_rank.get(c["subdomain"], []) for c in targets]
    return r.route_batch([clause_text(c, by_id) for c in targets], k=k, priors=priors)


# --------------------------------------------------------------------------
def cmd_profiles(args) -> int:
    profs = build_profiles()
    print(f"{len(profs)} family profiles\n")
    for p in profs:
        print(f"{p['family']:<4} {p['n_base']:>3} base / {p['n_total']:>4} total  {p['family_title']}")
        body = " ".join(p["profile"].split())
        print(f"     {body[:150]}{'…' if len(body) > 150 else ''}\n")
    return 0


def cmd_route(args) -> int:
    clauses = load_jsonl(CLAUSES, "Run `python sama.py parse` first.")
    targets = routable(clauses)
    print(f"routing {len(targets)} leaf clauses of {len(clauses)} total")

    r = Router(args.model, dense=not args.no_dense)
    routes = route_clauses(r, clauses, targets, args.k)

    with ROUTES.open("w", encoding="utf-8") as fh:
        for c, fams in zip(targets, routes):
            fh.write(
                json.dumps(
                    {
                        "clause_id": c["clause_id"],
                        "subdomain": c["subdomain"],
                        "families": [f for f, _ in fams],
                        "scores": [s for _, s in fams],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    picked = Counter(f for fams in routes for f, _ in fams)
    print(f"\nfamily selection frequency (top-{args.k})")
    for fam, n in picked.most_common():
        bar = "#" * max(1, n * 40 // max(picked.values()))
        print(f"  {fam:<4} {n:>4}  {bar}")
    never = [f for f in r.families if f not in picked]
    if never:
        print(f"\n  never selected: {', '.join(never)}")
        print("  (worth checking — a family no clause routes to is either")
        print("   genuinely absent from SAMA, or a routing blind spot)")

    controls = load_jsonl(CONTROLS, "")
    live = [c for c in controls if not c["withdrawn"]]
    per_fam = Counter(c["family"] for c in live)
    pairs = sum(sum(per_fam[f] for f, _ in fams) for fams in routes)
    full = len(targets) * len(live)
    print(f"\ncandidate pairs after routing : {pairs:,}")
    print(f"without routing               : {full:,}")
    print(f"reduction                     : {100 - 100*pairs//full}%")
    print(f"\nwrote {len(targets)} routes -> {ROUTES.relative_to(KB)}")
    return 0


def cmd_show(args) -> int:
    clauses = load_jsonl(CLAUSES, "Run `python sama.py parse` first.")
    hits = [c for c in clauses if c["clause_id"].lower() == args.clause_id.lower()]
    if not hits:
        print(f"not found: {args.clause_id}")
        return 1
    c = hits[0]
    r = Router(args.model, dense=not args.no_dense)
    route = route_clauses(r, clauses, [c], args.k)[0]
    by_id = {x["clause_id"]: x for x in clauses}
    print(f"{c['clause_id']}  ({c['subdomain']} {c['subdomain_title']})")
    print(f"\nrouting text: {clause_text(c, by_id)[:400]}\n")
    for n, (fam, score) in enumerate(route, 1):
        title = next(p["family_title"] for p in r.profiles if p["family"] == fam)
        print(f"  {n}. {fam:<4} {score:.5f}  {title}")
    anchor = ANCHORS.get(c["subdomain"])
    if anchor:
        got = {f for f, _ in route}
        mark = "HIT" if got & anchor else "MISS"
        print(f"\n  anchor for {c['subdomain']}: {', '.join(sorted(anchor))}  -> {mark}")
    return 0


def cmd_check(args) -> int:
    clauses = load_jsonl(CLAUSES, "Run `python sama.py parse` first.")
    targets = [c for c in routable(clauses) if c["subdomain"] in ANCHORS]
    print(f"scoring {len(targets)} clauses across {len({c['subdomain'] for c in targets})} "
          f"anchored subdomains\n")

    r = Router(args.model, dense=not args.no_dense)
    routes = route_clauses(r, clauses, targets, max(args.ks))

    print(f"{'K':>3}  {'recall':>7}   hits/total")
    for k in sorted(args.ks):
        hits = sum(
            1
            for c, fams in zip(targets, routes)
            if {f for f, _ in fams[:k]} & ANCHORS[c["subdomain"]]
        )
        print(f"{k:>3}  {hits/len(targets):>6.1%}   {hits}/{len(targets)}")

    k = args.report_k
    print(f"\nper-subdomain recall at K={k}")
    per: Dict[str, List[bool]] = defaultdict(list)
    for c, fams in zip(targets, routes):
        per[c["subdomain"]].append(bool({f for f, _ in fams[:k]} & ANCHORS[c["subdomain"]]))
    for sd in sorted(per, key=lambda s: sum(per[s]) / len(per[s])):
        hits, tot = sum(per[sd]), len(per[sd])
        flag = "  <-- weak" if hits / tot < 0.6 else ""
        print(f"  {sd:<9} {hits:>3}/{tot:<3} {hits/tot:>6.0%}  "
              f"expected {','.join(sorted(ANCHORS[sd]))}{flag}")

    print("\nThese anchors are a floor, not a ground truth: they cover 21 of 36")
    print("subdomains and only the uncontroversial cases. Report the number as")
    print("router recall on anchored subdomains, not as overall accuracy.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=ix.DEFAULT_MODEL)
    ap.add_argument("--no-dense", action="store_true", help="BM25 only")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("profiles", help="show family profiles")

    p = sub.add_parser("route", help="route all clauses")
    p.add_argument("--k", type=int, default=3)

    p = sub.add_parser("show", help="routing for one clause")
    p.add_argument("clause_id")
    p.add_argument("--k", type=int, default=3)

    p = sub.add_parser("check", help="recall@K against anchors")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5])
    p.add_argument("--report-k", type=int, default=3)

    args = ap.parse_args()
    return {"profiles": cmd_profiles, "route": cmd_route, "show": cmd_show,
            "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
