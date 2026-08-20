#!/usr/bin/env python3
"""
candidates.py — generate the control shortlist each clause will be judged against.

The router narrows 20 families to a handful. This stage narrows the controls
inside those families to a shortlist small enough to put in front of a language
model.

    all pairs                    412,698
    after family routing (K=5)   128,487
    after retrieval (N=10)         ~4,000   <- this stage
    after the acceptance gate      accepted findings only

Retrieval is the same hybrid used for the policy corpus — BM25 for identifiers
and exact regulatory terms, dense embeddings for paraphrase, fused by RRF —
but over control statements rather than document chunks, and restricted to the
families the router selected.

Withdrawn controls are excluded before retrieval, never after: a shortlist that
contains tombstones wastes model budget and invites an invalid finding.

Subcommands
-----------
  generate  clause -> shortlist, writing processed/candidates.jsonl
  show      shortlist for one clause, with the reason each control surfaced
  check     recall@N against anchor clause->control assignments
  stats     shortlist statistics and coverage of the control catalogue

  python candidates.py check
  python candidates.py generate --n 10
  python candidates.py show 3.3.13-4.b.6.d
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
import router as rt

KB = Path(__file__).resolve().parent
CANDIDATES = KB / "processed" / "candidates.jsonl"
CTRL_NPY = KB / "index" / "controls.npy"
CTRL_META = KB / "index" / "controls_meta.json"

# clause -> a control any reviewer would expect on the shortlist. Used only to
# measure recall. Deliberately few and obvious; this is a floor, not a mapping.
ANCHORS: Dict[str, set] = {
    "3.3.13-4.b.6.d": {"AC-7"},        # revoke after 3 successive failed PINs
    "3.3.5-4.f.1.a": {"AC-17", "IA-2"},  # MFA for all remote access
    "3.3.5-4.b.6": {"AC-2"},           # periodic review of access rights
    "3.1.7-1": {"AT-3"},               # role-based specialist training
    "3.1.6-1": {"AT-2"},               # awareness programme
    "3.3.15-1": {"IR-4", "IR-8"},      # incident management process
    "3.3.14-4.j": {"AU-6", "SI-4"},    # centralised log analysis, SIEM
    "3.3.17-3.f": {"SI-2", "RA-5"},    # patch management
    "3.3.9-4.c": {"SC-12"},            # encryption key management
    "3.3.2-3.a": {"PE-3"},             # physical entry controls
    "3.3.1-3.d": {"PS-3"},             # screening and background check
    "3.3.11-5": {"MP-6"},              # non-retrievable destruction
    "3.4.1-5.a": {"SA-4", "SR-5"},     # security requirements in procurement
    "3.3.8-6.e": {"SC-7"},             # network segmentation
}


def query_text(c: Dict[str, Any], by_id: Dict[str, Any], mode: str = "parent") -> str:
    """The query a clause is retrieved with.

    Routing and retrieval do NOT want the same query, and assuming they did was
    an error. Routing picks a topic, so the whole context chain helps: it tells
    you the clause is about electronic banking. Retrieval picks one control out
    of a few hundred, and that same context buries the clause. A fourteen-word
    clause carrying a hundred words of inherited context is retrieved as though
    it were about its heading.

        clause   the clause and its own sub-items only
        parent   plus the immediate stem, which resolves what the clause is a
                 sub-item of, and the subdomain title as a light topical anchor
        context  the full chain used for routing — kept for comparison
    """
    own = c.get("full_text") or c["text"]
    if mode == "clause":
        return own
    if mode == "context":
        return rt.clause_text(c, by_id)
    parts = [c.get("subdomain_title", "")]
    pid = c.get("parent_id")
    if pid and pid in by_id:
        parts.append(by_id[pid]["text"])
    parts.append(own)
    return " ".join(p for p in parts if p)


def control_text(c: Dict[str, Any], with_guidance: bool = False) -> str:
    """What a control is indexed on: identity, title, then the requirement.

    Whether to add the discussion prose is an empirical question, not an
    obvious one. It is real NIST content and widens the vocabulary a clause can
    match against — which matters because SAMA and NIST often name the same
    thing differently ("secure disposal" against "media sanitization"). It is
    also long and partly boilerplate, which can flatten the differences between
    controls. Measure it with `check --with-guidance` before choosing.
    """
    base = f"{c['control_id']} {c['title']}. {c['family_title']}. {c['statement']}"
    if with_guidance and c.get("guidance"):
        base += " " + c["guidance"]
    return base


class ControlIndex:
    def __init__(self, model_name: str = ix.DEFAULT_MODEL, dense: bool = True,
                 with_guidance: bool = False):
        rows = rt.load_jsonl(rt.CONTROLS, "Run `python oscal.py flatten` first.")
        self.controls = [c for c in rows if not c["withdrawn"]]
        self.by_id = {c["control_id"]: c for c in self.controls}
        self.with_guidance = with_guidance
        self.texts = [control_text(c, with_guidance) for c in self.controls]
        if with_guidance:
            print("  indexing statement + discussion")
        print(f"indexing {len(self.controls)} active controls "
              f"({len(rows) - len(self.controls)} withdrawn excluded)")

        self.family_rows: Dict[str, List[int]] = defaultdict(list)
        self.index_by_id: Dict[str, int] = {}
        for i, c in enumerate(self.controls):
            self.family_rows[c["family"]].append(i)
            self.index_by_id[c["control_id"]] = i
        n_enh = sum(1 for c in self.controls if c["is_enhancement"])
        print(f"  {n_enh} enhancements, {len(self.controls) - n_enh} base controls")

        self.bm25 = ix.BM25([ix.tokenize(t) for t in self.texts])
        self.model_name = model_name
        self._model = None
        self.emb = self._embeddings() if dense else None

    def _get_model(self):
        if self._model is None:
            self._model = ix._load_model(self.model_name)
        return self._model

    def _embeddings(self) -> np.ndarray:
        import hashlib

        fp = hashlib.sha256("".join(self.texts).encode()).hexdigest()[:16]
        key = {"model": self.model_name, "fingerprint": fp, "n": len(self.controls),
               "guidance": self.with_guidance}
        CTRL_NPY.parent.mkdir(parents=True, exist_ok=True)
        if CTRL_NPY.exists() and CTRL_META.exists():
            if json.loads(CTRL_META.read_text()) == key:
                emb = np.load(CTRL_NPY)
                print(f"  reusing cached control embeddings {emb.shape}")
                return emb
        model = self._get_model()
        print(f"  encoding {len(self.texts)} controls")
        emb = model.encode(
            self.texts, batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=True,
        ).astype(np.float32)
        np.save(CTRL_NPY, emb)
        CTRL_META.write_text(json.dumps(key))
        return emb

    def shortlist(
        self,
        query: str,
        families: Sequence[str],
        n: int = 10,
        qvec: Optional[np.ndarray] = None,
        base_quota: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Shortlist controls for one clause.

        Two corrections are applied to the raw fused ranking, both because
        enhancements outnumber base controls two to one and, being narrower and
        more specific, they match individual phrases more sharply:

        parent lift  an enhancement ranking well is evidence that its base
                     control is relevant, so the base is pulled into the pool
                     at the enhancement's rank. Without this, SI-4(13) and
                     SI-4(17) can both rank while SI-4 itself never appears.
        base quota   base controls are guaranteed a share of the slots. SAMA
                     clauses are written at principle level, so the base
                     control is usually the correct target and an enhancement
                     is a refinement of an answer, not the answer.
        """
        allowed = sorted({i for f in families for i in self.family_rows.get(f, [])})
        if not allowed:
            return []
        allowed_set = set(allowed)

        lex_full = self.bm25.top(query, len(self.controls))
        lex = [(i, s) for i, s in lex_full if i in allowed_set][: n * 5]

        den: List[Tuple[int, float]] = []
        if self.emb is not None:
            v = qvec if qvec is not None else self._get_model().encode(
                [query], normalize_embeddings=True, convert_to_numpy=True
            )[0].astype(np.float32)
            sub = self.emb[allowed] @ v
            order = np.argsort(-sub)[: n * 5]
            den = [(allowed[int(j)], float(sub[int(j)])) for j in order]

        lists = [lex] + ([den] if den else [])
        pre = ix.rrf(*lists)

        lift: Dict[int, int] = {}
        for rank, (idx, _s) in enumerate(pre, 1):
            c = self.controls[idx]
            pid = c.get("parent_id")
            if c["is_enhancement"] and pid:
                j = self.index_by_id.get(pid)
                if j is not None and j in allowed_set:
                    lift[j] = min(lift.get(j, 10**6), rank)
        if lift:
            lists.append([(i, 0.0) for i, _ in sorted(lift.items(), key=lambda kv: kv[1])])
        fused = ix.rrf(*lists)

        lr = {i: r for r, (i, _) in enumerate(lex, 1)}
        dr = {i: r for r, (i, _) in enumerate(den, 1)}

        n_base = max(1, int(round(n * base_quota)))
        base_sel: List[Tuple[int, float]] = []
        enh_sel: List[Tuple[int, float]] = []
        overflow: List[Tuple[int, float]] = []
        for idx, score in fused:
            if not self.controls[idx]["is_enhancement"]:
                (base_sel if len(base_sel) < n_base else overflow).append((idx, score))
            else:
                (enh_sel if len(enh_sel) < n - n_base else overflow).append((idx, score))
            if len(base_sel) + len(enh_sel) >= n:
                break
        picked = base_sel + enh_sel
        if len(picked) < n:
            picked += overflow[: n - len(picked)]
        order = {i: r for r, (i, _) in enumerate(fused, 1)}
        picked.sort(key=lambda kv: order.get(kv[0], 10**6))

        out = []
        for idx, score in picked[:n]:
            c = self.controls[idx]
            out.append(
                {
                    "control_id": c["control_id"],
                    "family": c["family"],
                    "title": c["title"],
                    "is_enhancement": c["is_enhancement"],
                    "n_params": c["n_params"],
                    "score": round(score, 6),
                    "bm25_rank": lr.get(idx),
                    "dense_rank": dr.get(idx),
                    "parent_lifted": idx in lift and not lr.get(idx) and not dr.get(idx),
                }
            )
        return out


# --------------------------------------------------------------------------
def _prepare(args, only: Optional[Sequence[str]] = None):
    clauses = rt.load_jsonl(rt.CLAUSES, "Run `python sama.py parse` first.")
    by_id = {c["clause_id"]: c for c in clauses}
    targets = rt.routable(clauses)
    if only:
        want = {o.lower() for o in only}
        targets = [c for c in targets if c["clause_id"].lower() in want]
        if not targets:
            sys.exit(f"no such clause: {', '.join(only)}")

    routes_file = rt.ROUTES
    if routes_file.exists() and not args.reroute:
        table = {json.loads(l)["clause_id"]: json.loads(l)["families"]
                 for l in routes_file.open(encoding="utf-8") if l.strip()}
        missing = [c for c in targets if c["clause_id"] not in table]
        if not missing:
            print(f"using routes from {routes_file.name}")
            return clauses, by_id, targets, [table[c["clause_id"]] for c in targets]
        print(f"{len(missing)} clause(s) missing from routes — re-routing")

    r = rt.Router(args.model, dense=not args.no_dense)
    routed = rt.route_clauses(r, clauses, targets, args.k)
    return clauses, by_id, targets, [[f for f, _ in row] for row in routed]


def cmd_generate(args) -> int:
    clauses, by_id, targets, fams = _prepare(args)
    ci = ControlIndex(args.model, dense=not args.no_dense, with_guidance=args.with_guidance)

    queries = [query_text(c, by_id, args.query_mode) for c in targets]
    qvecs = None
    if ci.emb is not None:
        qvecs = ci._get_model().encode(
            queries, batch_size=64, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=len(queries) > 100,
        ).astype(np.float32)

    n_written = 0
    surfaced: Counter = Counter()
    with CANDIDATES.open("w", encoding="utf-8") as fh:
        for i, (c, fl) in enumerate(zip(targets, fams)):
            sl = ci.shortlist(queries[i], fl, n=args.n, base_quota=args.base_quota,
                              qvec=None if qvecs is None else qvecs[i])
            for cand in sl:
                surfaced[cand["control_id"]] += 1
            fh.write(json.dumps({
                "clause_id": c["clause_id"],
                "subdomain": c["subdomain"],
                "clause_text": c["text"],
                "families": list(fl),
                "candidates": sl,
            }, ensure_ascii=False) + "\n")
            n_written += 1

    pairs = sum(len(json.loads(l)["candidates"])
                for l in CANDIDATES.open(encoding="utf-8") if l.strip())
    print(f"\nclauses          : {n_written}")
    print(f"shortlist size   : {args.n}")
    print(f"candidate pairs  : {pairs:,}")
    print(f"controls reached : {len(surfaced)} of {len(ci.controls)} "
          f"({100*len(surfaced)//len(ci.controls)}%)")
    print(f"\nmost frequently shortlisted")
    for cid, n in surfaced.most_common(10):
        print(f"  {cid:<12} {n:>4}  {ci.by_id[cid]['title'][:48]}")
    unreached = len(ci.controls) - len(surfaced)
    print(f"\n{unreached} control(s) never shortlisted. Some are genuinely")
    print("absent from SAMA — that set is the raw material for the coverage")
    print("gap report, so keep it rather than discarding it.")
    print(f"\nwrote {n_written} shortlists -> {CANDIDATES.relative_to(KB)}")
    return 0


def cmd_show(args) -> int:
    clauses, by_id, targets, fams = _prepare(args, only=[args.clause_id])
    c, fl = targets[0], fams[0]
    ci = ControlIndex(args.model, dense=not args.no_dense, with_guidance=args.with_guidance)
    sl = ci.shortlist(query_text(c, by_id, args.query_mode), fl, n=args.n,
                      base_quota=args.base_quota)

    print(f"\n{c['clause_id']}  ({c['subdomain']} {c['subdomain_title']})")
    print(f"{c['text']}\n")
    print(f"query ({args.query_mode}): {query_text(c, by_id, args.query_mode)[:260]}\n")
    if c["numeric_params"]:
        print(f"states a value: {', '.join(c['numeric_params'])}\n")
    print(f"routed families: {', '.join(fl)}\n")
    for n, cand in enumerate(sl, 1):
        why = "both" if cand["bm25_rank"] and cand["dense_rank"] else (
            "bm25" if cand["bm25_rank"] else "dense")
        odp = f"  {cand['n_params']} ODP" if cand["n_params"] else ""
        print(f"  {n:>2}. {cand['control_id']:<12} {cand['title'][:46]:<46} ({why}){odp}")

    anchor = ANCHORS.get(c["clause_id"])
    if anchor:
        got = {x["control_id"] for x in sl}
        print(f"\n  anchor: {', '.join(sorted(anchor))} -> "
              f"{'HIT' if got & anchor else 'MISS'}")
    if c["numeric_params"] and any(x["n_params"] for x in sl):
        print("\n  This clause fixes a value and at least one candidate leaves")
        print("  an open parameter: a parameter-gap finding, not a plain match.")
    return 0


def cmd_check(args) -> int:
    ids = list(ANCHORS)
    clauses, by_id, targets, fams = _prepare(args, only=ids)
    found = {c["clause_id"] for c in targets}
    missing = [i for i in ids if i not in found]
    if missing:
        print(f"note: {len(missing)} anchor clause(s) are not leaf clauses "
              f"and were skipped: {', '.join(missing)}\n")

    ci = ControlIndex(args.model, dense=not args.no_dense, with_guidance=args.with_guidance)
    maxn = max(args.ns)
    rows = []
    for c, fl in zip(targets, fams):
        sl = ci.shortlist(query_text(c, by_id, args.query_mode), fl, n=maxn,
                          base_quota=args.base_quota)
        rows.append((c, [x["control_id"] for x in sl]))

    print(f"\n{'N':>3}  {'recall':>7}   hits/total")
    for n in sorted(args.ns):
        hits = sum(1 for c, sl in rows if set(sl[:n]) & ANCHORS[c["clause_id"]])
        print(f"{n:>3}  {hits/len(rows):>6.1%}   {hits}/{len(rows)}")

    n = args.report_n
    print(f"\nper-clause at N={n}")
    for c, sl in rows:
        want = ANCHORS[c["clause_id"]]
        hit = set(sl[:n]) & want
        rank = next((i for i, x in enumerate(sl[:n], 1) if x in want), None)
        mark = f"rank {rank}" if hit else "MISS"
        print(f"  {c['clause_id']:<18} want {','.join(sorted(want)):<12} {mark:<8} "
              f"got {', '.join(sl[:4])}")
    print("\nA miss here is a retrieval failure the judge can never recover from,")
    print("so N should be chosen for recall. Precision is the judge's problem.")
    return 0


def cmd_stats(args) -> int:
    if not CANDIDATES.exists():
        sys.exit("Run `python candidates.py generate` first.")
    rows = [json.loads(l) for l in CANDIDATES.open(encoding="utf-8") if l.strip()]
    fam = Counter(x["family"] for r in rows for x in r["candidates"])
    enh = sum(1 for r in rows for x in r["candidates"] if x["is_enhancement"])
    tot = sum(len(r["candidates"]) for r in rows)
    print(f"shortlists        : {len(rows)}")
    print(f"candidate pairs   : {tot:,}")
    print(f"enhancements      : {enh:,} ({100*enh//max(tot,1)}%)")
    print("\nby family")
    for f, n in fam.most_common():
        print(f"  {f:<4} {n:>5}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=ix.DEFAULT_MODEL)
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--k", type=int, default=5, help="families per clause when re-routing")
    ap.add_argument("--reroute", action="store_true", help="ignore routes.jsonl")
    ap.add_argument("--query-mode", choices=["clause", "parent", "context"], default="parent",
                    help="how much inherited context to put in the retrieval query")
    ap.add_argument("--with-guidance", action="store_true",
                    help="index the control discussion as well as its statement")
    ap.add_argument("--base-quota", type=float, default=0.7,
                    help="share of shortlist slots reserved for base controls (0 disables)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate", help="build all shortlists")
    p.add_argument("--n", type=int, default=10)

    p = sub.add_parser("show", help="shortlist for one clause")
    p.add_argument("clause_id")
    p.add_argument("--n", type=int, default=10)

    p = sub.add_parser("check", help="recall@N against anchors")
    p.add_argument("--ns", type=int, nargs="+", default=[3, 5, 10, 20])
    p.add_argument("--report-n", type=int, default=10)

    sub.add_parser("stats", help="shortlist statistics")

    args = ap.parse_args()
    return {"generate": cmd_generate, "show": cmd_show,
            "check": cmd_check, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
