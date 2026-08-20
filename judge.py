#!/usr/bin/env python3
"""
judge.py — assign a relationship type to each candidate pair, with evidence.

This is the last stage of the funnel and the only one that uses a language
model. Its output is a finding: a SAMA clause, a NIST control, one of the five
NIST IR 8477 relationship types, a verbatim span of evidence from each side,
and a confidence.

Two design decisions carry most of the weight.

One call per clause, not per pair
    All ten candidates go into a single prompt. That is 407 calls instead of
    4,070, and the model sees the alternatives side by side, so it can say
    which control fits best rather than judging each in isolation against no
    baseline.

Nothing is accepted on the model's word
    Every judgment passes a deterministic gate before it is recorded: the
    control must be one that was actually offered, the relationship must be one
    of the five, and both evidence spans must appear verbatim in their source
    texts. A judgment that fails is rejected and kept as a rejection, never
    silently repaired. This gate is the Y.3172 policy node: it applies policy
    to model output to safeguard the sanity of the pipeline.

Relationship types (NIST IR 8477), with SAMA as the focal element:

    equal            same requirement, same scope
    subset_of        the SAMA clause is narrower; NIST covers it and more
    superset_of      the SAMA clause is broader; it covers the NIST control
                     and more
    intersects_with  they overlap, neither contains the other
    not_related      no meaningful relationship

Subcommands
-----------
  dry-run   print the prompt for one clause without calling anything
  judge     run the judge, writing processed/findings.jsonl (resumable)
  show      findings for one clause
  report    acceptance rates, relationship mix, parameter gaps

  python judge.py dry-run 3.3.13-4.b.6.d
  python judge.py judge --mock --limit 20      # plumbing test, no API
  python judge.py judge --limit 20             # real, 20 clauses
  python judge.py judge                        # full run, resumable
  python judge.py report
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import router as rt

KB = Path(__file__).resolve().parent
CANDIDATES = KB / "processed" / "candidates.jsonl"
FINDINGS = KB / "processed" / "findings.jsonl"

RELATIONS = ["equal", "subset_of", "superset_of", "intersects_with", "not_related"]
TAU_ACCEPT = 0.60          # below this a verified judgment is disputed, not accepted
MIN_EVIDENCE_CHARS = 15    # a three-word span is not evidence


# --------------------------------------------------------------------------
# normalisation for verbatim checking
# --------------------------------------------------------------------------
_QUOTES = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " ",
}


def norm(text: str) -> str:
    """Collapse whitespace and fold the typographic variants a PDF introduces.

    Without this, a span copied correctly by the model still fails the check
    because the source used a curly apostrophe or a non-breaking space.
    """
    text = unicodedata.normalize("NFKC", text or "")
    for a, b in _QUOTES.items():
        text = text.replace(a, b)
    return " ".join(text.split()).lower()


def contains(haystack: str, needle: str) -> bool:
    return norm(needle) in norm(haystack)


# --------------------------------------------------------------------------
# model access
# --------------------------------------------------------------------------
class Model:
    """Thin wrapper over one provider. Kept deliberately small: the pipeline
    should not care which model produced a judgment, only that it survived the
    gate."""

    def __init__(self, provider: str, name: str, mock: bool = False,
                 temperature: Optional[float] = None):
        self.provider, self.name, self.mock = provider, name, mock
        # Newer Anthropic models reject `temperature` outright, so it is sent
        # only when asked for. Determinism was never really on offer from it
        # anyway; what makes this pipeline reproducible is the gate, not the
        # sampling setting.
        self.temperature = temperature
        self._client = None
        if mock:
            return
        if provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                sys.exit("pip install -q anthropic")
            key = os.environ.get("ANTHROPIC_API_KEY") or _colab_secret("ANTHROPIC_API_KEY")
            if not key:
                sys.exit("Set ANTHROPIC_API_KEY (env var, or Colab secret of that name).")
            self._client = anthropic.Anthropic(api_key=key)
        elif provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                sys.exit("pip install -q openai")
            key = os.environ.get("OPENAI_API_KEY") or _colab_secret("OPENAI_API_KEY")
            if not key:
                sys.exit("Set OPENAI_API_KEY (env var, or Colab secret of that name).")
            self._client = OpenAI(api_key=key)
        else:
            sys.exit(f"unknown provider: {provider}")

    def complete(self, system: str, user: str, max_tokens: int = 4000) -> str:
        if self.provider == "anthropic":
            kw: Dict[str, Any] = dict(
                model=self.name, max_tokens=max_tokens,
                system=system, messages=[{"role": "user", "content": user}],
            )
            if self.temperature is not None:
                kw["temperature"] = self.temperature
            try:
                r = self._client.messages.create(**kw)
            except Exception as exc:  # noqa: BLE001
                if "temperature" in str(exc) and "temperature" in kw:
                    kw.pop("temperature")          # model refuses it; send without
                    self.temperature = None
                    r = self._client.messages.create(**kw)
                else:
                    raise
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")

        kw = dict(model=self.name, max_tokens=max_tokens,
                  messages=[{"role": "system", "content": system},
                            {"role": "user", "content": user}])
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        r = self._client.chat.completions.create(**kw)
        return r.choices[0].message.content or ""


def _colab_secret(name: str) -> Optional[str]:
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get(name)
    except Exception:
        return None


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------
SYSTEM = """You are a regulatory analyst mapping the SAMA Cyber Security Framework (Saudi Arabia, 2017) onto NIST SP 800-53 Rev 5.2.0.

You assign one of five relationship types, defined as in NIST IR 8477 with the SAMA clause as the focal element:

- equal: the two state the same requirement at the same scope.
- subset_of: the SAMA clause is narrower. The NIST control covers everything the SAMA clause requires, and more.
- superset_of: the SAMA clause is broader. It covers everything the NIST control requires, and more.
- intersects_with: they overlap in part; neither fully contains the other.
- not_related: no meaningful relationship.

Rules you must follow exactly:

1. Report only controls that stand in a real relationship. Omit the rest; omission means not_related.
2. Every judgment must quote evidence VERBATIM from the text given to you — an exact character-for-character substring, no paraphrase, no ellipsis, no correction of typography. Quote at least 15 characters from each side.
3. Judge only what the texts say. Do not use knowledge of either framework beyond the text provided.
4. If the SAMA clause states a concrete value (a count, a period, a frequency) and the NIST control leaves that value as an organization-defined parameter, set parameter_gap to true and name the parameter.
5. Prefer a base control over one of its enhancements unless the clause is specifically about what the enhancement adds.
6. confidence is your own probability that a domain expert would agree, from 0.0 to 1.0. Be calibrated: use values below 0.6 when the match is arguable.

Return ONLY a JSON array. No prose, no code fence. Each element:

{"control_id": "...", "relationship": "equal|subset_of|superset_of|intersects_with", "confidence": 0.0, "evidence_sama": "verbatim span from the SAMA clause", "evidence_nist": "verbatim span from the control statement", "parameter_gap": false, "parameter_note": "", "rationale": "one sentence"}

An empty array is a valid and expected answer when nothing on the list is related."""


def build_prompt(clause: Dict[str, Any], row: Dict[str, Any], controls: Dict[str, Any]) -> str:
    parts = [
        "## SAMA clause",
        f"Identifier: {clause['clause_id']}",
        f"Subdomain: {clause['subdomain']} {clause['subdomain_title']}",
        f"Domain: {clause['domain_title']}",
    ]
    if clause.get("principle"):
        parts.append(f"Subdomain principle: {clause['principle']}")
    if clause.get("parent_id"):
        parts.append(f"Governed by: {clause['parent_id']}")
    parts.append(f"\nClause text:\n{clause['text']}")
    if clause.get("n_children"):
        parts.append(f"\nWith its sub-items:\n{clause['full_text']}")
    if clause.get("numeric_params"):
        parts.append(f"\nThis clause states a concrete value: {', '.join(clause['numeric_params'])}")

    parts.append("\n## Candidate NIST SP 800-53 controls\n")
    for cand in row["candidates"]:
        c = controls.get(cand["control_id"])
        if not c:
            continue
        parts.append(f"### {c['control_id']} — {c['title']}")
        parts.append(f"Family: {c['family']} ({c['family_title']})")
        if c["is_enhancement"]:
            parts.append(f"Enhancement of {c['parent_id']}")
        parts.append(f"Statement: {c['statement'] or '(no statement text)'}")
        if c["params"]:
            parts.append("Organization-defined parameters: " + "; ".join(str(p) for p in c["params"]))
        parts.append("")
    parts.append("Return the JSON array now.")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# the acceptance gate — Y.3172 policy node
# --------------------------------------------------------------------------
def verify(
    j: Dict[str, Any], clause: Dict[str, Any], offered: Sequence[str], controls: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    errs: List[str] = []

    cid = str(j.get("control_id", "")).strip().upper()
    if cid not in {o.upper() for o in offered}:
        errs.append(f"control {cid or '(missing)'} was not offered as a candidate")

    rel = str(j.get("relationship", "")).strip().lower()
    if rel not in RELATIONS or rel == "not_related":
        errs.append(f"relationship '{rel}' is not a reportable type")

    try:
        conf = float(j.get("confidence"))
        if not 0.0 <= conf <= 1.0:
            raise ValueError
    except (TypeError, ValueError):
        errs.append("confidence is missing or out of range")
        conf = 0.0

    ev_s = str(j.get("evidence_sama", "") or "")
    ev_n = str(j.get("evidence_nist", "") or "")
    haystack_s = f"{clause['text']} {clause.get('full_text','')}"
    ctrl = controls.get(cid)
    haystack_n = f"{ctrl['statement']} {ctrl.get('guidance','')}" if ctrl else ""

    if len(norm(ev_s)) < MIN_EVIDENCE_CHARS:
        errs.append("SAMA evidence too short to be evidence")
    elif not contains(haystack_s, ev_s):
        errs.append("SAMA evidence is not a verbatim span of the clause")

    if len(norm(ev_n)) < MIN_EVIDENCE_CHARS:
        errs.append("NIST evidence too short to be evidence")
    elif not contains(haystack_n, ev_n):
        errs.append("NIST evidence is not a verbatim span of the control")

    return (not errs), errs


def grade(j: Dict[str, Any], ok: bool, errs: Sequence[str], tau: float) -> str:
    if not ok:
        return "rejected"
    return "accepted" if float(j.get("confidence", 0)) >= tau else "disputed"


def parse_json_array(text: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
    i, k = t.find("["), t.rfind("]")
    if i == -1 or k == -1 or k < i:
        return None, "no JSON array in the response"
    try:
        data = json.loads(t[i : k + 1])
    except json.JSONDecodeError as e:
        return None, f"malformed JSON: {e}"
    if not isinstance(data, list):
        return None, "top level is not an array"
    return [d for d in data if isinstance(d, dict)], ""


# --------------------------------------------------------------------------
def mock_response(clause: Dict[str, Any], row: Dict[str, Any], controls: Dict[str, Any]) -> str:
    """Deterministic stand-in so the whole pipeline — verification, grading,
    checkpointing, reporting — can be exercised without spending API budget."""
    out = []
    for rank, cand in enumerate(row["candidates"][:3], 1):
        c = controls.get(cand["control_id"])
        if not c or not c["statement"]:
            continue
        src = clause.get("full_text") or clause["text"]
        ev_s = " ".join(src.split()[:14])
        ev_n = " ".join(c["statement"].split()[:14])
        if rank == 3:                      # exercise the rejection path
            ev_n = "a span that does not appear anywhere in this control text"
        out.append({
            "control_id": c["control_id"],
            "relationship": ["subset_of", "intersects_with", "equal"][rank - 1],
            "confidence": [0.82, 0.55, 0.71][rank - 1],
            "evidence_sama": ev_s,
            "evidence_nist": ev_n,
            "parameter_gap": bool(clause.get("numeric_params") and c["n_params"]),
            "parameter_note": (c["params"][0] if c["params"] else ""),
            "rationale": "mock judgment for plumbing test",
        })
    return json.dumps(out)


# --------------------------------------------------------------------------
def load_all():
    if not CANDIDATES.exists():
        sys.exit("ERROR: candidates.jsonl not found. Run `python candidates.py generate` first.")
    rows = [json.loads(l) for l in CANDIDATES.open(encoding="utf-8") if l.strip()]
    clauses = {c["clause_id"]: c for c in rt.load_jsonl(rt.CLAUSES, "Run sama.py parse.")}
    controls = {c["control_id"]: c for c in rt.load_jsonl(rt.CONTROLS, "Run oscal.py flatten.")}
    return rows, clauses, controls


def cmd_dry_run(args) -> int:
    rows, clauses, controls = load_all()
    row = next((r for r in rows if r["clause_id"].lower() == args.clause_id.lower()), None)
    if not row:
        sys.exit(f"no shortlist for {args.clause_id}")
    prompt = build_prompt(clauses[row["clause_id"]], row, controls)
    print("=" * 70 + "\nSYSTEM\n" + "=" * 70)
    print(SYSTEM)
    print("\n" + "=" * 70 + "\nUSER\n" + "=" * 70)
    print(prompt)
    print("\n" + "=" * 70)
    print(f"approx input tokens: {(len(SYSTEM) + len(prompt)) // 4:,}")
    return 0


def cmd_judge(args) -> int:
    rows, clauses, controls = load_all()
    if args.subdomain:
        rows = [r for r in rows if r["subdomain"] == args.subdomain]
    done: set = set()
    if FINDINGS.exists() and not args.restart:
        kept, failed = [], []
        for l in FINDINGS.open(encoding="utf-8"):
            if not l.strip():
                continue
            rec = json.loads(l)
            (failed if rec.get("parse_error") else kept).append(rec)
        if failed and not args.keep_failed:
            # A failed call is not a result, so it must not count as done.
            # Its record is dropped here so the clause returns to the queue
            # instead of being silently skipped forever.
            with FINDINGS.open("w", encoding="utf-8") as fh:
                for rec in kept:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"dropped {len(failed)} failed record(s) for retry: "
                  + ", ".join(r["clause_id"] for r in failed[:5])
                  + (" …" if len(failed) > 5 else ""))
        done = {r["clause_id"] for r in kept}
        if args.keep_failed:
            done |= {r["clause_id"] for r in failed}
        if done:
            print(f"resuming: {len(done)} clause(s) already judged")
    todo = [r for r in rows if r["clause_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("nothing to do.")
        return 0

    model = Model(args.provider, args.model, mock=args.mock,
                  temperature=args.temperature)
    tag = "MOCK" if args.mock else f"{args.provider}:{args.model}"
    print(f"judging {len(todo)} clause(s) with {tag}, tau={args.tau}\n")

    stats: Counter = Counter()
    t0 = time.time()
    with FINDINGS.open("a", encoding="utf-8") as fh:
        for n, row in enumerate(todo, 1):
            cid = row["clause_id"]
            clause = clauses.get(cid)
            if clause is None:
                continue
            offered = [c["control_id"] for c in row["candidates"]]
            prompt = build_prompt(clause, row, controls)

            try:
                raw = (mock_response(clause, row, controls) if args.mock
                       else model.complete(SYSTEM, prompt, max_tokens=args.max_tokens))
                err = ""
            except Exception as exc:  # noqa: BLE001 — one bad call must not end the run
                raw, err = "", f"{type(exc).__name__}: {exc}"

            judgments, perr = ([], err) if err else parse_json_array(raw)
            if judgments is None:
                judgments, perr = [], perr or "unparseable response"

            findings = []
            for j in judgments:
                ok, errs = verify(j, clause, offered, controls)
                status = grade(j, ok, errs, args.tau)
                stats[status] += 1
                if ok:
                    stats[f"rel:{j.get('relationship')}"] += 1
                    if j.get("parameter_gap"):
                        stats["parameter_gap"] += 1
                findings.append({
                    "control_id": str(j.get("control_id", "")).upper(),
                    "relationship": str(j.get("relationship", "")).lower(),
                    "confidence": j.get("confidence"),
                    "evidence_sama": j.get("evidence_sama", ""),
                    "evidence_nist": j.get("evidence_nist", ""),
                    "parameter_gap": bool(j.get("parameter_gap")),
                    "parameter_note": j.get("parameter_note", ""),
                    "rationale": j.get("rationale", ""),
                    "status": status,
                    "gate_errors": errs,
                })
            if perr:
                stats["call_failed"] += 1
                if stats["call_failed"] <= 3:
                    print(f"  ! {cid}: {perr[:160]}")
            elif not findings:
                stats["no_relation"] += 1

            fh.write(json.dumps({
                "clause_id": cid,
                "subdomain": row["subdomain"],
                "clause_text": clause["text"],
                "offered": offered,
                "findings": findings,
                "model": tag,
                "parse_error": perr,
            }, ensure_ascii=False) + "\n")
            fh.flush()   # checkpoint every clause; Colab sessions end abruptly

            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                eta = el / n * (len(todo) - n)
                print(f"  {n}/{len(todo)}  accepted {stats['accepted']} "
                      f"disputed {stats['disputed']} rejected {stats['rejected']}  "
                      f"eta {eta/60:.1f} min")

    print(f"\naccepted {stats['accepted']} | disputed {stats['disputed']} "
          f"| rejected {stats['rejected']} | no relation {stats['no_relation']} "
          f"| CALLS FAILED {stats['call_failed']}")
    if stats["call_failed"]:
        print("\n*** Failed calls are not findings. Nothing was judged for those")
        print("*** clauses. Fix the cause, delete findings.jsonl, and re-run.")
        print("*** Inspect the reasons with:  python judge.py errors")
    if stats["parameter_gap"]:
        print(f"parameter-gap findings: {stats['parameter_gap']}")
    if stats["rejected"]:
        print("\nRejections are kept, not discarded. They are the evidence that")
        print("the gate does something, and their reasons are worth reporting.")
    return 0


def cmd_show(args) -> int:
    if not FINDINGS.exists():
        sys.exit("Run `python judge.py judge` first.")
    for l in FINDINGS.open(encoding="utf-8"):
        r = json.loads(l)
        if r["clause_id"].lower() != args.clause_id.lower():
            continue
        print("=" * 70)
        print(f"{r['clause_id']}   ({r['subdomain']})")
        print(f"{r['clause_text']}\n")
        if r.get("parse_error"):
            print(f"parse error: {r['parse_error']}\n")
        if not r["findings"]:
            print("no related control found among the candidates offered.")
            return 0
        for f in r["findings"]:
            mark = {"accepted": "[+]", "disputed": "[?]", "rejected": "[-]"}[f["status"]]
            print(f"{mark} {f['control_id']:<12} {f['relationship']:<16} "
                  f"conf {f['confidence']}")
            print(f"      SAMA: \u201c{f['evidence_sama'][:110]}\u201d")
            print(f"      NIST: \u201c{f['evidence_nist'][:110]}\u201d")
            if f["parameter_gap"]:
                print(f"      PARAMETER GAP: {f['parameter_note']}")
            if f["gate_errors"]:
                for e in f["gate_errors"]:
                    print(f"      gate: {e}")
            print(f"      {f['rationale'][:110]}")
            print()
        return 0
    print(f"no findings recorded for {args.clause_id}")
    return 1


def cmd_errors(args) -> int:
    """Show clauses whose call or parse failed, with the reason."""
    if not FINDINGS.exists():
        sys.exit("Run `python judge.py judge` first.")
    rows = [json.loads(l) for l in FINDINGS.open(encoding="utf-8") if l.strip()]
    bad = [r for r in rows if r.get("parse_error")]
    if not bad:
        print(f"no call or parse errors in {len(rows)} record(s).")
        empty = [r for r in rows if not r["findings"]]
        print(f"{len(empty)} clause(s) returned an empty array — that is a real")
        print("answer from the model, not a failure.")
        return 0
    print(f"{len(bad)} of {len(rows)} clause(s) failed\n")
    for kind, n in Counter(r["parse_error"].split(":")[0] for r in bad).most_common():
        print(f"  {n:>4}  {kind}")
    print("\nfirst few:")
    for r in bad[:5]:
        print(f"  {r['clause_id']:<18} {r['parse_error'][:150]}")
    return 0


def cmd_report(args) -> int:
    if not FINDINGS.exists():
        sys.exit("Run `python judge.py judge` first.")
    rows = [json.loads(l) for l in FINDINGS.open(encoding="utf-8") if l.strip()]
    failed = [r for r in rows if r.get("parse_error")]
    if failed:
        print(f"WARNING: {len(failed)} of {len(rows)} clause(s) had a failed call or")
        print("unparseable response. Those are not findings. See `judge.py errors`.\n")
    allf = [f for r in rows for f in r["findings"]]
    st = Counter(f["status"] for f in allf)
    total = max(len(allf), 1)

    print("=" * 62)
    print("JUDGMENT REPORT")
    print("=" * 62)
    print(f"clauses judged   : {len(rows)}")
    print(f"judgments        : {len(allf)}")
    for s in ("accepted", "disputed", "rejected"):
        print(f"  {s:<14} {st[s]:>5}  ({100*st[s]//total}%)")
    print(f"clauses with no relation found: "
          f"{sum(1 for r in rows if not r['findings'])}")

    ok = [f for f in allf if f["status"] in ("accepted", "disputed")]
    print("\nrelationship mix (verified judgments)")
    for rel, n in Counter(f["relationship"] for f in ok).most_common():
        print(f"  {rel:<18} {n:>5}")

    acc = [f for f in allf if f["status"] == "accepted"]
    print(f"\ndistinct controls matched : {len({f['control_id'] for f in acc})}")
    fam = Counter(f["control_id"].split("-")[0] for f in acc)
    print("by family")
    for f, n in fam.most_common():
        print(f"  {f:<4} {n:>5}")

    pg = [f for f in acc if f["parameter_gap"]]
    if pg:
        print(f"\nparameter-gap findings: {len(pg)}")
        for f in pg[:10]:
            print(f"  {f['control_id']:<12} {f['parameter_note'][:60]}")

    rej = [f for f in allf if f["status"] == "rejected"]
    if rej:
        print(f"\nwhy the gate rejected ({len(rej)})")
        for e, n in Counter(e for f in rej for e in f["gate_errors"]).most_common():
            print(f"  {n:>5}  {e}")

    covered = {r["clause_id"] for r in rows if any(
        f["status"] == "accepted" for f in r["findings"])}
    print(f"\nclauses with at least one accepted finding: {len(covered)}/{len(rows)} "
          f"({100*len(covered)//max(len(rows),1)}%)")
    print("\nA clause with no accepted finding is a coverage gap candidate — but")
    print("check whether it is referential before counting it as one.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dry-run", help="print the prompt for one clause")
    p.add_argument("clause_id")

    p = sub.add_parser("judge", help="run the judge (resumable)")
    p.add_argument("--provider", default="anthropic", choices=["anthropic", "openai"])
    p.add_argument("--model", default="claude-sonnet-5",
                   help="anthropic: claude-sonnet-5 | claude-haiku-4-5-20251001 | claude-opus-5")
    p.add_argument("--mock", action="store_true", help="no API; exercise the pipeline")
    p.add_argument("--limit", type=int, help="only this many clauses")
    p.add_argument("--subdomain", help="only this subdomain, e.g. 3.3.5")
    p.add_argument("--tau", type=float, default=TAU_ACCEPT)
    p.add_argument("--max-tokens", type=int, default=4000)
    p.add_argument("--temperature", type=float, default=None,
                   help="omitted by default; newer models reject it")
    p.add_argument("--restart", action="store_true", help="ignore existing findings")
    p.add_argument("--keep-failed", action="store_true",
                   help="do not retry clauses whose call failed")

    p = sub.add_parser("show", help="findings for one clause")
    p.add_argument("clause_id")

    sub.add_parser("errors", help="show failed calls and their reasons")
    sub.add_parser("report", help="acceptance rates and relationship mix")

    args = ap.parse_args()
    return {"dry-run": cmd_dry_run, "judge": cmd_judge, "show": cmd_show,
            "errors": cmd_errors, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
