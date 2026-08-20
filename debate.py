#!/usr/bin/env python3
"""
debate.py — an adversarial layer over the judgments that were not decisive.

judge.py is a single judge: it proposes a relationship and evidence, and a
deterministic gate either accepts that or does not. It has no way to be argued
with, and 61% of its verified judgments came back `intersects_with` — the
answer available when a case is not thought through hard enough to say which
side contains the other.

This layer adds two more agents and lets them disagree.

    proposer   the existing finding, taken as the opening position
    opponent   a DIFFERENT model, told to attack it: argue the relationship is
               wrong, or that the containment runs the other way, or that
               there is no relationship at all
    arbiter    a third call that sees both and rules, or declines to rule

What makes this agentic rather than a longer pipeline is that the opponent's
output changes what the arbiter decides, and both may abstain — the opponent by
finding no objection, the arbiter by ruling the case unresolved. Abstention is a
decision, not a failure to execute.

Two models by design. An opponent running on the same model as the proposer
tends to agree with it: same priors, same blind spots. Cross-provider debate is
the difference between adversarial review and a model marking its own work.

Scope is deliberately narrow. Findings already accepted with high confidence
are not re-litigated; the debate runs on the provisional band and on clauses
where nothing was found, which is exactly where a single judge is weakest.

Nothing here can widen what the gate allows. Every arbiter ruling passes the
same verification as the original: the control must have been a candidate, the
relationship must be one of the five, and both evidence spans must appear
verbatim. The debate can change a verdict; it cannot lower the bar.

Subcommands
-----------
  dry-run   print the opponent and arbiter prompts for one clause
  run       debate the selected findings -> processed/findings_v2.jsonl
  report    what changed, and whether it helped
  compare   findings.jsonl against findings_v2.jsonl, side by side

  python debate.py dry-run 3.3.5-4.b.4
  python debate.py run --mock --limit 10
  python debate.py run --band provisional --limit 25
  python debate.py report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import judge as J
import router as rt

KB = Path(__file__).resolve().parent
FINDINGS = KB / "processed" / "findings.jsonl"
CANDIDATES = KB / "processed" / "candidates.jsonl"
OUT = KB / "processed" / "findings_v2.jsonl"

PROVISIONAL_LOW, PROVISIONAL_HIGH = 0.50, 0.70


# --------------------------------------------------------------------------
# opponent
# --------------------------------------------------------------------------
OPPONENT_SYSTEM = """You are a regulatory analyst reviewing someone else's mapping between the SAMA Cyber Security Framework (Saudi Arabia, 2017) and NIST SP 800-53 Rev 5.2.0. Your role is adversarial: find what is wrong with the proposed mapping.

Relationship types, with the SAMA clause as the focal element:
- equal: same requirement, same scope.
- subset_of: the SAMA clause is narrower; the NIST control covers it and more.
- superset_of: the SAMA clause is broader; it covers the control and more.
- intersects_with: they overlap; neither contains the other.
- not_related: no meaningful relationship.

Attack the proposal on whichever of these grounds applies:

1. WRONG TYPE. `intersects_with` is the answer people reach for when they have not worked out which side is broader. If everything the SAMA clause requires is covered by the control, the relationship is subset_of, not intersects_with. To defend intersects_with, you must be able to name something each side requires that the other does not. If you cannot name both, the type is wrong.
2. WRONG DIRECTION. Check that subset_of and superset_of are not reversed.
3. NOT RELATED AT ALL. The two may share vocabulary while requiring different things.
4. WRONG CONTROL. A different control from the candidate list may fit better.
5. MISREAD EVIDENCE. The quoted span may not support what is claimed of it.

If the proposal is sound, say so. A reviewer who objects to everything is as useless as one who objects to nothing.

Return ONLY a JSON object, no prose, no code fence:

{"objection": true, "grounds": "wrong_type|wrong_direction|not_related|wrong_control|misread_evidence", "argument": "two sentences at most, citing the texts", "proposed_relationship": "equal|subset_of|superset_of|intersects_with|not_related", "proposed_control_id": "only if grounds is wrong_control, else empty", "strength": 0.0}

To raise no objection: {"objection": false, "grounds": "", "argument": "why it is sound", "proposed_relationship": "", "proposed_control_id": "", "strength": 0.0}

`strength` is how likely you think a domain expert would side with you, 0.0 to 1.0."""


ARBITER_SYSTEM = """You are the deciding analyst. You see a proposed mapping between the SAMA Cyber Security Framework and NIST SP 800-53, and an objection to it. Rule on which is right.

Relationship types, with the SAMA clause as the focal element:
- equal: same requirement, same scope.
- subset_of: the SAMA clause is narrower; the NIST control covers it and more.
- superset_of: the SAMA clause is broader; it covers the control and more.
- intersects_with: they overlap; neither contains the other. Only correct when you can name something each side requires that the other does not.
- not_related: no meaningful relationship.

Rules:

1. Decide from the texts in front of you, not from knowledge of either framework.
2. Your evidence must be quoted VERBATIM — an exact character-for-character substring of the text given, at least 15 characters from each side. No paraphrase, no ellipsis.
3. Only the controls listed as candidates may be cited.
4. If neither position is well supported, rule `unresolved`. That is a legitimate outcome and is preferred to a confident answer you cannot evidence.
5. confidence is your probability that a domain expert would agree with your ruling.

Return ONLY a JSON object, no prose, no code fence:

{"ruling": "uphold|revise|reject|unresolved", "relationship": "equal|subset_of|superset_of|intersects_with|not_related", "control_id": "...", "confidence": 0.0, "evidence_sama": "verbatim span", "evidence_nist": "verbatim span", "reasoning": "one or two sentences"}

uphold: the original stands. revise: the relationship or control changes. reject: there is no relationship. unresolved: the texts do not settle it."""


# --------------------------------------------------------------------------
def control_block(c: Dict[str, Any]) -> str:
    out = [f"### {c['control_id']} — {c['title']}",
           f"Family: {c['family']} ({c['family_title']})"]
    if c["is_enhancement"]:
        out.append(f"Enhancement of {c['parent_id']}")
    out.append(f"Statement: {c['statement'] or '(none)'}")
    if c["params"]:
        out.append("Organization-defined parameters: " + "; ".join(str(p) for p in c["params"]))
    return "\n".join(out)


def clause_block(cl: Dict[str, Any]) -> str:
    out = [f"Identifier: {cl['clause_id']}",
           f"Subdomain: {cl['subdomain']} {cl['subdomain_title']}"]
    if cl.get("principle"):
        out.append(f"Subdomain principle: {cl['principle']}")
    out.append(f"\nClause text:\n{cl['text']}")
    if cl.get("n_children"):
        out.append(f"\nWith its sub-items:\n{cl['full_text']}")
    if cl.get("numeric_params"):
        out.append(f"\nStates a concrete value: {', '.join(cl['numeric_params'])}")
    return "\n".join(out)


def opponent_prompt(cl, finding, controls, offered) -> str:
    ctrl = controls.get(finding["control_id"], {})
    parts = ["## SAMA clause", clause_block(cl),
             "\n## The proposed mapping",
             f"Control: {finding['control_id']}",
             f"Relationship: {finding['relationship']}",
             f"Confidence: {finding.get('confidence')}",
             f"Evidence from SAMA: \u201c{finding.get('evidence_sama','')}\u201d",
             f"Evidence from NIST: \u201c{finding.get('evidence_nist','')}\u201d",
             f"Stated reason: {finding.get('rationale','')}",
             "\n## The control as proposed"]
    if ctrl:
        parts.append(control_block(ctrl))
    others = [o for o in offered if o != finding["control_id"]][:6]
    if others:
        parts.append("\n## Other controls that were available")
        for cid in others:
            c = controls.get(cid)
            if c:
                parts.append(f"- {c['control_id']} — {c['title']}: {c['statement'][:220]}")
    parts.append("\nReturn the JSON object now.")
    return "\n".join(parts)


def arbiter_prompt(cl, finding, objection, controls, offered) -> str:
    parts = ["## SAMA clause", clause_block(cl),
             "\n## Position A — the proposal",
             f"Control: {finding['control_id']}  ·  Relationship: {finding['relationship']}",
             f"Evidence (SAMA): \u201c{finding.get('evidence_sama','')}\u201d",
             f"Evidence (NIST): \u201c{finding.get('evidence_nist','')}\u201d",
             f"Reason: {finding.get('rationale','')}",
             "\n## Position B — the objection"]
    if objection.get("objection"):
        parts += [f"Grounds: {objection.get('grounds')}",
                  f"Argument: {objection.get('argument')}",
                  f"Proposed instead: {objection.get('proposed_relationship') or '(unchanged)'}"
                  + (f" on {objection['proposed_control_id']}"
                     if objection.get("proposed_control_id") else ""),
                  f"Stated strength: {objection.get('strength')}"]
    else:
        parts.append(f"No objection raised. Reason given: {objection.get('argument','')}")

    parts.append("\n## Candidate controls (only these may be cited)")
    shown = [finding["control_id"]]
    if objection.get("proposed_control_id"):
        shown.append(objection["proposed_control_id"])
    shown += [o for o in offered if o not in shown][:5]
    for cid in shown:
        c = controls.get(cid)
        if c:
            parts.append(control_block(c) + "\n")
    parts.append("Return the JSON object now.")
    return "\n".join(parts)


# --------------------------------------------------------------------------
def parse_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    import re

    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip()).strip()
    i, k = t.find("{"), t.rfind("}")
    if i == -1 or k == -1 or k < i:
        return None, "no JSON object in the response"
    try:
        obj = json.loads(t[i : k + 1])
    except json.JSONDecodeError as e:
        return None, f"malformed JSON: {e}"
    return (obj, "") if isinstance(obj, dict) else (None, "not an object")


def select(findings_rows, band: str, tau: float) -> List[Tuple[Dict, Dict]]:
    """Which findings go to debate.

    provisional  verified but under-confident — where a single judge is weakest
    disputed     everything below tau
    intersects   only the intersects_with findings, the specific failure mode
    all          every verified finding
    """
    out = []
    for r in findings_rows:
        for f in r["findings"]:
            if f["status"] == "rejected":
                continue
            conf = f.get("confidence") or 0
            keep = (
                band == "all"
                or (band == "disputed" and conf < tau)
                or (band == "provisional" and PROVISIONAL_LOW <= conf < PROVISIONAL_HIGH)
                or (band == "intersects" and f["relationship"] == "intersects_with")
            )
            if keep:
                out.append((r, f))
    return out


# --------------------------------------------------------------------------
def mock_objection(finding) -> str:
    if finding["relationship"] == "intersects_with":
        return json.dumps({
            "objection": True, "grounds": "wrong_type",
            "argument": "mock: nothing is named that SAMA requires and the control does not",
            "proposed_relationship": "subset_of", "proposed_control_id": "", "strength": 0.7})
    return json.dumps({"objection": False, "grounds": "", "argument": "mock: sound",
                       "proposed_relationship": "", "proposed_control_id": "", "strength": 0.0})


def mock_ruling(cl, finding, objection) -> str:
    src = cl.get("full_text") or cl["text"]
    if objection.get("objection"):
        return json.dumps({
            "ruling": "revise",
            "relationship": objection.get("proposed_relationship") or finding["relationship"],
            "control_id": finding["control_id"], "confidence": 0.72,
            "evidence_sama": " ".join(src.split()[:14]),
            "evidence_nist": finding.get("evidence_nist", ""),
            "reasoning": "mock ruling"})
    return json.dumps({
        "ruling": "uphold", "relationship": finding["relationship"],
        "control_id": finding["control_id"],
        "confidence": min(1.0, (finding.get("confidence") or 0.5) + 0.1),
        "evidence_sama": finding.get("evidence_sama", ""),
        "evidence_nist": finding.get("evidence_nist", ""),
        "reasoning": "mock ruling"})


# --------------------------------------------------------------------------
def cmd_dry_run(args) -> int:
    rows = [json.loads(l) for l in FINDINGS.open(encoding="utf-8") if l.strip()]
    clauses = {c["clause_id"]: c for c in rt.load_jsonl(rt.CLAUSES, "")}
    controls = {c["control_id"]: c for c in rt.load_jsonl(rt.CONTROLS, "")}
    row = next((r for r in rows if r["clause_id"].lower() == args.clause_id.lower()), None)
    if not row or not row["findings"]:
        sys.exit(f"no findings for {args.clause_id}")
    f = row["findings"][0]
    cl = clauses[row["clause_id"]]

    print("=" * 70 + "\nOPPONENT — system\n" + "=" * 70)
    print(OPPONENT_SYSTEM)
    print("\n" + "=" * 70 + "\nOPPONENT — user\n" + "=" * 70)
    p1 = opponent_prompt(cl, f, controls, row["offered"])
    print(p1)
    obj = json.loads(mock_objection(f))
    print("\n" + "=" * 70 + "\nARBITER — user (with a mock objection)\n" + "=" * 70)
    p2 = arbiter_prompt(cl, f, obj, controls, row["offered"])
    print(p2)
    print("\n" + "=" * 70)
    print(f"approx input tokens: opponent {(len(OPPONENT_SYSTEM)+len(p1))//4:,}, "
          f"arbiter {(len(ARBITER_SYSTEM)+len(p2))//4:,}")
    return 0


def cmd_run(args) -> int:
    if not FINDINGS.exists():
        sys.exit("Run `python judge.py judge` first.")
    rows = [json.loads(l) for l in FINDINGS.open(encoding="utf-8") if l.strip()]
    clauses = {c["clause_id"]: c for c in rt.load_jsonl(rt.CLAUSES, "")}
    controls = {c["control_id"]: c for c in rt.load_jsonl(rt.CONTROLS, "")}

    picked = select(rows, args.band, args.tau)
    done: set = set()
    if OUT.exists() and not args.restart:
        for l in OUT.open(encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                done.add((d["clause_id"], d["control_id"]))
        if done:
            print(f"resuming: {len(done)} already debated")
    todo = [(r, f) for r, f in picked if (r["clause_id"], f["control_id"]) not in done]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("nothing to do.")
        return 0

    opp = J.Model(args.opponent_provider, args.opponent_model, mock=args.mock)
    arb = J.Model(args.arbiter_provider, args.arbiter_model, mock=args.mock)
    print(f"debating {len(todo)} finding(s) from band '{args.band}'")
    print(f"  opponent: {'MOCK' if args.mock else args.opponent_provider + ':' + args.opponent_model}")
    print(f"  arbiter : {'MOCK' if args.mock else args.arbiter_provider + ':' + args.arbiter_model}\n")

    stats: Counter = Counter()
    t0 = time.time()
    with OUT.open("a", encoding="utf-8") as fh:
        for n, (row, f) in enumerate(todo, 1):
            cl = clauses.get(row["clause_id"])
            if cl is None:
                continue
            offered = row.get("offered") or []
            err = ""

            # --- opponent
            try:
                raw = (mock_objection(f) if args.mock
                       else opp.complete(OPPONENT_SYSTEM,
                                         opponent_prompt(cl, f, controls, offered),
                                         max_tokens=1200))
            except Exception as exc:  # noqa: BLE001
                raw, err = "", f"opponent: {type(exc).__name__}: {exc}"
            obj, perr = ({}, err) if err else parse_json_object(raw)
            if obj is None:
                obj, perr = {}, perr or "opponent unparseable"

            objected = bool(obj.get("objection"))
            stats["objection" if objected else "no_objection"] += 1

            # --- arbiter
            ruling: Dict[str, Any] = {}
            if not perr:
                try:
                    raw2 = (mock_ruling(cl, f, obj) if args.mock
                            else arb.complete(ARBITER_SYSTEM,
                                              arbiter_prompt(cl, f, obj, controls, offered),
                                              max_tokens=1500))
                    ruling, perr = parse_json_object(raw2)
                    ruling = ruling or {}
                except Exception as exc:  # noqa: BLE001
                    perr = f"arbiter: {type(exc).__name__}: {exc}"

            verdict = str(ruling.get("ruling", "")).lower()
            status, errs = "unresolved", []
            if verdict in ("uphold", "revise"):
                # the arbiter's ruling faces the same gate as the original
                cand = {
                    "control_id": ruling.get("control_id") or f["control_id"],
                    "relationship": ruling.get("relationship") or f["relationship"],
                    "confidence": ruling.get("confidence"),
                    "evidence_sama": ruling.get("evidence_sama", ""),
                    "evidence_nist": ruling.get("evidence_nist", ""),
                }
                ok, errs = J.verify(cand, cl, offered, controls)
                status = J.grade(cand, ok, errs, args.tau)
            elif verdict == "reject":
                status = "rejected_by_debate"
            stats[f"verdict:{verdict or 'none'}"] += 1
            stats[f"status:{status}"] += 1

            changed = (verdict == "revise"
                       and (ruling.get("relationship") != f["relationship"]
                            or (ruling.get("control_id") or f["control_id"]) != f["control_id"]))
            if changed:
                stats["changed"] += 1
                stats[f"shift:{f['relationship']}->{ruling.get('relationship')}"] += 1

            fh.write(json.dumps({
                "clause_id": row["clause_id"],
                "subdomain": row["subdomain"],
                "control_id": f["control_id"],
                "original": {k: f.get(k) for k in
                             ("relationship", "confidence", "status", "rationale")},
                "objection": obj,
                "ruling": ruling,
                "verdict": verdict,
                "final": {
                    "control_id": ruling.get("control_id") or f["control_id"],
                    "relationship": ruling.get("relationship") or f["relationship"],
                    "confidence": ruling.get("confidence", f.get("confidence")),
                    "evidence_sama": ruling.get("evidence_sama", f.get("evidence_sama", "")),
                    "evidence_nist": ruling.get("evidence_nist", f.get("evidence_nist", "")),
                    "status": status,
                },
                "changed": changed,
                "gate_errors": errs,
                "error": perr,
                "models": {"opponent": ("mock" if args.mock else args.opponent_model),
                           "arbiter": ("mock" if args.mock else args.arbiter_model)},
            }, ensure_ascii=False) + "\n")
            fh.flush()

            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  objections {stats['objection']}  "
                      f"changed {stats['changed']}  "
                      f"eta {(el/n*(len(todo)-n))/60:.1f} min")

    print(f"\nobjections raised : {stats['objection']} of {len(todo)}")
    print(f"verdicts          : " + ", ".join(
        f"{k.split(':')[1]}={v}" for k, v in stats.items() if k.startswith("verdict:")))
    print(f"verdict changed the mapping: {stats['changed']}")
    shifts = {k.split(':', 1)[1]: v for k, v in stats.items() if k.startswith("shift:")}
    if shifts:
        print("\nrelationship shifts")
        for k, v in sorted(shifts.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<40} {v}")
    print(f"\nwrote -> {OUT.relative_to(KB)}")
    print("\nfindings.jsonl is untouched. Compare with `python debate.py compare`.")
    return 0


def cmd_report(args) -> int:
    if not OUT.exists():
        sys.exit("Run `python debate.py run` first.")
    rows = [json.loads(l) for l in OUT.open(encoding="utf-8") if l.strip()]
    n = len(rows)
    obj = sum(1 for r in rows if r["objection"].get("objection"))
    changed = sum(1 for r in rows if r["changed"])

    print("=" * 62)
    print("DEBATE REPORT")
    print("=" * 62)
    print(f"findings debated   : {n}")
    print(f"objections raised  : {obj} ({100*obj//max(n,1)}%)")
    print(f"mappings changed   : {changed} ({100*changed//max(n,1)}%)")

    print("\nverdicts")
    for k, v in Counter(r["verdict"] or "(none)" for r in rows).most_common():
        print(f"  {k:<20} {v:>4}")

    print("\ngrounds of objection")
    for k, v in Counter(r["objection"].get("grounds", "") for r in rows
                        if r["objection"].get("objection")).most_common():
        print(f"  {k or '(unstated)':<20} {v:>4}")

    print("\nrelationship before -> after (changed only)")
    for r in rows:
        if r["changed"]:
            print(f"  {r['clause_id']:<18} {r['control_id']:<10} "
                  f"{r['original']['relationship']:<16} -> {r['final']['relationship']}")

    print("\nstatus after the gate")
    for k, v in Counter(r["final"]["status"] for r in rows).most_common():
        print(f"  {k:<22} {v:>4}")

    ge = [e for r in rows for e in r["gate_errors"]]
    if ge:
        print(f"\nrulings blocked by the gate: {len(ge)}")
        for k, v in Counter(ge).most_common(5):
            print(f"  {v:>4}  {k}")
        print("  The debate can change a verdict; it cannot lower the evidence bar.")

    err = [r for r in rows if r.get("error")]
    if err:
        print(f"\ncalls that failed: {len(err)}")
        for r in err[:3]:
            print(f"  {r['clause_id']}: {r['error'][:120]}")
    return 0


def cmd_compare(args) -> int:
    if not OUT.exists():
        sys.exit("Run `python debate.py run` first.")
    rows = [json.loads(l) for l in OUT.open(encoding="utf-8") if l.strip()]
    before = Counter(r["original"]["relationship"] for r in rows)
    after = Counter(r["final"]["relationship"] for r in rows)
    keys = sorted(set(before) | set(after))
    print(f"{'relationship':<20}{'single judge':>14}{'after debate':>14}{'delta':>8}")
    for k in keys:
        d = after[k] - before[k]
        print(f"  {k:<18}{before[k]:>14}{after[k]:>14}{d:>+8}")
    print(f"\n{'confidence':<20}{'before':>14}{'after':>14}")
    b = [r["original"]["confidence"] or 0 for r in rows]
    a = [r["final"]["confidence"] or 0 for r in rows]
    print(f"  {'mean':<18}{sum(b)/max(len(b),1):>14.3f}{sum(a)/max(len(a),1):>14.3f}")
    acc_b = sum(1 for r in rows if r["original"]["status"] == "accepted")
    acc_a = sum(1 for r in rows if r["final"]["status"] == "accepted")
    print(f"  {'accepted':<18}{acc_b:>14}{acc_a:>14}{acc_a-acc_b:>+8}")
    print("\nA shift out of intersects_with is the result to look for: it is the")
    print("verdict a single judge reaches when it has not worked out which side")
    print("contains the other.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dry-run", help="print both prompts for one clause")
    p.add_argument("clause_id")

    p = sub.add_parser("run", help="debate selected findings")
    p.add_argument("--band", default="provisional",
                   choices=["provisional", "disputed", "intersects", "all"])
    p.add_argument("--opponent-provider", default="openai", choices=["anthropic", "openai"])
    p.add_argument("--opponent-model", default="gpt-4o")
    p.add_argument("--arbiter-provider", default="anthropic", choices=["anthropic", "openai"])
    p.add_argument("--arbiter-model", default="claude-sonnet-5")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--tau", type=float, default=J.TAU_ACCEPT)
    p.add_argument("--restart", action="store_true")

    sub.add_parser("report", help="what the debate changed")
    sub.add_parser("compare", help="single judge against debate, side by side")

    args = ap.parse_args()
    return {"dry-run": cmd_dry_run, "run": cmd_run,
            "report": cmd_report, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
