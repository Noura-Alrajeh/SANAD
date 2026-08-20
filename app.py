#!/usr/bin/env python3
"""
app.py — SANAD dashboard.

Four tabs over work that is already finished. Nothing here calls a language
model or an embedding model except the Ask tab, and that one loads lazily. The
consequence is the point: if the network dies during a demo, three of the four
tabs keep working, because they are reading files.

    Gaps          what SAMA does not cover, and what it covers differently
    Mapping       browse a clause and the controls matched to it, with evidence
    Ask           grounded question answering over the policy corpus
    Knowledge base  the manifest and its ITU factor and dimension coverage

Every figure shown traces to a file on disk: findings.jsonl, sama_clauses.jsonl,
controls.jsonl, manifest.csv. A number nobody can trace is a number nobody
should believe, so each tab names the file it came from.

  python app.py selftest    # data layer only, no gradio, no network
  python app.py             # launch
"""

from __future__ import annotations

import csv
import json
import re
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

KB = Path(__file__).resolve().parent
P = KB / "processed"

# Families SAMA hands to another instrument rather than leaving uncovered.
# Excluding these from the gap count is the difference between a defensible
# report and an indefensible one.
DOCUMENTED_EXCLUSIONS = {
    "CP": "SAMA CSF 1.3 refers business continuity to the SAMA Business "
          "Continuity Minimum Requirements, a separate instrument.",
}

# Subdomains whose clauses point at other standards instead of stating a
# requirement, but which the parser scored as substantive because they are
# grammatically ordinary numbered items. Reported rather than silently fixed.
REFERENTIAL_IN_SUBSTANCE = {
    "3.2.3": "States only that the organisation shall comply with PCI-DSS, EMV "
             "and SWIFT CSCF. It delegates rather than requires, so its "
             "clauses have no mappable content.",
}

# Controls added in Rev 5.2.0 (August 2025), eight years after SAMA CSF.
REV_520_ADDITIONS = {"SA-15(13)", "SA-24", "SI-02(07)", "SI-2(7)"}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


class Data:
    """Everything the dashboard shows, loaded once."""

    def __init__(self) -> None:
        # findings.jsonl holds the proposer's verdicts. Where a finding was
        # debated, findings_v2.jsonl holds what the arbiter ruled after the
        # opponent argued against it, and that ruling is the system's output.
        # Merging here rather than in the pipeline keeps both files intact, so
        # the ablation can still be computed from the same directory.
        self.findings = _jsonl(P / "findings.jsonl")
        self.debate = _jsonl(P / "findings_v2.jsonl")
        self.n_debated = self._apply_debate()
        self.clauses = {c["clause_id"]: c for c in _jsonl(P / "sama_clauses.jsonl")}
        self.controls = {c["control_id"]: c for c in _jsonl(P / "controls.jsonl")}
        self.candidates = {r["clause_id"]: r for r in _jsonl(P / "candidates.jsonl")}
        self.manifest = self._manifest()

        self.live = {k: v for k, v in self.controls.items() if not v["withdrawn"]}
        self.by_clause: Dict[str, List[Dict[str, Any]]] = {
            r["clause_id"]: r["findings"] for r in self.findings
        }
        self.accepted = [
            (r["clause_id"], f)
            for r in self.findings
            for f in r["findings"]
            if f["status"] == "accepted"
        ]
        self.verified = [
            (r["clause_id"], f)
            for r in self.findings
            for f in r["findings"]
            if f["status"] in ("accepted", "disputed")
        ]

    def _apply_debate(self) -> int:
        """Overlay arbiter rulings onto the proposer's findings.

        Keyed on (clause, control): the arbiter may change the relationship,
        the confidence and the evidence, and may reject outright. A ruling that
        failed the gate is left as the proposer had it, since a blocked ruling
        is not an improvement on anything.
        """
        if not self.debate:
            return 0
        ruled = {}
        for r in self.debate:
            if r.get("error"):
                continue
            final = r.get("final", {})
            if final.get("status") in (None, "unresolved"):
                continue
            ruled[(r["clause_id"], r["control_id"])] = (final, r.get("verdict", ""))

        applied = 0
        for row in self.findings:
            for f in row["findings"]:
                key = (row["clause_id"], f["control_id"])
                if key not in ruled:
                    continue
                final, verdict = ruled[key]
                f["proposer"] = {k: f.get(k) for k in
                                 ("relationship", "confidence", "status")}
                f["relationship"] = final.get("relationship", f["relationship"])
                f["confidence"] = final.get("confidence", f.get("confidence"))
                f["evidence_sama"] = final.get("evidence_sama") or f.get("evidence_sama", "")
                f["evidence_nist"] = final.get("evidence_nist") or f.get("evidence_nist", "")
                f["status"] = ("rejected" if final["status"] == "rejected_by_debate"
                               else final["status"])
                f["verdict"] = verdict
                f["debated"] = True
                applied += 1
        return applied

    @staticmethod
    def _manifest() -> List[Dict[str, str]]:
        path = KB / "manifest.csv"
        if not path.exists():
            return []
        with path.open(encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))

    def ready(self) -> Tuple[bool, str]:
        missing = [
            n for n, d in (("findings.jsonl", self.findings),
                           ("sama_clauses.jsonl", self.clauses),
                           ("controls.jsonl", self.controls))
            if not d
        ]
        if missing:
            return False, "missing: " + ", ".join(missing)
        return True, ""


# --------------------------------------------------------------------------
# tab 1 — gaps
# --------------------------------------------------------------------------
# A clause whose shortlist contained nothing from a family plausibly related to
# its subdomain was never really tested. Calling that a gap asserts absence from
# the catalogue on the strength of a retrieval failure.
SUBDOMAIN_EXPECTED: Dict[str, set] = {
    "3.1.5": {"SA", "PL"}, "3.1.6": {"AT"}, "3.1.7": {"AT"},
    "3.2.1": {"RA", "PM"}, "3.2.5": {"CA", "AU"}, "3.3.1": {"PS"},
    "3.3.2": {"PE"}, "3.3.3": {"CM"}, "3.3.5": {"AC", "IA"},
    "3.3.6": {"SA", "SI"}, "3.3.7": {"CM"}, "3.3.8": {"SC", "CM"},
    "3.3.9": {"SC"}, "3.3.11": {"MP"}, "3.3.14": {"AU", "SI"},
    "3.3.15": {"IR"}, "3.3.16": {"RA", "SI"}, "3.3.17": {"RA", "SI"},
    "3.4.1": {"SA", "SR"}, "3.4.2": {"SA", "SR"}, "3.4.3": {"SA", "SC"},
}


def enumerative(c: Dict[str, Any]) -> bool:
    """A list item, not a requirement.

    "cyber security specialists;" carries no obligation of its own — its stem
    does ("risk management activities should involve:"). Treating such items as
    independent requirements is what produced most of the apparent coverage
    gaps: they cannot match a control, and they should never have been asked to.
    """
    txt = (c.get("text") or "").strip()
    if not (c.get("deontic_inherited") and c.get("parent_id")
            and len(txt) < 70 and txt.endswith((";", "."))):
        return False
    return not re.search(
        r"\b(is|are|was|were|be|been|shall|should|must|may|will|can)\b"
        r"|\b\w+(?:ing|ed|es|s)\b\s+(?:for|to|by|with|in|on|that|the)\b",
        txt, re.I)


def evidenced_absence(d: Data, cid: str, findings: List[Dict[str, Any]]) -> bool:
    """Was the absence actually tested?

    Absence is the one verdict that cannot be shown by a citation, so it has to
    be earned: either the judge saw a candidate and rejected it, or the
    shortlist at least reached a family the subdomain plausibly belongs to. With
    neither, the correct statement is that the clause was not tested, not that
    the catalogue lacks a counterpart.
    """
    if findings:
        return True
    row = d.candidates.get(cid, {})
    fams = {x["control_id"].split("-")[0] for x in row.get("candidates", [])}
    expected = SUBDOMAIN_EXPECTED.get(d.clauses.get(cid, {}).get("subdomain", ""))
    return bool(expected and (fams & expected))


def clause_gaps(d: Data) -> List[Dict[str, Any]]:
    """SAMA clauses for which nothing was accepted.

    Referential clauses are excluded: 3.3.12 Payment Systems states no
    requirement of its own, so counting it as uncovered would be wrong.

    The controls that were offered and not taken are carried through, because
    "the judge saw good candidates and rejected them" and "nothing plausible
    reached the judge" are very different pieces of evidence and only the first
    supports a gap claim. A reviewer can tell them apart at a glance; a bare
    count cannot.
    """
    out = []
    for r in d.findings:
        cid = r["clause_id"]
        c = d.clauses.get(cid, {})
        if c.get("clause_type") == "referential" or enumerative(c):
            continue
        if any(f["status"] == "accepted" for f in r["findings"]):
            continue
        disputed = [f for f in r["findings"] if f["status"] == "disputed"]
        offered = [x["control_id"] for x in
                   d.candidates.get(cid, {}).get("candidates", [])][:5]
        if r["findings"]:
            severity = "below threshold"
        elif evidenced_absence(d, cid, r["findings"]):
            severity = "evidenced absence"
        else:
            severity = "not tested"
        out.append({
            "clause_id": cid,
            "subdomain": f"{c.get('subdomain','')} {c.get('subdomain_title','')}".strip(),
            "sd_key": c.get("subdomain", ""),
            "severity": severity,
            "best": max((f.get("confidence") or 0 for f in disputed), default=0.0),
            "offered": ", ".join(offered),
            "text": c.get("text", "")[:300],
        })

    # Surface the subdomains with the most uncovered clauses first: a reader
    # should meet the worst-covered area, not whichever sorts first by id.
    order = {"evidenced absence": 0, "not tested": 1, "below threshold": 2}
    density = Counter(g["sd_key"] for g in out if g["severity"] == "evidenced absence")
    out.sort(key=lambda g: (order[g["severity"]], -density[g["sd_key"]],
                            g["sd_key"], g["best"]))
    return out


def local_specificity(d: Data) -> List[Dict[str, Any]]:
    """Clauses broader than the control they match.

    A `superset_of` finding says the national instrument requires something the
    international reference does not. That is the most policy-relevant result
    the system produces and the easiest to lose in an aggregate, so it gets its
    own view.
    """
    rows = []
    for cid, f in d.verified:
        if f["relationship"] != "superset_of":
            continue
        c = d.clauses.get(cid, {})
        rows.append({
            "clause_id": cid,
            "subdomain": f"{c.get('subdomain','')} {c.get('subdomain_title','')}".strip(),
            "control_id": f["control_id"],
            "confidence": f.get("confidence"),
            "status": f["status"],
            "clause": c.get("text", "")[:220],
            "rationale": f.get("rationale", "")[:200],
        })
    rows.sort(key=lambda r: -(r["confidence"] or 0))
    return rows


def gap_by_subdomain(d: Data) -> List[Dict[str, Any]]:
    """Gaps rolled up to the subdomain, which is the unit a regulator acts on.

    Reporting 249 uncovered clauses overstates the case: many are sub-items of
    one stem ("committee objectives;", "minimum number of meeting
    participants;") and are not independent requirements.
    """
    total: Counter = Counter()
    accepted: Counter = Counter()
    disputed_only: Counter = Counter()
    nothing: Counter = Counter()
    untested: Counter = Counter()
    titles: Dict[str, str] = {}
    for r in d.findings:
        c = d.clauses.get(r["clause_id"], {})
        if c.get("clause_type") == "referential" or enumerative(c):
            continue
        sd = c.get("subdomain", "")
        titles[sd] = c.get("subdomain_title", "")
        total[sd] += 1
        if any(f["status"] == "accepted" for f in r["findings"]):
            accepted[sd] += 1
        elif any(f["status"] == "disputed" for f in r["findings"]):
            disputed_only[sd] += 1
        elif evidenced_absence(d, r["clause_id"], r["findings"]):
            nothing[sd] += 1
        else:
            untested[sd] += 1

    # A subdomain where every clause matched something but nothing cleared the
    # threshold is not the same as one where nothing matched at all. The first
    # is an artefact of tau; only the second is evidence of a coverage gap.
    rows = [{
        "subdomain": sd,
        "title": titles.get(sd, ""),
        "accepted": accepted[sd],
        "review": disputed_only[sd],
        "nothing": nothing[sd],
        "untested": untested[sd],
        "total": total[sd],
        "pct_nothing": round(100 * nothing[sd] / max(total[sd], 1)),
        "kind": ("evidenced absence" if nothing[sd] > max(disputed_only[sd], untested[sd])
                 else "not tested" if untested[sd] > disputed_only[sd]
                 else "below threshold" if disputed_only[sd] else "covered"),
    } for sd in sorted(total)]
    rows.sort(key=lambda r: (-r["nothing"], -r["review"]))
    return rows


def control_gaps(d: Data) -> List[Dict[str, Any]]:
    """Active NIST controls no SAMA clause matched, by family.

    Counted twice. A control matched only below the confidence threshold is
    still a match; counting it as unreached would repeat, at family level, the
    error the subdomain table was corrected for. Only the verified column
    supports a claim that nothing in SAMA corresponds.
    """
    acc = {f["control_id"] for _, f in d.accepted}
    ver = {f["control_id"] for _, f in d.verified}
    rows = []
    for fam in sorted({c["family"] for c in d.live.values()}):
        fam_ctrls = [c for c in d.live.values() if c["family"] == fam]
        n = len(fam_ctrls)
        miss_acc = sum(1 for c in fam_ctrls if c["control_id"] not in acc)
        miss_ver = sum(1 for c in fam_ctrls if c["control_id"] not in ver)
        rows.append({
            "family": fam,
            "family_title": fam_ctrls[0]["family_title"],
            "unmatched": miss_acc,
            "unreached": miss_ver,
            "total": n,
            "pct": round(100 * miss_acc / max(n, 1)),
            "pct_unreached": round(100 * miss_ver / max(n, 1)),
            "note": DOCUMENTED_EXCLUSIONS.get(fam, ""),
        })
    rows.sort(key=lambda r: (-r["pct_unreached"], -r["pct"]))
    return rows


def parameter_gaps(d: Data) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split parameter gaps into the two kinds, which are not the same finding.

    hard  SAMA fixes a concrete value where NIST leaves an open parameter.
          This is instantiation: the regulator has answered a question NIST
          deliberately left to the adopter.
    soft  Neither side commits. SAMA says "periodically", NIST says
          [organization-defined frequency]. That is a shared silence, and it
          is a different observation about regulatory specificity.
    """
    hard, soft = [], []
    for cid, f in d.accepted:
        if not f.get("parameter_gap"):
            continue
        c = d.clauses.get(cid, {})
        row = {
            "clause_id": cid,
            "control_id": f["control_id"],
            "values": ", ".join(c.get("numeric_params", [])),
            "note": f.get("parameter_note", "")[:160],
            "clause": c.get("text", "")[:200],
        }
        (hard if c.get("has_numeric_param") else soft).append(row)
    return hard, soft


def drift(d: Data) -> Dict[str, Any]:
    """Controls postdating SAMA CSF 2017, and how they fared."""
    matched = {f["control_id"] for _, f in d.accepted}
    out: Dict[str, Any] = {}
    for fam in ("SR", "PT"):
        ctrls = [c for c in d.live.values() if c["family"] == fam]
        out[fam] = {
            "total": len(ctrls),
            "matched": sum(1 for c in ctrls if c["control_id"] in matched),
            "title": ctrls[0]["family_title"] if ctrls else "",
        }
    new = [c for c in REV_520_ADDITIONS if c in d.live]
    out["rev520"] = {"known": new, "matched": [c for c in new if c in matched]}
    return out


# --------------------------------------------------------------------------
# tab 2 — mapping
# --------------------------------------------------------------------------
def best_example(d: Data) -> Optional[str]:
    """Open the Mapping tab on a clause that shows the system working.

    The first clause by identifier happens to have no accepted match, so
    defaulting to it puts an empty table in front of a reader in the first
    thirty seconds. Prefer a clause with an accepted finding, and among those
    prefer one carrying a parameter gap.
    """
    scored = []
    for cid, fs in d.by_clause.items():
        acc = [f for f in fs if f["status"] == "accepted"]
        if not acc:
            continue
        scored.append((any(f.get("parameter_gap") for f in acc), len(acc),
                       max(f.get("confidence") or 0 for f in acc), cid))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][3]


def clause_choices(d: Data) -> List[str]:
    def key(cid: str):
        c = d.clauses.get(cid, {})
        sd = c.get("subdomain", "9.9.9")
        return ([int(x) for x in sd.split(".")], cid)
    return sorted(d.by_clause, key=key)


def clause_detail(d: Data, clause_id: str) -> Tuple[str, List[List[str]]]:
    c = d.clauses.get(clause_id)
    if not c:
        return f"No clause {clause_id}.", []
    head = [
        f"### {clause_id} — {c['subdomain']} {c['subdomain_title']}",
        f"*{c['domain_title']}*  ·  level {c['level']}  ·  {c['clause_type']}"
        f"  ·  page {c['page']}",
        "",
        f"> {c['text']}",
    ]
    if c.get("numeric_params"):
        head.append(f"\n**States a value:** {', '.join(c['numeric_params'])}")
    if c.get("principle"):
        head.append(f"\n**Subdomain principle.** {c['principle'][:400]}")

    rows = []
    for f in d.by_clause.get(clause_id, []):
        mark = {"accepted": "accepted", "disputed": "needs review", "rejected": "rejected"}
        ctrl = d.controls.get(f["control_id"], {})
        if f.get("debated"):
            was = f["proposer"]["relationship"]
            history = (f"{f.get('verdict','')} · proposed {was}"
                       if was != f["relationship"] else f.get("verdict", ""))
        else:
            history = "not debated"
        rows.append([
            f["control_id"],
            ctrl.get("title", ""),
            f["relationship"],
            f"{f.get('confidence', '')}",
            mark.get(f["status"], f["status"]),
            history,
            f.get("evidence_sama", "")[:150],
            f.get("evidence_nist", "")[:150],
        ])
    if not rows:
        head.append("\n**No related control accepted.** Coverage gap candidate.")
    return "\n".join(head), rows


MAPPING_COLS = ["control", "title", "relationship", "confidence", "status",
                "debate", "evidence (SAMA)", "evidence (NIST)"]


# --------------------------------------------------------------------------
# tab 4 — knowledge base
# --------------------------------------------------------------------------
FACTORS = ["data", "research", "deployment_support", "standards", "open_source", "sandbox"]


def kb_table(d: Data) -> List[List[str]]:
    rows = []
    for m in d.manifest:
        if m.get("tier") == "method":
            continue
        rows.append([
            m.get("doc_id", ""), m.get("tier", ""), m.get("title_en", "")[:60],
            m.get("issuing_body", ""), m.get("status", ""), m.get("year", ""),
            (m.get("sha256") or "")[:12] or "\u2014",
            m.get("reference_pair", "") or "\u2014", m.get("url", ""),
        ])
    return rows


KB_COLS = ["doc_id", "tier", "title", "issuer", "legal status", "year", "sha256",
           "analysed against", "url"]


def itu_coverage(d: Data) -> Tuple[List[List[str]], List[List[str]]]:
    content = [m for m in d.manifest if m.get("tier") != "method"]
    fc: Counter = Counter()
    dc: Counter = Counter()
    for m in content:
        for f in (m.get("itu_factors") or "").split(";"):
            if f.strip():
                fc[f.strip()] += 1
        for x in (m.get("itu_dimensions") or "").split(";"):
            if x.strip():
                dc[x.strip()] += 1
    frows = [[f, str(fc.get(f, 0)),
              "covered by the system itself, not by a document"
              if f in ("open_source", "sandbox") and not fc.get(f) else ""]
             for f in FACTORS]
    drows = [[str(i), str(dc.get(str(i), 0))] for i in range(1, 14)]
    return frows, drows


# --------------------------------------------------------------------------
# tab 3 — ask
# --------------------------------------------------------------------------
_RETRIEVER = None


def ask(question: str, k: int = 6, status: Optional[str] = None) -> str:
    """Retrieve grounded passages. Loads the retriever on first use only."""
    global _RETRIEVER
    if not (question or "").strip():
        return "_Ask a question about the policy corpus._"
    try:
        if _RETRIEVER is None:
            import index as ix

            _RETRIEVER = ix.Retriever()
        hits = _RETRIEVER.search(question, k=k,
                                 status=status if status in ("binding", "advisory") else None)
    except Exception as exc:  # noqa: BLE001
        return f"Retrieval unavailable: `{type(exc).__name__}: {exc}`"
    if not hits:
        return ("No passage in the knowledge base matches that. The system does "
                "not answer from outside the corpus.")
    out = [f"**{len(hits)} passages.** Every claim below is a quotation; the "
           f"system does not paraphrase and does not answer unsourced.\n"]
    for n, h in enumerate(hits, 1):
        out.append(
            f"**{n}. {h.get('title_en','')}** — {h.get('issuing_body','')} "
            f"[{h.get('status','')}]  ·  page {h['page']}\n\n"
            f"> {' '.join(h['text'].split())[:700]}\n\n"
            f"`{h['chunk_id']}`  ·  [source]({h.get('url','')})\n"
        )
    return "\n".join(out)


# --------------------------------------------------------------------------
def summary(d: Data) -> str:
    total = len([f for r in d.findings for f in r["findings"]])
    acc = len(d.accepted)
    n_deb = d.n_debated
    dis = sum(1 for r in d.findings for f in r["findings"] if f["status"] == "disputed")
    rej = total - acc - dis
    rels = Counter(f["relationship"] for _, f in d.verified)
    aligned = rels["equal"] + rels["subset_of"]
    hard, soft = parameter_gaps(d)
    gaps = clause_gaps(d)
    nothing = sum(1 for g in gaps if g["severity"] == "evidenced absence")
    untested = sum(1 for g in gaps if g["severity"] == "not tested")
    below = sum(1 for g in gaps if g["severity"] == "below threshold")
    mappable = sum(1 for r in d.findings
                   if d.clauses.get(r["clause_id"], {}).get("clause_type") != "referential"
                   and not enumerative(d.clauses.get(r["clause_id"], {})))
    excluded = len(d.findings) - mappable
    sup = local_specificity(d)
    covered = len({cid for cid, _ in d.accepted
                   if not enumerative(d.clauses.get(cid, {}))
                   and d.clauses.get(cid, {}).get("clause_type") != "referential"})
    n_clauses = len(d.findings)

    return f"""## SANAD — evidence-anchored regulatory alignment

**SAMA Cyber Security Framework (2017)** against **NIST SP 800-53 Rev 5.2.0**,
with every claim anchored to a verbatim span of both sources.

| | |
|---|---|
| clauses analysed | **{mappable}** mappable, {excluded} excluded as list items or referential |
| judgments | **{total}** — {acc} accepted, {dis} for review, {rej} rejected at the gate |
| debated | {n_deb} argued by three agents on two models |
| clauses with an accepted match | **{covered}** ({100*covered//max(mappable,1)}% of mappable) |
| evidenced absence | **{nothing}** — a candidate was offered and rejected |
| not tested | {untested} — no plausible candidate reached the judge |
| matched only below threshold | {below} |
| broader than the reference | **{len(sup)}** `superset_of` findings |
| parameter gaps | **{len(hard)}** hard, {len(soft)} shared-silence |
| alignment (equal + subset_of) | **{aligned}** of {len(d.verified)} verified |

The system is a selective classifier, not an oracle. It accepts what it can
evidence and routes the rest to human review rather than guessing — the
{dis} judgments marked for review are a feature of the design, not a shortfall
of it.
"""


# --------------------------------------------------------------------------
def selftest() -> int:
    d = Data()
    ok, why = d.ready()
    print(f"data ready: {ok} {why}")
    if not ok:
        return 1
    print(f"  clauses {len(d.clauses)}  controls {len(d.controls)} "
          f"({len(d.live)} active)  findings rows {len(d.findings)}")
    print(f"  accepted {len(d.accepted)}  verified {len(d.verified)}")
    print(f"  arbiter rulings applied: {d.n_debated}")
    g = clause_gaps(d)
    from collections import Counter as _C
    print(f"  clause gaps {len(g)}: " +
          ", ".join(f"{k}={v}" for k, v in _C(x["severity"] for x in g).most_common()))
    print(f"  local specificity (superset_of): {len(local_specificity(d))}")
    excl = sum(1 for r in d.findings
               if d.clauses.get(r["clause_id"], {}).get("clause_type") == "referential"
               or enumerative(d.clauses.get(r["clause_id"], {})))
    print(f"  excluded as list items / referential: {excl} of {len(d.findings)}")
    sd = gap_by_subdomain(d)
    if sd:
        w = sd[0]
        print(f"  weakest subdomain: {w['subdomain']} — {w['nothing']} no match, "
              f"{w['review']} for review, of {w['total']}  [{w['kind']}]")
    print(f"  mapping opens on: {best_example(d)}")
    cg = control_gaps(d)
    print(f"  control gaps: worst family {cg[0]['family']} {cg[0]['pct']}%")
    h, s = parameter_gaps(d)
    print(f"  parameter gaps: hard {len(h)} soft {len(s)}")
    print(f"  drift: {drift(d)}")
    ch = clause_choices(d)
    print(f"  clause choices {len(ch)}")
    md, rows = clause_detail(d, ch[0]) if ch else ("", [])
    print(f"  detail rows for {ch[0] if ch else '-'}: {len(rows)}")
    print(f"  kb rows {len(kb_table(d))}")
    fr, dr = itu_coverage(d)
    print(f"  itu factors {len(fr)} dimensions {len(dr)}")
    print("\n" + summary(d)[:400])
    return 0


def launch(share: bool = True) -> int:
    try:
        import gradio as gr
    except ImportError:
        sys.exit("pip install -q gradio")

    d = Data()
    ok, why = d.ready()
    if not ok:
        sys.exit(f"Cannot start: {why}. Run the pipeline first.")

    gaps = clause_gaps(d)
    sdgaps = gap_by_subdomain(d)
    cgaps = control_gaps(d)
    sup = local_specificity(d)
    sup = local_specificity(d)
    hard, soft = parameter_gaps(d)
    dr = drift(d)
    choices = clause_choices(d)
    default = best_example(d) or (choices[0] if choices else None)
    frows, drows = itu_coverage(d)

    with gr.Blocks(title="SANAD — AI Readiness, Saudi financial sector") as app:
        gr.Markdown(summary(d))

        with gr.Tabs():
            with gr.Tab("Gaps"):
                gr.Markdown(
                    "### Coverage gaps\n"
                    "Absence is the one verdict no citation can establish, so it "
                    "has to be earned. **Evidenced absence** means a plausible "
                    "control was offered to the judge and rejected. **Not "
                    "tested** means the shortlist never reached a control family "
                    "the subdomain plausibly belongs to — a "
                    "retrieval failure, not a finding about the catalogue. "
                    "**Below threshold** is a match, not a gap. The "
                    "evidenced-absence / not-tested split depends on a "
                    "hand-curated family map covering the routed subdomains; "
                    "broadening that map would reclassify some of the 53, not "
                    "change any judgment.\n\n"
                    "Referential clauses (3.3.12 Payment Systems, which "
                    "delegates to SARIE and mada) and enumerative list items "
                    "(\"cyber security specialists;\") are excluded: neither "
                    "states a requirement that could have a counterpart."
                )
                gr.Dataframe(
                    [[g["clause_id"], g["subdomain"], g["severity"],
                      f"{g['best']:.2f}" if g["best"] else "\u2014", g["offered"], g["text"]]
                     for g in gaps],
                    headers=["clause", "subdomain", "severity", "conf",
                             "shortlisted", "clause text"],
                    column_widths=["11%", "19%", "13%", "6%", "19%", "32%"],
                    wrap=True,
                )

                gr.Markdown(
                    "### Weakest subdomains\n"
                    "Rolled up to the subdomain, which is the unit a regulator "
                    "acts on. Many uncovered clauses are sub-items of a single "
                    "stem and are not independent requirements, so the clause "
                    "count above overstates the case and this table does not."
                )
                gr.Dataframe(
                    [[r["subdomain"], r["title"], r["accepted"], r["review"],
                      r["nothing"], r["untested"], r["total"], r["kind"]]
                     for r in sdgaps],
                    headers=["sub", "title", "acc", "rev", "gap", "untd", "all", "reading"],
                    column_widths=["9%", "24%", "6%", "6%", "6%", "7%", "6%", "36%"],
                    wrap=True,
                )
                gr.Markdown(
                    "*for review* means a control was matched but below the "
                    "0.60 threshold — an artefact of where the threshold sits, "
                    "not a gap. Only *no match* is evidence that nothing in the "
                    "catalogue corresponds.\n\n"
                    + "\n".join(f"**{k}** — {v}" for k, v in REFERENTIAL_IN_SUBSTANCE.items())
                )

                gr.Markdown(
                    "### Unmatched NIST controls, by family\n"
                    "High percentages are candidates, not conclusions. A family "
                    "can be unmatched because SAMA delegates it elsewhere, "
                    "because SAMA genuinely omits it, or because the router "
                    "failed to reach it — three different findings."
                )
                gr.Dataframe(
                    [[r["family"], r["family_title"], r["total"],
                      f"{r['unmatched']} ({r['pct']}%)",
                      f"{r['unreached']} ({r['pct_unreached']}%)", r["note"]]
                     for r in cgaps],
                    headers=["fam", "title", "n", "no acc.", "unreached", "note"],
                    column_widths=["6%", "26%", "6%", "11%", "12%", "39%"],
                    wrap=True,
                )
                gr.Markdown(
                    "The last column is the one that supports a gap claim. The "
                    "one before it includes controls matched below threshold, "
                    "which are matches, not gaps."
                )

                gr.Markdown(
                    f"### Broader than the reference — {len(sup)} findings\n"
                    "Clauses where the Saudi instrument requires something NIST "
                    "does not. A `superset_of` result is a substantive finding, "
                    "not a null one: it is where national regulation has gone "
                    "further than the standard it was built from, and it is the "
                    "first thing a harmonisation exercise would erase by "
                    "accident."
                )
                gr.Dataframe(
                    [[r["clause_id"], r["subdomain"], r["control_id"],
                      f"{r['confidence']}", r["status"], r["clause"]] for r in sup],
                    headers=["clause", "subdomain", "control", "confidence",
                             "status", "clause text"],
                    wrap=True,
                )

                gr.Markdown(
                    f"### Regulatory drift\n"
                    f"SAMA CSF is from 2017. NIST SP 800-53 reached Rev 5 in 2020 "
                    f"and Release 5.2.0 in August 2025.\n\n"
                    f"- **SR** ({dr['SR']['title']}): {dr['SR']['matched']} of "
                    f"{dr['SR']['total']} controls matched\n"
                    f"- **PT** ({dr['PT']['title']}): {dr['PT']['matched']} of "
                    f"{dr['PT']['total']} controls matched\n\n"
                    f"Neither family existed as such when SAMA CSF was written. "
                    f"A low match rate is evidence of drift; a high one shows the "
                    f"concept was already covered under another heading, which is "
                    f"why the two families should not be reported together."
                )

                gr.Markdown(
                    f"### Parameter gaps — {len(hard)} hard\n"
                    "SAMA fixes a concrete value where NIST leaves an "
                    "organization-defined parameter open. This is the regulator "
                    "answering a question NIST deliberately left to the adopter."
                )
                gr.Dataframe(
                    [[h["clause_id"], h["control_id"], h["values"], h["note"], h["clause"]]
                     for h in hard],
                    headers=["clause", "control", "SAMA value", "note", "clause text"],
                    wrap=True,
                )
                gr.Markdown(
                    f"**{len(soft)} further parameter gaps** are shared silence: "
                    "SAMA says *periodically*, NIST says *[organization-defined "
                    "frequency]*, and neither commits. That is an observation "
                    "about specificity, not about instantiation."
                )

            with gr.Tab("Mapping"):
                gr.Markdown(
                    "Pick a clause to see the controls matched to it, the "
                    "relationship type from NIST IR 8477, and the verbatim "
                    "evidence from each side. Nothing was accepted without both "
                    "spans passing a literal substring check.\n\n"
                    "The **debate** column shows what the three agents did: "
                    "`uphold` means the opponent's objection was rejected, "
                    "`revise` that the arbiter changed the verdict, and the "
                    "proposer's original relationship is named where it "
                    "differs."
                )
                pick = gr.Dropdown(choices, value=default,
                                   label="SAMA clause", filterable=True)
                info = gr.Markdown()
                table = gr.Dataframe(headers=MAPPING_COLS, wrap=True)

                def _show(cid):
                    return clause_detail(d, cid)

                pick.change(_show, pick, [info, table])
                if default:
                    app.load(lambda: clause_detail(d, default), None, [info, table])
                gr.Markdown(
                    "A clause with no rows above has no accepted mapping: see "
                    "the Gaps tab, which shows whether a candidate was offered "
                    "and rejected, or none reached the agents at all."
                )

            with gr.Tab("Ask"):
                gr.Markdown(
                    "Question answering over the eight-document policy corpus. "
                    "Answers are passages, not prose: every result carries its "
                    "document, page, chunk identifier and source link. When "
                    "nothing matches, the system says so rather than composing "
                    "an answer from outside the corpus."
                )
                q = gr.Textbox(label="Question", placeholder="who must approve the cyber security policy")
                with gr.Row():
                    kk = gr.Slider(3, 12, value=6, step=1, label="passages")
                    st = gr.Radio(["any", "binding", "advisory"], value="any",
                                  label="legal status")
                out = gr.Markdown()
                gr.Button("Search", variant="primary").click(
                    lambda a, b, c: ask(a, int(b), None if c == "any" else c),
                    [q, kk, st], out)
                q.submit(lambda a, b, c: ask(a, int(b), None if c == "any" else c),
                         [q, kk, st], out)

            with gr.Tab("Knowledge base"):
                gr.Markdown(
                    "Every document public; core documents fetched from their "
                    "issuing bodies and pinned by SHA-256, so a finding traces "
                    "to the exact version that produced it. Referenced rows "
                    "are cited with a verified URL but not retrieved. The SAMA "
                    "framework was downloaded by hand from its rulebook page: "
                    "SAMA publishes no stable direct file URL."
                )
                gr.Dataframe(kb_table(d), headers=KB_COLS, wrap=True)
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**ITU AI Readiness factors**")
                        gr.Dataframe(frows, headers=["factor", "documents", "note"], wrap=True)
                    with gr.Column():
                        gr.Markdown("**ITU AI Readiness dimensions**")
                        gr.Dataframe(drows, headers=["dimension", "documents"])
                gr.Markdown(
                    "Methodological references (ITU-T Y.3172, the ITU AI Ready "
                    "report, NIST IR 8477) are held in the manifest but excluded "
                    "from these counts. They are how the work was done, not "
                    "national policy content."
                )

    # On Hugging Face Spaces the platform provides the public URL; asking
    # Gradio for a share tunnel there fails. Detect the platform rather than
    # requiring a different launch command.
    on_spaces = bool(os.environ.get("SPACE_ID"))
    app.launch(share=share and not on_spaces,
               server_name="0.0.0.0" if on_spaces else None,
               debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "selftest" in sys.argv else launch("--no-share" not in sys.argv))
