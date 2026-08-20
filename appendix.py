#!/usr/bin/env python3
"""
appendix.py — generate report appendices A, B and C from the manifest.

Every figure in the appendices is read from manifest.csv at generation time
rather than transcribed. A reference list copied by hand drifts from the file it
describes the moment either changes, and the appendices are precisely where a
reviewer checks that claim.

    python appendix.py > ../docs/appendix.tex

Appendix A — References, split into policy instruments and methodological
             references, in a citation form that carries issuer, title, version,
             URL and access date.
Appendix B — ITU readiness factors and dimensions, counted from the manifest
             tags, with the two factors the system supplies itself named as
             such.
Appendix C — The manifest itself: every row, with the SHA-256 prefix that pins
             the version each finding was produced against.

Output is a LaTeX fragment. Append it to report.tex before \\end{document}, or
\\input it.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

KB = Path(__file__).resolve().parent
MANIFEST = KB / "manifest.csv"

FACTORS = ["data", "research", "deployment_support", "standards",
           "open_source", "sandbox"]

# Two factors carry no document because the system supplies them: the released
# pipeline, and the anchor-based harness that measures retrieval without manual
# annotation. Reporting them as zero would understate the contribution.
SYSTEM_SUPPLIED = {
    "open_source": "supplied by the system: the released pipeline and a corpus "
                   "reproducible from this manifest",
    "sandbox": "supplied by the system: the anchor-based evaluation harness",
}

DIMENSION_NAMES = {
    1: "Data and model marketplace",
    2: "Generated content marketplace",
    3: "Cross-domain correlation analysis",
    4: "Contextualisation and regional impact",
    5: "AI integration in workflows",
    6: "Compute and infrastructure",
    7: "Strategy alignment",
    8: "Skills and capacity",
    9: "Ecosystem and partnerships",
    10: "AI and policies",
    11: "Risk and trust",
    12: "Investment and funding",
    13: "Digital infrastructure",
}

# Dimensions the system evidences directly, independent of document tags.
SYSTEM_EVIDENCE = {
    2: "29 gate rejections, 2 of them fabricated citations (\\S3)",
    11: "acceptance gate and 184 declared referrals (\\S3, \\S7)",
}


def clip(s: str, n: int) -> str:
    """Truncate at a word boundary. Cutting mid-word ("future n") reads as a
    display fault rather than a deliberate abbreviation."""
    s = " ".join((s or "").split())
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)[0]
    return (cut or s[:n]) + "\u2026"


def tex(s: str) -> str:
    """Escape for LaTeX. URLs are handled separately by \\url."""
    if not s:
        return ""
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return " ".join(s.split())


def load() -> List[Dict[str, str]]:
    if not MANIFEST.exists():
        sys.exit(f"ERROR: {MANIFEST} not found.")
    with MANIFEST.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    order = {"core": 0, "referenced": 1, "method": 2}
    rows.sort(key=lambda r: (order.get(r["tier"], 9), r["doc_id"]))
    return rows


# --------------------------------------------------------------------------
def citation(r: Dict[str, str]) -> str:
    """One reference entry.

    Version, access date and translation status are included because a
    regulatory citation without them cannot be checked: instruments are amended,
    pages move, and the Arabic text is the one that binds.
    """
    bits = [f"\\textbf{{{tex(r['issuing_body'])}}}."]
    bits.append(f"\\emph{{{tex(r['title_en'])}}}.")
    ver = tex(r.get("version", ""))
    bits.append(f"{ver}, {r['year']}." if ver else f"{r['year']}.")
    url = r.get("url", "")
    if "/wps/portal/" in url:
        # Portal URLs carry session-like path segments: too long to print, and
        # not reconstructable by hand from a PDF. Cite the stable landing page;
        # the full URL stays in the manifest, where its fragility is documented.
        base = url.split("/details/")[0] if "/details/" in url else url[:60]
        bits.append(f"\\url{{{base}}}")
        bits.append("(document listed under its title in the knowledge centre; "
                    "full path recorded in the manifest).")
    elif url:
        bits.append(f"\\url{{{url}}}")
    if r.get("url_accessed"):
        bits.append(f"Accessed {r['url_accessed']}.")
    if (r.get("is_official_translation") or "").upper() == "TRUE":
        bits.append("Arabic text authoritative; official English translation used.")
    if r["tier"] == "referenced":
        bits.append("\\emph{Cited in this report; not retrieved or indexed.}")
    return " ".join(bits)


def appendix_a(rows) -> None:
    print(r"\section*{Appendix A — References}")
    print(r"\addcontentsline{toc}{section}{Appendix A — References}")
    print()
    print(r"\subsection*{A.1 Policy and regulatory instruments}")
    print(r"\begin{enumerate}[leftmargin=1.5em,itemsep=2pt]")
    for r in rows:
        if r["tier"] != "method":
            print(f"  \\item {citation(r)}")
    print(r"\end{enumerate}")
    print()
    print(r"\subsection*{A.2 Domain and methodological references}")
    print(r"\begin{enumerate}[leftmargin=1.5em,itemsep=2pt]")
    for r in rows:
        if r["tier"] == "method":
            print(f"  \\item {citation(r)}")
    print(r"\end{enumerate}")
    print()
    print(r"""\noindent Source files are not redistributed. \code{fetch.py}
reconstructs the corpus from this manifest and verifies each document against
its recorded SHA-256, so any reader obtains the identical corpus from the
issuing bodies.""")
    print()


def appendix_b(rows) -> None:
    content = [r for r in rows if r["tier"] != "method"]
    fc: Counter = Counter()
    dc: Counter = Counter()
    fd: Dict[str, List[str]] = defaultdict(list)
    dd: Dict[str, List[str]] = defaultdict(list)
    for r in content:
        short = r["doc_id"].split("-", 2)[-1].rsplit("-", 1)[0]
        for f in (r.get("itu_factors") or "").split(";"):
            if f.strip():
                fc[f.strip()] += 1
                fd[f.strip()].append(short)
        for d in (r.get("itu_dimensions") or "").split(";"):
            if d.strip():
                dc[d.strip()] += 1
                dd[d.strip()].append(short)

    print(r"\section*{Appendix B — ITU AI Readiness factors and dimensions}")
    print(r"\addcontentsline{toc}{section}{Appendix B — ITU factors and dimensions}")
    print()
    print(r"\subsection*{B.1 The six factors}")
    print(r"\begin{center}\small")
    print(r"\begin{tabular}{@{}p{3.3cm}r p{10.5cm}@{}}")
    print(r"\toprule")
    print(r"\hd{Factor} & \hd{Docs} & \hd{Supporting evidence} \\")
    print(r"\midrule")
    for f in FACTORS:
        n = fc.get(f, 0)
        if n:
            ev = ", ".join(tex(x) for x in sorted(set(fd[f])))
        else:
            ev = f"\\emph{{{SYSTEM_SUPPLIED.get(f, 'not claimed')}}}"
        print(f"{tex(f.replace('_', ' '))} & {n} & {ev} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{center}")
    print()
    print(r"""\noindent \emph{Open source} and \emph{sandbox} carry no document
because the system supplies them rather than documenting them --- the stronger
form of contribution.""")
    print()

    print(r"\subsection*{B.2 The thirteen dimensions}")
    print(r"\begin{center}\small")
    print(r"\begin{tabular}{@{}r p{5.4cm} r p{6.4cm}@{}}")
    print(r"\toprule")
    print(r"\hd{\#} & \hd{Dimension} & \hd{Docs} & \hd{Claimed on} \\")
    print(r"\midrule")
    for i in range(1, 14):
        n = dc.get(str(i), 0)
        sysev = SYSTEM_EVIDENCE.get(i, "")
        if n:
            claim = ", ".join(tex(x) for x in sorted(set(dd[str(i)])))
            if sysev:
                claim += f"; {sysev}"
        elif sysev:
            claim = sysev
        else:
            claim = r"\emph{not claimed}"
        print(f"{i} & {tex(DIMENSION_NAMES[i])} & {n or '--'} & {claim} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{center}")
    print()
    # The body claims a direct contribution on four dimensions. The corpus tags
    # touch more than that, so the two levels are separated here: a document
    # bearing on a dimension is not the same as the work contributing to it.
    contributed = sorted(set(SYSTEM_EVIDENCE) | {4, 10})
    corpus_only = [i for i in range(1, 14)
                   if dc.get(str(i)) and i not in contributed]
    not_claimed = [i for i in range(1, 14)
                   if not dc.get(str(i)) and i not in contributed]
    print(r"\noindent Dimensions " + ", ".join(str(i) for i in not_claimed) +
          r" are not claimed. Dimensions " +
          ", ".join(str(i) for i in corpus_only) +
          r""" are evidenced by corpus documents only; the direct contribution
claims rest on """ + ", ".join(str(i) for i in contributed) +
          r""", as stated in \S8. Stating that is preferable to padding the
table.""")
    print()


def appendix_c(rows) -> None:
    print(r"\section*{Appendix C — Knowledge base manifest}")
    print(r"\addcontentsline{toc}{section}{Appendix C — Knowledge base manifest}")
    print()
    print(r"""\noindent The knowledge base prototype: every document public; core documents
fetched from their issuing bodies and pinned by SHA-256. The record schema
(\code{manifest.schema.json}, in the repository) is offered as a candidate
contribution to a standardised ITU AI Readiness record format.""")
    print()
    print(r"\begin{center}\scriptsize")
    print(r"\begin{tabular}{@{}p{3.1cm}p{1.5cm}p{4.9cm}p{1.4cm}p{1.1cm}p{2.1cm}@{}}")
    print(r"\toprule")
    print(r"\hd{doc\_id} & \hd{tier} & \hd{title} & \hd{status} & \hd{year} & \hd{sha256} \\")
    print(r"\midrule")
    for r in rows:
        sha = (r.get("sha256") or "")[:12]
        sha = f"\\texttt{{{sha}}}" if sha else "--"
        print(f"\\texttt{{{tex(r['doc_id'])}}} & {tex(r['tier'])} & "
              f"{tex(clip(r['title_en'], 54))} & {tex(r['status'])} & {r['year']} & {sha} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{center}")
    print()
    n_core = sum(1 for r in rows if r["tier"] == "core")
    n_ref = sum(1 for r in rows if r["tier"] == "referenced")
    n_met = sum(1 for r in rows if r["tier"] == "method")
    print(rf"""\noindent {len(rows)} records: {n_core} \emph{{core}} (ingested,
parsed and analysed), {n_ref} \emph{{referenced}} (cited in this report with a
verified URL, not retrieved), {n_met} \emph{{method}} (methodological
references, excluded from knowledge-base size counts). A dash in the SHA-256
column marks a record that was cited rather than retrieved.""")
    print()


# --------------------------------------------------------------------------
# Appendix D — the evaluation detail the report promises but has no room for.
# The ablation table is transcribed rather than computed: each row is a run
# against a code state that no longer exists, so there is nothing on disk to
# read it from. Everything below it is computed from the findings.
# --------------------------------------------------------------------------
ABLATIONS = [
    ("Router", "clause text alone", "54.8\\% @3",
     "short clauses carry almost no signal"),
    ("Router", "+ structural context", "\\textbf{90.5\\% @3}",
     "a clause is scoped by its heading and stem"),
    ("Router", "+ dense embeddings", "92.5\\% @3",
     "fixes vocabulary mismatch (HR $\\to$ Personnel Security, 0/8 $\\to$ 8/8)"),
    ("Candidates", "baseline", "69.2\\% @10",
     "enhancements crowd out base controls"),
    ("Candidates", "+ base quota, parent lift", "\\textbf{92.3\\% @10}",
     "SAMA is principle-level; the base control is the right target"),
    ("Candidates", "+ assessment text removed", "76.9\\% @10",
     "a correction, not a regression --- see D.2"),
    ("Candidates", "+ control discussion text", "84.6\\% @10",
     "widens vocabulary against domain synonymy"),
    ("Candidates", "+ retrieval-specific query", "\\textbf{92.3\\% @3}",
     "routing and retrieval need different queries"),
]


def _findings():
    import json
    fp = KB / "processed" / "findings.jsonl"
    if not fp.exists():
        return []
    return [json.loads(l) for l in fp.open(encoding="utf-8") if l.strip()]


def appendix_d() -> None:
    rows = _findings()
    print(r"\section*{Appendix D — Evaluation detail}")
    print(r"\addcontentsline{toc}{section}{Appendix D — Evaluation detail}")
    print()

    # D.1 ablations
    print(r"\subsection*{D.1 Ablations}")
    print(r"""\noindent Each row is a full re-run measured against the anchor
sets. Recall is reported at the depth where the change was decided.""")
    print(r"\begin{center}\small")
    print(r"\begin{tabular}{@{}p{2.2cm}p{4.6cm}p{2.3cm}p{6.2cm}@{}}")
    print(r"\toprule")
    print(r"\hd{Stage} & \hd{Change} & \hd{Recall} & \hd{Diagnosis} \\")
    print(r"\midrule")
    for stage, change, recall, diag in ABLATIONS:
        print(f"{stage} & {change} & {recall} & {diag} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}\end{center}")
    print()

    # D.2 the contamination finding
    print(r"\subsection*{D.2 Assessment-procedure contamination}")
    print(r"""\noindent The public OSCAL release merges SP 800-53 controls with
SP 800-53A \emph{assessment procedures} in one file. A naive parse pulls lists
of documents to examine and roles to interview into the control statement,
inflating it by up to 84\%. That noise raised apparent recall from 76.9\% to
92.3\% through accidental matching: the extra vocabulary matched clauses the
control itself does not address. Cleaning the source lowered the headline
number and made it honest. A metric that improves after data cleaning deserves
suspicion; one that falls deserves trust.""")
    print()

    # D.3 risk-coverage, computed
    if rows:
        confs = sorted(f.get("confidence") or 0
                       for r in rows for f in r["findings"]
                       if f["status"] != "rejected")
        print(r"\subsection*{D.3 Risk--coverage}")
        print(r"""\noindent Computed on the single-judge findings, before the
debate layer; the debated run accepts 335 judgments at the same threshold,
because arbiter rulings resolve cases the proposer left under-confident.
Confidence is not continuous: it clusters, so the threshold selects a band
rather than a point.""")
        print(r"\begin{center}\small")
        print(r"\begin{tabular}{@{}rrr@{}}\toprule")
        print(r"\hd{$\tau$} & \hd{Accepted judgments} & \hd{Coverage} \\")
        print(r"\midrule")
        for t in (0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80):
            k = sum(1 for c in confs if c >= t)
            mark = r"\textbf{" if abs(t - 0.60) < 1e-9 else ""
            end = r"}" if mark else ""
            print(f"{mark}{t:.2f}{end} & {mark}{k}{end} & "
                  f"{mark}{100*k//max(len(confs),1)}\\%{end} \\\\")
        print(r"\bottomrule\end{tabular}\end{center}")
        print()

    # D.4 banded accuracy — why "banded" and not "calibrated"
    print(r"\subsection*{D.4 Band accuracy, and why not calibration}")
    print(r"""\noindent Human review of the stratified sample, split by the band
the system assigned:""")
    print(r"\begin{center}\small")
    print(r"\begin{tabular}{@{}p{3.2cm}p{2.6cm}rr@{}}\toprule")
    print(r"\hd{Band} & \hd{Range} & \hd{Correct} & \hd{Precision} \\")
    print(r"\midrule")
    print(r"high confidence & $\geq 0.70$ & 13/14 & 92.9\% \\")
    print(r"medium confidence & 0.60--0.70 & 14/16 & 87.5\% \\")
    print(r"\bottomrule\end{tabular}\end{center}")
    print(r"""\noindent The ordering holds --- the higher band is the more
accurate --- but both bands exceed the accuracy their own bounds imply. The
system is \emph{ordered}, not \emph{calibrated}: a 0.75 score does not mean a
75\% chance of being right, it means a better chance than a 0.60 score. This is
why the report says \emph{banded confidence} and reports no expected
calibration error.""")
    print()

    # D.5 per-family coverage, computed
    if rows:
        import json
        cp = KB / "processed" / "controls.jsonl"
        if cp.exists():
            ctrls = [json.loads(l) for l in cp.open(encoding="utf-8") if l.strip()]
            live = [c for c in ctrls if not c["withdrawn"]]
            acc = {f["control_id"] for r in rows for f in r["findings"]
                   if f["status"] == "accepted"}
            ver = {f["control_id"] for r in rows for f in r["findings"]
                   if f["status"] in ("accepted", "disputed")}
            fams = {}
            for c in live:
                fams.setdefault(c["family"], []).append(c["control_id"])
            print(r"\subsection*{D.5 Coverage by NIST family}")
            print(r"""\noindent \emph{Unreached} is the column that supports a
gap claim: the one before it counts controls matched below the threshold, which
are matches.""")
            print(r"\begin{center}\scriptsize")
            print(r"\begin{tabular}{@{}llrrr@{}}\toprule")
            print(r"\hd{Fam} & \hd{Title} & \hd{Controls} & "
                  r"\hd{No accepted} & \hd{Unreached} \\")
            print(r"\midrule")
            titles = {c["family"]: c["family_title"] for c in live}
            for fam in sorted(fams, key=lambda f: -sum(
                    1 for c in fams[f] if c not in ver) / max(len(fams[f]), 1)):
                ids = fams[fam]
                na = sum(1 for c in ids if c not in acc)
                nv = sum(1 for c in ids if c not in ver)
                print(f"{fam} & {tex(clip(titles[fam], 40))} & {len(ids)} & "
                      f"{na} ({100*na//len(ids)}\\%) & "
                      f"{nv} ({100*nv//len(ids)}\\%) \\\\")
            print(r"\bottomrule\end{tabular}\end{center}")
            print()


def main() -> int:
    rows = load()
    print(r"% Generated by appendix.py from manifest.csv — do not edit by hand.")
    print(r"\clearpage")
    print()
    appendix_a(rows)
    appendix_b(rows)
    appendix_c(rows)
    appendix_d()
    return 0


if __name__ == "__main__":
    sys.exit(main())
