# SANAD — evidence-anchored regulatory alignment

SAMA Cyber Security Framework (2017) mapped against NIST SP 800-53 Release 5.2.0,
with every accepted claim anchored to a verbatim span of both sources. Three agents
on two models argue each mapping; a deterministic gate rejects any verdict whose
evidence is not a verbatim quote of both texts. The name is the method: in the
Arabic scholarly tradition, a *sanad* is the chain of transmission that
authenticates a text, and this system accepts no finding without one.

Report: `report.pdf` · Demo video: link included in the submitted report · Dashboard: `python app.py` (or open `launch.ipynb`)

## Quickstart (free-tier Colab or any Python 3.10+)

```bash
git clone https://github.com/Noura-Alrajeh/SANAD.git && cd SANAD
pip install -q -r requirements.txt
python app.py selftest     # prints every headline figure, computed live
python app.py              # launches the Gradio dashboard
```

`selftest` computes the headline figures **from the pipeline's own output files**
— there is no second copy of the numbers to drift out of sync. Reconcile its
output against §3/§6 of the report; the demo video opens with this command.

## Pipeline (run order)

| # | Stage | Script | Output |
|---|-------|--------|--------|
| 1 | Fetch & pin sources | `fetch.py` | raw documents + SHA-256 in `manifest.csv` |
| 2 | Parse NIST OSCAL | `oscal.py` | 1,014 active controls (withdrawn tombstoned) |
| 3 | Parse SAMA PDF | `sama.py` | 493 clauses (407 leaf; identifiers extracted, never invented) |
| 4 | Index corpus | `index.py` | 683 anchored passages, 8 core documents |
| 5 | Route families | `router.py` | 412,698 pairs → 128,487 (K=5) |
| 6 | Retrieve candidates | `candidates.py` | → 4,070 (N=10 per clause) |
| 7 | Judge | `judge.py` | 407 calls → `findings.jsonl` (single-judge) |
| 8 | Debate | `debate.py` | proposer/opponent/arbiter → `findings_v2.jsonl` |
| 9 | Evaluate | `precision.py` | stratified human sample, Wilson intervals |
| 10 | Present | `app.py` | dashboard + `selftest` |

`fetch.py verify --upstream` re-hashes every source against its recorded SHA-256;
a changed catalogue re-enters at stage 2. One row (SAMA CSF) mismatches **by
design** — its URL serves a page, not the file; see `manifest.csv` notes.

## Every figure ← its command

| Figure in the report | Command |
|---|---|
| Funnel, gate counts (541 / 335 / 177 / 29), gaps 119 = 17+53+49 | `python app.py selftest` |
| Debate mechanics: objections, verdicts, relationship shifts | `python debate.py report` |
| Precision 90% (95% CI 74–97), band accuracy | `python precision.py score` |
| Drift: SR 2/27, PT 1/21, Rev 5.2.0 matches | `python app.py selftest` (drift block) |
| Dashboard tables (Gaps / Mapping / Ask / Knowledge base) | `python app.py` |

**Two counting units** (report §6): a *judgment* is one clause–control pair; a
*clause* may carry several. `debate.py report` and `compare` count **debated
judgments only** (a narrower scope than the 541 total); system-wide totals come
from `selftest`. `judge.py report` reads the pre-debate `findings.jsonl` and is
diagnostic only.

## Knowledge base

`manifest.csv` — 15 records, one per row (8 core, fetched from issuing bodies and pinned by
SHA-256 · 4 referenced, cited with verified URLs, not retrieved · 3 method).
Schema: `manifest.schema.json`, offered as a candidate contribution to a
standardized ITU AI Readiness record format. Validate with
`python validate_manifest.py`.

Raw source documents are **not** redistributed; `fetch.py` re-obtains the
identical corpus from the issuing bodies and verifies each hash.

## Honest limitations (see report §7 and Appendix D)

Anchored-set recall is an optimistic floor (21/36 subdomains covered).
The evidenced-absence / not-tested split relies on a hand-curated family map;
broadening it would reclassify some of the 53, not change any judgment.
Four clause identifiers are duplicated in the source PDF (§6); corrected-
denominator coverage is 57.0% vs the reported 57.7%. An initial run left 12
debates unresolved (recorded as abstentions); the completed run moved one
clause (161→162) and no other headline figure.
