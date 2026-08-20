#!/usr/bin/env python3
"""validate_manifest.py — guards manifest.csv against the failure modes that
actually occurred during editing: swallowed rows, embedded newlines, stale
tiers, broken hashes, duplicated note sentences. Run before every commit."""
import csv, re, sys, pathlib

PATH = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "manifest.csv")
EXPECTED_ROWS = 15
TIERS = {"core", "referenced", "method"}
DOC_ID = re.compile(r"^[A-Z]{2,4}-[A-Z0-9]{2,12}-[A-Z0-9]{2,20}-[0-9]{4}$")
SHA = re.compile(r"^[a-f0-9]{64}$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

errors = []
text = PATH.read_text(encoding="utf-8-sig")
rows = list(csv.DictReader(text.splitlines()))

# 1. row count
if len(rows) != EXPECTED_ROWS:
    errors.append(f"row count {len(rows)} != {EXPECTED_ROWS}")
# 2. no embedded newlines inside any field
for r in rows:
    for k, v in r.items():
        if v and "\n" in v:
            errors.append(f"{r['doc_id']}: embedded newline in {k}")
# 3. doc_id pattern + uniqueness
ids = [r["doc_id"] for r in rows]
if len(set(ids)) != len(ids):
    errors.append("duplicate doc_id")
for i in ids:
    if not DOC_ID.match(i):
        errors.append(f"bad doc_id: {i}")
# 4. tier enum
for r in rows:
    if r["tier"] not in TIERS:
        errors.append(f"{r['doc_id']}: tier '{r['tier']}' not in {sorted(TIERS)}")
# 5. core rows complete: url + sha256 + local_path + url_accessed
for r in rows:
    if r["tier"] == "core":
        for f in ("url", "sha256", "local_path", "url_accessed"):
            if not r.get(f):
                errors.append(f"{r['doc_id']}: core row missing {f}")
# 6. sha format wherever present
for r in rows:
    if r.get("sha256") and not SHA.match(r["sha256"]):
        errors.append(f"{r['doc_id']}: malformed sha256")
# 7. url_accessed date format wherever present
for r in rows:
    if r.get("url_accessed") and not DATE.match(r["url_accessed"]):
        errors.append(f"{r['doc_id']}: bad url_accessed '{r['url_accessed']}'")
# 8. no duplicated sentences inside notes (paste artefacts)
for r in rows:
    sents = [s.strip() for s in (r.get("notes") or "").split(".") if s.strip()]
    if any(sents.count(s) > 1 for s in set(sents)):
        errors.append(f"{r['doc_id']}: duplicated sentence in notes")
# 9. referenced/method rows carry a URL (mentor rule: reference links required)
for r in rows:
    if r["tier"] in ("referenced", "method") and not r.get("url"):
        errors.append(f"{r['doc_id']}: {r['tier']} row missing url")
# 10. tier census printed for the human eye
census = {t: sum(1 for r in rows if r["tier"] == t) for t in sorted(TIERS)}

if errors:
    print("FAIL")
    for e in errors:
        print(" -", e)
    sys.exit(1)
print(f"OK — {len(rows)} rows, tiers {census}")
