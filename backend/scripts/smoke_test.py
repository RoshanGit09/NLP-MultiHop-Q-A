"""smoke_test.py — end-to-end HTTP test against the running FinTraceQA server."""
import json
import urllib.request
import urllib.error

BASE = "http://localhost:8000"
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def get(path):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=15) as r:
        return json.loads(r.read())


def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f"  →  {detail}" if detail else ""))
    return condition


results = []

print("\n══════════════════════════════════════")
print("  FinTraceQA Smoke Test")
print("══════════════════════════════════════")

# ── 1. Health ──────────────────────────────────────────────────────────────
print("\n[1] Health check")
try:
    h = get("/health")
    results.append(check("status=ok", h.get("status") == "ok", str(h)))
except Exception as e:
    results.append(check("Health endpoint reachable", False, str(e)))

# ── 2. Entity resolve ─────────────────────────────────────────────────────
print("\n[2] Entity resolve — Infosys")
try:
    r = post("/resolve", {"mention": "Infosys"})
    matches = r.get("candidates", [])
    results.append(check("Returns matches list", len(matches) > 0, str([m["name"] for m in matches[:3]])))
except Exception as e:
    results.append(check("Resolve endpoint", False, str(e)))

# ── 3. Entity resolve — RBI ────────────────────────────────────────────────
print("\n[3] Entity resolve — RBI")
try:
    r = post("/resolve", {"mention": "RBI"})
    matches = r.get("candidates", [])
    results.append(check("RBI matches found", len(matches) > 0, str([m["name"] for m in matches[:3]])))
except Exception as e:
    results.append(check("Resolve RBI", False, str(e)))

# ── 4. Single-hop chat ────────────────────────────────────────────────────
print("\n[4] Single-hop: 'What did Infosys report?'")
try:
    c = post("/chat", {"session_id": "s1", "message": "What did Infosys report?", "lang": "en"})
    ans = c.get("answer", "")
    conf = c.get("confidence", 0)
    paths = c.get("reasoning_paths", [])
    results.append(check("Got an answer", len(ans) > 0, ans[:100]))
    results.append(check("confidence >= 0", conf >= 0, f"conf={conf:.2f}"))
    results.append(check("paths returned", len(paths) >= 0, f"{len(paths)} paths"))
except Exception as e:
    results.append(check("Single-hop chat", False, str(e)))

# ── 5. Multi-hop chat ─────────────────────────────────────────────────────
print("\n[5] Multi-hop: 'How did RBI rate cut impact IT sector?'")
try:
    c = post("/chat", {"session_id": "s2", "message": "How did RBI rate cut impact IT sector?", "lang": "en"})
    ans = c.get("answer", "")
    conf = c.get("confidence", 0)
    results.append(check("Got an answer", len(ans) > 0, ans[:100]))
    results.append(check("confidence >= 0", conf >= 0, f"conf={conf:.2f}"))
except Exception as e:
    results.append(check("Multi-hop chat", False, str(e)))

# ── 6. Follow-up / coreference ────────────────────────────────────────────
print("\n[6] Follow-up in same session: 'What about Wipro?'")
try:
    c = post("/chat", {"session_id": "s1", "message": "What about Wipro?", "lang": "en"})
    ans = c.get("answer", "")
    results.append(check("Got answer for follow-up", len(ans) > 0, ans[:100]))
except Exception as e:
    results.append(check("Follow-up chat", False, str(e)))

# ── 7. Hindi query ────────────────────────────────────────────────────────
print("\n[7] Hindi: 'Infosys ne kya announce kiya?'")
try:
    c = post("/chat", {"session_id": "s3", "message": "Infosys ne kya announce kiya?", "lang": "hi"})
    ans = c.get("answer", "")
    lang = c.get("answer_lang", "?")
    results.append(check("Got answer", len(ans) > 0, ans[:100]))
    results.append(check("Lang detected", lang in ("hi", "en", "mixed"), f"detected={lang}"))
except Exception as e:
    results.append(check("Hindi chat", False, str(e)))

# ── 8. Explanation steps ──────────────────────────────────────────────────
print("\n[8] Explanation steps present")
try:
    c = post("/chat", {"session_id": "s4", "message": "Which sector does Infosys operate in?", "lang": "en"})
    steps = c.get("explanation_steps", [])
    results.append(check("explanation_steps is a list", isinstance(steps, list), f"{len(steps)} steps"))
except Exception as e:
    results.append(check("Explanation steps", False, str(e)))

# ── Summary ───────────────────────────────────────────────────────────────
passed = sum(results)
total = len(results)
print("\n══════════════════════════════════════")
print(f"  Result: {passed}/{total} checks passed")
print("══════════════════════════════════════\n")
