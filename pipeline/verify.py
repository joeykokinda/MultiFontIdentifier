"""
Self-verification and confidence.

The matcher already scores every candidate font by shape similarity. This
module turns those scores into an honest verdict per line: how strong the
top match is, and how clearly it beats the runner-up. Sibling fonts (League
Spartan vs Jost) score close together, so a small relative margin correctly
reads as "likely" rather than "verified" instead of asserting a guess.
"""


def font_verdict(top5: list[dict]) -> dict:
    """
    Turn a line's ranked font candidates into {verdict, confidence,
    match_score, alternatives}. Confidence reflects how far the top match
    separates from the next-best DISTINCT family.
    """
    if not top5:
        return {"verdict": "uncertain", "confidence": 0.0,
                "match_score": 0.0, "alternatives": []}

    s1 = float(top5[0]["confidence"])
    runner = next((c for c in top5[1:] if c["font"] != top5[0]["font"]), None)
    s2 = float(runner["confidence"]) if runner else 0.0
    rel_margin = (s1 - s2) / s1 if s1 > 0 else 0.0

    if s1 < 0.18:
        verdict = "uncertain"          # nothing matched the shape well
    elif rel_margin >= 0.20:
        verdict = "verified"
    elif rel_margin >= 0.06:
        verdict = "likely"
    else:
        verdict = "uncertain"          # top two are a toss-up

    confidence = round(min(1.0, rel_margin * 3.0) * min(1.0, s1 / 0.35), 2)
    alternatives = []
    for c in top5[1:]:
        name = c["font"]
        if name != top5[0]["font"] and name not in alternatives:
            alternatives.append(name)
        if len(alternatives) >= 3:
            break

    return {"verdict": verdict, "confidence": confidence,
            "match_score": round(s1, 4), "alternatives": alternatives}


def overall_verdict(line_verdicts: list[dict]) -> dict:
    """Aggregate per-line verdicts into one summary for the whole image."""
    if not line_verdicts:
        return {"verdict": "uncertain", "confidence": 0.0}
    order = {"verified": 2, "likely": 1, "uncertain": 0}
    worst = min(line_verdicts, key=lambda v: order[v["verdict"]])["verdict"]
    avg = round(sum(v["confidence"] for v in line_verdicts) / len(line_verdicts), 2)
    return {"verdict": worst, "confidence": avg}
