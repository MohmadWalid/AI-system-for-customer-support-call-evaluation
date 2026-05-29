"""
src/coaching.py — Generates per-call coaching reports from results.json.
"""
import json
import time
from pathlib import Path

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL

COACHING_DIR = Path("data/coaching")


def generate_coaching_report(call_result: dict) -> dict:
    fine_label      = call_result["fine_label_predicted"]
    overall_summary = call_result.get("overall_summary", "")
    violations      = call_result.get("violations", [])

    if violations:
        violation_lines = []
        for v in violations:
            violation_lines.append(
                f'- Turn {v.get("turn", "?")} : "{v.get("evidence", "")}"\n'
                f'  Policy  : {v.get("violated_policy", "")}\n'
                f'  Reason  : {v.get("reason", "")}'
            )
        violation_text = "\n".join(violation_lines)
    else:
        violation_text = "- (none)"

    score = call_result.get("score") or {}
    policy_compliance = score.get("policy_compliance", "N/A")
    resolution_score = score.get("issue_resolution", {}).get("score", "N/A")
    resolution_reason = score.get("issue_resolution", {}).get("reason", "")
    comm_score = score.get("communication", {}).get("score", "N/A")
    comm_note = score.get("communication", {}).get("note", "")
    final_score = score.get("final_score", "N/A")
    grade = score.get("grade", "N/A")

    recovered = call_result.get("recovered", False)
    recovery_note = call_result.get("recovery_note", "")

    recovery_addition = ""
    if recovered:
        recovery_addition = f"\n- Note: Agent recovered from an earlier mistake.\n    {recovery_note}"

    prompt = (
        f"You are a banking call center coach.\n\n"
        f"Issue class: {fine_label}\n\n"
        f"PERFORMANCE SCORES:\n"
        f"- Final Grade: {grade} ({final_score}/100)\n"
        f"- Policy Compliance: {policy_compliance}%\n"
        f"- Resolution: {resolution_score} — {resolution_reason}\n"
        f"- Communication: {comm_score} — {comm_note}"
        f"{recovery_addition}\n\n"
        f"VIOLATIONS:\n"
        f"{violation_text}\n\n"
        f"Write a coaching report with exactly these three sections:\n"
        f"1. STRENGTHS: 2-3 bullet points of what the agent did well\n"
        f"2. IMPROVEMENTS: 2-3 bullet points of specific behaviors to fix\n"
        f"3. REPHRASING: For each violation, rewrite a better agent response\n\n"
        f"Reply ONLY in this JSON format:\n"
        f'{{"strengths": ["...", "..."], "improvements": ["...", "..."], '
        f'"rephrasing": [{{"original": "...", "better": "..."}}]}}'
    )

    client   = Groq(api_key=GROQ_API_KEY)
    
    for api_attempt in range(1, 10):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=600,
            )
            break
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e) or (
                hasattr(e, "status_code") and e.status_code == 429
            ):
                print(f"  [Groq-Coaching] Rate limit hit. Sleeping 5s (attempt {api_attempt}/9)...")
                time.sleep(5)
            else:
                raise e

    raw = response.choices[0].message.content.strip()

    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        parsed = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        parsed = {
            "strengths":    ["(parse error — raw response below)"],
            "improvements": [],
            "rephrasing":   [],
            "raw":          raw,
        }

    return {
        "call_id":              call_result["call_id"],
        "fine_label_predicted": fine_label,
        "recovered":            recovered,
        "recovery_note":        recovery_note,
        "verdict":              call_result.get("verdict"),
        "violations_count":     len(violations),
        "strengths":            parsed.get("strengths", []),
        "improvements":         parsed.get("improvements", []),
        "rephrasing":           parsed.get("rephrasing", []),
    }



def generate_all_reports(results_path: str = "data/results.json") -> None:
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))

    COACHING_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating coaching reports for {len(results)} calls...\n")

    for call_result in results:
        call_id = call_result["call_id"]
        print(f"  {call_id} ...", end=" ", flush=True)

        report      = generate_coaching_report(call_result)
        output_path = COACHING_DIR / f"{call_id}.json"
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"OK  ({report['violations_count']} violations)")

    print(f"\nDone. Reports saved to: {COACHING_DIR.resolve()}")


if __name__ == "__main__":
    generate_all_reports()
