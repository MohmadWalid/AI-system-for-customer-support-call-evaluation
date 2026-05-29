"""
src/scoring.py — LLM-powered call quality scoring.
"""
import json
import re
import time
from pathlib import Path

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL

DISMISSIVE_PHRASES = [
    "nothing we can do",
    "contractual",
    "not our fault",
    "your fault",
]


def assess_resolution(agent_turns: list[str], fine_label: str) -> dict:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(agent_turns))
    prompt = (
        f"You are a banking call center QA evaluator.\n"
        f"Issue class: {fine_label}\n\n"
        f"Agent turns:\n{numbered}\n\n"
        f"Did the agent successfully resolve the customer's issue?\n"
        f"Consider: did they take action, provide a solution, or give clear next steps?\n\n"
        f'Reply ONLY in this JSON format:\n'
        f'{{"resolved": true or false, "reason": "one sentence"}}'
    )

    client = Groq(api_key=GROQ_API_KEY)
    parsed = None

    for attempt in range(1, 4):
        for api_attempt in range(1, 10):
            try:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=150,
                )
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e) or (hasattr(e, "status_code") and e.status_code == 429):
                    print(f"  [Groq-Scoring] Rate limit hit. Sleeping 5s before retry (Attempt {api_attempt}/9)...")
                    time.sleep(5)
                else:
                    raise e

        raw = response.choices[0].message.content.strip()
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            raw_json = raw[start:end]
            parsed = json.loads(raw_json)
            break                           # success — exit retry loop
        except (ValueError, json.JSONDecodeError):
            match = re.search(r'"resolved"\s*:\s*(true|false)', raw, re.IGNORECASE)
            if not match:
                match = re.search(r'resolved\b.*?\b(true|false|yes|no)', raw, re.IGNORECASE)
            if match:
                val = match.group(1).lower()
                resolved_val = val in ["true", "yes"]
                parsed = {"resolved": resolved_val, "reason": "(extracted via fuzzy regex)"}
                break
            else:
                print(f"  [assess_resolution] attempt {attempt}/3 failed to parse JSON — retrying...")
                parsed = None

    if parsed is None:
        parsed = {"resolved": False, "reason": "(parse error after 3 attempts)"}

    resolved = bool(parsed.get("resolved", False))
    return {
        "resolved": resolved,
        "reason": parsed.get("reason", ""),
        "score": 100 if resolved else 40,
    }


def assess_communication(violations: list[dict]) -> dict:
    if not violations:
        return {"score": 100, "note": "No communication violations detected."}

    evidence_text = " ".join(v.get("evidence", "") for v in violations).lower()

    for phrase in DISMISSIVE_PHRASES:
        if phrase in evidence_text:
            return {
                "score": 40,
                "note": f"Dismissive language detected: '{phrase}'.",
            }

    return {
        "score": 70,
        "note": "Violations present but no dismissive language found.",
    }


def compute_score(call_result: dict, agent_turns: list[str]) -> dict:
    violations = call_result.get("violations", [])
    total_turns = len(agent_turns)
    ok_turns = total_turns - len(violations)
    policy_compliance = (ok_turns / total_turns * 100) if total_turns else 0.0

    resolution = assess_resolution(agent_turns, call_result.get("fine_label_predicted", ""))
    communication = assess_communication(violations)

    final_score = (
        policy_compliance * 0.50
        + resolution["score"] * 0.30
        + communication["score"] * 0.20
    )

    if final_score >= 90:
        grade = "A"
    elif final_score >= 75:
        grade = "B"
    elif final_score >= 60:
        grade = "C"
    else:
        grade = "D"

    return {
        "call_id": call_result["call_id"],
        "policy_compliance": round(policy_compliance, 1),
        "issue_resolution": resolution,
        "communication": communication,
        "final_score": round(final_score, 1),
        "grade": grade,
    }


def score_all_calls(results_path: str, transcripts: list[dict]) -> None:
    path = Path(results_path)
    results = json.loads(path.read_text(encoding="utf-8"))

    transcript_map = {t["call_id"]: t for t in transcripts}

    scored = []
    print(f"\n{'Call ID':<12} {'Compliance':>10} {'Resolution':>11} {'Comm':>6} {'Final':>7} {'Grade':>6}")
    print("-" * 55)

    for call_result in results:
        call_id = call_result["call_id"]
        transcript = transcript_map.get(call_id)
        if transcript is None:
            print(f"  WARNING: no transcript found for {call_id}, skipping.")
            scored.append(call_result)
            continue

        agent_turns = [
            u["text"] for u in transcript["utterances"] if u["speaker"] == "agent"
        ]

        scoring = compute_score(call_result, agent_turns)
        call_result["score"] = scoring

        print(
            f"{call_id:<12}"
            f"{scoring['policy_compliance']:>10.1f}"
            f"{scoring['issue_resolution']['score']:>11}"
            f"{scoring['communication']['score']:>6}"
            f"{scoring['final_score']:>7.1f}"
            f"  {scoring['grade']}"
        )
        scored.append(call_result)

    path.write_text(json.dumps(scored, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults with scores saved to: {path.resolve()}")
