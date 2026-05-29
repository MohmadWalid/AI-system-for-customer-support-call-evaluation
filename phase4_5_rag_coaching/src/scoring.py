"""
src/scoring.py — LLM-powered call quality scoring.
"""
import json
import time
from pathlib import Path

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL

def assess_quality(agent_turns: list[str], fine_label: str) -> dict:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(agent_turns))
    prompt = (
        f"You are a banking call center QA evaluator.\n"
        f"Issue class: {fine_label}\n\n"
        f"Agent turns:\n{numbered}\n\n"
        f"Evaluate the agent on two dimensions:\n\n"
        f"1. RESOLUTION: Did the agent successfully resolve the customer's issue?\n"
        f"   Consider: did they take action, provide a solution, or give clear next steps?\n\n"
        f"2. COMMUNICATION: How was the agent's communication quality?\n"
        f"   Look for: dismissive language, lack of empathy, rude tone, "
        f"unhelpful responses, or unprofessional behavior.\n"
        f"   100 = professional, empathetic, clear\n"
        f"   70  = acceptable but could improve\n"
        f"   40  = dismissive, rude, or unprofessional\n\n"
        f"Reply ONLY in this JSON format:\n"
        f'{{"resolved": true or false, '
        f'"resolution_reason": "one sentence", '
        f'"communication_score": 40 or 70 or 100, '
        f'"communication_note": "one sentence"}}'
    )

    client = Groq(api_key=GROQ_API_KEY)

    for api_attempt in range(1, 10):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200,
            )
            break
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e) or (
                hasattr(e, "status_code") and e.status_code == 429
            ):
                print(f"  [Groq-Quality] Rate limit hit. Sleeping 5s (attempt {api_attempt}/9)...")
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
            "resolved": False,
            "resolution_reason": "(parse error)",
            "communication_score": 70,
            "communication_note": "(parse error)",
        }

    resolved = bool(parsed.get("resolved", False))
    return {
        "resolution": {
            "resolved": resolved,
            "reason":   parsed.get("resolution_reason", ""),
            "score":    100 if resolved else 40,
        },
        "communication": {
            "score": parsed.get("communication_score", 70),
            "note":  parsed.get("communication_note", ""),
        },
    }


def compute_score(call_result: dict, agent_turns: list[str]) -> dict:
    violations = call_result.get("violations", [])
    total_turns = len(agent_turns)
    ok_turns = total_turns - len(violations)
    policy_compliance = (ok_turns / total_turns * 100) if total_turns else 0.0

    fine_label = call_result.get("fine_label_predicted", "")
    quality = assess_quality(agent_turns, fine_label)
    resolution = quality["resolution"]
    communication = quality["communication"]

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
