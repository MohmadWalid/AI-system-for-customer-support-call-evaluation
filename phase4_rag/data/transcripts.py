"""
data/transcripts.py — Mock call transcript dataset.

Each transcript has:
  - call_id:    unique identifier for the call
  - agent:      agent name / ID
  - utterances: list of dicts with speaker ("agent" | "customer") and text
"""

TRANSCRIPTS = [
    {
        "call_id": "CALL-001",
        "agent": "Agent_Alice",
        "utterances": [
            {
                "speaker": "agent",
                "text": (
                    "Hi, I'm calling from FinanceXpert. I can guarantee that "
                    "this investment will give you a 15% annual return — "
                    "there's absolutely no risk involved."
                ),
            },
            {
                "speaker": "customer",
                "text": "That sounds too good to be true. Are you sure?",
            },
            {
                "speaker": "agent",
                "text": (
                    "One hundred percent sure. We've never had a client lose "
                    "money. You should sign up before the offer expires tonight."
                ),
            },
        ],
    },
    {
        "call_id": "CALL-002",
        "agent": "Agent_Bob",
        "utterances": [
            {
                "speaker": "agent",
                "text": (
                    "Before we go any further, I want to make sure you're "
                    "aware of all the fees associated with this product. "
                    "There's a 1.5% annual management charge and a 0.5% "
                    "entry fee. Are you happy to proceed?"
                ),
            },
            {
                "speaker": "customer",
                "text": "Yes, I understand the fees. Please go ahead.",
            },
            {
                "speaker": "agent",
                "text": (
                    "Great. Also, just to confirm you have a 14-day cooling-off "
                    "period during which you can cancel without any penalty."
                ),
            },
        ],
    },
    {
        "call_id": "CALL-003",
        "agent": "Agent_Carol",
        "utterances": [
            {
                "speaker": "customer",
                "text": "I'm not interested, please don't call me again.",
            },
            {
                "speaker": "agent",
                "text": (
                    "I understand, but just let me tell you about one more "
                    "product — I promise this is the last thing and it could "
                    "really benefit you. It'll only take two minutes."
                ),
            },
        ],
    },
    {
        "call_id": "CALL-004",
        "agent": "Agent_Dave",
        "utterances": [
            {
                "speaker": "agent",
                "text": (
                    "Based on your income and risk appetite, I'd recommend our "
                    "balanced portfolio. It's well-diversified and historically "
                    "has performed in line with market benchmarks."
                ),
            },
            {
                "speaker": "customer",
                "text": "What if the market drops?",
            },
            {
                "speaker": "agent",
                "text": (
                    "That's a fair concern. All investments carry risk and "
                    "past performance is not a guarantee of future results. "
                    "The balanced portfolio is designed to mitigate downside "
                    "risk, but losses are possible."
                ),
            },
        ],
    },
]
