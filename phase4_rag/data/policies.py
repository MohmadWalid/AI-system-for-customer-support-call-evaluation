"""
data/policies.py — Mock policy dataset.

Each policy has:
  - id:       unique identifier
  - name:     short human-readable label
  - tag:      one of the five call-stage categories
  - text:     full policy text (used for embedding & LLM context)
"""

POLICIES = [
    # ── ADVICE ────────────────────────────────────────────────────────────────
    {
        "id": "ADV-001",
        "name": "Suitable Advice Only",
        "tag": "advice",
        "text": (
            "Agents must provide advice that is appropriate to the customer's "
            "stated financial situation and needs. Recommending products that "
            "are clearly unsuitable for the customer's profile is prohibited."
        ),
    },
    {
        "id": "ADV-002",
        "name": "No Guaranteed Returns",
        "tag": "advice",
        "text": (
            "Agents must never guarantee investment returns or imply that a "
            "product carries no risk. All forward-looking statements must be "
            "accompanied by appropriate risk disclaimers."
        ),
    },
    {
        "id": "ADV-003",
        "name": "Fact-Based Recommendations",
        "tag": "advice",
        "text": (
            "Advice must be based on verified facts and documented customer "
            "information. Agents may not fabricate or exaggerate product "
            "features to influence a purchase decision."
        ),
    },

    # ── DISCLOSURE ─────────────────────────────────────────────────────────────
    {
        "id": "DIS-001",
        "name": "Fee Disclosure",
        "tag": "disclosure",
        "text": (
            "All applicable fees, charges, and commissions must be disclosed "
            "clearly before the customer commits to any product or service. "
            "Omitting or understating fees is a compliance violation."
        ),
    },
    {
        "id": "DIS-002",
        "name": "Conflict of Interest Disclosure",
        "tag": "disclosure",
        "text": (
            "Agents must disclose any personal or organisational conflict of "
            "interest that may influence their recommendations. Failure to "
            "disclose known conflicts is prohibited."
        ),
    },
    {
        "id": "DIS-003",
        "name": "Regulatory Status Disclosure",
        "tag": "disclosure",
        "text": (
            "Agents must identify themselves and their regulatory status at "
            "the start of every call. Misrepresenting authorisation or "
            "qualifications is strictly prohibited."
        ),
    },

    # ── RESOLUTION ─────────────────────────────────────────────────────────────
    {
        "id": "RES-001",
        "name": "Timely Complaint Resolution",
        "tag": "resolution",
        "text": (
            "Customer complaints must be acknowledged within one business day "
            "and fully resolved within the regulatory deadline. Agents may not "
            "dismiss or delay complaints without documented justification."
        ),
    },
    {
        "id": "RES-002",
        "name": "Accurate Complaint Logging",
        "tag": "resolution",
        "text": (
            "Every complaint raised during a call must be logged accurately in "
            "the CRM system. Failing to record a complaint, or recording "
            "inaccurate details, is a compliance violation."
        ),
    },

    # ── CLOSING ────────────────────────────────────────────────────────────────
    {
        "id": "CLO-001",
        "name": "Customer Consent at Closing",
        "tag": "closing",
        "text": (
            "Before completing a sale, agents must obtain explicit verbal "
            "consent from the customer confirming they understand and agree to "
            "all product terms. Proceeding without confirmed consent is prohibited."
        ),
    },
    {
        "id": "CLO-002",
        "name": "Cooling-Off Period Reminder",
        "tag": "closing",
        "text": (
            "Agents must inform customers of their right to cancel within the "
            "statutory cooling-off period at the point of sale. Omitting this "
            "information is a compliance violation."
        ),
    },
    {
        "id": "CLO-003",
        "name": "No Pressure Closing",
        "tag": "closing",
        "text": (
            "Agents must not use high-pressure or manipulative tactics to push "
            "a customer into closing a sale. This includes creating false urgency "
            "or threatening the withdrawal of an offer."
        ),
    },

    # ── OBJECTION HANDLING ─────────────────────────────────────────────────────
    {
        "id": "OBJ-001",
        "name": "Honest Objection Response",
        "tag": "objection_handling",
        "text": (
            "When a customer raises an objection, agents must respond honestly "
            "and accurately. Providing misleading information to overcome an "
            "objection is a compliance violation."
        ),
    },
    {
        "id": "OBJ-002",
        "name": "Respect Customer Refusal",
        "tag": "objection_handling",
        "text": (
            "If a customer clearly states they do not wish to proceed or do not "
            "want to hear further information, the agent must respect that decision "
            "immediately without continued persuasion attempts."
        ),
    },
    {
        "id": "OBJ-003",
        "name": "No False Scarcity",
        "tag": "objection_handling",
        "text": (
            "Agents must not fabricate limited availability or artificial "
            "deadlines to pressure customers into overcoming their objections. "
            "All scarcity claims must be factually accurate."
        ),
    },
]
