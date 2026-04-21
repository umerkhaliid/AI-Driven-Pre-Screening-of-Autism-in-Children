"""
Scoring logic for the combined Q-CHAT-10 + M-CHAT-R screening tool.

24 unique questions total:
  a1-a10:  Q-CHAT-10 items (parent answers A-E, mapped to binary 0/1)
  a11-a24: M-CHAT-R unique items (parent answers Yes/No, mapped to binary 0/1)

Total screening score: 0-24 (sum of at-risk responses)
"""

# --- Q-CHAT-10 scoring (questions 1-10, answers A-E) ---

def qchat_score_item(question_number: int, answer_letter: str) -> int:
    """
    Q-CHAT-10 scoring rules:
      Q1-Q9: C, D, E => 1 (at risk)  |  A, B => 0 (typical)
      Q10:   A, B, C => 1 (at risk)  |  D, E => 0 (typical)
    """
    answer_letter = answer_letter.strip().upper()

    if question_number in range(1, 10):
        return 1 if answer_letter in ["C", "D", "E"] else 0

    if question_number == 10:
        return 1 if answer_letter in ["A", "B", "C"] else 0

    raise ValueError(f"Q-CHAT item number must be 1-10, got {question_number}")


# --- M-CHAT-R scoring (questions 11-24, answers Yes/No) ---

# Reverse-scored items: "Yes" = at risk (1), "No" = typical (0)
MCHAT_REVERSE_ITEMS = {12, 14, 18}

# Normal items: "No" = at risk (1), "Yes" = typical (0)
MCHAT_NORMAL_ITEMS = {11, 13, 15, 16, 17, 19, 20, 21, 22, 23, 24}


def mchat_score_item(question_number: int, answer_yes_no: str) -> int:
    """
    M-CHAT-R scoring for questions 11-24.

    Normal items (11,13,15-17,19-24):  No => 1 (at risk), Yes => 0
    Reverse items (12,14,18):          Yes => 1 (at risk), No => 0
    """
    answer = str(answer_yes_no).strip().lower()
    if answer not in ["yes", "no"]:
        raise ValueError(f"M-CHAT answer must be 'Yes' or 'No', got '{answer_yes_no}'")

    is_yes = answer == "yes"

    if question_number in MCHAT_REVERSE_ITEMS:
        return 1 if is_yes else 0

    if question_number in MCHAT_NORMAL_ITEMS:
        return 0 if is_yes else 1

    raise ValueError(f"M-CHAT item number must be 11-24, got {question_number}")


# --- Combined screening score ---

def compute_screening_score(qchat_answers: dict, mchat_answers: dict) -> int:
    """
    Computes total screening score (0-24).
    qchat_answers: {1: "A", 2: "C", ..., 10: "B"}
    mchat_answers: {11: "Yes", 12: "No", ..., 24: "Yes"}
    """
    total = 0
    for q in range(1, 11):
        total += qchat_score_item(q, qchat_answers[q])
    for q in range(11, 25):
        total += mchat_score_item(q, mchat_answers[q])
    return total


# --- Risk level interpretation ---

def screening_risk_level(total_score: int) -> str:
    """
    Maps total screening score (0-24) to 4-tier risk level.
    These thresholds are aligned with the synthetic dataset generation.
    """
    if total_score <= 3:
        return "No Risk"
    elif total_score <= 8:
        return "Mild Risk"
    elif total_score <= 15:
        return "Moderate Risk"
    else:
        return "Severe Risk"


def screening_referral_interpretation(total_score: int) -> str:
    """Referral recommendation based on screening score."""
    if total_score <= 3:
        return "Below Referral Threshold"
    elif total_score <= 8:
        return "Monitoring Recommended"
    elif total_score <= 15:
        return "Referral Suggested"
    else:
        return "Urgent Referral Recommended"
