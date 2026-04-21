"""
Maps parent answers to binary features for the combined screening tool.

Q-CHAT-10 (a1-a10): A/B/C/D/E answers -> 0/1
M-CHAT-R  (a11-a24): Yes/No answers -> 0/1
"""

from typing import Dict
from src.scoring import qchat_score_item, mchat_score_item


def map_qchat_answers_to_features(qchat_answers: Dict[int, str]) -> Dict[str, int]:
    """
    Maps Q-CHAT-10 answers (A-E) to binary features a1..a10.

    Input:  {1: "A", 2: "C", ..., 10: "B"}
    Output: {"a1": 0, "a2": 1, ..., "a10": 1}
    """
    features = {}
    for q in range(1, 11):
        if q not in qchat_answers:
            raise ValueError(f"Missing Q-CHAT answer for question {q}")
        features[f"a{q}"] = qchat_score_item(q, qchat_answers[q])
    return features


def map_mchat_answers_to_features(mchat_answers: Dict[int, str]) -> Dict[str, int]:
    """
    Maps M-CHAT-R answers (Yes/No) to binary features a11..a24.

    Input:  {11: "Yes", 12: "No", ..., 24: "Yes"}
    Output: {"a11": 0, "a12": 0, ..., "a24": 0}
    """
    features = {}
    for q in range(11, 25):
        if q not in mchat_answers:
            raise ValueError(f"Missing M-CHAT-R answer for question {q}")
        features[f"a{q}"] = mchat_score_item(q, mchat_answers[q])
    return features


def map_all_answers_to_features(
    qchat_answers: Dict[int, str],
    mchat_answers: Dict[int, str]
) -> Dict[str, int]:
    """
    Combines Q-CHAT-10 + M-CHAT-R answers into a single feature dict (a1..a24).
    """
    features = map_qchat_answers_to_features(qchat_answers)
    features.update(map_mchat_answers_to_features(mchat_answers))
    return features


def compute_total_score(mapped_features: Dict[str, int]) -> int:
    """Computes total screening score (0-24) from mapped binary features."""
    return sum(mapped_features[f"a{i}"] for i in range(1, 25))
