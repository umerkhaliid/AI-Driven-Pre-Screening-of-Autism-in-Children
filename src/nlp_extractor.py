"""
NLP-Based Symptom Extraction from Free-Text Parent Input
=========================================================
Pipeline (3 stages):
  1. Tokenisation + sentence segmentation — spaCy en_core_web_sm
  2. Keyword/phrase matching — curated PhraseMatcher per symptom category
  3. Negation detection — negspaCy (NegEx algorithm) with regex fallback

Output: 10-dimensional binary flag vector
  1 = symptom signal present (after negation resolved)
  0 = no signal detected / not mentioned

The 10 flags span DSM-5 symptom domains:
  Social communication : nlp_eye_contact_absent, nlp_name_response_absent,
                         nlp_social_interest_absent
  Joint attention      : nlp_pointing_absent
  Repetitive behaviour : nlp_echolalia_present, nlp_repetitive_behaviour
  Sensory processing   : nlp_sensory_sensitivity
  Communication        : nlp_language_delay, nlp_gesture_absent
  Imagination          : nlp_pretend_play_absent

References:
  - Robins et al. (2014). Validation of the Modified Checklist for Autism in
    Toddlers, Revised with Follow-Up (M-CHAT-R/F). Pediatrics.
  - Allison et al. (2012). The Q-CHAT (Quantitative CHecklist for Autism in
    Toddlers): A normally distributed quantitative measure. J Autism Dev Disord.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Ordered list of NLP feature column names ─────────────────────────────────
# ORDER IS FIXED — must match synthetic dataset columns and model training.
NLP_FEATURE_NAMES: List[str] = [
    "nlp_eye_contact_absent",
    "nlp_name_response_absent",
    "nlp_pointing_absent",
    "nlp_echolalia_present",
    "nlp_repetitive_behaviour",
    "nlp_sensory_sensitivity",
    "nlp_social_interest_absent",
    "nlp_language_delay",
    "nlp_gesture_absent",
    "nlp_pretend_play_absent",
]

# ── Curated keyword/phrase patterns per symptom category ─────────────────────
# Phrases are lowercase; matching is case-insensitive.
# Sorted longest-first per category to avoid partial-match shadowing.
SYMPTOM_PATTERNS: Dict[str, List[str]] = {
    "nlp_eye_contact_absent": [
        "no eye contact",
        "poor eye contact",
        "lack of eye contact",
        "avoids eye contact",
        "never makes eye contact",
        "avoids looking",
        "looks away",
        "won't look",
        "doesn't look at us",
        "doesn't look at me",
        "doesn't look at my face",
        "looking in my eyes",
        "look in the eye",
        "look at us",
        "look at me",
        "eye contact",
    ],
    "nlp_name_response_absent": [
        "doesn't respond to his name",
        "doesn't respond to her name",
        "doesn't respond to their name",
        "no response to name",
        "not responding to name",
        "ignores when called",
        "ignores us when",
        "ignores me when",
        "doesn't react when we call",
        "doesn't turn when",
        "doesn't respond when called",
        "doesn't respond",
        "ignores us",
        "ignores me",
        "respond to name",
        "response to name",
        "call his name",
        "call her name",
        "call their name",
    ],
    "nlp_pointing_absent": [
        "doesn't point at things",
        "does not point at things",
        "never points at things",
        "no pointing",
        "not pointing",
        "doesn't use finger to point",
        "doesn't point",
        "does not point",
        "never points",
        "won't point",
        "point at things",
        "pointing at",
    ],
    "nlp_echolalia_present": [
        "repeats everything we say",
        "repeats what we say",
        "repeats words over and over",
        "repeats phrases",
        "repeats words",
        "memorised phrases",
        "memorized phrases",
        "scripted phrases",
        "scripted speech",
        "quotes from tv",
        "quotes from movies",
        "repeating lines",
        "echoes what",
        "echolalia",
        "echoing",
        "parrots",
        "parrot",
    ],
    "nlp_repetitive_behaviour": [
        "lines things up",
        "lines up toys",
        "lines up objects",
        "lines up",
        "same routine every day",
        "insists on the same",
        "insists on same",
        "upset when routine changes",
        "tantrums when routine",
        "hand flapping",
        "flaps his hands",
        "flaps her hands",
        "flapping hands",
        "flapping",
        "spinning objects",
        "spins in circles",
        "spinning",
        "rocks back and forth",
        "rocking motion",
        "rocking",
        "head banging",
        "head-banging",
        "repetitive behavior",
        "repetitive behaviour",
        "same thing over and over",
        "fixated on",
        "fixation on",
        "arranges toys",
        "arranges objects",
        "arranges",
    ],
    "nlp_sensory_sensitivity": [
        "sensitive to loud noises",
        "sensitive to noise",
        "sensitive to sound",
        "sensitive to light",
        "bothered by loud",
        "bothered by noise",
        "bothered by sound",
        "covers his ears",
        "covers her ears",
        "covers ears",
        "plugs ears",
        "overreacts to sounds",
        "overreacts to noise",
        "overreacts to",
        "extreme reaction to",
        "hates certain textures",
        "refuses certain textures",
        "doesn't like being touched",
        "dislikes touch",
        "smells everything",
        "sniffs objects",
        "sensory issues",
        "sensory problems",
        "sensory",
    ],
    "nlp_social_interest_absent": [
        "not interested in other children",
        "no interest in other children",
        "doesn't play with other children",
        "doesn't interact with other children",
        "prefers to be alone",
        "plays alone all the time",
        "plays alone",
        "ignores other kids",
        "doesn't notice other children",
        "doesn't seek attention",
        "doesn't initiate",
        "no friends",
        "doesn't make friends",
        "very withdrawn",
        "withdrawn",
        "not social",
    ],
    "nlp_language_delay": [
        "no words at all",
        "no words",
        "lost words",
        "lost language",
        "lost speech",
        "speech regression",
        "language regression",
        "stopped talking",
        "used to say words",
        "delayed speech",
        "delayed language",
        "speech delay",
        "language delay",
        "few words",
        "limited words",
        "only a few words",
        "barely talks",
        "hardly speaks",
        "not talking",
        "not speaking yet",
        "not speaking",
        "doesn't talk",
        "non-verbal",
        "nonverbal",
    ],
    "nlp_gesture_absent": [
        "doesn't wave goodbye",
        "doesn't wave hello",
        "doesn't wave hi",
        "never waves goodbye",
        "no waving",
        "won't wave",
        "doesn't wave",
        "does not wave",
        "never waves",
        "no gestures",
        "doesn't use gestures",
        "doesn't clap",
        "won't clap",
        "doesn't nod",
        "no nodding",
    ],
    "nlp_pretend_play_absent": [
        "no pretend play",
        "no imaginative play",
        "no make-believe",
        "not imaginative",
        "lacks imagination",
        "doesn't pretend to cook",
        "doesn't feed dolls",
        "doesn't use toys properly",
        "doesn't play with toys",
        "doesn't pretend",
        "does not pretend",
        "doesn't imitate",
        "doesn't copy",
        "no symbolic play",
    ],
}


# ── Pre-compile all phrase patterns ──────────────────────────────────────────

def _build_compiled_patterns() -> Dict[str, re.Pattern]:
    compiled = {}
    for flag, phrases in SYMPTOM_PATTERNS.items():
        sorted_phrases = sorted(phrases, key=len, reverse=True)
        pattern = "|".join(re.escape(p) for p in sorted_phrases)
        compiled[flag] = re.compile(pattern, re.IGNORECASE)
    return compiled


_COMPILED_PATTERNS: Dict[str, re.Pattern] = _build_compiled_patterns()

# ── NegEx-style negation trigger terms ───────────────────────────────────────
_NEGATION_PATTERN = re.compile(
    r"\b(not|no|never|won't|wouldn't|doesn't|does not|don't|do not|"
    r"cannot|can't|hasn't|haven't|without|fails to|unable to|nor|"
    r"neither|refuses|denies|absent|lacks|lack of|rarely|seldom)\b",
    re.IGNORECASE,
)


# ── spaCy + negspaCy lazy loader ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_nlp():
    """
    Lazily load spaCy pipeline + negspaCy component.
    Returns (nlp, has_negex: bool).
    If spaCy is unavailable, returns (None, False) — regex fallback is used.
    """
    try:
        import spacy
    except ImportError:
        logger.warning(
            "spaCy not installed. NLP extraction uses regex fallback. "
            "Install: pip install spacy && python -m spacy download en_core_web_sm"
        )
        return None, False

    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "attribute_ruler", "lemmatizer"])
    except OSError:
        logger.warning(
            "spaCy model 'en_core_web_sm' not found. "
            "Install: python -m spacy download en_core_web_sm"
        )
        return None, False

    has_negex = False
    try:
        import negspacy  # noqa: F401
        if "negex" not in nlp.pipe_names:
            nlp.add_pipe(
                "negex",
                config={"neg_termset": "en_clinical_sensitive"},
                last=True,
            )
        has_negex = True
    except (ImportError, Exception) as e:
        logger.warning(
            f"negspaCy not available ({e}). Negation detection uses regex fallback. "
            "Install: pip install negspacy"
        )

    return nlp, has_negex


# ── Core helpers ─────────────────────────────────────────────────────────────

def _default_flags() -> Dict[str, int]:
    """Returns all-zero flag dict (safe default when text is empty/unavailable)."""
    return {name: 0 for name in NLP_FEATURE_NAMES}


def _is_negated_regex(sentence: str) -> bool:
    """
    Regex-based NegEx-inspired negation check.
    Returns True if the sentence contains a negation trigger within 6 tokens
    before the matched phrase (approximated by whole-sentence check here).
    """
    return bool(_NEGATION_PATTERN.search(sentence))


def _extract_regex_fallback(text: str) -> Dict[str, int]:
    """
    Pure-regex extraction pipeline used when spaCy is unavailable.
    Splits text into clauses, then runs negation + phrase matching per clause.
    """
    flags = _default_flags()
    # Split on sentence/clause boundaries
    clauses = re.split(r"[.!?;]|\band\b|\bbut\b|\balso\b|\bhowever\b", text)
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        negated = _is_negated_regex(clause)
        for flag, pattern in _COMPILED_PATTERNS.items():
            if pattern.search(clause) and not negated:
                flags[flag] = 1
    return flags


def _extract_with_spacy(text: str, nlp, has_negex: bool) -> Dict[str, int]:
    """
    Full spaCy pipeline:
    1. Tokenise and sentencise
    2. Per sentence: match keyword phrases
    3. Per sentence: check negation (negspaCy entity attribute if available,
       else regex NegEx fallback)
    """
    flags = _default_flags()
    doc = nlp(text)

    for sent in doc.sents:
        sent_text = sent.text

        for flag, pattern in _COMPILED_PATTERNS.items():
            match = pattern.search(sent_text)
            if not match:
                continue

            # Determine negation
            if has_negex:
                is_negated = _negex_check(sent, sent_text, match)
            else:
                is_negated = _is_negated_regex(sent_text)

            if not is_negated:
                flags[flag] = 1

    return flags


def _negex_check(sent, sent_text: str, match: re.Match) -> bool:
    """
    Check negation using negspaCy entity attribute first, then regex fallback.
    negspaCy attaches _.negex = True on Span objects for negated entities.
    """
    try:
        for ent in sent.ents:
            # Check if the matched phrase overlaps with a negated entity
            if ent.start_char <= sent.start_char + match.start() and \
               ent.end_char >= sent.start_char + match.end():
                if hasattr(ent._, "negex") and ent._.negex:
                    return True
    except Exception:
        pass
    return _is_negated_regex(sent_text)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_symptom_flags(text: Optional[str]) -> Dict[str, int]:
    """
    Main entry point. Given a parent's free-text description, returns a dict
    of 10 binary autism-relevant symptom flags.

    Always safe to call — returns all-zeros if:
      - text is None/empty
      - spaCy dependencies are missing (regex fallback is used instead)
      - any internal error occurs (graceful degradation)

    Negation is handled: "he does NOT avoid eye contact" → eye_contact_absent = 0

    Args:
        text: Parent's free-text description (any length, can be None/empty)

    Returns:
        Dict[str, int] mapping each NLP_FEATURE_NAME to 0 or 1
    """
    if not text or not str(text).strip():
        return _default_flags()

    cleaned = str(text).strip()
    if len(cleaned) < 5:
        return _default_flags()

    try:
        nlp, has_negex = _load_nlp()
        if nlp is None:
            return _extract_regex_fallback(cleaned)
        return _extract_with_spacy(cleaned, nlp, has_negex)
    except Exception as exc:
        logger.error(f"NLP extraction failed: {exc}. Using regex fallback.")
        try:
            return _extract_regex_fallback(cleaned)
        except Exception as exc2:
            logger.error(f"Regex fallback also failed: {exc2}. Returning zeros.")
            return _default_flags()


def get_nlp_feature_names() -> List[str]:
    """Returns the ordered list of 10 NLP feature column names."""
    return list(NLP_FEATURE_NAMES)


def nlp_flags_to_clinical_summary(flags: Dict[str, int]) -> str:
    """
    Converts binary NLP flags to a human-readable clinical summary.
    Used for enriching the LLM report prompt with free-text context.
    """
    LABEL_MAP = {
        "nlp_eye_contact_absent":     "Reduced/absent eye contact",
        "nlp_name_response_absent":   "Limited response to name",
        "nlp_pointing_absent":        "Absent pointing or showing",
        "nlp_echolalia_present":      "Echolalia (word/phrase repetition)",
        "nlp_repetitive_behaviour":   "Repetitive behaviours or routines",
        "nlp_sensory_sensitivity":    "Sensory sensitivities",
        "nlp_social_interest_absent": "Limited social interest",
        "nlp_language_delay":         "Language delay or regression",
        "nlp_gesture_absent":         "Limited gestures (e.g., waving)",
        "nlp_pretend_play_absent":    "Limited pretend or imaginative play",
    }
    detected = [LABEL_MAP[k] for k, v in flags.items() if v == 1]
    if not detected:
        return "No specific symptom signals detected in the free-text description."
    return (
        "Signals detected from parent's free-text description: "
        + "; ".join(detected) + "."
    )
