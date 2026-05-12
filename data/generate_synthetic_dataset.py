"""
Synthetic Dataset Generator for Autism Pre-Screening Tool (Extended)
====================================================================
Combines Q-CHAT-10 (10 questions) + M-CHAT-R (20 questions) into a unified
24-question dataset after removing 6 duplicate questions.

Target: 4-class risk level (No Risk, Mild Risk, Moderate Risk, Severe Risk)
Size: 5,000 samples
"""

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

# -- Total samples & class distribution ---------------------------------------
N = 5000
CLASS_DIST = {
    0: 0.35,   # No Risk       -- 1750
    1: 0.30,   # Mild Risk     -- 1500
    2: 0.20,   # Moderate Risk -- 1000
    3: 0.15,   # Severe Risk   --  750
}
CLASS_LABELS = {0: "No Risk", 1: "Mild Risk", 2: "Moderate Risk", 3: "Severe Risk"}

# -- 24 Unique Questions -------------------------------------------------------
# Source: Q-CHAT-10 (a1-a10) + 14 unique M-CHAT-R items (a11-a24)
# Binary encoding: 1 = at-risk response, 0 = typical response
#
# Duplicates removed (Q-CHAT kept, M-CHAT dropped):
#   Q-CHAT Q1 ~ M-CHAT Q10 (name response)
#   Q-CHAT Q2 ~ M-CHAT Q14 (eye contact)
#   Q-CHAT Q3 ~ M-CHAT Q6  (pointing to request)
#   Q-CHAT Q4 ~ M-CHAT Q7  (pointing to share)
#   Q-CHAT Q5 ~ M-CHAT Q3  (pretend play)
#   Q-CHAT Q6 ~ M-CHAT Q16 (gaze following)

QUESTIONS = {
    # -- Q-CHAT-10 (a1-a10) ---------------------------------------------------
    "a1":  "Does your child look at you when you call his/her name?",
    "a2":  "How easy is it for you to get eye contact with your child?",
    "a3":  "Does your child point to indicate that s/he wants something?",
    "a4":  "Does your child point to share interest with you?",
    "a5":  "Does your child pretend? (e.g. care for dolls, talk on toy phone)",
    "a6":  "Does your child follow where you're looking?",
    "a7":  "Does your child comfort upset family members?",
    "a8":  "Would you describe your child's first words as typical?",
    "a9":  "Does your child use simple gestures? (e.g. wave goodbye)",
    "a10": "Does your child stare at nothing with no apparent purpose?",  # REVERSE

    # -- M-CHAT-R unique items (a11-a24) --------------------------------------
    "a11": "If you point at something across the room, does your child look at it?",
    "a12": "Have you ever wondered if your child might be deaf?",                # REVERSE
    "a13": "Does your child like climbing on things?",
    "a14": "Does your child make unusual finger movements near his/her eyes?",   # REVERSE
    "a15": "Is your child interested in other children?",
    "a16": "Does your child show you things by bringing them to you?",
    "a17": "When you smile at your child, does he/she smile back at you?",
    "a18": "Does your child get upset by everyday noises?",                      # REVERSE
    "a19": "Does your child walk?",
    "a20": "Does your child try to copy what you do?",
    "a21": "Does your child try to get you to watch him/her?",
    "a22": "Does your child understand when you tell him/her to do something?",
    "a23": "Does your child look at your face to see how you feel about something new?",
    "a24": "Does your child like movement activities?",
}

QUESTION_COLS = [f"a{i}" for i in range(1, 25)]

# -- NLP flag columns — 10 synthetic flags augmenting the feature vector ------
# Each flag represents a free-text symptom signal extracted by the NLP pipeline.
# Probabilities: [P(flag=1|NoRisk), P(flag=1|Mild), P(flag=1|Moderate), P(flag=1|Severe)]
#
# Logic: Higher severity -> parent more likely to describe these in free text.

NLP_FEATURE_NAMES = [
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

NLP_ITEM_PROBS = {
    #                                    NoRisk  Mild   Moderate  Severe
    "nlp_eye_contact_absent":           [0.03,  0.20,  0.55,     0.85],
    "nlp_name_response_absent":         [0.02,  0.18,  0.50,     0.82],
    "nlp_pointing_absent":              [0.03,  0.15,  0.48,     0.80],
    "nlp_echolalia_present":            [0.02,  0.12,  0.38,     0.70],
    "nlp_repetitive_behaviour":         [0.03,  0.14,  0.42,     0.75],
    "nlp_sensory_sensitivity":          [0.04,  0.16,  0.40,     0.68],
    "nlp_social_interest_absent":       [0.03,  0.17,  0.50,     0.80],
    "nlp_language_delay":               [0.02,  0.22,  0.52,     0.84],
    "nlp_gesture_absent":               [0.03,  0.16,  0.46,     0.78],
    "nlp_pretend_play_absent":          [0.03,  0.15,  0.44,     0.76],
}

NLP_COLS = NLP_FEATURE_NAMES  # alias for clarity

# -- Per-question at-risk probability by class --------------------------------
# Each question has [P(1|NoRisk), P(1|Mild), P(1|Moderate), P(1|Severe)]

ITEM_PROBS = {
    #                          NoRisk  Mild   Moderate  Severe
    # --- Q-CHAT-10 items ---
    "a1":  [0.04, 0.22, 0.55, 0.85],   # Name response
    "a2":  [0.05, 0.25, 0.58, 0.88],   # Eye contact
    "a3":  [0.06, 0.20, 0.50, 0.82],   # Pointing to request
    "a4":  [0.05, 0.25, 0.55, 0.86],   # Pointing to share
    "a5":  [0.06, 0.22, 0.52, 0.80],   # Pretend play
    "a6":  [0.05, 0.23, 0.53, 0.84],   # Gaze following
    "a7":  [0.07, 0.20, 0.48, 0.78],   # Comfort others
    "a8":  [0.05, 0.18, 0.45, 0.76],   # First words
    "a9":  [0.04, 0.19, 0.50, 0.82],   # Simple gestures
    "a10": [0.06, 0.20, 0.48, 0.80],   # Stare at nothing (REVERSE)

    # --- M-CHAT-R unique items ---
    "a11": [0.04, 0.18, 0.48, 0.82],   # Follow pointing
    "a12": [0.03, 0.15, 0.40, 0.70],   # Wonder if deaf (REVERSE)
    "a13": [0.08, 0.14, 0.28, 0.45],   # Climbing (motor)
    "a14": [0.03, 0.16, 0.42, 0.72],   # Unusual finger movements (REVERSE)
    "a15": [0.05, 0.22, 0.52, 0.83],   # Interest in children
    "a16": [0.05, 0.20, 0.50, 0.82],   # Show things
    "a17": [0.03, 0.15, 0.45, 0.80],   # Smile back
    "a18": [0.07, 0.18, 0.40, 0.65],   # Upset by noises (REVERSE)
    "a19": [0.08, 0.12, 0.22, 0.38],   # Walking (motor)
    "a20": [0.05, 0.20, 0.50, 0.80],   # Copy/imitate
    "a21": [0.05, 0.22, 0.52, 0.82],   # Get you to watch
    "a22": [0.04, 0.18, 0.48, 0.78],   # Understand instructions
    "a23": [0.05, 0.20, 0.50, 0.80],   # Social referencing
    "a24": [0.08, 0.14, 0.25, 0.42],   # Movement activities (motor)
}

# -- Demographic distributions by class ---------------------------------------
DEMO_CONFIG = {
    # age_mons: (mean, std) per class
    "age_mons": {
        0: (32, 10),
        1: (28, 10),
        2: (26, 10),
        3: (24, 10),
    },
    # gender: P(male=1) per class — ASD ~4:1 male bias
    "gender_male_prob": {
        0: 0.50,
        1: 0.58,
        2: 0.65,
        3: 0.72,
    },
    # jaundice: P(yes=1) per class — slight elevation in ASD
    "jaundice_prob": {
        0: 0.12,
        1: 0.16,
        2: 0.20,
        3: 0.25,
    },
    # family_mem_with_asd: P(yes=1) per class — strong genetic component
    "family_asd_prob": {
        0: 0.05,
        1: 0.12,
        2: 0.22,
        3: 0.35,
    },
    # ethnicity (same across classes for fairness)
    "ethnicity": [
        "White European", "Middle Eastern", "South Asian", "Asian",
        "Black", "Hispanic", "Native Indian", "Others", "Mixed",
    ],
    "ethnicity_probs": [0.22, 0.18, 0.14, 0.12, 0.10, 0.10, 0.04, 0.05, 0.05],
}


def generate_dataset(n=N, seed=SEED):
    """Generate the full synthetic dataset with 37 features (24 Q + 3 demo + 10 NLP)."""
    rng = np.random.default_rng(seed)

    # -- Step 1: Assign classes -----------------------------------------------
    class_sizes = {c: int(n * p) for c, p in CLASS_DIST.items()}
    # Adjust rounding to hit exact N
    diff = n - sum(class_sizes.values())
    class_sizes[0] += diff

    classes = np.concatenate([np.full(sz, c) for c, sz in class_sizes.items()])
    rng.shuffle(classes)

    # -- Step 2: Generate question responses ----------------------------------
    data = {"risk_class": classes}

    for q in QUESTION_COLS:
        probs = np.array(ITEM_PROBS[q])
        sample_probs = probs[classes]
        noise = rng.normal(0, 0.03, size=n)
        sample_probs = np.clip(sample_probs + noise, 0.01, 0.99)
        data[q] = rng.binomial(1, sample_probs)

    # -- Step 3: Inject inter-question correlations ---------------------------
    correlation_groups = [
        ["a1", "a2", "a4", "a6", "a11", "a17"],   # Joint attention
        ["a3", "a8", "a9", "a22"],                  # Communication
        ["a7", "a15", "a16", "a21", "a23"],         # Social interest
        ["a10", "a12", "a14", "a18"],               # Repetitive/sensory
        ["a13", "a19", "a24"],                      # Motor
    ]

    for group in correlation_groups:
        group_vals = np.column_stack([data[q] for q in group])
        group_mean = group_vals.mean(axis=1)

        for q in group:
            flip_mask = rng.random(n) < 0.30
            group_majority = (group_mean > 0.5).astype(int)
            data[q] = np.where(flip_mask, group_majority, data[q])

    # -- Step 4: Generate demographics ----------------------------------------
    # Age (clipped to 12-60 months)
    age = np.zeros(n, dtype=int)
    for c in range(4):
        mask = classes == c
        mu, sigma = DEMO_CONFIG["age_mons"][c]
        age[mask] = rng.normal(mu, sigma, size=mask.sum()).astype(int)
    data["age_mons"] = np.clip(age, 12, 60)

    # Gender
    gender_prob = np.array([DEMO_CONFIG["gender_male_prob"][c] for c in classes])
    data["gender"] = rng.binomial(1, gender_prob)

    # Jaundice
    jaundice_prob = np.array([DEMO_CONFIG["jaundice_prob"][c] for c in classes])
    data["jaundice"] = rng.binomial(1, jaundice_prob)

    # Family member with ASD
    fam_prob = np.array([DEMO_CONFIG["family_asd_prob"][c] for c in classes])
    data["family_mem_with_asd"] = rng.binomial(1, fam_prob)

    # Ethnicity
    ethnicity = rng.choice(
        DEMO_CONFIG["ethnicity"],
        size=n,
        p=DEMO_CONFIG["ethnicity_probs"],
    )
    data["ethnicity"] = ethnicity

    # -- Step 5: Generate synthetic NLP flag columns --------------------------
    for nlp_col in NLP_COLS:
        probs_by_class = np.array(NLP_ITEM_PROBS[nlp_col])
        sample_probs = probs_by_class[classes]
        noise = rng.normal(0, 0.02, size=n)
        sample_probs = np.clip(sample_probs + noise, 0.01, 0.99)
        data[nlp_col] = rng.binomial(1, sample_probs)

    # -- Step 6: Compute derived scores ---------------------------------------
    df = pd.DataFrame(data)
    df["screening_score"] = df[QUESTION_COLS].sum(axis=1)

    # -- Step 7: Add controlled label noise (5%) for realism ------------------
    noise_mask = rng.random(n) < 0.05
    noisy_classes = df["risk_class"].values.copy()

    for i in np.where(noise_mask)[0]:
        current = noisy_classes[i]
        if current == 0:
            noisy_classes[i] = 1
        elif current == 3:
            noisy_classes[i] = 2
        else:
            noisy_classes[i] = current + rng.choice([-1, 1])

    df["risk_class"] = noisy_classes

    # -- Step 8: Reorder columns ----------------------------------------------
    col_order = (
        QUESTION_COLS
        + ["age_mons", "gender", "ethnicity", "jaundice", "family_mem_with_asd"]
        + NLP_COLS
        + ["screening_score", "risk_class"]
    )
    # Safety guard: only keep columns that actually exist
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    return df


def validate_dataset(df: pd.DataFrame):
    """Print a comprehensive validation report for the generated dataset."""
    print("SYNTHETIC DATASET VALIDATION REPORT")
    print("=" * 70)

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    # Class distribution
    print("\n--- Class Distribution ---")
    dist = df["risk_class"].value_counts().sort_index()
    for cls, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {cls} ({CLASS_LABELS[cls]:>15s}): {count:>5d} ({pct:5.1f}%)")

    # Score statistics by class
    print("\n--- Screening Score by Class ---")
    for cls in range(4):
        subset = df[df["risk_class"] == cls]["screening_score"]
        print(f"  {CLASS_LABELS[cls]:>15s}: "
              f"mean={subset.mean():.1f}, std={subset.std():.1f}, "
              f"min={subset.min()}, max={subset.max()}")

    # Demographics by class
    print("\n--- Demographics by Class ---")
    for cls in range(4):
        subset = df[df["risk_class"] == cls]
        print(f"  {CLASS_LABELS[cls]:>15s}: "
              f"age={subset['age_mons'].mean():.1f} +/- {subset['age_mons'].std():.1f}, "
              f"male={subset['gender'].mean():.2f}, "
              f"jaundice={subset['jaundice'].mean():.2f}, "
              f"fam_asd={subset['family_mem_with_asd'].mean():.2f}")

    # Per-question at-risk rate by class
    print("\n--- At-Risk Response Rate per Question ---")
    print(f"  {'Q':>4s}  {'NoRisk':>8s}  {'Mild':>8s}  {'Moderate':>8s}  {'Severe':>8s}")
    for q in QUESTION_COLS:
        rates = []
        for cls in range(4):
            rate = df[df["risk_class"] == cls][q].mean()
            rates.append(f"{rate:.2f}")
        print(f"  {q:>4s}  {'  '.join(f'{r:>8s}' for r in rates)}")

    # NLP flag rates by class
    print("\n--- NLP Flag Rate by Class ---")
    print(f"  {'Flag':>35s}  {'NoRisk':>8s}  {'Mild':>8s}  {'Moderate':>8s}  {'Severe':>8s}")
    for flag in NLP_COLS:
        if flag in df.columns:
            rates = []
            for cls in range(4):
                rate = df[df["risk_class"] == cls][flag].mean()
                rates.append(f"{rate:.2f}")
            print(f"  {flag:>35s}  {'  '.join(f'{r:>8s}' for r in rates)}")

    # Missing values
    print(f"\n--- Missing Values: {df.isnull().sum().sum()} ---")

    # Data types
    print("\n--- Data Types ---")
    for col in df.columns:
        print(f"  {col:>25s}: {df[col].dtype}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print("Generating synthetic autism pre-screening dataset...")
    df = generate_dataset()
    validate_dataset(df)

    # Save
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "raw", "synthetic_autism_screening_5000.csv")
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    print(f"Final shape: {df.shape}")
