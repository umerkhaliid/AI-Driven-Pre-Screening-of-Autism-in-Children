from src.inference import predict_autism_risk

sample_payload = {
    "age_mons": 28,
    "gender": "male",
    "jaundice": "no",
    "family_mem_with_asd": "yes",
    "qchat_answers": {
        1: "A", 2: "C", 3: "B", 4: "D", 5: "C",
        6: "A", 7: "E", 8: "B", 9: "C", 10: "B"
    },
    "mchat_answers": {
        11: "Yes", 12: "No", 13: "Yes", 14: "No", 15: "Yes",
        16: "Yes", 17: "Yes", 18: "No", 19: "Yes", 20: "Yes",
        21: "Yes", 22: "Yes", 23: "Yes", 24: "Yes"
    }
}

result = predict_autism_risk(sample_payload)

print("Screening Score:", result["screening_score"], "/", result["screening_score_max"])
print("Score Risk Level:", result["score_risk_level"])
print("Referral:", result["referral_interpretation"])
print()
print("Default Prediction:", result["prediction_default"]["predicted_label"])
print("Screening Prediction:", result["prediction_screening"]["predicted_label"])
print("Thresholds Used:", result["prediction_screening"]["thresholds_used"])
print()
print("Class Probabilities:", result["class_probabilities"])
