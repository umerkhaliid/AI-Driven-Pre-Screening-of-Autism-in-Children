from src.inference import predict_autism_risk
from src.llm_report_groq import generate_risk_report


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
report = generate_risk_report(result)

print("\n" + "=" * 80)
print(report)
print("=" * 80)
