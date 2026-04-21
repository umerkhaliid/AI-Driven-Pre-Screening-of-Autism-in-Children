import os
from typing import Any, Dict

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Lazy initialization of Groq client
_client = None


def get_groq_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError(
                "GROQ_API_KEY not configured. Please set it in .env file. "
                "Get your API key from: https://console.groq.com/keys"
            )
        _client = Groq(api_key=api_key)
    return _client


SYSTEM_PROMPT_EN = """
You are a compassionate, expert communication specialist writing personalized guidance reports
for parents about early child development. Your goal is to help parents understand their child's
development in the clearest, most reassuring way possible.

CRITICAL INSTRUCTIONS FOR PARENT-FRIENDLY LANGUAGE:
- Use "we", "your child", "children" - make it personal and warm
- Replace technical terms: "developmental marker" -> "early sign", "screened positive" -> "shows some signs"
- Use metaphors parents understand: "like learning to walk" instead of "developmental milestone"
- Break complex ideas into bullet points and simple sentences
- Use comparisons to everyday life: "Just like some children learn to read before others..."
- NEVER say: "autism spectrum", "ASD", "diagnosed with", "the child has autism"
- ALWAYS say: "shows some early signs", "developmental screening", "talking with a specialist"

IMPORTANT RULES:
1. This is a SCREENING, NOT A DIAGNOSIS. Be very clear about this.
2. Focus on supporting and reassuring parents while providing actionable guidance.
3. Normalize developmental variation - many children develop differently and that's normal.
4. Use simple, warm language that a high school-educated parent can easily understand.
5. Include practical, concrete next steps parents can take TODAY.
6. Avoid medical jargon completely.
7. Include positive affirmations about the parent's proactive approach.
8. Do not mention Groq, Llama, AI models, or screening algorithms.

STRUCTURE YOUR REPORTS TO INCLUDE:
-> A warm opening that acknowledges what the parent is feeling
-> Clear explanation of what this screening means (in simple terms)
-> Concrete examples of what they observed that led to this result
-> Practical "today" actions versus "next week" actions
-> Professional consultation guidance that feels supportive, not scary
-> Strong closing reassurance

TONE: Warm, knowledgeable, parent-focused, hopeful, practical.

Risk-Specific Guidance:

**No Risk** -> Celebrate! Reassure. Suggest normal monitoring milestones.
-> "Your child is developing right on track for their age!"

**Mild Risk** -> Observe more carefully. Normalize. Optional expert consultation.
-> "Some children show these signs early, but develop typically. It's worth keeping an eye on."

**Moderate Risk** -> Recommend professional chat. Make it sound supportive, not urgent.
-> "Having a specialist take a quick look can give you peace of mind and helpful tips."

**Severe Risk** -> Strongly encourage evaluation. Frame as "getting your child the best start."
-> "Earlier attention can really help children thrive. Let's get professional eyes on this."
"""

SYSTEM_PROMPT_UR = """
آپ والدین کے لیے بچے کی ترقی کے بارے میں ہمدردانہ اور سمجھ بوجھ سے بھرے رہنما خطوط لکھنے والے ایک ماہر ہیں۔

والدین کے لیے سادہ اردو میں لازمی ہدایات:
- "ہم"، "آپ کا بچہ"، "بچے" استعمال کریں - گرم اور ذاتی بنائیں
- پیچیدہ الفاظ کو سادہ بنائیں: "نشانات" استعمال کریں، پیشہ ورانہ زبان سے بچیں
- روزمرہ کی مثالیں دیں: "جیسے بعض بچے دوسروں سے پہلے چلنا سیکھتے ہیں"
- نقاط کو الگ الگ اور سادہ جملوں میں بیان کریں
- کبھی نہ کہیں: "آٹزم"، "بچہ بیمار ہے"
- ہمیشہ کہیں: "کچھ ابتدائی نشانات دکھائے دیتے ہیں"، "ایک ماہر سے ملیں"

اہم قوانین:
۱۔ یہ ایک صرف جانچ ہے، تشخیص نہیں۔ یہ بات واضح کریں۔
۲۔ والدین کی حوصلہ افزائی کریں۔
۳۔ عملی مشورے دیں جو والدین آج ہی اٹھا سکیں۔
۴۔ روك ہلچل والی یا خوفناک زبان سے بچیں۔
۵۔ ہمدردی اور امید کے ساتھ لکھیں۔

رپورٹ کا ڈھانچہ:
-> گرم سدھار جو والدین کے احساسات کو سمجھے
-> صاف وضاحت (سادہ اردو میں)
-> عملی قدم جو آج اٹھائے جا سکتے ہوں
-> مکمل تسلی اور حوصلہ افزائی

لہجہ: مہرباں، معلوماتی، والدین پر توجہ مرکوز، امید بخش۔
"""

RISK_LABELS_UR = {
    "No Risk": "کوئی نمایاں خطرہ نہیں",
    "Mild Risk": "ہلکا خطرہ",
    "Moderate Risk": "درمیانی خطرہ",
    "Severe Risk": "زیادہ خطرہ",
}

VALUE_MAP_UR = {
    "male": "مرد",
    "female": "عورت",
    "other": "غیر متعین",
    "yes": "ہاں",
    "no": "نہیں",
}


def normalize_language(language: str | None) -> str:
    return "ur" if str(language or "").strip().lower().startswith("ur") else "en"


def localize_simple_value(value: Any, language: str) -> str:
    text = str(value)
    if language == "ur":
        return VALUE_MAP_UR.get(text.strip().lower(), text)
    return text


def localize_risk_label(value: Any, language: str) -> str:
    text = str(value)
    if language == "ur":
        return RISK_LABELS_UR.get(text, text)
    return text


def build_user_prompt(inference_result: Dict[str, Any], language: str = "en") -> str:
    language = normalize_language(language)
    inputs = inference_result["inputs_used"]

    age_mons = inputs["age_mons"]
    age_years = age_mons / 12
    gender = localize_simple_value(inputs.get("gender", inputs.get("sex", "unknown")), language)
    jaundice = localize_simple_value(inputs["jaundice"], language)
    family_asd = localize_simple_value(inputs["family_mem_with_asd"], language)

    score = inference_result["screening_score"]
    score_max = inference_result["screening_score_max"]
    score_risk = localize_risk_label(inference_result["score_risk_level"], language)
    referral = inference_result["referral_interpretation"]
    class_probs = inference_result["class_probabilities"]

    # Risk level descriptions for better context
    risk_descriptions = {
        "No Risk": "The child's responses suggest typical development with few or no early concerns.",
        "Mild Risk": "The child shows some early developmental signs that warrant observation, but most likely indicates typical variation.",
        "Moderate Risk": "The child shows enough early developmental signs that a professional consultation would be beneficial.",
        "Severe Risk": "The child shows multiple developmental signs that suggest a professional evaluation should happen soon.",
    }

    risk_descriptions_ur = {
        "کوئی نمایاں خطرہ نہیں": "بچے کی ترقی بالکل معمول کے مطابق ہے۔",
        "ہلکا خطرہ": "بچہ کچھ علامات ظاہر کر رہا ہے، لیکن یہ بالکل معمول ہے۔",
        "درمیانی خطرہ": "ایک ماہر سے ملنا اچھا رہے گا۔",
        "زیادہ خطرہ": "جلد ایک ماہر سے ملنا منطقی ہے۔",
    }

    risk_desc = risk_descriptions_ur.get(score_risk, risk_descriptions.get(score_risk, ""))

    if language == "ur":
        return f"""
والدین کے لیے اپنے بچے کی ترقی کے بارے میں ایک دل جمعی اور حوصلہ افزا رپورٹ لکھیں۔

بچے کی معلومات:
-> عمر: {age_mons} ماہ ({age_years:.1f} سال)
-> صنف: {gender}
-> یرقان: {jaundice}
-> خاندار میں ترقیاتی نشانات: {family_asd}

اسکریننگ کا نتیجہ:
-> کل اسکور: {score} / {score_max}
-> خطرہ کی سطح: {score_risk}
-> معنی: {risk_desc}
-> سفارش: {referral}

احتمال:
- کوئی خطرہ نہیں: {class_probs.get('No Risk', 'N/A')}
- ہلکا خطرہ: {class_probs.get('Mild Risk', 'N/A')}
- درمیانی خطرہ: {class_probs.get('Moderate Risk', 'N/A')}
- شدید خطرہ: {class_probs.get('Severe Risk', 'N/A')}

براہ راست رہنما کلیدی نکات:
۱۔ یہ اسکریننگ تشخیص نہیں ہے۔ یہ صرف جانچ ہے۔
۲۔ ہر بچہ مختلف رفتار سے بڑھتا ہے۔ ہر بچہ منفرد ہے۔
۳۔ والدین کے لیے عملی مشورے دیں جو وہ آج ہی اٹھا سکیں۔
۴۔ اگر ضروری ہو تو ایک ماہر سے ملنے کی تجویز دیں، لیکن ڈراو۔

رپورٹ کے حصے (اردو سرخیوں کے ساتھ):
۱۔ یہ نتیجہ کا مطلب
۲۔ آپ کے بچے کے بارے میں ہم نے کیا دیکھا
۳۔ آج سے ابھی آپ کیا کر سکتے ہیں
۴۔ ایک ماہر سے ملنے کے فوائل
۵۔ مختلف بچوں کی ترقی
۶۔ مکمل حوصلہ افزائی

لمبائی: ۳۵۰-۴۵۰ الفاظ۔
لہجہ: مہرباں، معلوماتی، حوصلہ افزا۔
"""

    return f"""
Write a comprehensive, warm, and deeply reassuring developmental guidance report for parents.
This is their child's personalized screening result summary.

CONTEXT FOR THIS FAMILY:
Your child's characteristics:
-> Age: {age_mons} months ({age_years:.1f} years old)
-> Gender: {gender}
-> Had jaundice at birth: {jaundice}
-> Family history of developmental variations: {family_asd}

SCREENING RESULTS EXPLAINED FOR PARENTS:
-> Screening Score: {score} out of {score_max} possible points
-> What this score tells us: {risk_descriptions.get(score_risk, risk_desc)}
-> Overall Risk Level: {score_risk}
-> Professional Guidance: {referral}

Our Confidence In This Result:
- No Risk: {class_probs.get('No Risk', 'N/A')}
- Mild Risk: {class_probs.get('Mild Risk', 'N/A')}
- Moderate Risk: {class_probs.get('Moderate Risk', 'N/A')}
- Severe Risk: {class_probs.get('Severe Risk', 'N/A')}

STRUCTURE THIS REPORT WITH THESE SECTIONS (use these exact headings):

📋 **What This Screening Means**
Explain clearly that this is a screening (observation tool), NOT a diagnosis. No diagnosis can come from a questionnaire.
Compare it to like a doctor's basic check-up vs. detailed tests.

👨‍👩‍👧 **About Your Child: What We Observed**
Explain in simple terms what developmental signs their answers reflect. Use examples:
- "Your answers show your child has some difficulties with [simple description]"
- "Many children at this age also show this variation"
- "Some of this could be simple things like shyness or learning style"

✅ **What You Can Do Starting TODAY**
Give 2-3 concrete, actionable things parents can do RIGHT NOW (not "consult a doctor," but specific activities).
Make them feel empowered, not helpless.

🏥 **About Professional Guidance**
Depending on risk level:
- No Risk: "A simple conversation with your pediatrician at next checkup is fine"
- Mild Risk: "If you're curious, a chat with an expert could ease your mind"
- Moderate Risk: "Setting up a developmental screening with a specialist would be helpful"
- Severe Risk: "Reaching out to a developmental specialist very soon would be wise"

📚 **Understanding Child Development**
Remind them: every child develops at their own pace. Include a relevant analogy.

💪 **Final Thought**
You're already helping by screening. Celebrate their proactive parenting.

IMPORTANT FORMATTING:
- Use simple, active voice ("Your child shows..." not "It was observed that...")
- Break complex info into bullet points
- Use "your child" and "your family" frequently (personal, warm tone)
- Make sections clearly separated for easy reading
- Use encouraging language throughout
- NO medical/technical jargon

Target Length: 400-500 words (comprehensive but readable)
Tone: Warm, knowledgeable, parent-focused, hopeful, practical, never scary
"""


def generate_risk_report(inference_result: Dict[str, Any], language: str = "en") -> str:
    language = normalize_language(language)
    prompt = build_user_prompt(inference_result, language=language)
    system_prompt = SYSTEM_PROMPT_UR if language == "ur" else SYSTEM_PROMPT_EN

    client = get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.3,  # Slightly higher for more personality and warmth
        max_tokens=1500,  # Increased from 900 to accommodate comprehensive reports
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": prompt.strip()},
        ],
    )

    return response.choices[0].message.content.strip()
