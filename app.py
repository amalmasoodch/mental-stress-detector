"""
MindSense AI - Mental Stress Detector (Client Interface Runner)
Supports 7-class mental state classification using TF-IDF (1,2 N-Grams)
and Logistic Regression defined in model.py.
"""

MENTAL_HEALTH_CLASSES = [
    "Anxiety",
    "Bipolar",
    "Depression",
    "Normal",
    "Personality disorder",
    "Stress",
    "Suicidal",
]

GUIDANCE_MESSAGES = {
    "Normal": "Balanced state detected. Continue maintaining healthy daily routines.",
    "Stress": "Elevated stress indicators detected. Consider structured breaks and workload pacing.",
    "Anxiety": "Anxiety signals detected. Grounding exercises and calm breathing routines may help.",
    "Depression": "Low mood / depressive signals detected. Reaching out to loved ones or a professional can help.",
    "Bipolar": "Mood fluctuation indicators detected. Consider structured tracking and clinical consultation.",
    "Personality disorder": "Interpersonal distress patterns detected. Professional psychological support is recommended.",
    "Suicidal": "Critical distress signals detected. Please contact a mental health professional or crisis helpline immediately.",
}


def format_diagnostic_report(statement: str, predicted_class: str, confidence: float) -> str:
    guidance = GUIDANCE_MESSAGES.get(predicted_class, "Consult a qualified mental health specialist for guidance.")
    lines = [
        "=" * 65,
        "MindSense AI - Diagnostic Assessment Report",
        "=" * 65,
        f"Input Statement : {statement}",
        f"Detected State  : {predicted_class}",
        f"Confidence      : {confidence:.1f}%",
        f"Guidance        : {guidance}",
        "=" * 65,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sample_text = "Everything feels heavy and exhausting lately, I have zero motivation to get out of bed."
    print(format_diagnostic_report(sample_text, "Depression", 60.8))
    print("System ready. Connect your trained vectorizer and classifier from model.py for live inference.")
