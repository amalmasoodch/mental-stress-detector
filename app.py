"""
MindSense AI - Mental Stress Detector (Client Runner)
Connects user inputs with your trained logic in model.py
"""

def display_banner():
    print("=" * 55)
    print("   MindSense AI | Mental Stress Detector v1.0")
    print("=" * 55)

def classify_stress_score(score: float) -> str:
    if score <= 35:
        return "Low Stress (Optimal)"
    elif score <= 70:
        return "Moderate Stress (Monitor)"
    return "High Stress (Action Recommended)"

if __name__ == "__main__":
    display_banner()
    print("System ready. Import your prediction function from model.py to run live inference.")
