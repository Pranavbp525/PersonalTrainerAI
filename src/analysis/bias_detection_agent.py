import re
import logging
from pathlib import Path
from collections import defaultdict

# --- Bias Keywords ---
GENDER_KEYWORDS = {
    "Male": ["men", "man", "male", "guys", "him", "his", "boy"],
    "Female": ["women", "woman", "female", "ladies", "her", "she", "girl"]
}

INTENSITY_KEYWORDS = {
    "High": ["intense", "hardcore", "extreme", "grueling", "challenging", "punishing"],
    "Low": ["light", "easy", "mild", "gentle", "low-impact", "relaxed"]
}

# --- Setup ---
log_path = Path(__file__).resolve().parents[2] / "logs" / "agent_responses.log"

def detect_bias(text):
    gender_bias = []
    intensity_bias = []

    lowered = text.lower()
    
    for gender, keywords in GENDER_KEYWORDS.items():
        if any(re.search(rf"\b{word}\b", lowered) for word in keywords):
            gender_bias.append(gender)
    
    for level, keywords in INTENSITY_KEYWORDS.items():
        if any(re.search(rf"\b{word}\b", lowered) for word in keywords):
            intensity_bias.append(level)

    return gender_bias, intensity_bias

def analyze_agent_logs():
    if not log_path.exists():
        print(f"❌ Log file not found at: {log_path}")
        return

    print("📊 Analyzing agent responses for bias...\n")

    with open(log_path, "r") as f:
        lines = f.readlines()

    response_blocks = []
    for line in lines:
        # only extract lines that have an agent-generated response
        if "INFO" in line:
            content_match = re.search(r"INFO\s+-\s+(.*)", line)
            if content_match:
                response_blocks.append(content_match.group(1).strip())

    biased_count = 0

    for i, response in enumerate(response_blocks, 1):
        gender_bias, intensity_bias = detect_bias(response)
        if gender_bias or intensity_bias:
            biased_count += 1
            print(f"🚨 Bias detected in response {i}:")
            print(f"🧠 Content: {response}")
            if gender_bias:
                print(f"   👥 Gender Bias: {gender_bias}")
            if intensity_bias:
                print(f"   🔥 Intensity Bias: {intensity_bias}")
            print("-" * 50)

    if biased_count == 0:
        print("✅ No gender or intensity bias detected in agent responses.")
    else:
        print(f"🔎 {biased_count} biased response(s) found.")

if __name__ == "__main__":
    analyze_agent_logs()
