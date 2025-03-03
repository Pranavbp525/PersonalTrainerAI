import pandas as pd
import re
import json
import logging
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# **Step 1: Load JSON Data**
def load_workout_data(json_file_path):
    """Loads workout data from a JSON file into a Pandas DataFrame."""
    try:
        with open(json_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        df = pd.DataFrame(data)
        logger.info(f"Successfully loaded {len(df)} workout records from {json_file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")
        return None

# **Step 2: Detect Gender Bias in Workouts**
def detect_gender_bias(df):
    """Analyzes gender-related mentions in workout descriptions."""
    def detect_gender(text):
        if re.search(r"\b(men|male|him|he|boy)\b", text, re.IGNORECASE):
            return "Male"
        elif re.search(r"\b(women|female|her|she|girl)\b", text, re.IGNORECASE):
            return "Female"
        else:
            return "Unisex"

    df["Gender_Slice"] = df["description"].apply(detect_gender)
    gender_counts = df["Gender_Slice"].value_counts()
    logger.info(f" Gender-Based Workout Distribution:\n{gender_counts}")
    return df

# 🔹 **Step 3: Analyze Workout Intensity Bias**
def detect_workout_intensity(df):
    """Categorizes workouts into 'High', 'Medium', or 'Low' intensity based on descriptions."""
    def detect_intensity(text):
        if re.search(r"\b(advanced|heavy|high intensity|elite)\b", text, re.IGNORECASE):
            return "High"
        elif re.search(r"\b(intermediate|moderate|balanced)\b", text, re.IGNORECASE):
            return "Medium"
        else:
            return "Low"

    df["Workout_Intensity"] = df["description"].apply(detect_intensity)
    intensity_distribution = df.groupby(["Gender_Slice", "Workout_Intensity"]).size().unstack()
    logger.info(f" Workout Intensity Distribution by Gender:\n{intensity_distribution}")
    return df

# **Step 4: Measure Bias in Chatbot Recommendations**
def evaluate_bias(df):
    """Uses Fairlearn to check for accuracy differences between groups."""
    # Simulated predictions (1 = recommended workout, 0 = not recommended)
    df["true_label"] = 1  # Assume workouts should be recommended
    df["predicted_label"] = df["Workout_Intensity"].apply(lambda x: 1 if x == "High" else 0)

    groups = df["Gender_Slice"]
    metric_frame = MetricFrame(
        metrics=accuracy_score,
        y_true=df["true_label"],
        y_pred=df["predicted_label"],
        sensitive_features=groups
    )

    logger.info(f" Bias Evaluation Metrics:\n{metric_frame.by_group}")

# **Step 5: Run Bias Detection (No Data is Saved)**
def run_bias_detection_pipeline(json_path):
    """Runs the bias detection process without modifying data."""
    df = load_workout_data(json_path)
    if df is None:
        return

    df = detect_gender_bias(df)
    df = detect_workout_intensity(df)
    evaluate_bias(df)

# Run the Pipeline (No Data is Modified)
if __name__ == "__main__":
    input_json = "data/preprocessed_json_data/pdf_data.json"
    run_bias_detection_pipeline(input_json)
    input_json = "data/preprocessed_json_data/ms_data.json"
    run_bias_detection_pipeline(input_json)
    input_json = "data/preprocessed_json_data/blogs.json"
    run_bias_detection_pipeline(input_json)