import json
import unicodedata
import logging
import os

#  Configure logging for each file, writing to the same file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs/preprocessing.log"))),  # Logs go into the same file
        logging.StreamHandler()  #  Also print logs to the console
    ]
)

logger = logging.getLogger(__name__)  

def clean_text(text):
    """Normalize text encoding to remove special characters."""
    normalized = unicodedata.normalize("NFKD", text)
    return ''.join(c for c in normalized if not unicodedata.combining(c))

def format_summary(summary):
    """Formats the summary dictionary into a readable text format, excluding 'Workout PDF'."""
    summary_text = []
    for key, value in summary.items():
        if key != "Workout PDF":  #  Exclude "Workout PDF"
            summary_text.append(f"{key}: {clean_text(value)}")
    
    return "\n".join(summary_text)  # Join all summary fields into text

def extract_relevant_workout_data(json_data):
    """Extracts relevant data while preserving source, title, and URL from each workout entry."""
    logger.info("Starting workout data extraction")  # Log start of extraction
    
    workouts = json_data.get("workouts", [])  # Extract workouts list
    
    extracted_data = []
    
    for index, workout in enumerate(workouts):
        try:
            # Extract required fields
            source = clean_text(workout.get("source", "Unknown Source"))
            title = clean_text(workout.get("title", "No Title"))
            url = workout.get("url", "No URL")

            # Extract summary (without "Workout PDF")
            summary_data = workout.get("summary", {})
            formatted_summary = format_summary(summary_data)

            # Extract and combine workout + exercise descriptions
            workout_description = clean_text(workout.get("description", ""))
            exercises = workout.get("exercises", [])

            # Combine summary + description
            combined_description = f"{formatted_summary}\n\n{workout_description}"
            
            for exercise in exercises:
                exercise_description = clean_text(exercise.get("description", ""))
                combined_description += f"\n\n{exercise_description}"
            
            # Append structured workout data
            extracted_data.append({
                "source": source,
                "title": title,
                "url": url,
                "description": combined_description
            })

            logger.info(f"Processed workout {index + 1}/{len(workouts)}: {title}")  # Log each processed workout

        except Exception as e:
            logger.error(f"Error processing workout {index + 1}: {str(e)}", exc_info=True)

    logger.info("Workout data extraction completed")  # Log completion
    return extracted_data


def ms_preprocessing():

    logger.info("Reading input file: workouts.json")  # Log file reading start

    try:
        input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw_json_data/ms_data.json"))
        with open(input_file, "r", encoding="utf-8") as file:
            json_data = json.load(file)
        logger.info("Successfully read workouts.json")  # Log success
    except Exception as e:
        logger.error(f"Failed to read workouts.json: {str(e)}", exc_info=True)
        exit(1)  # Exit if file cannot be read

    processed_workouts = extract_relevant_workout_data(json_data)

    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/preprocessed_json_data/ms_data.json"))
    try:
        # Make sure the directory exists
        directory = os.path.dirname(output_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_workouts, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully saved processed data to {output_file}")  # Log save success
    except Exception as e:
        logger.error(f"Failed to save {output_file}: {str(e)}", exc_info=True)
