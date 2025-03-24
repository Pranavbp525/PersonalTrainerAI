import pytest
from datetime import datetime
from pydantic import ValidationError
from src.chatbot.agent.agent_models import (
    UserProfile, UserModel, AgentState, TrainingPrinciples, TrainingApproaches, 
    TrainingApproach, Citation, Citations, AdherenceRate, ProgressMetrics, 
    IssuesList, Adjustment, AdjustmentsList, BasicRoutine, SetExtract, 
    ExerciseExtract, RoutineExtract, SetUpdate, ExerciseUpdate, WorkoutUpdate, 
    WorkoutUpdateRequest, SetRoutineUpdate, ExerciseRoutineUpdate, RoutineUpdate, 
    RoutineUpdateRequest, SetRoutineCreate, ExerciseRoutineCreate, RoutineCreate, 
    RoutineCreateRequest
)

def test_user_profile():
    profile = UserProfile(
        name="John Doe",
        age=30,
        gender="Male",
        goals=["Lose weight", "Build muscle"],
        preferences=["Morning workouts"],
        constraints=["No heavy lifting"],
        fitness_level="Intermediate",
        motivation_factors=["Health", "Appearance"],
        learning_style="Visual",
        confidence_scores={"strength": 0.8},
        available_equipment=["Dumbbells", "Treadmill"],
        training_environment="Home",
        schedule={"Monday": "7am-8am"},
        measurements={"waist": 32.0},
        height=180.0,
        weight=75.0,
        workout_history=["Routine A", "Routine B"]
    )
    assert profile.name == "John Doe"
    assert profile.age == 30
    assert profile.fitness_level == "Intermediate"

def test_user_profile_invalid_age():
    with pytest.raises(ValidationError):
        UserProfile(age="invalid_age")

def test_user_model():
    user_model = UserModel(
        name="Jane Doe",
        age=25,
        gender="Female",
        goals=["Run a marathon"],
        last_updated=datetime.utcnow(),
        model_version=1,
        missing_fields=["height", "weight"]
    )
    assert user_model.name == "Jane Doe"
    assert user_model.age == 25
    assert user_model.goals == ["Run a marathon"]

def test_training_principles():
    principles = TrainingPrinciples(principles=["Progressive Overload", "Specificity"])
    assert "Progressive Overload" in principles.principles

def test_training_approaches():
    approach = TrainingApproach(name="HIIT", description="High-Intensity Interval Training")
    approaches = TrainingApproaches(approaches=[approach])
    assert approaches.approaches[0].name == "HIIT"

def test_citations():
    citation = Citation(source="Journal of Sports Science", content="Strength training improves endurance.")
    citations = Citations(citations=[citation])
    assert citations.citations[0].source == "Journal of Sports Science"

def test_adherence_rate():
    adherence_rate = AdherenceRate(rate=0.85)
    assert adherence_rate.rate == pytest.approx(0.85, 0.01)

def test_routine_create_request():
    set_routine_create = SetRoutineCreate(type="Warmup", weight_kg=0, reps=10)
    exercise_routine_create = ExerciseRoutineCreate(
        exercise_template_id="123",
        rest_seconds=60,
        notes="Warm up exercise",
        sets=[set_routine_create]
    )
    routine_create = RoutineCreate(
        title="Morning Routine",
        notes="A simple morning exercise routine",
        exercises=[exercise_routine_create]
    )
    routine_create_request = RoutineCreateRequest(routine=routine_create)
    assert routine_create_request.routine.title == "Morning Routine"import pytest
from datetime import datetime
from pydantic import ValidationError
from src.chatbot.agent.agent_models import (
    UserProfile, UserModel, AgentState, TrainingPrinciples, TrainingApproaches, 
    TrainingApproach, Citation, Citations, AdherenceRate, ProgressMetrics, 
    IssuesList, Adjustment, AdjustmentsList, BasicRoutine, SetExtract, 
    ExerciseExtract, RoutineExtract, SetUpdate, ExerciseUpdate, WorkoutUpdate, 
    WorkoutUpdateRequest, SetRoutineUpdate, ExerciseRoutineUpdate, RoutineUpdate, 
    RoutineUpdateRequest, SetRoutineCreate, ExerciseRoutineCreate, RoutineCreate, 
    RoutineCreateRequest
)

def test_user_profile():
    profile = UserProfile(
        name="John Doe",
        age=30,
        gender="Male",
        goals=["Lose weight", "Build muscle"],
        preferences=["Morning workouts"],
        constraints=["No heavy lifting"],
        fitness_level="Intermediate",
        motivation_factors=["Health", "Appearance"],
        learning_style="Visual",
        confidence_scores={"strength": 0.8},
        available_equipment=["Dumbbells", "Treadmill"],
        training_environment="Home",
        schedule={"Monday": "7am-8am"},
        measurements={"waist": 32.0},
        height=180.0,
        weight=75.0,
        workout_history=["Routine A", "Routine B"]
    )
    assert profile.name == "John Doe"
    assert profile.age == 30
    assert profile.fitness_level == "Intermediate"

def test_user_profile_invalid_age():
    with pytest.raises(ValidationError):
        UserProfile(age="invalid_age")

def test_user_model():
    user_model = UserModel(
        name="Jane Doe",
        age=25,
        gender="Female",
        goals=["Run a marathon"],
        last_updated=datetime.utcnow(),
        model_version=1,
        missing_fields=["height", "weight"]
    )
    assert user_model.name == "Jane Doe"
    assert user_model.age == 25
    assert user_model.goals == ["Run a marathon"]

def test_training_principles():
    principles = TrainingPrinciples(principles=["Progressive Overload", "Specificity"])
    assert "Progressive Overload" in principles.principles

def test_training_approaches():
    approach = TrainingApproach(name="HIIT", description="High-Intensity Interval Training")
    approaches = TrainingApproaches(approaches=[approach])
    assert approaches.approaches[0].name == "HIIT"

def test_citations():
    citation = Citation(source="Journal of Sports Science", content="Strength training improves endurance.")
    citations = Citations(citations=[citation])
    assert citations.citations[0].source == "Journal of Sports Science"

def test_adherence_rate():
    adherence_rate = AdherenceRate(rate=0.85)
    assert adherence_rate.rate == pytest.approx(0.85, 0.01)

def test_routine_create_request():
    set_routine_create = SetRoutineCreate(type="Warmup", weight_kg=0, reps=10)
    exercise_routine_create = ExerciseRoutineCreate(
        exercise_template_id="123",
        rest_seconds=60,
        notes="Warm up exercise",
        sets=[set_routine_create]
    )
    routine_create = RoutineCreate(
        title="Morning Routine",
        notes="A simple morning exercise routine",
        exercises=[exercise_routine_create]
    )
    routine_create_request = RoutineCreateRequest(routine=routine_create)
    assert routine_create_request.routine.title == "Morning Routine"
