# Personal Trainer AI

The PersonalTrainerAI project aims to revolutionize personal fitness training by leveraging artificial intelligence. This project will develop an AI-powered personal trainer that provides customized workout plans, real-time feedback, and progress tracking to users. By utilizing advanced AI Agent systems, the AI trainer will adapt to individual fitness levels and goals, ensuring a personalized and effective training experience.

The primary objective of this project is to make personal training accessible and affordable for everyone. With the AI trainer, users can receive professional guidance and support without the need for expensive gym memberships or personal trainers. This solution is designed to promote a healthier lifestyle and help users achieve their fitness goals efficiently.

Our project includes developing a comprehensive Machine Learning Operations (MLOps) pipeline, encompassing data collection, preprocessing, model training, and deployment. The AI trainer will be available as a user-friendly mobile application, allowing users to conveniently access their personalized workout plans and track their progress anytime, anywhere.

## Documentation

For more details on specific parts of the project, refer to the following documentation:

- [Agent Documentation](./readme/agent.md): Details on the different levels of the fitness trainer agent.
- [Data Pipeline Setup](./readme/data_pipeline.md): Instructions on setting up and using the data pipeline.
- [Deployment Guide](./readme/deployment.md): Steps to deploy the system via Docker and CI/CD pipelines.
- [Model Development](./readme/model_development.md): Information about model design, training, and evaluation.


## 📂 Project Directory Structure

```
.
├── .dvc/
│   ├── .gitignore
│   └── config
│
├── .github/
│   └── workflows/
│       ├── data_pipeline_to_gcp.yml
│       └── python-tests.yml
│
├── alembic/
│   ├── README
│   ├── env.py
│   ├── script.py.mako
│
├── dags/
│   └── data_pipeline_airflow.py
│
├── data/
│   ├── preprocessed_json_data/
│   │   ├── .gitignore
│   │   ├── blogs.json.dvc
│   │   ├── ms_data.json.dvc
│   │   └── pdf_data.json.dvc
│   └── raw_json_data/
│       ├── .gitignore
│       ├── blogs.json.dvc
│       ├── ms_data.json.dvc
│       └── pdf_data.json.dvc
│
├── experiments/
│   └── pranav/
│       └── agent/
│           ├── assets/
│           ├── README.md
│           ├── __init__.py
│           ├── agent.ipynb
│           ├── basic_agent.py
│           ├── cognitive_agent.py
│           ├── multi_agent.py
│           ├── new_agent_architecture.py
│           ├── orchestrator_worker_agent.py
│           └── stage_based_agent.py
│
├── logs/
│   ├── preprocessing.log
│   ├── scraper.log
│   └── vectordb.log
│
├── logstash/
│   ├── config/
│   │   ├── logstash.yml
│   │   └── pipelines.yml
│   └── pipeline/
│       ├── fitness-chatbot.conf
│       └── minimal.conf
│
├── result/
│   ├── advanced_evaluation_results.json
│   ├── fitness_domain_metrics_comparison.png
│   ├── human_evaluation_metrics_comparison.png
│   ├── overall_comparison.png
│   ├── ragas_metrics_comparison.png
│   ├── response_time_comparison.png
│   └── retrieval_metrics_comparison.png
│
├── results/
│   ├── evaluation_results.json
│   ├── metrics_comparison.png
│   └── response_time_comparison.png
│
├── src/
│   ├── chatbot/
│   │   └── agent/
│   │       ├── __init__.py
│   │       ├── agent_models.py
│   │       ├── graph.py
│   │       ├── hevy_api.py
│   │       ├── hevy_exercises.json
│   │       ├── llm_tools.py
│   │       ├── personal_trainer_agent.py
│   │       ├── prompts.py
│   │       ├── test_api.py
│   │       └── utils.py
│   ├── agent_eval/
│   │   └── eval.py
│   ├── alembic/
│   │   └── versions/
│   ├── README.md
│   ├── __init__.py
│   ├── alembic.ini
│   ├── chat_client.py
│   ├── config.py
│   ├── elk_logging.py
│   ├── experiments.ipynb
│   ├── hevy_exercises.json
│   ├── main.py
│   ├── models.py
│   └── redis_utils.py
│
├── data_pipeline/
│   ├── __init__.py
│   ├── blogs.py
│   ├── ms.py
│   ├── ms_preprocess.py
│   ├── other_preprocesing.py
│   ├── pdfs.py
│   └── vector_db.py
│
├── other/
│   ├── __init__.py
│   └── bias_detection.py
│
├── rag_model/
│   ├── .DS_Store
│   └── __init__.py
│
├── tests/
│   ├── advanced_rag_evaluation_test.py
│   ├── advanced_rag_test.py
│   ├── modular_rag_test.py
│   ├── raptor_rag_test.py
│   ├── test_ms.py
│   ├── test_ms_preprocess.py
│   ├── test_other_preprocessing.py
│   ├── test_pdf_scraper.py
│   └── test_vectdb.py
│
├── .dockerignore
├── .dvcignore
├── .env.example
├── .env.local
├── .gitignore
├── 2.8.0
├── Dockerfile
├── Dockerfile.frontend
├── Dockerfile.test
├── ELK-INTEGRATION.md
├── README.md
├── Scoping.md
├── alembic.ini
├── docker-compose-elk.yml
├── docker-compose.chatbot.yml
├── docker-compose.local.yml
├── docker-compose.yaml
├── image.png
├── kibana-dashboard-setup.md
└── requirements.frontend.txt
```

