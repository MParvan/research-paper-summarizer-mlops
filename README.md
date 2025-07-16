# 🧠 Research Paper Summarizer with LLMs and MLOps

A production-grade NLP system that summarizes research papers using state-of-the-art LLMs. Includes a full MLOps pipeline and interactive demo.

## 🔧 Features
- Upload full PDF or arXiv link
- Summarization with BART/T5/SciBERT
- Streamlit frontend for easy interaction
- CI/CD with GitHub Actions
- Model tracking and versioning via MLflow
- Dockerized REST API with FastAPI
- Cloud-ready (GCP / AWS / Hugging Face Spaces)

## 🚀 Tech Stack
- Python, PyTorch, HuggingFace Transformers
- FastAPI, Streamlit, MLflow, Docker, Airflow
- GCP/AWS (optional for deployment)

## 📁 Project Structure

```
research-paper-summarizer-mlops/
├── data/                # Raw and processed datasets
│   ├── raw/
│   └── processed/
├── notebooks/           # Exploration and visualization notebooks
│   └── data_preparation.ipynb
├── src/
│   ├── data/            # Preprocessing & dataset scripts
│   ├── model/           # Model training, inference
│   ├── pipeline/        # ML pipeline & orchestration logic
│   └── api/             # FastAPI app for REST endpoints
├── docker/              # Dockerfiles for services
├── scripts/             # CLI tools and automation scripts
├── tests/               # Unit/integration tests
├── .github/             # CI/CD workflows
├── streamlit_app/       # Frontend UI
├── requirements.txt
├── docker-compose.yml
├── README.md
└── pyproject.toml
```


## 📊 Dataset


## 🛠️ Getting Started


## 🤖 Inference API


## 🧪 Tests


## 🧠 Authors & License
Milad Parvan — [GitHub](https://github.com/MParvan) | [LinkedIn](https://www.linkedin.com/in/milad-parvan-6ba485221/)
MIT License
