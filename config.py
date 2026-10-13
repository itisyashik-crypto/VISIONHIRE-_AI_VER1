import os
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")


class AppConfig:
    PROJECT_NAME = "VISIONHIRE AI"
    TAGLINE = "AI Resume Screening & Interview Intelligence Platform"
    SUBTITLE = "Next-Generation Talent Intelligence System"
    FOOTER = "Created by: Yashik Kalra"

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("Gemini_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL") or os.getenv("Gemini_MODEL", "gemini-1.5-flash")
    DEFAULT_ROLE = "Software Engineer"
    JOB_ROLES = [
        "Data Analyst",
        "ML Engineer",
        "Frontend Developer",
        "Backend Developer",
        "Java Developer",
        "DevOps Engineer",
        "Cloud Engineer",
        "Data Scientist",
        "Business Analyst",
        "Software Engineer",
    ]

    UPLOADED_DATASET_PATH = ROOT_DIR / "dataset" / "resume_screening_dataset_500rows.csv"
    DATASET_PATH = UPLOADED_DATASET_PATH
    MODEL_PATH = ROOT_DIR / "models" / "candidate_model.joblib"
    VECTOR_DB_PATH = ROOT_DIR / "vector_db"
    KNOWLEDGE_BASE_PATH = ROOT_DIR / "knowledge_base"
    ASSETS_PATH = ROOT_DIR / "assets"


def validate_startup() -> Dict[str, Any]:
    status = {
        "api_loaded": bool(AppConfig.GOOGLE_API_KEY),
        "gemini_connected": False,
        "model_active": False,
        "knowledge_base_loaded": False,
        "faiss_active": False,
        "embeddings_active": False,
        "dataset_loaded": AppConfig.DATASET_PATH.exists(),
    }

    try:
        from model_manager import ModelManager

        status["model_active"] = ModelManager().ensure_current_model()
    except Exception:
        status["model_active"] = False

    try:
        from gemini_client import GeminiClient

        status["gemini_connected"] = GeminiClient().check_connection()
    except Exception:
        status["gemini_connected"] = False

    try:
        from vector_store import VectorStoreManager

        vector_store = VectorStoreManager()
        status["faiss_active"] = vector_store.vector_store is not None
        status["embeddings_active"] = vector_store.embeddings is not None
        status["knowledge_base_loaded"] = bool(vector_store.documents)
    except Exception:
        status["faiss_active"] = False
        status["embeddings_active"] = False
        status["knowledge_base_loaded"] = False

    return status


CONFIG = AppConfig
