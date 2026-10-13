from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from model_manager import ModelManager


class CandidatePredictor:
    def __init__(self, model_path: str | Path | None = None):
        self.model_manager = ModelManager(model_path)
        self.model_manager.model = self.model_manager.load_model()

    def predict(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        return self.model_manager.predict_candidate(profile)

    @property
    def available_roles(self) -> list[str]:
        if self.model_manager.model is None:
            self.model_manager.ensure_current_model()
        return self.model_manager.available_roles


if __name__ == "__main__":
    profile = {
        "years_experience": 4,
        "resume_score": 87,
        "skills": ["Python", "SQL", "Machine Learning"],
        "projects": ["AI platform"],
        "certifications": ["AWS"],
        "role_match_score": 82,
        "technical_skill_strength": 80,
        "interview_readiness": 88,
        "career_readiness": 76,
        "role_fit": 83,
    }
    print(CandidatePredictor().predict(profile))
