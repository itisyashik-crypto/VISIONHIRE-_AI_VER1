from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from config import AppConfig


FEATURE_COLUMNS = [
    "skills_text",
    "experience_years",
    "projects_count",
    "certifications_count",
    "target_role",
]
NUMERIC_FEATURES = ["experience_years", "projects_count", "certifications_count"]
ALIASES = {
    "skills_text": ("skills", "skills_text"),
    "experience_years": ("experience_years", "years_experience", "experience"),
    "projects_count": ("projects_count", "project_count"),
    "certifications_count": ("certifications_count", "certification_count"),
    "target_role": ("target_role", "job_role", "role"),
    "selected": ("selected", "suitability_label", "selection_label", "is_selected"),
}


class ModelManager:
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path or AppConfig.MODEL_PATH)
        self.model = None
        self.metadata: Dict[str, Any] = {}

    @staticmethod
    def _column_lookup(frame: pd.DataFrame) -> Dict[str, str]:
        return {str(column).strip().lower(): column for column in frame.columns}

    @classmethod
    def _canonicalize_dataset(cls, frame: pd.DataFrame) -> pd.DataFrame:
        lookup = cls._column_lookup(frame)
        canonical = pd.DataFrame(index=frame.index)
        for target, aliases in ALIASES.items():
            source = next((lookup[name] for name in aliases if name in lookup), None)
            if source is not None:
                canonical[target] = frame[source]
        required = set(FEATURE_COLUMNS + ["selected"])
        missing = required - set(canonical.columns)
        if missing:
            raise ValueError(f"Dataset is missing required fields: {', '.join(sorted(missing))}")

        canonical["skills_text"] = canonical["skills_text"].fillna("").astype(str)
        canonical["target_role"] = canonical["target_role"].fillna("").astype(str).str.strip()
        for column in NUMERIC_FEATURES:
            canonical[column] = pd.to_numeric(canonical[column], errors="coerce")
        labels = canonical["selected"]
        if labels.dtype == object:
            normalized = labels.astype(str).str.strip().str.lower()
            labels = normalized.map({
                "1": 1, "true": 1, "yes": 1, "selected": 1,
                "0": 0, "false": 0, "no": 0, "rejected": 0,
            })
        canonical["selected"] = pd.to_numeric(labels, errors="coerce")
        canonical = canonical.dropna(subset=["selected"])
        canonical["selected"] = canonical["selected"].astype(int)
        return canonical

    def train_model(self, csv_path: str | Path | None = None) -> Dict[str, Any]:
        dataset_path = Path(csv_path or AppConfig.DATASET_PATH)
        if not dataset_path.exists():
            return {"status": "dataset_missing", "message": "No candidate dataset was found; Gemini and business rules remain available."}

        try:
            frame = self._canonicalize_dataset(pd.read_csv(dataset_path))
        except (ValueError, pd.errors.ParserError) as exc:
            return {"status": "invalid_dataset", "message": str(exc)}
        if frame.empty or frame["selected"].nunique() < 2 or frame["selected"].value_counts().min() < 2:
            return {"status": "insufficient_labels", "message": "Training requires at least two examples for each selection outcome."}

        preprocessor = ColumnTransformer([
            ("skills", TfidfVectorizer(ngram_range=(1, 2), max_features=1500), "skills_text"),
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]), NUMERIC_FEATURES),
            ("role", OneHotEncoder(handle_unknown="ignore"), ["target_role"]),
        ])
        pipeline = Pipeline([
            ("features", preprocessor),
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
        ])
        X = frame[FEATURE_COLUMNS]
        y = frame["selected"]
        roles = frame["target_role"]
        X_train, X_test, y_train, y_test, roles_train, roles_test = train_test_split(
            X, y, roles, test_size=0.2, random_state=42, stratify=y
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        role_preprocessor = ColumnTransformer([
            ("skills", TfidfVectorizer(ngram_range=(1, 2), max_features=1500), "skills_text"),
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]), NUMERIC_FEATURES),
        ])
        role_pipeline = Pipeline([
            ("features", role_preprocessor),
            ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
        ])
        role_features_train = X_train.drop(columns=["target_role"])
        role_features_test = X_test.drop(columns=["target_role"])
        role_pipeline.fit(role_features_train, roles_train)
        role_predictions = role_pipeline.predict(role_features_test)
        training_vectors = pipeline.named_steps["features"].transform(X_train)
        neighbor_count = min(5, len(X_train))
        neighbor_index = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine")
        neighbor_index.fit(training_vectors)
        metrics = {
            "status": "trained",
            "rows_used": int(len(frame)),
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
            "role_match_accuracy": round(float(accuracy_score(roles_test, role_predictions)), 4),
            "message": f"Selection model trained from {len(frame)} labeled rows in {dataset_path.name}.",
        }
        self.metadata = {
            "pipeline": pipeline,
            "role_pipeline": role_pipeline,
            "neighbor_index": neighbor_index,
            "training_selected": y_train.astype(int).tolist(),
            "training_roles": roles_train.astype(str).tolist(),
            "roles": sorted(frame["target_role"].dropna().unique().tolist()),
            "dataset_path": str(dataset_path.resolve()),
            "dataset_mtime_ns": dataset_path.stat().st_mtime_ns,
            "metrics": metrics,
        }
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.metadata, self.model_path)
        self.model = pipeline
        return metrics

    def load_model(self):
        if not self.model_path.exists():
            return None
        try:
            payload = joblib.load(self.model_path)
            if not isinstance(payload, dict) or "pipeline" not in payload:
                return None
            self.metadata = payload
            self.model = payload["pipeline"]
            return self.model
        except Exception:
            self.model = None
            self.metadata = {}
            return None

    def ensure_current_model(self) -> bool:
        dataset_path = Path(AppConfig.DATASET_PATH)
        if not dataset_path.exists():
            self.model = None
            self.metadata = {}
            return False
        if self.model is None:
            self.load_model()
        is_current = (
            self.model is not None
            and self.metadata.get("dataset_path") == str(dataset_path.resolve())
            and self.metadata.get("dataset_mtime_ns") == dataset_path.stat().st_mtime_ns
        )
        if not is_current:
            return self.train_model(dataset_path).get("status") == "trained"
        return True

    @property
    def available_roles(self) -> list[str]:
        return list(self.metadata.get("roles", []))

    def predict_candidate(self, profile: Dict[str, Any]) -> Dict[str, Any] | None:
        if not self.ensure_current_model():
            return None
        features = {
            "skills_text": " ".join(profile.get("skills", [])),
            "experience_years": profile.get("experience_years"),
            "projects_count": profile.get("projects_count", len(profile.get("projects", []))),
            "certifications_count": profile.get("certifications_count", len(profile.get("certifications", []))),
            "target_role": profile.get("target_role", ""),
        }
        frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        classifier = self.model.named_steps["classifier"]
        positive_index = list(classifier.classes_).index(1)
        probability = float(self.model.predict_proba(frame)[0][positive_index])
        role_pipeline = self.metadata.get("role_pipeline")
        role_probabilities: Dict[str, float] = {}
        if role_pipeline is not None:
            role_model = role_pipeline.named_steps["classifier"]
            role_features = frame.drop(columns=["target_role"])
            role_probabilities = {
                str(role): float(score)
                for role, score in zip(
                    role_model.classes_,
                    role_pipeline.predict_proba(role_features)[0],
                )
            }

        candidate_similarity = None
        neighbor_index = self.metadata.get("neighbor_index")
        if neighbor_index is not None:
            query_vector = self.model.named_steps["features"].transform(frame)
            distances, indices = neighbor_index.kneighbors(query_vector)
            similar_profiles = []
            training_labels = self.metadata["training_selected"]
            training_roles = self.metadata["training_roles"]
            for distance, index in zip(distances[0], indices[0]):
                similar_profiles.append({
                    "similarity": round(max(0.0, 1.0 - float(distance)), 4),
                    "role": training_roles[index],
                    "selected": bool(training_labels[index]),
                })
            candidate_similarity = {
                "nearest_profiles": similar_profiles,
                "historical_selected_rate": round(
                    sum(profile["selected"] for profile in similar_profiles) / len(similar_profiles),
                    4,
                ) if similar_profiles else None,
                "reference_count": len(similar_profiles),
            }

        target_role = features["target_role"]
        return {
            "selection_probability": probability,
            "predicted_selected": bool(probability >= 0.5),
            "role_match_probability": role_probabilities.get(target_role),
            "recommended_role": max(role_probabilities, key=role_probabilities.get) if role_probabilities else None,
            "role_probabilities": role_probabilities,
            "candidate_similarity": candidate_similarity,
            "target_role": target_role,
            "model_dataset": Path(self.metadata["dataset_path"]).name,
        }
