from __future__ import annotations

from typing import Dict, List, Any


class CareerCoach:
    def __init__(self):
        self.learning_paths = {
            "Data Analyst": ["SQL", "Tableau", "Statistics", "A/B Testing"],
            "ML Engineer": ["ML systems", "MLOps", "Model deployment", "Feature engineering"],
            "Data Scientist": ["Advanced statistics", "Experimentation", "Model monitoring", "Deep learning"],
            "Frontend Developer": ["React", "TypeScript", "UI architecture", "Performance optimization"],
            "Backend Developer": ["API design", "Databases", "System design", "Testing"],
            "Java Developer": ["Spring Boot", "Microservices", "Kafka", "Cloud integration"],
            "DevOps Engineer": ["Docker", "Kubernetes", "CI/CD", "Observability"],
            "Cloud Engineer": ["AWS", "Azure", "Networking", "Security"],
            "Business Analyst": ["Requirements modeling", "Stakeholder management", "Metrics", "Agile"],
            "Software Engineer": ["System design", "Distributed systems", "Security", "Leadership"],
        }

    def generate_guidance(self, role: str, candidate_score: float, skills: List[str], gaps: List[str]) -> Dict[str, Any]:
        path = self.learning_paths.get(role, ["Systems thinking", "Communication", "Execution strategy"])
        recommendations = [
            f"Target high-impact projects that demonstrate mastery in {', '.join(skills[:3]) or 'your strongest technical areas'}.",
            f"Close the skill gap by improving {', '.join(gaps[:3]) or 'core role fundamentals'} through structured practice and hands-on building.",
            "Create a visible portfolio with measurable outcomes and business impact to strengthen employer confidence.",
        ]
        certs = {
            "Data Analyst": ["Tableau Desktop Specialist", "Microsoft PL-300"],
            "ML Engineer": ["AWS ML Specialty", "Google ML Engineer"],
            "Data Scientist": ["Databricks Data Scientist", "TensorFlow Developer"],
            "Frontend Developer": ["Meta Front-End Developer", "Google UX Design"],
            "Backend Developer": ["AWS Developer Associate", "Spring Professional"],
            "Java Developer": ["Oracle Java SE", "Spring Professional"],
            "DevOps Engineer": ["AWS DevOps Engineer", "CKA"],
            "Cloud Engineer": ["AWS Solutions Architect", "Azure Administrator"],
            "Business Analyst": ["CBAP", "IIBA ECBA"],
            "Software Engineer": ["AWS Developer Associate", "Google Cloud Professional"],
        }

        return {
            "role": role,
            "readiness": "Interview Ready" if candidate_score >= 90 else "Almost Ready" if candidate_score >= 70 else "Needs Preparation",
            "learning_path": path,
            "certifications": certs.get(role, ["Industry-recognized professional certifications"]),
            "career_direction": f"Advance toward a senior {role.lower()} track with cross-functional leadership and product ownership responsibilities.",
            "recommendations": recommendations,
        }
