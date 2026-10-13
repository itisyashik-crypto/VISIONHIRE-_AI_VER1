from __future__ import annotations

from typing import Any, Dict, List

from config import AppConfig
from gemini_client import GeminiClient
from vector_store import VectorStoreManager


class RAGEngine:
    def __init__(self, knowledge_base_dir: str | None = None):
        self.vector_store = VectorStoreManager(knowledge_base_dir or AppConfig.KNOWLEDGE_BASE_PATH)
        self.gemini = GeminiClient()

    def retrieve_knowledge(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        return self.vector_store.search(query, k=k)

    def generate_candidate_insights(
        self,
        resume_info: Dict[str, Any],
        role: str,
        dataset_prediction: Dict[str, Any] | None = None,
        role_predictions: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        skills = resume_info.get("skills", [])
        query = " ".join([
            f"{role} job requirements essential skills interview topics recruiter expectations",
            "resume ATS project experience scoring candidate evaluation framework",
            "career growth learning roadmap certifications skill gap recommendations",
            "candidate skills: " + ", ".join(skills),
            "candidate projects: " + " ".join(resume_info.get("projects", [])),
        ])
        context = self.retrieve_knowledge(query, k=5)
        career_context = self.retrieve_knowledge(
            f"career growth learning roadmap certifications skill gaps role transition to {role}",
            k=2,
        )
        context_keys = {(item.get("source"), item.get("content")) for item in context}
        context.extend(
            item for item in career_context
            if (item.get("source"), item.get("content")) not in context_keys
        )
        gaps = self._infer_missing_skills(
            role,
            skills,
            resume_info.get("required_skills"),
        )
        advice = "\n".join(
            f"[{item.get('source', 'knowledge base')}] {item['content']}"
            for item in context
        )
        rag_confidence = "high" if len(context) >= 3 else "moderate" if context else "unavailable"
        resume_evidence_fields = (
            "skills", "projects", "experience", "experience_years", "certifications",
            "education", "achievements", "technologies", "required_skills", "matched_skills",
            "missing_skills", "skill_match", "project_evidence", "resume_completeness",
            "business_rules_score", "resume_score", "role_match_score",
        )
        resume_evidence = {
            field: resume_info[field]
            for field in resume_evidence_fields
            if field in resume_info
        }

        prompt = f"""You are a fair, evidence-based recruitment and career intelligence assistant.
Assess only evidence present in the resume fields. Do not infer protected traits or invent qualifications.
Return only valid JSON with these keys: gemini_score (number 0-100 or null), rag_score (number 0-100 or null),
interview_readiness_score (number 0-100 or null), resume_feedback (string), strengths (array of strings),
weaknesses (array of strings), improvement_areas (array of strings), recommendations (array of strings),
hiring_risk_analysis (string), readiness_level (string), career_recommendations (array of strings),
learning_roadmap (array of strings), suggested_certifications (array of strings), recruiter_summary (string),
hiring_recommendation (string), recommended_role (string or null).

Score the candidate evidence for the target role. Explain readiness based on demonstrated skills,
Use null for any score that cannot be supported by the available evidence. Assess role fit, resume quality,
and interview readiness separately. `rag_score` is your assessment against the retrieved frameworks, not
retrieval similarity. If no documents were retrieved, set rag_score to null. Do not let the historical
dataset probability determine the recommendation; discuss disagreements and uncertainty. Keep a human
recruiter responsible for the final decision.

Target role: {role}
Resume evidence (contact details and raw document text excluded): {resume_evidence}
Dataset prediction for selected role: {dataset_prediction or 'Unavailable'}
Dataset probabilities for represented roles: {role_predictions or 'Unavailable'}
Business-rule evidence: role skill coverage={resume_info.get('skill_match')}; matched skills={resume_info.get('matched_skills')}; missing skills={gaps}; resume completeness={resume_info.get('resume_completeness')}; project evidence={resume_info.get('project_evidence')}; business score={resume_info.get('business_rules_score')}.
Retrieved knowledge ({rag_confidence} coverage): {advice or 'No knowledge-base documents were retrieved.'}
"""
        try:
            import json

            response = self.gemini.generate_text(prompt)
            if response.startswith("```") and response.endswith("```"):
                response = response[3:-3].strip()
                if response.lower().startswith("json"):
                    response = response[4:].strip()
            generated = json.loads(response)
            required_fields = {
                "gemini_score", "interview_readiness_score", "resume_feedback",
                "strengths", "weaknesses", "improvement_areas", "recommendations",
                "hiring_risk_analysis", "readiness_level", "career_recommendations",
                "rag_score", "learning_roadmap", "suggested_certifications",
                "recruiter_summary", "hiring_recommendation", "recommended_role",
            }
            if required_fields.issubset(generated):
                score_fields = ["gemini_score", "interview_readiness_score", "rag_score"]
                valid_scores = all(
                    generated.get(field) is None
                    or (
                        isinstance(generated[field], (int, float))
                        and not isinstance(generated[field], bool)
                        and 0 <= generated[field] <= 100
                    )
                    for field in score_fields
                )
                if valid_scores and (context or generated["rag_score"] is None):
                    generated["rag_score"] = generated.get("rag_score") if context else None
                    generated["knowledge_context"] = advice
                    generated["knowledge_sources"] = [item.get("source") for item in context]
                    generated["retrieval_method"] = sorted({item.get("retrieval_method", "unknown") for item in context})
                    generated["rag_confidence"] = rag_confidence
                    generated["ai_generated"] = True
                    return generated
        except Exception:
            pass

        strengths = []
        if skills:
            strengths.append(f"Resume lists {len(skills)} extracted skills, including {', '.join(skills[:3])}.")
        projects = resume_info.get("projects", [])
        if projects:
            strengths.append(f"Resume contains {len(projects)} extracted project evidence item(s).")
        years = resume_info.get("experience_years")
        if years is not None:
            strengths.append(f"Resume explicitly states {years:g} years of experience.")
        if resume_info.get("certifications"):
            strengths.append(f"Resume lists {len(resume_info['certifications'])} extracted certification(s).")

        return {
            "gemini_score": None,
            "interview_readiness_score": None,
            "rag_score": None,
            "resume_score": resume_info.get("resume_score"),
            "role_match_score": resume_info.get("role_match_score"),
            "resume_feedback": "Gemini feedback is unavailable; showing only extracted resume evidence.",
            "strengths": strengths,
            "weaknesses": [f"Resume does not show the extracted role requirement: {skill}." for skill in gaps],
            "improvement_areas": gaps,
            "recommendations": [f"Add verifiable evidence of {skill} if you have that experience." for skill in gaps],
            "career_recommendations": [],
            "learning_roadmap": [],
            "suggested_certifications": [],
            "recruiter_summary": "Automated assessment was unavailable. A recruiter should review the extracted evidence.",
            "hiring_recommendation": "Recruiter review required; no automated recommendation is available.",
            "recommended_role": None,
            "hiring_risk_analysis": "Unavailable without Gemini analysis; review the extracted evidence and source documents.",
            "readiness_level": "Unavailable without Gemini analysis",
            "knowledge_context": advice,
            "knowledge_sources": [item.get("source") for item in context],
            "retrieval_method": sorted({item.get("retrieval_method", "unknown") for item in context}),
            "rag_confidence": rag_confidence,
            "ai_generated": False,
        }

    def _infer_missing_skills(
        self,
        role: str,
        skills: List[str],
        required_skills: List[str] | None = None,
    ) -> List[str]:
        role_map = {
            "Data Analyst": ["SQL", "Power BI", "Excel", "Statistics"],
            "ML Engineer": ["Model deployment", "Feature engineering", "MLOps", "Deep learning"],
            "Data Scientist": ["Statistics", "Machine learning", "Experiment design", "Visualization"],
            "Frontend Developer": ["React", "TypeScript", "CSS", "Performance optimization"],
            "Backend Developer": ["APIs", "Databases", "System design", "Testing"],
            "Java Developer": ["Spring Boot", "Microservices", "Concurrency", "REST"],
            "DevOps Engineer": ["Docker", "Kubernetes", "CI/CD", "Infrastructure as code"],
            "Cloud Engineer": ["AWS", "Azure", "Networking", "Security"],
            "Business Analyst": ["Requirements", "Stakeholder management", "KPIs", "Process mapping"],
            "Software Engineer": ["System design", "Testing", "Distributed systems", "Debugging"],
        }
        required = required_skills or role_map.get(role, ["Core engineering fundamentals"])
        lower_skills = {skill.lower() for skill in skills}
        missing = [
            skill for skill in required
            if not any(skill.lower() in candidate_skill for candidate_skill in lower_skills)
        ]
        return missing[:5]
