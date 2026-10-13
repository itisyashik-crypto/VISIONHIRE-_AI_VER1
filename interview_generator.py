from __future__ import annotations

import json
from typing import Dict, List, Any

from gemini_client import GeminiClient

ROLE_KNOWLEDGE = {
    "Software Engineer": ["system design", "data structures", "api design", "scalability", "testing"],
    "Data Scientist": ["statistics", "machine learning", "experimentation", "visualization", "model evaluation"],
    "ML Engineer": ["feature engineering", "model deployment", "mLOps", "model validation", "monitoring"],
    "Data Analyst": ["sql", "dashboarding", "business reporting", "data cleaning", "insights"],
    "Frontend Developer": ["html", "css", "javascript", "ui architecture", "performance"],
    "Backend Developer": ["api design", "database modeling", "scalability", "security", "testing"],
    "Java Developer": ["oop", "spring boot", "microservices", "rest apis", "concurrency"],
    "DevOps Engineer": ["ci/cd", "docker", "kubernetes", "infrastructure", "monitoring"],
    "Cloud Engineer": ["cloud architecture", "networking", "security", "containers", "cost optimization"],
    "Business Analyst": ["requirements", "stakeholder management", "process mapping", "kpis", "analysis"],
}


def generate_interview_questions(
    role: str,
    skills: List[str],
    experience: str,
    projects: List[str],
    knowledge_context: str = "",
) -> List[Dict[str, Any]]:
    base_skills = skills or ["core technical skills"]
    role_topics = ROLE_KNOWLEDGE.get(role, ["role fundamentals", "execution strategy"])

    prompt = f"""Create 6 personalized interview questions for the target role.
Return only a JSON array. Each object must have type, difficulty, and question keys.
Use a mix of Technical, Behavioral, Scenario-Based, Project-Based, and System Design questions;
vary difficulty between Beginner, Intermediate, and Advanced. Base questions on the candidate evidence,
do not invent experience, and use retrieved hiring guidance where available.

Role: {role}
Skills: {skills}
Experience: {experience}
Projects: {projects}
Retrieved guidance: {knowledge_context or 'No knowledge-base documents retrieved.'}
Role topics: {role_topics}
"""
    try:
        response = GeminiClient().generate_text(prompt)
        if response.startswith("```") and response.endswith("```"):
            response = response[3:-3].strip()
            if response.lower().startswith("json"):
                response = response[4:].strip()
        questions = json.loads(response)
        valid_questions = [
            {
                "type": item["type"],
                "difficulty": item["difficulty"],
                "question": item["question"],
                "source": "Gemini",
            }
            for item in questions
            if isinstance(item, dict)
            and all(item.get(field) for field in ("type", "difficulty", "question"))
        ]
        if valid_questions:
            return valid_questions
    except Exception:
        pass

    questions: List[Dict[str, Any]] = []

    questions.append({
        "type": "Technical",
        "difficulty": "Intermediate",
        "question": f"Explain how you would design and optimize a solution for {role.lower()} responsibilities using {', '.join(base_skills[:3])}.",
        "source": "Rules-based fallback",
    })
    questions.append({
        "type": "Behavioral",
        "difficulty": "Advanced",
        "question": f"Describe a complex challenge from {experience or 'your project experience'} and how you prioritized trade-offs in a delivery-driven environment.",
        "source": "Rules-based fallback",
    })
    questions.append({
        "type": "Scenario-Based",
        "difficulty": "Advanced",
        "question": f"A production issue impacts user experience in a live system. Walk through your diagnostic and remediation plan for a {role.lower()} team.",
        "source": "Rules-based fallback",
    })
    questions.append({
        "type": "Project-Based",
        "difficulty": "Intermediate",
        "question": f"Review one of your recent projects: {projects[0] if projects else 'your most impactful project'}, and explain the architecture, trade-offs, and measurable outcomes.",
        "source": "Rules-based fallback",
    })
    questions.append({
        "type": "System Design",
        "difficulty": "Advanced",
        "question": f"Design a scalable system for a high-growth {role.lower()} use case leveraging {', '.join(role_topics[:3])} and strong operational reliability.",
        "source": "Rules-based fallback",
    })
    questions.append({
        "type": "Technical",
        "difficulty": "Beginner",
        "question": f"What are the core principles of {role.lower()} that you would use to debug a problem in a production environment?",
        "source": "Rules-based fallback",
    })

    return questions
