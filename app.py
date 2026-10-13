from __future__ import annotations

import streamlit as st

from presentation import render_app


ROLE_SKILL_MAP = {
    "Data Analyst": ["SQL", "Python", "Excel", "Power BI", "Statistics"],
    "ML Engineer": ["Python", "Machine Learning", "Scikit-Learn", "MLOps", "Deep Learning"],
    "Frontend Developer": ["JavaScript", "React", "CSS", "HTML", "UI"],
    "Backend Developer": ["Python", "APIs", "Databases", "System Design", "Testing"],
    "Java Developer": ["Java", "Spring Boot", "Microservices", "REST", "SQL"],
    "DevOps Engineer": ["Docker", "Kubernetes", "CI/CD", "Linux", "Cloud"],
    "Cloud Engineer": ["AWS", "Azure", "Terraform", "Networking", "Security"],
    "Data Scientist": ["Python", "Statistics", "Machine Learning", "SQL", "Visualization"],
    "Business Analyst": ["Requirements", "Stakeholder Management", "Excel", "KPIs", "SQL"],
    "Software Engineer": ["Python", "Java", "System Design", "Testing", "Software Engineering"],
}


def calculate_match_score(resume_data: dict, role: str, role_knowledge=None) -> dict:
    required_skills = []
    for item in role_knowledge or []:
        role_section_active = False
        for line in item.get("content", "").splitlines():
            if line.startswith("## "):
                role_section_active = line[3:].strip().casefold() == role.casefold()
            elif role_section_active and "**required skills:**" in line.lower():
                required_skills.extend(
                    skill.strip().rstrip(".")
                    for skill in line.split("**Required skills:**", 1)[1].split(",")
                    if skill.strip()
                )
    required_skills = list(dict.fromkeys(required_skills)) or ROLE_SKILL_MAP.get(role, [])
    skills = {str(skill).strip().lower() for skill in resume_data.get("skills", [])}
    matched_skills = [
        skill for skill in required_skills
        if any(skill.lower() in candidate_skill for candidate_skill in skills)
    ]
    missing_skills = [skill for skill in required_skills if skill not in matched_skills]
    skill_coverage = (
        len(matched_skills) / len(required_skills) * 100
        if required_skills else None
    )
    projects = resume_data.get("projects", [])
    project_evidence = min(len(projects), 3) / 3 * 100
    completeness_fields = [
        bool(resume_data.get("email") and resume_data["email"] != "Not available"),
        bool(resume_data.get("phone") and resume_data["phone"] != "Not available"),
        bool(resume_data.get("skills")),
        bool(projects),
        bool(resume_data.get("education")),
        bool(resume_data.get("certifications")),
        bool(resume_data.get("experience") and resume_data["experience"] != "No experience summary found"),
    ]
    completeness = sum(completeness_fields) / len(completeness_fields) * 100
    business_components = [completeness, project_evidence]
    if skill_coverage is not None:
        business_components.append(skill_coverage)
    business_score = round(sum(business_components) / len(business_components), 1)

    return {
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_match": round(skill_coverage, 1) if skill_coverage is not None else None,
        "role_match_score": round(skill_coverage, 1) if skill_coverage is not None else None,
        "project_evidence": round(project_evidence, 1),
        "resume_completeness": round(completeness, 1),
        "resume_score": business_score,
        "business_rules_score": business_score,
    }


st.set_page_config(
    page_title="VISIONHIRE AI | Recruitment Intelligence",
    page_icon="V",
    layout="wide",
)
render_app(calculate_match_score)
