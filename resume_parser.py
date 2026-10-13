import re
from pathlib import Path
from typing import Dict, List, Any

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Read a PDF resume and return plain text."""
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        return ""

    try:
        reader = PdfReader(str(pdf_file))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n".join(pages)
    except Exception:
        return ""


def extract_email(text: str) -> str:
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else "Not available"


def extract_phone(text: str) -> str:
    match = re.search(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)\d{3}[-.\s]?\d{4}", text)
    return match.group(0).strip() if match else "Not available"


def extract_name(text: str) -> str:
    first_line = text.strip().splitlines()[0].strip() if text.strip() else "Candidate"
    if len(first_line) < 80:
        return first_line or "Candidate"
    return "Candidate"


def extract_skills(text: str) -> List[str]:
    skill_keywords = [
        "python", "sql", "pandas", "numpy", "scikit-learn", "machine learning",
        "deep learning", "tensorflow", "pytorch", "java", "javascript", "react",
        "nodejs", "aws", "azure", "gcp", "docker", "kubernetes", "tableau",
        "power bi", "excel", "spark", "hadoop", "etl", "airflow", "sql", "nlp",
        "opencv", "computer vision", "streamlit", "flask", "fastapi", "c++",
        "c#", "mongodb", "postgresql", "redis", "dbt", "git", "ci/cd", "devops",
        "cloud", "data analysis", "statistics", "business analysis", "agile"
    ]
    found = []
    lower_text = text.lower()
    for skill in skill_keywords:
        if skill in lower_text:
            found.append(skill.title())
    return sorted(set(found))


def extract_projects(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    projects = []
    for line in lines:
        if any(keyword in line.lower() for keyword in ["project", "implemented", "built", "developed", "dashboard", "automation", "platform"]):
            projects.append(line)
    return projects[:5]


def extract_certifications(text: str) -> List[str]:
    certs = []
    patterns = ["aws certified", "azure", "google cloud", "tableau", "sql", "pmp", "scrum", "databricks"]
    lower_text = text.lower()
    for pattern in patterns:
        if pattern in lower_text:
            certs.append(pattern.title())
    return certs


def extract_education(text: str) -> List[str]:
    education = []
    lines = text.splitlines()
    for line in lines:
        if any(keyword in line.lower() for keyword in ["bachelor", "master", "phd", "degree", "engineering", "computer science", "mba"]):
            education.append(line.strip())
    return education[:5]


def extract_experience(text: str) -> str:
    lines = text.splitlines()
    experiences = []
    for line in lines:
        if any(keyword in line.lower() for keyword in ["years", "experience", "worked", "professional", "intern", "senior", "lead"]):
            experiences.append(line.strip())
    return " | ".join(experiences[:5]) if experiences else "No experience summary found"


def extract_years_experience(text: str) -> float | None:
    match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience\b",
        text,
        re.IGNORECASE,
    )
    return float(match.group(1)) if match else None


def parse_resume(pdf_path: str | Path) -> Dict[str, Any]:
    text = extract_text_from_pdf(pdf_path)
    return {
        "candidate_name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": extract_skills(text),
        "projects": extract_projects(text),
        "certifications": extract_certifications(text),
        "education": extract_education(text),
        "experience": extract_experience(text),
        "experience_years": extract_years_experience(text),
        "technologies": extract_skills(text),
        "achievements": extract_projects(text),
        "raw_text": text,
    }
