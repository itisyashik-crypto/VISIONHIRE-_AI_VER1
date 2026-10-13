from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from textwrap import shorten
from typing import Callable

import plotly.graph_objects as go
import streamlit as st

from config import CONFIG, validate_startup
from fusion_engine import fuse_candidate_scores
from interview_generator import generate_interview_questions
from predictor import CandidatePredictor
from rag_engine import RAGEngine
from resume_parser import parse_resume


def _startup_signatures():
    dataset = CONFIG.DATASET_PATH
    dataset_signature = (
        (dataset.stat().st_mtime_ns, dataset.stat().st_size)
        if dataset.exists() else None
    )
    knowledge_signature = tuple(
        (path.name, path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(CONFIG.KNOWLEDGE_BASE_PATH.iterdir())
        if path.is_file()
    ) if CONFIG.KNOWLEDGE_BASE_PATH.exists() else ()
    return dataset_signature, knowledge_signature


@st.cache_data(ttl=300, show_spinner=False)
def _startup_status(dataset_signature, knowledge_signature):
    return validate_startup()


def _plot_layout(fig, height=300):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans, sans-serif", "color": "#a9bad0", "size": 12},
        margin={"l": 40, "r": 40, "t": 36, "b": 36},
        showlegend=False,
    )
    return fig


def _build_recruiter_report(analysis, questions, career):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    output = BytesIO()
    generated_at = datetime.now().astimezone()
    ink = colors.HexColor("#13263b")
    muted = colors.HexColor("#5b6f83")
    blue = colors.HexColor("#2676b9")
    pale = colors.HexColor("#eef5fa")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="VHCoverTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=30, leading=36, textColor=ink, alignment=TA_LEFT, spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="VHSubTitle", parent=styles["Normal"], fontName="Helvetica",
        fontSize=13, leading=19, textColor=muted, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="VHSection", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=15, leading=19, textColor=ink, spaceBefore=15, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="VHBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9.5, leading=14, textColor=ink, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="VHSmall", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8, leading=11, textColor=muted,
    ))
    styles.add(ParagraphStyle(
        name="VHLabel", parent=styles["BodyText"], fontName="Helvetica-Bold",
        fontSize=8.5, leading=12, textColor=blue,
    ))

    def safe_text(value):
        normalized = str(value).replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
        normalized = normalized.replace("\u2014", "-").replace("\u2013", "-")
        return escape(normalized.encode("latin-1", errors="replace").decode("latin-1"))

    def paragraph(value, style="VHBody"):
        return Paragraph(safe_text(value).replace("\n", "<br/>"), styles[style])

    def bullet_section(story, title, items):
        story.append(paragraph(title, "VHSection"))
        if items:
            story.extend(paragraph(f"- {item}") for item in items)
        else:
            story.append(paragraph("No evidence was returned for this section.", "VHSmall"))

    candidate_name = analysis.get("candidate_name") or "Candidate profile"
    score_values = [
        ("Resume quality", analysis.get("resume_score")),
        ("Role match", analysis.get("role_match_score")),
        ("Interview readiness", analysis.get("insights", {}).get("interview_readiness_score")),
        ("Dataset selection probability", (analysis.get("dataset_prediction") or {}).get("selection_probability", None)),
    ]
    insights = analysis.get("insights", {})
    fusion = analysis.get("fusion", {})
    timestamp = generated_at.strftime("%B %d, %Y at %H:%M %Z")
    story = [
        Spacer(1, 1.1 * inch),
        paragraph("VISIONHIRE AI", "VHLabel"),
        Spacer(1, .2 * inch),
        paragraph("Recruitment Intelligence Report", "VHCoverTitle"),
        paragraph("Evidence-led candidate briefing", "VHSubTitle"),
        Spacer(1, .45 * inch),
        paragraph(f"Candidate: {candidate_name}", "VHSection"),
        paragraph(f"Target role: {analysis.get('selected_role', 'Not selected')}", "VHBody"),
        paragraph("Prepared by: VisionHire AI", "VHBody"),
        paragraph("Created by: Yashik Kalra", "VHBody"),
        paragraph(f"Generated: {timestamp}", "VHSmall"),
        Spacer(1, .35 * inch),
        paragraph("This report supports recruiter review; it is not an automated hiring decision.", "VHSmall"),
        PageBreak(),
        paragraph("Executive summary", "VHSection"),
        paragraph(f"Hiring recommendation: {insights.get('hiring_recommendation', fusion.get('recommendation', 'Recruiter review required'))}"),
        paragraph(insights.get("recruiter_summary", "Recruiter review is required to finalize this decision.")),
    ]

    overview_rows = [[paragraph("Candidate detail", "VHLabel"), paragraph("Extracted evidence", "VHLabel")]]
    for label, value in (
        ("Name", candidate_name),
        ("Target role", analysis.get("selected_role", "Not selected")),
        ("Experience", f"{analysis['experience_years']:g} years" if isinstance(analysis.get("experience_years"), (int, float)) else "Not stated"),
        ("Education", "; ".join(analysis.get("education", [])) or "Not listed"),
        ("Certifications", "; ".join(analysis.get("certifications", [])) or "None extracted"),
        ("Key skills", "; ".join(analysis.get("skills", [])) or "None extracted"),
    ):
        overview_rows.append([paragraph(label, "VHLabel"), paragraph(value)])
    overview_table = Table(overview_rows, colWidths=[1.55 * inch, 5.2 * inch], repeatRows=1)
    overview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), pale),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#d9e4ed")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([paragraph("Candidate overview", "VHSection"), overview_table])

    score_rows = [[paragraph("Measure", "VHLabel"), paragraph("Score", "VHLabel"), paragraph("Explanation", "VHLabel")]]
    explanations = {
        "Resume quality": "Resume completeness, extracted role skills, and project evidence.",
        "Role match": f"{len(analysis.get('matched_skills', []))} of {len(analysis.get('required_skills', []))} retrieved role requirements matched.",
        "Interview readiness": insights.get("readiness_level", "Based on Gemini candidate evidence analysis."),
        "Dataset selection probability": "Historical model signal; contextual evidence, not a decision.",
    }
    for label, value in score_values:
        if label == "Dataset selection probability" and value is not None:
            value = f"{value * 100:.1f}%"
        elif value is not None:
            value = f"{value:.1f}%"
        else:
            value = "Unavailable"
        score_rows.append([paragraph(label, "VHLabel"), paragraph(value), paragraph(explanations[label], "VHSmall")])
    score_table = Table(score_rows, colWidths=[1.7 * inch, 1.05 * inch, 4 * inch], repeatRows=1)
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), pale),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#d9e4ed")),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([paragraph("Assessment", "VHSection"), score_table])
    bullet_section(story, "Strengths", insights.get("strengths", []))
    bullet_section(story, "Weaknesses", insights.get("weaknesses", []))
    bullet_section(story, "Missing skills", insights.get("improvement_areas", analysis.get("missing_skills", [])))
    bullet_section(story, "Recommendations", insights.get("recommendations", []))
    bullet_section(story, "Learning plan", career.get("learning_roadmap", career.get("learning_path", [])))
    bullet_section(story, "Recommended certifications", career.get("certifications", []))
    bullet_section(story, "Career roadmap", insights.get("career_recommendations", []))

    grouped_questions = {}
    for question in questions:
        grouped_questions.setdefault(question.get("type", "Interview question"), []).append(question)
    for group in ("Technical", "Behavioral", "Scenario-Based", "Project-Based", "System Design"):
        if group not in grouped_questions:
            continue
        bullet_section(
            story,
            f"{group} interview questions",
            [f"({item.get('difficulty', 'Unspecified')}) {item.get('question', '')}" for item in grouped_questions[group]],
        )
    story.extend([
        paragraph("Recruiter decision", "VHSection"),
        paragraph(insights.get("hiring_recommendation", fusion.get("recommendation", "Recruiter review required"))),
        paragraph(insights.get("hiring_risk_analysis", "Review evidence and unknowns before deciding.")),
    ])

    def page_footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d9e4ed"))
        canvas.line(.75 * inch, .58 * inch, 7.75 * inch, .58 * inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(muted)
        canvas.drawString(.75 * inch, .38 * inch, "VISIONHIRE AI  ·  Created by: Yashik Kalra")
        canvas.drawRightString(7.75 * inch, .38 * inch, f"Page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=.75 * inch,
        leftMargin=.75 * inch,
        topMargin=.7 * inch,
        bottomMargin=.8 * inch,
        title=f"VisionHire Recruitment Intelligence Report - {candidate_name}",
        author="VisionHire AI",
    )
    document.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return output.getvalue(), generated_at


def _skills_radar(analysis):
    required = analysis.get("required_skills", [])
    matched = {skill.casefold() for skill in analysis.get("matched_skills", [])}
    if not required:
        return None
    values = [100 if skill.casefold() in matched else 0 for skill in required]
    figure = go.Figure(go.Scatterpolar(
        r=values,
        theta=[shorten(skill, width=15, placeholder="…") for skill in required],
        fill="toself",
        line={"color": "#70baff", "width": 2},
        fillcolor="rgba(90, 174, 239, .17)",
        marker={"color": "#61e2e3", "size": 7},
        hovertemplate="%{theta}<br>%{customdata}<extra></extra>",
        customdata=[
            f"{skill}: {'Found in resume' if value else 'Not found in extracted skills'}"
            for skill, value in zip(required, values)
        ],
    ))
    figure.update_layout(
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {"visible": True, "range": [0, 100], "tickvals": [0, 50, 100], "gridcolor": "rgba(154,172,194,.15)", "linecolor": "rgba(154,172,194,.15)"},
            "angularaxis": {"gridcolor": "rgba(154,172,194,.12)", "linecolor": "rgba(154,172,194,.15)", "tickfont": {"size": 10}},
        },
    )
    return _plot_layout(figure, 340)


def _readiness_gauge(value):
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"color": "#eef5ff", "size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#71849a", "tickfont": {"color": "#9aacc2"}},
            "bar": {"color": "#61e2e3", "thickness": .22},
            "bgcolor": "#101c2a",
            "bordercolor": "rgba(148,178,210,.15)",
            "borderwidth": 1,
            "steps": [
                {"range": [0, 50], "color": "rgba(105,128,155,.12)"},
                {"range": [50, 75], "color": "rgba(112,186,255,.10)"},
                {"range": [75, 100], "color": "rgba(97,226,227,.13)"},
            ],
        },
    ))
    return _plot_layout(figure, 260)


def _skill_gap_chart(analysis):
    required = analysis.get("required_skills", [])
    matched = {skill.casefold() for skill in analysis.get("matched_skills", [])}
    if not required:
        return None
    ordered = list(reversed(required))
    values = [100 if skill.casefold() in matched else 0 for skill in ordered]
    colors = ["#70baff" if value else "#f3c477" for value in values]
    labels = ["Found" if value else "Gap" for value in values]
    figure = go.Figure(go.Bar(
        x=values,
        y=ordered,
        orientation="h",
        marker={"color": colors, "cornerradius": 5},
        text=labels,
        textposition="outside",
        textfont={"color": "#b9c9dc", "size": 11},
        hovertemplate="%{y}: %{customdata}<extra></extra>",
        customdata=labels,
    ))
    figure.update_xaxes(range=[0, 118], showticklabels=False, showgrid=False, zeroline=False)
    figure.update_yaxes(showgrid=False, tickfont={"color": "#c4d2e2", "size": 12}, automargin=True)
    return _plot_layout(figure, max(220, 42 * len(required) + 40))


def _role_match_chart(analysis):
    probabilities = analysis.get("role_predictions", {})
    if not probabilities:
        return None
    roles = sorted(probabilities, key=probabilities.get)
    values = [probabilities[role] * 100 for role in roles]
    figure = go.Figure(go.Bar(
        x=values,
        y=roles,
        orientation="h",
        marker={"color": "#70baff", "cornerradius": 5},
        text=[f"{value:.0f}%" for value in values],
        textposition="outside",
        textfont={"color": "#c4d2e2", "size": 11},
        hovertemplate="%{y}: %{x:.1f}% role match probability<extra></extra>",
    ))
    figure.update_xaxes(range=[0, 112], ticksuffix="%", gridcolor="rgba(154,172,194,.12)", zeroline=False)
    figure.update_yaxes(showgrid=False, tickfont={"color": "#c4d2e2", "size": 12}, automargin=True)
    return _plot_layout(figure, max(230, 42 * len(roles) + 45))


def _metric_card(label: str, value: str, explanation: str, accent: str):
    numeric_value = re.fullmatch(r"(\d+(?:\.\d+)?)%", value)
    if numeric_value:
        target = round(float(numeric_value.group(1)))
        value_markup = (
            f"<div class='vh-kpi-value vh-counter' style='--vh-target:{target}'>"
            "<span class='vh-kpi-suffix'>%</span></div>"
        )
    else:
        value_markup = f"<div class='vh-kpi-value'>{escape(value)}</div>"
    with st.container(border=True):
        st.markdown(
            f"<div class='vh-kpi-label' style='color:{accent}'>{escape(label)}</div>"
            f"{value_markup}"
            f"<div class='vh-kpi-copy'>{escape(explanation)}</div>",
            unsafe_allow_html=True,
        )


def _section(title: str, description: str | None = None):
    st.markdown(
        f"<div class='vh-section-head'><div class='vh-overline'>Candidate intelligence</div>"
        f"<h2>{escape(title)}</h2>{f'<p>{escape(description)}</p>' if description else ''}</div>",
        unsafe_allow_html=True,
    )


def _profile_card(label: str, value: str, detail: str | None = None):
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"**{value or 'Not available'}**")
        if detail:
            st.markdown(f"<span class='vh-muted'>{escape(detail)}</span>", unsafe_allow_html=True)


def _skill_chips(skills):
    if not skills:
        st.write("No skills were extracted from the resume.")
        return
    chips = "".join(f"<span class='vh-skill'>{escape(str(skill))}</span>" for skill in skills)
    st.markdown(chips, unsafe_allow_html=True)


def _render_list_panel(title: str, items, empty_text: str):
    st.markdown(f"### {escape(title)}")
    if items:
        for item in items:
            st.markdown(f"- {item}")
    else:
        st.markdown(f"<span class='vh-muted'>{escape(empty_text)}</span>", unsafe_allow_html=True)


def _career_roadmap(analysis, career, insights):
    gaps = analysis.get("missing_skills", [])
    learning = career.get("learning_roadmap", []) or career.get("learning_path", [])
    certifications = career.get("certifications", [])
    recommended_role = analysis.get("recommended_role")
    future = recommended_role or (insights.get("career_recommendations") or ["Continue developing role depth."])[0]
    steps = [
        ("01", "Current position", analysis.get("selected_role", "Not selected")),
        ("02", "Priority skill gaps", ", ".join(gaps[:3]) if gaps else "No identified gaps"),
        ("03", "Recommended learning", "; ".join(learning[:2]) if learning else "No learning plan returned"),
        ("04", "Certification", ", ".join(certifications[:2]) if certifications else "Optional; none suggested"),
        ("05", "Future direction", future),
    ]
    items = "".join(
        f"<li><span class='vh-roadmap-step'>{step}</span><strong>{escape(str(title))}</strong>"
        f"<p>{escape(str(detail))}</p></li>"
        for step, title, detail in steps
    )
    st.markdown(f"<ol class='vh-roadmap'>{items}</ol>", unsafe_allow_html=True)


def _analyze(
    uploaded,
    selected_role: str,
    calculate_match_score: Callable,
    on_stage: Callable | None = None,
):
    def report_stage(label, fraction):
        if on_stage:
            on_stage(label, fraction)

    report_stage("Resume received", .06)
    with NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(uploaded.getvalue())
        resume_data = parse_resume(tmp.name)
    report_stage("Candidate profile extracted", .20)

    rag_engine = RAGEngine()
    role_knowledge = rag_engine.retrieve_knowledge(
        f"{selected_role} job role overview responsibilities required skills evaluation criteria",
        k=4,
    )
    match = calculate_match_score(resume_data, selected_role, role_knowledge)
    report_stage("Role requirements and skills reviewed", .36)
    predictor = CandidatePredictor()
    dataset_roles = predictor.available_roles
    reference_role = selected_role if selected_role in dataset_roles else (dataset_roles[0] if dataset_roles else None)
    dataset_reference = (
        predictor.predict({**resume_data, "target_role": reference_role})
        if reference_role else None
    )
    selected_prediction = dataset_reference if selected_role in dataset_roles else None
    role_predictions = dataset_reference.get("role_probabilities", {}) if dataset_reference else {}
    dataset_signals = (
        [selected_prediction.get("selection_probability"), selected_prediction.get("role_match_probability")]
        if selected_prediction else []
    )
    dataset_signals = [score for score in dataset_signals if score is not None]
    dataset_component_score = (
        sum(dataset_signals) / len(dataset_signals) * 100
        if dataset_signals else None
    )
    report_stage("Historical role signals evaluated", .52)
    insights = rag_engine.generate_candidate_insights(
        {**resume_data, **match},
        selected_role,
        selected_prediction,
        role_predictions,
    )
    report_stage("Knowledge and candidate intelligence assembled", .72)
    fusion = fuse_candidate_scores({
        "dataset": dataset_component_score,
        "gemini": insights.get("gemini_score"),
        "rag": insights.get("rag_score"),
        "business_rules": match["business_rules_score"],
    })
    analysis = {
        **resume_data,
        **match,
        "selected_role": selected_role,
        "dataset_prediction": selected_prediction,
        "dataset_component_score": dataset_component_score,
        "role_predictions": role_predictions,
        "recommended_role": (
            insights.get("recommended_role")
            if insights.get("recommended_role") in CONFIG.JOB_ROLES
            else dataset_reference.get("recommended_role") if dataset_reference else None
        ),
        "recommended_role_source": (
            "Gemini" if insights.get("recommended_role") in CONFIG.JOB_ROLES else "Dataset"
        ),
        "candidate_similarity": dataset_reference.get("candidate_similarity") if dataset_reference else None,
        "insights": insights,
        "fusion": fusion,
    }
    interview_knowledge = rag_engine.retrieve_knowledge(
        f"{selected_role} structured interview technical behavioral STAR project system design questions",
        k=4,
    )
    questions = generate_interview_questions(
        selected_role,
        resume_data.get("skills", []),
        resume_data.get("experience", ""),
        resume_data.get("projects", []),
        "\n".join(f"[{item['source']}] {item['content']}" for item in interview_knowledge),
    )
    report_stage("Interview and career guidance prepared", .90)
    career = {
        "readiness": insights.get("readiness_level", "Unavailable"),
        "career_direction": (insights.get("career_recommendations") or ["No career recommendation was returned."])[0],
        "learning_path": match["missing_skills"],
        "certifications": insights.get("suggested_certifications", []),
        "learning_roadmap": insights.get("learning_roadmap", []),
        "recommendations": insights.get("recommendations", []),
        "ai_generated": insights.get("ai_generated", False),
    }
    report_stage("Recruiter report ready", 1.0)
    return analysis, questions, career


def render_app(calculate_match_score: Callable):
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --canvas:#080e18; --surface:#0d1724; --surface-raised:#111f30; --line:rgba(148,178,210,.16); --ink:#eef5ff; --muted:#9aacc2; --dim:#6c819b; --electric:#70baff; --cyan:#61e2e3; --violet:#a994ff; --amber:#f3c477; }
        *,*::before,*::after { box-sizing:border-box; }
        html,body,[data-testid="stAppViewContainer"] { width:100%; min-width:0; overflow-x:hidden; background:linear-gradient(145deg,#080e18 0%,#0a1320 52%,#09111c 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
        [data-testid="stAppViewContainer"]>.main { background-image:repeating-linear-gradient(0deg,transparent,transparent 47px,rgba(158,190,222,.018) 48px),repeating-linear-gradient(90deg,transparent,transparent 47px,rgba(158,190,222,.018) 48px); }
        [data-testid="stHeader"] { background:transparent; } [data-testid="stToolbar"] { right:1rem; }
        .block-container { width:97vw; max-width:3600px; min-width:0; margin-inline:auto; padding:1rem clamp(16px,2vw,48px) 3.5rem; }
        [data-testid="stHorizontalBlock"],[data-testid="column"],[data-testid="stVerticalBlock"],[data-testid="stVerticalBlockBorderWrapper"] { min-width:0!important; }
        [data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] * { max-width:100%; overflow-wrap:anywhere; word-break:normal; }
        [data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li { line-height:1.55; }
        h1,h2,h3,h4,[data-testid="stMetricValue"] { color:var(--ink); font-family:'Space Grotesk',sans-serif; letter-spacing:0; }
        h2 { font-size:1.72rem; } h3 { font-size:1.12rem; }
        p,li,label,[data-testid="stMarkdownContainer"] { color:var(--muted); }
        [data-testid="stVerticalBlockBorderWrapper"] { background:linear-gradient(145deg,rgba(17,31,48,.78),rgba(11,22,35,.82)); border:1px solid var(--line); border-radius:16px; box-shadow:0 18px 45px rgba(0,0,0,.16); backdrop-filter:blur(18px); transition:border-color .2s ease,transform .2s ease,box-shadow .2s ease; animation:vh-reveal .45s ease both; }
        [data-testid="stVerticalBlockBorderWrapper"]:hover { border-color:rgba(112,186,255,.34); transform:translateY(-2px) perspective(900px) rotateX(.12deg); box-shadow:0 22px 50px rgba(0,0,0,.23),0 0 24px rgba(55,153,235,.06); }
        [data-testid="stFileUploaderDropzone"] { width:100%; min-width:0; min-height:168px; border:1px dashed rgba(112,186,255,.58)!important; border-radius:18px!important; background:linear-gradient(135deg,rgba(31,69,104,.26),rgba(17,36,57,.58))!important; box-shadow:inset 0 0 28px rgba(76,173,255,.05),0 0 24px rgba(76,173,255,.07); transition:border-color .2s ease,box-shadow .2s ease,background .2s ease; }
        [data-testid="stFileUploaderDropzone"]:hover { border-color:var(--cyan)!important; background:linear-gradient(135deg,rgba(36,91,134,.34),rgba(17,36,57,.70))!important; box-shadow:inset 0 0 34px rgba(76,173,255,.08),0 0 28px rgba(76,173,255,.12); }
        [data-testid="stFileUploaderDropzone"] button,div.stButton>button,[data-testid="stDownloadButton"] button { display:flex; align-items:center; justify-content:center; width:100%; min-width:0; min-height:48px; padding:10px 16px; color:#06111e!important; text-align:center; white-space:normal; line-height:1.25; background:linear-gradient(100deg,#8ac9ff,#69e2e3); border:0; border-radius:10px; font-family:'DM Sans',sans-serif; font-weight:700; box-shadow:0 8px 24px rgba(78,174,240,.18); transition:transform .18s ease,box-shadow .18s ease,filter .18s ease; }
        div.stButton>button p,[data-testid="stDownloadButton"] button p,[data-testid="stFileUploaderDropzone"] button p { color:inherit!important; margin:0; white-space:normal; overflow-wrap:anywhere; }
        div.stButton>button:hover { color:#06111e; filter:brightness(1.07); transform:translateY(-1px); box-shadow:0 12px 30px rgba(78,174,240,.28); }
        div.stButton>button[kind="secondary"], [data-testid="stDownloadButton"] button { color:#cfe5fa; background:rgba(21,39,58,.82); border:1px solid rgba(112,186,255,.28); box-shadow:none; }
        div.stButton>button[kind="secondary"]:hover, [data-testid="stDownloadButton"] button:hover { color:#eef7ff; background:rgba(32,61,87,.9); border-color:rgba(97,226,227,.52); }
        [data-testid="stFileUploader"] label,[data-testid="stSelectbox"] label { display:block; min-height:1.5rem; margin:0 0 8px; padding:0 2px; line-height:1.4; }
        [data-testid="stFileUploader"],[data-testid="stSelectbox"],[data-testid="stFileUploader"]>div,[data-testid="stSelectbox"]>div { min-width:0; width:100%; }
        div[data-testid="stSelectbox"]>div>div { min-height:48px; background:rgba(12,24,38,.9); border-color:var(--line); border-radius:10px; color:var(--ink); }
        [data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; background:rgba(12,23,36,.72); }
        hr { border-color:var(--line); }
        .vh-brand { display:flex;align-items:center;gap:12px;color:var(--ink);font:700 15px 'Space Grotesk',sans-serif; }
        .vh-mark { width:34px;height:34px;display:grid;place-items:center;border-radius:10px;color:#08121d;font:700 12px 'Space Grotesk',sans-serif;background:linear-gradient(145deg,#a0d4ff,#61e2e3);box-shadow:0 0 20px rgba(97,226,227,.19); }
        .vh-topline { display:flex;align-items:center;justify-content:space-between;gap:16px;padding:8px 0 20px;border-bottom:1px solid var(--line); }
        .vh-brand-block { display:flex;min-width:0;flex-direction:column;gap:4px; }
        .vh-brand-subtitle { padding-left:46px;color:var(--muted);font-size:12px;line-height:1.4;overflow-wrap:anywhere; }
        .vh-overline { color:var(--cyan);text-transform:uppercase;font-size:11px;font-weight:700; }
        .vh-hero { padding:56px 0 30px;max-width:850px;animation:vh-rise .55s ease-out both; }
        .vh-hero h1 { font:600 58px/1.04 'Space Grotesk',sans-serif;margin:12px 0 14px; }
        .vh-hero p { color:#a9bad0;font-size:17px;line-height:1.6;max-width:620px; }
        .vh-section-head { padding:clamp(26px,3vw,44px) 0 10px; } .vh-section-head p { margin:4px 0 0;color:var(--muted); }
        .vh-kpi-label { color:var(--muted);font-size:12px;text-transform:uppercase;font-weight:700; }
        .vh-kpi-value { color:var(--ink);font:600 clamp(22px,1.55vw,34px) 'Space Grotesk',sans-serif;margin:12px 0 8px; }
        @property --vh-counter-value { syntax:'<integer>'; inherits:false; initial-value:0; }
        .vh-counter:before { counter-reset:vhscore var(--vh-counter-value);content:counter(vhscore);animation:vh-count-up 1s cubic-bezier(.2,.72,.28,1) .08s forwards; }
        .vh-kpi-suffix { margin-left:2px; }
        .vh-kpi-copy { color:var(--muted);font-size:13px;line-height:1.5; }
        .vh-skill { display:inline-block;margin:4px 5px 4px 0;padding:6px 10px;border:1px solid rgba(112,186,255,.24);border-radius:999px;color:#c8e4ff;background:rgba(67,139,205,.09);font-size:12px; }
        .vh-muted { color:var(--dim);font-size:13px; } .vh-footer { border-top:1px solid var(--line);margin-top:50px;padding-top:18px;color:var(--dim);font-size:12px; }
        .vh-executive { display:grid;grid-template-columns:minmax(0,1fr) minmax(190px,.45fr);align-items:center;gap:24px;padding:clamp(20px,2.4vw,36px);border:1px solid rgba(112,186,255,.26);border-radius:20px;background:linear-gradient(120deg,rgba(21,45,68,.86),rgba(12,25,39,.92));box-shadow:0 24px 70px rgba(0,0,0,.2),inset 0 1px rgba(255,255,255,.035); }
        .vh-executive>* { min-width:0;overflow-wrap:anywhere; }
        .vh-executive-name { color:var(--ink);font:600 clamp(25px,2vw,36px) 'Space Grotesk',sans-serif;overflow-wrap:anywhere; }
        .vh-executive-result { text-align:right;min-width:0; }
        .vh-result-label { color:var(--muted);font-size:11px;text-transform:uppercase;font-weight:700; }
        .vh-result-value { color:var(--cyan);font:600 21px 'Space Grotesk',sans-serif;margin-top:7px;overflow-wrap:anywhere; }
        .vh-executive-details { display:flex;gap:clamp(14px,2vw,30px);flex-wrap:wrap;margin-top:20px;padding-top:16px;border-top:1px solid var(--line); }
        .vh-detail-label { display:block;color:var(--dim);font-size:11px;text-transform:uppercase;margin-bottom:5px; }
        .vh-detail-value { color:#d5e4f3;font-size:14px;overflow-wrap:anywhere; }
        .vh-kpi-strip { display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px; }
        .vh-roadmap { display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:0;list-style:none;padding:18px 0 0;margin:0; }
        .vh-roadmap li { position:relative;min-width:0;padding:18px 14px 12px 0;border-top:1px solid rgba(112,186,255,.35); }
        .vh-roadmap li:before { content:'';position:absolute;top:-5px;left:0;width:9px;height:9px;border-radius:50%;background:var(--cyan);box-shadow:0 0 13px rgba(97,226,227,.35); }
        .vh-roadmap-step { display:block;color:var(--cyan);font-size:10px;font-weight:700;text-transform:uppercase;margin-bottom:7px; }
        .vh-roadmap strong { color:var(--ink);font-size:13px;overflow-wrap:anywhere; }
        .vh-roadmap p { margin:6px 10px 0 0;font-size:12px;line-height:1.5;overflow-wrap:anywhere; }
        .vh-analysis-progress { color:#a9bad0;font-size:13px;padding:4px 0; }
        .vh-progress-done { color:var(--cyan); }
        [data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,#70baff,#61e2e3);box-shadow:0 0 12px rgba(97,226,227,.28); }
        @keyframes vh-rise { from {opacity:0;transform:translateY(10px)} to {opacity:1;transform:translateY(0)} }
        @keyframes vh-reveal { from {opacity:0;transform:translateY(5px)} to {opacity:1;transform:translateY(0)} }
        @keyframes vh-count-up { to { --vh-counter-value:var(--vh-target); } }
        @media(max-width:1100px) {
            [data-testid="stHorizontalBlock"] { flex-wrap:wrap;gap:14px; }
            [data-testid="column"] { min-width:min(100%,280px)!important; }
            .vh-kpi-strip { grid-template-columns:repeat(2,minmax(0,1fr)); }
        }
        @media(max-width:680px) {
            .block-container{width:100%;padding:.65rem 1rem 2.4rem}
            [data-testid="stHorizontalBlock"] { gap:8px; }
            [data-testid="column"] { min-width:100%!important; }
            .vh-hero{padding:34px 0 20px}.vh-hero h1{font-size:42px}.vh-hero p{font-size:15px}
            .vh-executive { grid-template-columns:1fr;gap:16px; }
            .vh-executive-result { text-align:left; }
            .vh-kpi-strip { grid-template-columns:1fr; }
            .vh-roadmap { grid-template-columns:1fr;padding-left:12px; }
            .vh-roadmap li { border-top:0;border-left:1px solid rgba(112,186,255,.35);padding:2px 12px 18px 20px; }
            .vh-roadmap li:before { top:3px;left:-5px; }
        }
        @media(prefers-reduced-motion:reduce) {
            *,*::before,*::after { animation-duration:.01ms!important;animation-iteration-count:1!important;scroll-behavior:auto!important;transition-duration:.01ms!important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for key, default in (
        ("analysis", None),
        ("questions", []),
        ("coach", None),
        ("recruiter_report_pdf", None),
        ("recruiter_report_filename", None),
    ):
        if key not in st.session_state:
            st.session_state[key] = default

    st.markdown(
        "<div class='vh-topline'><div class='vh-brand-block'>"
        "<div class='vh-brand'><span class='vh-mark'>VH</span>VISIONHIRE AI</div>"
        "<div class='vh-brand-subtitle'>AI Recruitment Intelligence Platform · Created by: Yashik Kalra</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='vh-hero'><div class='vh-overline'>AI Recruitment Intelligence Platform</div>"
        "<h1>VISIONHIRE AI</h1>"
        "<p><strong>See the candidate behind the resume.</strong></p>"
        "<p>Turn a resume into an executive-ready view of fit, readiness, and next steps.</p></div>",
        unsafe_allow_html=True,
    )

    upload_col, role_col = st.columns([1.45, .8], gap="large")
    with upload_col:
        uploaded = st.file_uploader(
            "Upload Resume · PDF",
            type=["pdf"],
            help="PDF resumes only",
            key="visionhire_resume_upload",
        )
    with role_col:
        selected_role = st.selectbox(
            "Target role",
            CONFIG.JOB_ROLES,
            index=CONFIG.JOB_ROLES.index(CONFIG.DEFAULT_ROLE) if CONFIG.DEFAULT_ROLE in CONFIG.JOB_ROLES else 0,
        )
        st.markdown("<div class='vh-muted'>Candidate profile and role evidence appear below.</div>", unsafe_allow_html=True)

    if st.button("Analyze Resume", type="primary", use_container_width=True, disabled=uploaded is None):
        st.session_state.analysis = None
        st.session_state.questions = []
        st.session_state.coach = None
        st.session_state.recruiter_report_pdf = None
        progress_status = st.status("Preparing candidate analysis", expanded=True)
        progress_bar = st.progress(0)

        def update_progress(label, fraction):
            progress_status.update(label=label, state="running")
            progress_bar.progress(int(fraction * 100), text=label)

        try:
            analysis, questions, career = _analyze(
                uploaded,
                selected_role,
                calculate_match_score,
                update_progress,
            )
            st.session_state.analysis = analysis
            st.session_state.questions = questions
            st.session_state.coach = career
            progress_status.update(label="Recruiter report ready", state="complete", expanded=False)
        except Exception:
            progress_status.update(label="Analysis needs attention", state="error", expanded=False)
            st.error("We couldn't complete this resume review. Check that the PDF contains selectable text, then try again.")

    analysis = st.session_state.analysis
    if analysis:
        insights = analysis["insights"]
        fusion = analysis["fusion"]
        prediction = analysis.get("dataset_prediction")
        st.markdown("<div class='vh-rule'></div>", unsafe_allow_html=True)
        _section("Executive summary", "A concise briefing grounded in the resume and role evidence")

        candidate_name = analysis.get("candidate_name")
        if not candidate_name or candidate_name == "Candidate":
            candidate_name = "Candidate profile"
        contact = [value for value in (analysis.get("email"), analysis.get("phone")) if value and value != "Not available"]
        recommended_role = analysis.get("recommended_role") or analysis.get("selected_role", "Not selected")
        recommendation = fusion.get("recommendation", "Insufficient evidence")
        experience_years = analysis.get("experience_years")
        experience_label = f"{experience_years:g} years" if isinstance(experience_years, (int, float)) else "Not stated"
        education = " · ".join(analysis.get("education", [])) or "Not listed"
        certifications = " · ".join(analysis.get("certifications", [])) or "None extracted"
        contact_line = "  ·  ".join(contact)
        st.markdown(
            f"<div class='vh-executive'><div><div class='vh-overline'>Candidate profile</div>"
            f"<div class='vh-executive-name'>{escape(str(candidate_name))}</div>"
            f"<div class='vh-muted'>{escape(contact_line)}</div></div>"
            f"<div class='vh-executive-result'><div class='vh-result-label'>Final hiring recommendation</div>"
            f"<div class='vh-result-value'>{escape(str(recommendation))}</div>"
            f"<div class='vh-muted'>Human recruiter review remains essential</div></div>"
            f"<div class='vh-executive-details'>"
            f"<div><span class='vh-detail-label'>Recommended role</span><span class='vh-detail-value'>{escape(str(recommended_role))}</span></div>"
            f"<div><span class='vh-detail-label'>Target role</span><span class='vh-detail-value'>{escape(str(analysis.get('selected_role', 'Not selected')))}</span></div>"
            f"<div><span class='vh-detail-label'>Experience</span><span class='vh-detail-value'>{escape(experience_label)}</span></div>"
            f"<div><span class='vh-detail-label'>Education</span><span class='vh-detail-value'>{escape(education)}</span></div>"
            f"<div><span class='vh-detail-label'>Certifications</span><span class='vh-detail-value'>{escape(certifications)}</span></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Top skills**")
        _skill_chips(analysis.get("skills", []))

        if analysis.get("projects"):
            with st.expander("Projects and experience evidence", expanded=False):
                for project in analysis["projects"]:
                    st.markdown(f"- {project}")

        _section("Candidate intelligence", "Four decision signals, each tied to evidence")
        resume_score = analysis.get("resume_score")
        role_score = analysis.get("role_match_score")
        readiness = insights.get("interview_readiness_score")
        resume_explanation = "Resume completeness, detected role skills, and project evidence."
        if analysis.get("role_match_score") is not None:
            resume_explanation = f"Evidence from {len(analysis.get('matched_skills', []))} matched role requirements, project history, and resume completeness."
        role_explanation = (
            f"{len(analysis.get('matched_skills', []))} of {len(analysis.get('required_skills', []))} retrieved role requirements appear in extracted skills."
            if role_score is not None else "Role requirements could not be scored from the available evidence."
        )
        readiness_explanation = (
            f"Gemini assessment: {insights.get('readiness_level', 'Readiness assessed from candidate evidence.')}"
            if readiness is not None else "No interview-readiness score was returned for this candidate."
        )
        recommendation_explanation = "Combines historical model signals, Gemini reasoning, retrieved standards, and resume evidence. Recruiter review remains essential."
        metric_cols = st.columns(4, gap="medium")
        metric_data = [
            ("Resume quality", f"{resume_score:.0f}%" if resume_score is not None else "Unavailable", resume_explanation, "#70baff"),
            ("Role match", f"{role_score:.0f}%" if role_score is not None else "Unavailable", role_explanation, "#61e2e3"),
            ("Interview readiness", f"{readiness:.0f}%" if readiness is not None else "Unavailable", readiness_explanation, "#a994ff"),
            ("Hiring recommendation", str(recommendation), recommendation_explanation, "#f3c477"),
        ]
        for column, values in zip(metric_cols, metric_data):
            with column:
                _metric_card(*values)

        st.markdown("<div class='vh-rule'></div>", unsafe_allow_html=True)
        _section("AI insights", "Evidence-grounded strengths, gaps, and next steps")
        if insights.get("resume_feedback"):
            with st.container(border=True):
                st.caption("Resume feedback")
                st.write(insights["resume_feedback"])

        insight_cols = st.columns(2, gap="medium")
        panels = [
            ("Strengths", insights.get("strengths", []), "No strengths were returned."),
            ("Weaknesses", insights.get("weaknesses", []), "No weaknesses were returned."),
            ("Missing skills", insights.get("improvement_areas", analysis.get("missing_skills", [])), "No role-skill gaps were identified in the extracted evidence."),
            ("Recommendations", insights.get("recommendations", []), "No recommendations were returned."),
        ]
        for index, (title, items, empty) in enumerate(panels):
            with insight_cols[index % 2]:
                _render_list_panel(title, items, empty)

        radar, gap_chart = st.columns(2, gap="large")
        with radar:
            with st.container(border=True):
                st.markdown("### Skills overview")
                st.caption("Detected in resume against retrieved role requirements")
                figure = _skills_radar(analysis)
                if figure:
                    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.markdown("<span class='vh-muted'>No role requirements were available for this visualization.</span>", unsafe_allow_html=True)
        with gap_chart:
            with st.container(border=True):
                st.markdown("### Skill gap analysis")
                st.caption("Presence or absence in extracted resume skills; not a proficiency rating")
                figure = _skill_gap_chart(analysis)
                if figure:
                    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.markdown("<span class='vh-muted'>No role requirements were available for this visualization.</span>", unsafe_allow_html=True)

        _section("Interview readiness", "Personalized prompts grounded in role requirements and candidate projects")
        interview_cols = st.columns([.75, 1.25], gap="large")
        with interview_cols[0]:
            with st.container(border=True):
                st.markdown("### Readiness signal")
                if readiness is not None:
                    st.plotly_chart(_readiness_gauge(float(readiness)), use_container_width=True, config={"displayModeBar": False})
                else:
                    st.write("Interview readiness is unavailable without a valid Gemini assessment.")
        with interview_cols[1]:
            with st.container(border=True):
                st.markdown("### Role match by supported role")
                role_figure = _role_match_chart(analysis)
                if role_figure:
                    st.plotly_chart(role_figure, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.write("No dataset role-match probabilities are available.")

        question_groups = {}
        for question in st.session_state.questions:
            question_groups.setdefault(question.get("type", "Interview question"), []).append(question)
        preferred_order = ["Technical", "Behavioral", "Scenario-Based", "Project-Based", "System Design"]
        ordered_groups = [name for name in preferred_order if name in question_groups]
        ordered_groups.extend(name for name in question_groups if name not in ordered_groups)
        question_cols = st.columns(2, gap="medium")
        for index, group in enumerate(ordered_groups):
            with question_cols[index % 2]:
                with st.expander(f"{group} · {len(question_groups[group])}", expanded=False):
                    for question in question_groups[group]:
                        st.caption(question.get("difficulty", "Question"))
                        st.write(question.get("question", ""))

        _section("Career roadmap", "Practical next steps based on the candidate's current evidence")
        career = st.session_state.coach or {}
        _career_roadmap(analysis, career, insights)

        with st.container(border=True):
            st.markdown("### Recruiter recommendation")
            st.markdown(f"**{escape(str(insights.get('hiring_recommendation', recommendation)))}**")
            st.write(insights.get("recruiter_summary", "Recruiter review is required to finalize this decision."))
            st.caption(insights.get("hiring_risk_analysis", "Review role evidence and unknowns before deciding."))

        report_button_col, report_status_col = st.columns([.7, 1.3], gap="medium")
        with report_button_col:
            if st.button("Generate Recruiter Report", type="secondary", use_container_width=True):
                try:
                    pdf_bytes, generated_at = _build_recruiter_report(
                        analysis,
                        st.session_state.questions,
                        career,
                    )
                    candidate_slug = re.sub(r"[^A-Za-z0-9]+", "-", str(candidate_name)).strip("-") or "Candidate"
                    st.session_state.recruiter_report_pdf = pdf_bytes
                    st.session_state.recruiter_report_filename = f"VisionHire-{candidate_slug}-Report.pdf"
                    st.session_state.recruiter_report_generated_at = generated_at.strftime("%B %d, %Y at %H:%M %Z")
                except Exception:
                    st.error("The recruiter report could not be generated. Please retry.")
        if st.session_state.recruiter_report_pdf:
            with report_status_col:
                st.download_button(
                    "Download Recruiter Report",
                    data=st.session_state.recruiter_report_pdf,
                    file_name=st.session_state.recruiter_report_filename or "VisionHire-Recruiter-Report.pdf",
                    mime="application/pdf",
                    type="secondary",
                    use_container_width=True,
                )
                st.caption(f"PDF generated {st.session_state.get('recruiter_report_generated_at', '')}")

    st.markdown(
        "<div class='vh-footer'>VISIONHIRE AI <span style='float:right'>Created by: Yashik Kalra</span></div>",
        unsafe_allow_html=True,
    )

    with st.expander("Developer mode", expanded=False):
        developer_enabled = st.checkbox("Show system diagnostics", key="visionhire_developer_mode")
        if developer_enabled:
            status = _startup_status(*_startup_signatures())
            diagnostic_cols = st.columns(3)
            for column, label, key in zip(
                diagnostic_cols * 2,
                ["Gemini API", "Gemini model", "Candidate dataset", "Knowledge documents", "FAISS index", "Embeddings"],
                ["api_loaded", "gemini_connected", "dataset_loaded", "knowledge_base_loaded", "faiss_active", "embeddings_active"],
            ):
                column.write(f"{label}: {'Active' if status[key] else 'Unavailable'}")
            st.caption(f"Active dataset: {CONFIG.DATASET_PATH.name if CONFIG.DATASET_PATH.exists() else 'not found'}")

    st.stop()
