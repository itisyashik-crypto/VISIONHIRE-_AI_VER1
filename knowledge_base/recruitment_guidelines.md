# Recruitment Guidelines

## Purpose and Fairness
Use structured, job-related evidence to support consistent hiring decisions. Assess what the candidate has demonstrated against published role requirements. Do not infer capability, motivation, reliability, culture fit, or future performance from protected characteristics, names, photos, age proxies, address, or other irrelevant personal data. A score is decision support, not a hiring decision; a qualified human reviewer remains accountable.

## Modern Recruitment Lifecycle
1. Define the role outcome, essential responsibilities, seniority, and measurable success criteria with the hiring manager.
2. Separate essential requirements from preferred capabilities and identify reasonable equivalent experience.
3. Publish an accessible role description with compensation range, process stages, and accommodation route.
4. Apply consistent intake, ATS parsing, and minimum-qualification checks; preserve the original resume for review.
5. Screen against an approved scorecard; record evidence and uncertainty rather than impressions.
6. Use structured technical, work-sample, and behavioral assessments relevant to actual work.
7. Calibrate interviewers against shared anchors before candidate interviews.
8. Compare evidence across candidates using the same criteria and interview stages.
9. Hold a documented debrief, resolve conflicting evidence, and record a rationale for the decision.
10. Communicate outcomes respectfully, retain records only as permitted, and review funnel outcomes for adverse impact and process quality.

## ATS Screening
- Treat extracted fields and keyword matches as noisy signals. Confirm important facts in source text.
- Normalize equivalent terms (for example, PostgreSQL and Postgres) without treating a missing synonym as a disqualification.
- Do not reject solely because a parser missed a skill, title, date, or section.
- Ignore formatting, file name, photo, and demographic clues when assessing job capability.
- Keep essential screening questions explicit, role-related, consistently applied, and auditable.
- Flag low extraction confidence for manual review; never convert missing data into a negative answer.

## Resume Screening Standards
Evaluate demonstrated skills, scope and recency of relevant work, project ownership, outcomes, and learning. Distinguish a skill named in a list from a skill evidenced by a project or responsibility. Consider transferable experience and nontraditional pathways. Do not penalize career breaks or unfamiliar employers without job-related evidence.

## Technical Evaluation
Choose a work sample or structured discussion representative of the role. Define expected signals before the exercise: approach, correctness, trade-offs, testing, maintainability, security, and communication where relevant. Give candidates equivalent instructions, time, tools, and accommodation. Score the reasoning and result against anchors; avoid trivia unrelated to the work.

## Behavioral Evaluation
Ask every candidate the same core questions and use neutral follow-ups. Seek a specific situation, the candidate's own actions, rationale, collaboration, outcome, and learning. Separate individual contribution from team results. Avoid leading questions and personality judgments.

## Communication Assessment
Assess clarity, listening, audience adaptation, and ability to explain decisions when those behaviors are material to the role. Accent, speaking style, confidence display, disability-related communication differences, or extroversion are not proxies for competence. Offer suitable formats and accommodations.

## Hiring Scorecard
Set weights before reviewing candidates. A practical default framework is:

| Dimension | Suggested share | Evidence |
|---|---:|---|
| Essential role skills | 30% | Demonstration, work sample, or specific delivery |
| Relevant experience and scope | 20% | Ownership, complexity, and recency |
| Project outcomes and problem solving | 20% | Approach, trade-offs, impact, and learning |
| Collaboration and communication | 15% | Structured examples relevant to the role |
| Growth and learning agility | 15% | Evidence of feedback use and skill development |

Weights are illustrative; teams should calibrate them to the job and use the same scorecard for all candidates. Mark a dimension “not observed” instead of inventing a score. A weighted total is valid only when evidence coverage and scoring rules are stated.

### Rating Anchors
- **1 — No evidence / materially below requirement:** required evidence was sought but not demonstrated. Missing resume information alone is not proof of inability.
- **2 — Partial:** some relevant evidence, with material gaps or substantial support required.
- **3 — Meets:** sufficient role-relevant evidence for the expected level.
- **4 — Strong:** clear ownership, sound judgment, and outcomes beyond the baseline.
- **5 — Exceptional:** repeated, high-quality evidence at unusual scope or complexity for the level.

Require written evidence for ratings. Do not average incomparable or unobserved dimensions.

## Candidate Ranking and Selection Framework
1. Verify essential requirements using reliable evidence.
2. Compare candidates only on the published, job-related scorecard.
3. Review uncertainty, missing data, assessment accommodations, and contradictory evidence.
4. Use model probabilities as historical-pattern indicators, not ground truth or a threshold for automatic rejection.
5. Check whether the model's training population, role, and input features are applicable to this candidate.
6. Apply the same human review and interview standard to candidates with similar evidence.
7. Record the decision, evidence, risks, and any override rationale.

### Decision Categories
- **Strong Hire:** repeated, directly observed evidence materially exceeds essential requirements; risks are understood and manageable.
- **Hire:** meets essential requirements with credible evidence; remaining gaps are trainable and proportionate.
- **Consider:** mixed or incomplete evidence; identify a specific follow-up assessment rather than guessing.
- **Reject:** reliable evidence shows an essential requirement is not met after a fair opportunity to demonstrate it, or a documented job-related requirement is unmet.

These are calibrated human-review labels, not automatic score bands. No single model score determines a category.

## Hiring Risks
- Parser errors, missing data, and resume-writing differences can create false negatives.
- Historical labels may encode prior bias, inconsistent interviewer practice, or changing role definitions.
- Small samples and unsupported roles produce unreliable probabilities.
- Keyword overlap may miss equivalent or transferable skills.
- Self-reported claims need appropriate verification, not assumptions.
- Unstructured interviews invite halo, affinity, anchoring, and recency effects.
- A model confidence value is not evidence of validity or fairness.

Mitigate through structured assessment, source verification, human review, calibration, monitoring, and a documented appeal or correction path.

## Recruiter Insights
Write concise observations tied to evidence: requirement, evidence source, strength or gap, uncertainty, and next action. Separate facts from interpretation. Prefer “The project section describes a Python ETL pipeline and names no deployment outcome” over “not production-ready.”

## Decision Rules
- Use no protected or irrelevant attributes in ranking or recommendation.
- Treat absent evidence as unknown unless a fair assessment established the gap.
- Never allow dataset output to make a final selection decision.
- Explain any score with its contributing sources and unavailable components.
- Apply thresholds only when validated, job-related, approved, and monitored.
- Escalate conflicting evidence and low-confidence extraction to a human reviewer.
- Provide a concrete, role-related next step for every “Consider” outcome.

## Worked Examples
**Example A — mixed model signal:** A model estimates a moderate selection probability, while a structured project review demonstrates strong role skills and clear ownership. Record both signals, inspect model applicability and evidence, and advance if the human scorecard supports it; do not reject on probability alone.

**Example B — missing certification:** A certification is preferred, not essential. The candidate demonstrates the relevant capability in a recent project. Score the demonstrated capability and note the credential as absent; do not treat it as a disqualifier.

**Example C — low resume detail:** A resume omits outcomes. Mark impact evidence as unobserved and ask a consistent project follow-up. Do not assign zero impact without asking.
