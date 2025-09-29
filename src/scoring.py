from src.text_normalizer import normalize
from src.skills import BASE_SKILLS

def extract_skills(text: str) -> set:
    nt = normalize(text)
    found = set()
    for skill, syns in BASE_SKILLS.items():
        if any(s in nt for s in syns):
            found.add(skill)
    return found

def ratio(required_skills: set, candidate_skills: set) -> float:
    if not required_skills:
        return 0.0
    return len(required_skills & candidate_skills) / len(required_skills)

def classify(score: float, threshold: float = 0.6) -> str:
    return "Atende" if score >= threshold else "Não atende"

def baseline_predict(vaga_texto: str, cv_texto: str, threshold: float = 0.6):
    req = extract_skills(vaga_texto)
    have = extract_skills(cv_texto)
    s = ratio(req, have)
    return {
        "score_baseline": round(s, 4),
        "classificacao_baseline": classify(s, threshold),
        "skills_vaga": sorted(req),
        "skills_cv": sorted(have),
        "skills_match": sorted(req & have),
        "threshold": threshold
    }
