from src.scoring import extract_skills, ratio, classify

def test_extract_skills_basic():
    s = extract_skills("Experiência com Python, Docker e AWS.")
    assert {"python","docker","aws"} <= s

def test_ratio_and_classify():
    req = {"python", "sql", "aws"}
    have = {"python", "docker", "aws"}
    r = ratio(req, have)
    assert abs(r - 2/3) < 1e-6
    assert classify(r, 0.6) == "Atende"
