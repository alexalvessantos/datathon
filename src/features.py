from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load
import numpy as np, os
from src.text_normalizer import normalize
from src.scoring import extract_skills

ART_TFIDF = "artifacts/tfidf.joblib"

def fit_tfidf(corpus, max_features=50000, ngram_range=(1,1)):
    vec = TfidfVectorizer(
        min_df=2, max_df=0.95,
        ngram_range=ngram_range,
        max_features=max_features,
        strip_accents="unicode",
    )
    vec.fit(corpus)
    return vec

def save_tfidf(vec, path: str = ART_TFIDF):
    os.makedirs("artifacts", exist_ok=True)
    dump(vec, path)

def load_tfidf(path: str = ART_TFIDF):
    return load(path)

def build_features(vec: TfidfVectorizer, vaga_texto: str, cv_texto: str):
    nv, nc = normalize(vaga_texto), normalize(cv_texto)
    X = vec.transform([nv, nc])
    cos = float(cosine_similarity(X[0], X[1])[0, 0])
    req = extract_skills(nv); have = extract_skills(nc)
    sratio = (len(req & have) / len(req)) if req else 0.0
    return np.array([[cos, sratio]]), cos
