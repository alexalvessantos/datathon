import os, pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from joblib import dump
from src.text_normalizer import normalize
from src.features import fit_tfidf, save_tfidf, build_features

ART_DIR = "artifacts"
ART_MODEL = f"{ART_DIR}/model.joblib"

SAMPLE_N = int(os.getenv("SAMPLE_N", "0"))

def main():
    df = pd.read_csv("data/train_pairs.csv")

    if SAMPLE_N and SAMPLE_N < len(df):
        df = df.sample(SAMPLE_N, random_state=42).reset_index(drop=True)

    # -------- evitar leakage: se houver QUALQUER label fraca, vamos treinar sem a feature de skills --------
    has_weak = "label_source" in df.columns and (df["label_source"] == "weak").any()
    print(f"Leakage guard: has_weak_labels={has_weak} -> usando apenas TF-IDF cosine" if has_weak else "Usando TF-IDF cosine + skill-ratio")

    df["vaga_n"] = df["vaga_texto"].fillna("").map(normalize)
    df["cv_n"]   = df["cv_texto"].fillna("").map(normalize)

    corpus = pd.concat([df["vaga_n"], df["cv_n"]], axis=0).tolist()
    vec = fit_tfidf(corpus, max_features=50000, ngram_range=(1,1))

    # construir features
    feats = []
    for v, c in zip(df["vaga_n"], df["cv_n"]):
        X, _ = build_features(vec, v, c)   # X.shape == (1, 2) -> [cosine, skill_ratio]
        feats.append(X[0])
    X = np.vstack(feats)
    if has_weak:
        X = X[:, :1]  # mantém só a coluna 0 (TF-IDF cosine)

    y = df["label"].astype(int).values

    # split estratificado
    Xtr, Xte, ytr, yte, df_tr, df_te = train_test_split(
        X, y, df, test_size=0.25, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, ytr)

    ypred = clf.predict(Xte)
    f1 = f1_score(yte, ypred)
    print("F1-score (geral):", round(f1, 4))
    print(classification_report(yte, ypred, digits=4))

    # ---- métrica apenas nos casos 'status' (se existirem no teste) ----
    if "label_source" in df_te.columns and (df_te["label_source"] == "status").any():
        mask = (df_te["label_source"] == "status").values
        if mask.sum() > 0:
            print("\n[Subset: label_source = status]")
            print("F1:", round(f1_score(yte[mask], ypred[mask]), 4))
            print(classification_report(yte[mask], ypred[mask], digits=4))

    os.makedirs(ART_DIR, exist_ok=True)
    dump(clf, ART_MODEL)
    save_tfidf(vec)
    print("Artifacts salvos em:", ART_DIR)

if __name__ == "__main__":
    main()
