# api/main.py
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from joblib import load

from src.scoring import baseline_predict
from src.features import load_tfidf, build_features

# Permite customizar via ENV, mas mantém defaults do seu projeto
ART_MODEL = os.getenv("ART_MODEL", "artifacts/model.joblib")
ART_TFIDF = os.getenv("ART_TFIDF", "artifacts/tfidf.joblib")

app = FastAPI(title="Tech Fit API", version="1.0")

MODEL = None
VEC = None


def load_artifacts() -> None:
    """Tenta carregar modelo e TF-IDF; se falhar, mantém baseline."""
    global MODEL, VEC
    if os.path.exists(ART_MODEL) and os.path.exists(ART_TFIDF):
        try:
            MODEL = load(ART_MODEL)
            VEC = load_tfidf(ART_TFIDF)
            print(">> [startup] Modelo e TF-IDF carregados.")
        except Exception as e:
            MODEL, VEC = None, None
            print(">> [startup] Falha ao carregar artifacts, usando baseline. Erro:", e)
    else:
        print(">> [startup] Artifacts não encontrados, usando baseline.")


@app.on_event("startup")
def _on_startup():
    load_artifacts()


# ------------------------ Endpoints utilitários ------------------------

@app.get("/")
def root():
    return {"name": "Tech Fit API", "version": app.version}

@app.get("/metrics")
def metrics():
    """Endpoint simples para health check (HTTP 200)."""
    return {"status": "ok"}

@app.get("/readiness")
def readiness():
    """Indica se os artifacts foram carregados."""
    return {"model_loaded": MODEL is not None, "vector_loaded": VEC is not None}


# --------------------------- Predição ----------------------------------

class PredictIn(BaseModel):
    vaga_texto: str = Field(..., min_length=1)
    cv_texto: str = Field(..., min_length=1)
    threshold: float = Field(0.6, ge=0.0, le=1.0)


@app.post("/predict")
def predict(body: PredictIn):
    # baseline sempre disponível
    base = baseline_predict(body.vaga_texto, body.cv_texto, body.threshold)

    # Sem artifacts -> retorna baseline
    if MODEL is None or VEC is None:
        base["fonte"] = "baseline"
        return base

    # Monta features (vetor TF-IDF + similaridade cosseno)
    try:
        X, cos = build_features(VEC, body.vaga_texto, body.cv_texto)
    except Exception as e:
        # Se algo quebrar na montagem, devolve baseline (melhor que 500)
        base["fonte"] = "baseline"
        base["erro"] = f"falha_features: {e}"
        return base

    # Alinha número de colunas com o modelo (proteção)
    n_cols = X.shape[1]
    n_model = getattr(MODEL, "n_features_in_", n_cols)
    if n_cols != n_model:
        if n_cols > n_model:
            X = X[:, :n_model]  # corta excedente
        else:
            pad = np.zeros((X.shape[0], n_model - n_cols))
            X = np.hstack([X, pad])

    # Predição
    try:
        proba = float(MODEL.predict_proba(X)[0, 1])
    except AttributeError:
        # Modelos sem predict_proba (ex.: SVM linear sem prob)
        # Tenta decision_function como fallback
        df = float(MODEL.decision_function(X)[0])
        # Squash simples para [0,1] (opcional, só para não quebrar)
        proba = 1 / (1 + np.exp(-df))
    except Exception as e:
        base["fonte"] = "baseline"
        base["erro"] = f"falha_modelo: {e}"
        return base

    return {
        **base,
        "score_modelo": round(proba, 4),
        "similaridade_tfidf": round(float(cos), 4),
        "classificacao_modelo": "Atende" if proba >= body.threshold else "Não atende",
        "fonte": "modelo",
    }


# Permite rodar com: python api/main.py (além do CMD do Docker)
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)
