Tech Fit API – Match de Vagas e Currículos

API para calcular a compatibilidade entre o texto de uma vaga e o texto de um currículo.
Retorna um score baseline (regras simples) e, quando os artefatos treinados estão presentes, também o score do modelo.

Arquivos grandes (dados e artefatos) não são versionados no Git.
Baixe-os na aba Releases do repositório e coloque nas pastas indicadas abaixo.

Visão geral

Problema: priorizar candidatos para vagas com base em similaridade textual e sinais de habilidade.

Solução: pipeline de ML (limpeza → TF-IDF → treino → avaliação) + API em FastAPI.

Tecnologias: Python 3.11, scikit-learn, numpy/pandas, FastAPI/Uvicorn, Docker.
Deploy sugerido: AWS ECR + AWS App Runner.

Estrutura do projeto
.
api/
  main.py                # API (endpoints) e carga dos artefatos 
src/
  features.py            # TF-IDF e features auxiliares
  scoring.py             # baseline de matching
  text_normalizer.py     # limpeza/normalização de texto
  train.py               # treino e serialização do modelo
tests/
  test_api.py
  test_scoring.py
artifacts/                # modelo e vetorizar (baixar pelas Releases)
data/                     # dados brutos/curados (baixar pelas Releases)
build_dataset.py          # (opcional) gera pares de treino
requirements.txt
Dockerfile
README.md


Os arquivos grandes estão no .gitignore.

Pré-requisitos

Python 3.11+
Docker 24+
AWS CLI configurado e permissões para ECR/App Runner

Dados e artefatos (baixar pelas Releases)

Crie as pastas e salve os arquivos:

data/applicants.json
data/train_pairs.csv           
artifacts/model.joblib
artifacts/tfidf.joblib


Os links estão na aba Releases do repositório.
Sem esses arquivos, a API funciona apenas com o baseline (campo "fonte": "baseline").

Como rodar localmente
Ambiente e dependências

Windows (PowerShell)

python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt

5.2 API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Swagger: http://127.0.0.1:8000/docs

OpenAPI: http://127.0.0.1:8000/openapi.json

Health: GET /metrics (200 OK)

Exemplos de uso
POST /predict

JSON

{
  "vaga_texto": "Exige Python, SQL e AWS.",
  "cv_texto": "Experiência com Python e AWS; Docker.",
  "threshold": 0.6
}

curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"vaga_texto":"Exige Python, SQL e AWS.","cv_texto":"Experiência com Python e AWS; Docker.","threshold":0.6}'


Python (requests)

import requests
payload = {
  "vaga_texto": "Exige Python, SQL e AWS.",
  "cv_texto": "Experiência com Python e AWS; Docker.",
  "threshold": 0.6
}
r = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=30)
print(r.json())


Resposta típica

{
  "score_baseline": 0.6667,
  "classificacao_baseline": "Atende",
  "skills_vaga": ["aws", "python", "sql"],
  "skills_cv": ["aws", "docker", "python"],
  "skills_match": ["aws", "python"],
  "threshold": 0.6,
  "score_modelo": 0.524,
  "similaridade_tfidf": 0.3745,
  "classificacao_modelo": "Não atende",
  "fonte": "modelo"
}

Pipeline de ML

Limpeza de texto (text_normalizer.py)
Lowercase, remoção de pontuação/stopwords etc.

Features (features.py)
TF-IDF + similaridade e sinais simples.

Treino (train.py)
Classificador scikit-learn, F1-score como métrica principal.

Serialização
artifacts/model.joblib e artifacts/tfidf.joblib.

Comandos úteis

Gerar pares
python build_dataset.py

Treinar e salvar em artifacts/
python src/train.py

Testes rápidos
pytest -q

Docker
Build
docker build -t techfit-api:latest .

Run
docker run --rm -p 8000:8000 techfit-api:latest
http://127.0.0.1:8000/docs


A aplicação escuta na porta 8000 (padrão) e aceita PORT via variável de ambiente.
Health check: HTTP em /metrics.

Deploy na AWS (ECR + App Runner)
9.1 Enviar a imagem ao ECR
Ajuste estes valores
REGION=us-east-1
REPO=techfit-api
ACCOUNT_ID=Minha conta pessoal

Login no ECR
aws ecr get-login-password --region $REGION \
docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

Criar o repositório
aws ecr create-repository --repository-name $REPO --region $REGION

Tag + push
docker tag techfit-api:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO:latest

9.2 App Runner (console)

Source: Container registry → Amazon ECR → selecione o repositório e a tag latest.

Port: 8000

Health check: HTTP em /metrics, timeout 5s, interval 10s

ECR access role: crie/selecione uma role com permissões de leitura no ECR.

Problemas comuns

422/JSON inválido: revise os nomes dos campos (vaga_texto, cv_texto, threshold).

Só baseline aparecendo: o projeto não encontrou artifacts/model.joblib e artifacts/tfidf.joblib.

Health check falhando: porta/host/rota incorretos ou role do App Runner sem permissão no ECR.

Autoria

Projeto acadêmico para fins de demonstração técnica.

Autor: Alex Santos!
