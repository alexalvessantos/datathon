import os, json, re, unicodedata, traceback
import pandas as pd
from src.scoring import extract_skills

DATA_DIR = "data"
PATH_VAGAS = os.path.join(DATA_DIR, "vagas.json")
PATH_APPLICANTS = os.path.join(DATA_DIR, "applicants.json")
PATH_PROSPECTS = os.path.join(DATA_DIR, "prospects.json")
OUT_CSV = os.path.join(DATA_DIR, "train_pairs.csv")

def normalize(text: str) -> str:
    if not text: return ""
    t = unicodedata.normalize("NFKD", text).encode("ascii","ignore").decode("utf-8").lower()
    t = re.sub(r"[^a-z0-9#+ ]", " ", t)
    return re.sub(r"\s+"," ", t).strip()

def concat_text_fields(obj) -> str:
    parts = []
    def walk(x):
        if isinstance(x, str) and x.strip(): parts.append(x.strip())
        elif isinstance(x, (int,float)): parts.append(str(x))
        elif isinstance(x, list):
            for y in x: walk(y)
        elif isinstance(x, dict):
            for _,y in x.items(): walk(y)
    walk(obj)
    return normalize(" ".join(parts))

def read_json_dict(path):
    with open(path,"r",encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), f"{os.path.basename(path)} deve ser dict no topo"
    return data

def map_vagas_text():
    data = read_json_dict(PATH_VAGAS)
    print(f"[DEBUG] vagas.json itens: {len(data)}")
    return {str(job_id): concat_text_fields(rec) for job_id, rec in data.items()}

def map_applicants_text():
    data = read_json_dict(PATH_APPLICANTS)
    print(f"[DEBUG] applicants.json itens: {len(data)}")
    out = {}
    miss = 0
    for key_top, rec in data.items():
        cand_id = rec.get("infos_basicas", {}).get("codigo_profissional") or key_top
        if not rec.get("infos_basicas", {}).get("codigo_profissional"): miss += 1
        out[str(cand_id)] = concat_text_fields(rec)
    if miss:
        print(f"[DEBUG] applicants sem codigo_profissional: {miss} (usei chave do topo)")
    return out

def iter_prospects_links():
    data = read_json_dict(PATH_PROSPECTS)
    print(f"[DEBUG] prospects.json itens: {len(data)}")
    total = 0
    for job_id, obj in data.items():
        lst = obj.get("prospects", [])
        if not isinstance(lst, list): continue
        for p in lst:
            cand_id = p.get("codigo")
            if cand_id:
                total += 1
                yield str(job_id), str(cand_id), p
    print(f"[DEBUG] links encontrados em prospects: {total}")

def label_from_status(p: dict):
    st = (p.get("situacao_candidato") or p.get("status") or p.get("situacao") or "").strip().lower()
    if any(x in st for x in ["contrat","aprov","hired","finalista","selecion"]): return 1
    if any(x in st for x in ["reprov","descart","negado","não aprovado","nao aprovado"]): return 0
    return None

def main():
    print("[DEBUG] build_dataset.py INICIADO")
    vaga_text = map_vagas_text()
    cand_text = map_applicants_text()
    print(f"[DEBUG] mapeados: vagas={len(vaga_text)} | candidatos={len(cand_text)}")

    # cache de skills
    print("[DEBUG] pré-calculando skills...")
    vaga_sk = {jid: extract_skills(vt) for jid, vt in vaga_text.items()}
    cand_sk = {cid: extract_skills(ct) for cid, ct in cand_text.items()}

    rows = []
    pos = neg = count = 0
    for job_id, cand_id, link in iter_prospects_links():
        vt = vaga_text.get(job_id, ""); ct = cand_text.get(cand_id, "")
        if not vt or not ct: 
            continue

        label = label_from_status(link)
        if label is None:
            # rótulo fraco a partir de skills
            req = vaga_sk.get(job_id, set())
            have = cand_sk.get(cand_id, set())
            ratio = (len(req & have) / len(req)) if req else 0.0
            label = 1 if ratio >= 0.6 else 0
            label_source = "weak"
        else:
            label_source = "status"

        pos += (label==1); neg += (label==0)
        rows.append({
            "job_id": job_id,
            "cand_id": cand_id,
            "vaga_texto": vt,
            "cv_texto": ct,
            "label": label,
            "label_source": label_source
        })
        count += 1
        if count % 5000 == 0:
            print(f"[DEBUG] pares gerados: {count} (1s={pos} | 0s={neg})")

    if not rows:
        print("[WARN] Sem pares via prospects. Gerando fallback (10x30)...")
        from itertools import islice
        vagas_list = list(islice(vaga_text.values(), 10))
        cands_list = list(islice(cand_text.values(), 30))
        for vt in vagas_list:
            for ct in cands_list:
                req, have = extract_skills(vt), extract_skills(ct)
                ratio = (len(req & have) / len(req)) if req else 0.0
                label = 1 if ratio >= 0.6 else 0
                rows.append({"vaga_texto": vt, "cv_texto": ct, "label": label})

    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(OUT_CSV, index=False, encoding="utf-8")
        print(f"[OK] Salvo: {OUT_CSV} | linhas={len(df)} | labels={df['label'].value_counts().to_dict()}")
    else:
        print("[ERRO] Ainda zero linhas.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
