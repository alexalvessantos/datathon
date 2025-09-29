import json, os, itertools

DATA_DIR = "data"
FILES = ["vagas.json", "applicants.json", "prospects.json"]

def keys_of(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield prefix + k
            yield from keys_of(v, prefix + k + ".")
    elif isinstance(obj, list) and obj:
        yield from keys_of(obj[0], prefix)

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    return data if isinstance(data, list) else []

for fn in FILES:
    p = os.path.join(DATA_DIR, fn)
    data = load(p)
    print(f"\n== {fn} ==")
    print("itens:", len(data))
    if data:
        print("exemplo:", data[0])
        ks = sorted(set(list(keys_of(data[0]))))
        print("chaves topo/ninho (amostra):")
        for k in ks:
            print(" -", k)
