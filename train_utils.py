import pickle

def load_triples(file_path):
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            h, r, t = parts
            triples.append((str(h), str(r), str(t)))
    return triples
