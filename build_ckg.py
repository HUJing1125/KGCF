import json
import random
from pathlib import Path



DATA_DIR = Path("./data")
ML_DIR = DATA_DIR / "ml-1m"
OUTPUT_DIR = DATA_DIR / "ckg_ml1m"
MODEL_DIR = DATA_DIR / "mtrainmodel"

ML2FB_FILE = ML_DIR / "ml2fb2.txt"
FB_SUBGRAPH_FILE = ML_DIR / "mlsubfb.txt"
RATINGS_FILE = ML_DIR / "ratings.dat"
MOVIES_FILE = ML_DIR / "movies.dat"

ENTITY_INFO_FILE = OUTPUT_DIR / "node_info.json"
EDGE_FILE = OUTPUT_DIR / "graph_edges.txt"
REL2ID_FILE = MODEL_DIR / "relation_to_id.json"
TRAIN_FILE = OUTPUT_DIR / "train.txt"
VALID_FILE = OUTPUT_DIR / "train.txt"
TEST_FILE = OUTPUT_DIR / "test.txt"


def parse_uri(uri: str) -> str:
    return uri.strip('<>').split('/')[-1]

def load_movie_to_fb(path: Path):
    mapping = {}
    info = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 3)
            if len(parts) != 4:
                continue
            movie_id, fb_ent, title, year = parts
            mapping[movie_id] = fb_ent.strip()
            info[movie_id] = (title.strip(), year.strip())
    print(f"Loaded {len(mapping)} movie->FB mappings")
    return mapping, info

def load_fb_triples_and_names(path: Path):

    triples = []
    name_dict = {}
    desc_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 去掉末尾的句点
            if line.endswith('.'):
                line = line[:-1]
            parts = line.split('\t')
            if len(parts) != 3:
                parts = line.split()
                if len(parts) >= 3:
                    parts = parts[:3]
                else:
                    print(f"Skipping malformed line: {line[:100]}")
                    continue
            s, p, o = parts
            s_id = parse_uri(s)
            p_id = parse_uri(p)
            if o.startswith('"'):
                o_text = o.split('"')[1] if '"' in o else o
            else:
                o_text = o
            if p_id == 'type.object.name':
                name_dict[s_id] = o_text
            elif p_id == 'common.topic.description':
                desc_dict[s_id] = o_text
            else:
                triples.append((s_id, p_id, parse_uri(o)))
    print(f"Loaded {len(triples)} FB triples, {len(name_dict)} names, {len(desc_dict)} descriptions")
    return triples, name_dict, desc_dict

def load_ratings(path: Path, min_rating=4):
    ratings = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            uid, mid, rating, _ = line.split('::')
            if int(rating) >= min_rating:
                ratings.append((f"user_{uid}", f"movie_{mid}", int(rating)))
    print(f"Loaded {len(ratings)} positive interactions")
    return ratings

def load_movies_title(path: Path):
    title_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            mid, title, _ = line.split('::', 2)
            title_dict[mid] = title
    return title_dict

def load_movies_type(path: Path):
    type_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('::', 2)   # 最多分割2次，得到3部分：id, title, genres
            if len(parts) != 3:
                continue
            mid, _, genres = parts
            genres = genres.replace('|', ',')
            type_dict[mid] = genres
    return type_dict

# ------------------------- 构建 CKG -------------------------
def build_ckg():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


    movie2fb, movie_info = load_movie_to_fb(ML2FB_FILE)  # movie_info: {movie_id: (title, year)}
    fb_triples, fb_names, fb_desc = load_fb_triples_and_names(FB_SUBGRAPH_FILE)
    ratings = load_ratings(RATINGS_FILE, min_rating=4)
    movie_titles = load_movies_title(MOVIES_FILE)
    movie_types = load_movies_type(MOVIES_FILE)

    fb_ent_to_movie_title = {}
    for movie_id, fb_ent in movie2fb.items():
        title = movie_info.get(movie_id, ('', ''))[0]
        if not title:
            title = movie_titles.get(movie_id, f"Movie_{movie_id}")
        fb_ent_to_movie_title[fb_ent] = title


    user_ids = sorted({uid for uid, _, _ in ratings})
    print(f"Total users: {len(user_ids)}")


    fb_entities = set()
    for s, p, o in fb_triples:
        fb_entities.add(s)
        fb_entities.add(o)
    for fb_ent in movie2fb.values():
        fb_entities.add(fb_ent)


    nodes = []

    for uid in user_ids:
        nodes.append({"id": uid, "name": uid, "description": ""})


    for movie_id, fb_ent in movie2fb.items():

        title = movie_info.get(movie_id, ('', ''))[0]
        if not title:
            title = movie_titles.get(movie_id, f"Movie_{movie_id}")

        genres = movie_types.get(movie_id, "")
        year = movie_info.get(movie_id, ('', ''))[1]
        desc_parts = []
        if year:
            desc_parts.append(f"Year: {year}")
        if genres:
            desc_parts.append(f"Genres: {genres}")
        desc = ", ".join(desc_parts) if desc_parts else f"Movie {movie_id}"
        nodes.append({"id": f"movie_{movie_id}", "name": title, "description": desc})


    for ent in fb_entities:
        if ent in fb_ent_to_movie_title:
            name = fb_ent_to_movie_title[ent]
        else:
            name = fb_names.get(ent, ent)
        desc = fb_desc.get(ent, f"Freebase entity {ent}")
        nodes.append({"id": ent, "name": name, "description": desc})


    unique_nodes = {node["id"]: node for node in nodes}.values()
    with open(ENTITY_INFO_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(unique_nodes), f, ensure_ascii=False, indent=2)
    print(f"Saved entity_info.json with {len(unique_nodes)} nodes")


    edges = []
    relations = set()


    for uid, mid, _ in ratings:
        movie_node = f"movie_{mid}"
        edges.append((uid, "interact", movie_node, "head"))
        edges.append((movie_node, "interact_inv", uid, "tail"))
        relations.add("interact")
        relations.add("interact_inv")


    for movie_id, fb_ent in movie2fb.items():
        movie_node = f"movie_{movie_id}"
        edges.append((movie_node, "has_entity", fb_ent, "head"))
        edges.append((fb_ent, "belongs_to", movie_node, "tail"))
        relations.add("has_entity")
        relations.add("belongs_to")


    for s, p, o in fb_triples:
        edges.append((s, p, o, "head"))
        edges.append((o, f"inv_{p}", s, "tail"))
        relations.add(p)
        relations.add(f"inv_{p}")

    with open(EDGE_FILE, 'w', encoding='utf-8') as f:
        for h, r, t, d in edges:
            f.write(f"{h}\t{r}\t{t}\t{d}\n")
    print(f"Saved graph_edges.txt with {len(edges)} edges")


    rel_list = sorted(relations)
    rel2id = {rel: i for i, rel in enumerate(rel_list)}
    with open(REL2ID_FILE, 'w') as f:
        json.dump(rel2id, f, indent=2)
    print(f"Saved relation_to_id.json with {len(rel2id)} relations")


    random.shuffle(ratings)
    n = len(ratings)
    train = ratings[:int(0.8*n)]
    valid = ratings[int(0.8*n):int(0.9*n)]
    test = ratings[int(0.9*n):]

    def save_triples(triples, filename):
        with open(OUTPUT_DIR / filename, 'w') as f:
            for uid, mid, _ in triples:
                f.write(f"{uid}\tinteract\t{mid}\n")

    save_triples(train, "train.txt")
    save_triples(valid, "valid.txt")
    save_triples(test, "test.txt")
    print(f"Saved train/valid/test: {len(train)}/{len(valid)}/{len(test)}")

if __name__ == "__main__":
    build_ckg()