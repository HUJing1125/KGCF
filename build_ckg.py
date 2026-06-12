import json
import random
from pathlib import Path
from collections import defaultdict

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
VALID_FILE = OUTPUT_DIR / "valid.txt"
TEST_FILE = OUTPUT_DIR / "test.txt"

MIN_RATING = 4  # Positive sample threshold

# ------------------------- Helper Functions -------------------------
def parse_uri(uri: str) -> str:
    return uri.strip('<>').split('/')[-1]

def load_movie_to_fb(path: Path):
    """Read ml2fb2.txt, return {movie_id: fb_entity_id} and {movie_id: (title, year)}"""
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
    """Read Freebase subgraph, return list of triples, name dictionary, description dictionary"""
    triples = []
    name_dict = {}
    desc_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith('.'):
                line = line[:-1]
            parts = line.split('\t')
            if len(parts) != 3:
                parts = line.split()
                if len(parts) >= 3:
                    parts = parts[:3]
                else:
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

def load_all_movies_from_ratings_with_ts(path: Path, min_rating):
    """
    Read all movie IDs that appear in ratings.dat (regardless of threshold),
    also return positive sample triples (with timestamps) for temporal splitting.
    """
    all_movies = set()
    ratings_pos_ts = []  # (user_id, movie_id, rating, timestamp)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            uid, mid, rating, ts = line.split('::')
            rating_int = int(rating)
            all_movies.add(mid)
            if rating_int >= min_rating:
                ratings_pos_ts.append((f"user_{uid}", f"movie_{mid}", rating_int, int(ts)))
    print(f"Total unique movies in ratings: {len(all_movies)}")
    print(f"Positive interactions (rating >= {min_rating}): {len(ratings_pos_ts)}")
    return all_movies, ratings_pos_ts

def load_movies_title(path: Path):
    """Read movies.dat, return {movie_id: title}"""
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
    """Read movies.dat, return {movie_id: genres} (comma separated)"""
    type_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('::', 2)
            if len(parts) != 3:
                continue
            mid, _, genres = parts
            genres = genres.replace('|', ',')
            type_dict[mid] = genres
    return type_dict

# ------------------------- Build CKG -------------------------
def build_ckg():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    movie2fb, movie_info = load_movie_to_fb(ML2FB_FILE)        # Mapping and auxiliary info (title, year)
    fb_triples, fb_names, fb_desc = load_fb_triples_and_names(FB_SUBGRAPH_FILE)
    all_movies, ratings_pos_ts = load_all_movies_from_ratings_with_ts(RATINGS_FILE, MIN_RATING)
    movie_titles = load_movies_title(MOVIES_FILE)
    movie_types = load_movies_type(MOVIES_FILE)

    # 2. Collect user IDs (based on positive samples)
    user_ids = sorted({uid for uid, _, _, _ in ratings_pos_ts})
    print(f"Total users: {len(user_ids)}")

    # 3. Collect all Freebase entities
    fb_entities = set()
    for s, p, o in fb_triples:
        fb_entities.add(s)
        fb_entities.add(o)
    for fb_ent in movie2fb.values():
        fb_entities.add(fb_ent)

    # 4. Build node info list node_info.json
    nodes = []

    # User nodes
    for uid in user_ids:
        nodes.append({"id": uid, "name": uid, "description": ""})

    # Movie nodes: include all movies appearing in ratings
    for mid in all_movies:
        # Get title (prefer from ml2fb2.txt, then from movies.dat)
        title = movie_info.get(mid, ('', ''))[0]
        if not title:
            title = movie_titles.get(mid)
        if not title:
            title = f"Movie_{mid}"

        # Get genres and year
        genres = movie_types.get(mid, "")
        year = movie_info.get(mid, ('', ''))[1]

        # Build description: year + genres, append Freebase description if available
        desc_parts = []
        if year:
            desc_parts.append(f"Year: {year}")
        if genres:
            desc_parts.append(f"Genres: {genres}")

        # If there is a Freebase mapping, try to add Freebase description
        fb_ent = movie2fb.get(mid)
        if fb_ent and fb_desc.get(fb_ent):
            desc_parts.append(fb_desc[fb_ent])

        desc = ". ".join(desc_parts) if desc_parts else f"Movie {mid}"
        nodes.append({"id": f"movie_{mid}", "name": title, "description": desc})

    # Freebase entity nodes
    fb_ent_to_movie_title = {}
    for mid, fb_ent in movie2fb.items():
        title = movie_titles.get(mid) or movie_info.get(mid, ('', ''))[0]
        if title:
            fb_ent_to_movie_title[fb_ent] = title

    for ent in fb_entities:
        if ent in fb_ent_to_movie_title:
            name = fb_ent_to_movie_title[ent]
        else:
            name = fb_names.get(ent, ent)
        desc = fb_desc.get(ent, f"Freebase entity {ent}")
        nodes.append({"id": ent, "name": name, "description": desc})

    # Deduplicate
    unique_nodes = {node["id"]: node for node in nodes}.values()
    with open(ENTITY_INFO_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(unique_nodes), f, ensure_ascii=False, indent=2)
    print(f"Saved node_info.json with {len(unique_nodes)} nodes")

    # 5. Build edge list (including inverse edges)
    edges = []
    relations = set()

    # User-item interaction edges (using positive samples, no timestamp needed)
    for uid, mid, _, _ in ratings_pos_ts:
        movie_node = f"movie_{mid}"
        edges.append((uid, "interact", movie_node, "head"))
        edges.append((movie_node, "interact_inv", uid, "tail"))
        relations.add("interact")
        relations.add("interact_inv")

    # Item-entity link edges (only for movies that have Freebase mapping)
    for mid, fb_ent in movie2fb.items():
        movie_node = f"movie_{mid}"
        edges.append((movie_node, "has_entity", fb_ent, "head"))
        edges.append((fb_ent, "belongs_to", movie_node, "tail"))
        relations.add("has_entity")
        relations.add("belongs_to")

    # Freebase entity-entity edges
    for s, p, o in fb_triples:
        edges.append((s, p, o, "head"))
        edges.append((o, f"inv_{p}", s, "tail"))
        relations.add(p)
        relations.add(f"inv_{p}")

    with open(EDGE_FILE, 'w', encoding='utf-8') as f:
        for h, r, t, d in edges:
            f.write(f"{h}\t{r}\t{t}\t{d}\n")
    print(f"Saved graph_edges.txt with {len(edges)} edges")

    # 6. Save relation mapping
    rel_list = sorted(relations)
    rel2id = {rel: i for i, rel in enumerate(rel_list)}
    with open(REL2ID_FILE, 'w') as f:
        json.dump(rel2id, f, indent=2)
    print(f"Saved relation_to_id.json with {len(rel2id)} relations")

    # 7. Split by timestamp into train/validation/test sets
    # Sort by timestamp ascending
    ratings_pos_ts.sort(key=lambda x: x[3])
    n = len(ratings_pos_ts)
    train = ratings_pos_ts[:int(0.8*n)]
    valid = ratings_pos_ts[int(0.8*n):int(0.9*n)]
    test = ratings_pos_ts[int(0.9*n):]

    def save_triples(triples, filename):
        with open(OUTPUT_DIR / filename, 'w') as f:
            for uid, mid, _, _ in triples:
                f.write(f"{uid}\tinteract\t{mid}\n")

    save_triples(train, "train.txt")
    save_triples(valid, "valid.txt")
    save_triples(test, "test.txt")
    print(f"Saved train/valid/test: {len(train)}/{len(valid)}/{len(test)}")

if __name__ == "__main__":
    build_ckg()