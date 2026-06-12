# build_book_ckg.py (Modified)
import json
import random
from pathlib import Path
from collections import defaultdict

# ------------------------- Configuration Paths -------------------------
DATA_DIR = Path("./data")
BOOK_DIR = DATA_DIR / "book"                # Amazon book raw data directory
OUTPUT_DIR = DATA_DIR / "ckg_book"          # CKG output directory
MODEL_DIR = DATA_DIR / "btrainmodel"        # Model training directory

# Input files
AB2FB_FILE = BOOK_DIR / "ab2fb2.txt"
FB_SUBGRAPH_FILE = BOOK_DIR / "absubfb.txt"
RATINGS_FILE = BOOK_DIR / "Books.csv"
META_FILE = BOOK_DIR / "BooksMeta.json"
MIN_RATING = 3.0

# Output files
ENTITY_INFO_FILE = OUTPUT_DIR / "node_info.json"
EDGE_FILE = OUTPUT_DIR / "graph_edges.txt"
REL2ID_FILE = MODEL_DIR / "relation_to_id.json"
TRAIN_FILE = OUTPUT_DIR / "train.txt"
VALID_FILE = OUTPUT_DIR / "valid.txt"
TEST_FILE = OUTPUT_DIR / "test.txt"

# ------------------------- Helper Functions -------------------------
def parse_uri(uri: str) -> str:
    """Extract entity ID from Freebase URI (remove prefix and angle brackets)"""
    return uri.strip('<>').split('/')[-1]

def load_book_to_fb(path: Path):
    """
    Read ab2fb2.txt, format: book_asin, fb_entity_id, book_title
    Returns: {book_asin: fb_entity_id} and {book_asin: book_title}
    """
    mapping = {}
    titles = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 2)   # Split at most 2 times to get 3 parts
            if len(parts) != 3:
                continue
            asin, fb_ent, title = parts
            mapping[asin] = fb_ent.strip()
            titles[asin] = title.strip()
    print(f"Loaded {len(mapping)} book->FB mappings")
    return mapping, titles

def load_fb_triples_and_names(path: Path):
    """
    Read absubfb.txt, same format as mlsubfb.txt (tab-separated, may end with a dot)
    Returns: list of triples, name dictionary, description dictionary
    """
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
            # Extract object plain text (remove quotes and language tags)
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

def load_all_books_from_ratings(path: Path, min_rating):
    """
    Read all book asins from Books.csv (without rating threshold)
    Also return positive sample triples (for graph construction)
    """
    all_asins = set()
    ratings_pos = []  # Positive triples (user_id, book_asin, rating)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 4:
                continue
            asin, user, rating_str, ts = parts[0], parts[1], parts[2], parts[3]
            try:
                rating = float(rating_str)
            except:
                continue
            all_asins.add(asin)
            if rating >= min_rating:
                ratings_pos.append((f"user_{user}", f"book_{asin}", rating))
    print(f"Total unique books in ratings: {len(all_asins)}")
    print(f"Positive interactions (rating >= {min_rating}): {len(ratings_pos)}")
    return all_asins, ratings_pos

def load_book_metadata(path: Path):
    """
    Read BooksMeta.json (one JSON object per line)
    Returns: {asin: {'title': ..., 'description': ..., 'category': ...}}
    """
    meta = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except:
                continue
            asin = data.get('asin')
            if not asin:
                continue
            title = data.get('title', '')
            # description may be a list, take the first element
            desc_list = data.get('description', [])
            description = desc_list[0] if desc_list else ''
            category_list = data.get('category', [])
            category = ' > '.join(category_list) if category_list else ''
            meta[asin] = {'title': title, 'description': description, 'category': category}
    print(f"Loaded metadata for {len(meta)} books")
    return meta

def load_ratings_with_ts(path: Path, min_rating):
    """Load ratings with timestamps for temporal splitting"""
    ratings_ts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 4:
                continue
            asin, user, rating_str, ts = parts[0], parts[1], parts[2], parts[3]
            try:
                rating = float(rating_str)
                timestamp = int(ts)
            except:
                continue
            if rating >= min_rating:
                ratings_ts.append((f"user_{user}", f"book_{asin}", rating, timestamp))
    return ratings_ts

def build_ckg():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    book2fb, book_titles_from_fb = load_book_to_fb(AB2FB_FILE)  # Existing mapping
    fb_triples, fb_names, fb_descs = load_fb_triples_and_names(FB_SUBGRAPH_FILE)
    all_books_from_ratings, ratings_pos = load_all_books_from_ratings(RATINGS_FILE, MIN_RATING)
    book_meta = load_book_metadata(META_FILE)

    # 2. Collect user IDs
    user_ids = sorted({uid for uid, _, _ in ratings_pos})
    print(f"Total users: {len(user_ids)}")

    # 3. Collect all Freebase entities (from triples and mapping)
    fb_entities = set()
    for s, p, o in fb_triples:
        fb_entities.add(s)
        fb_entities.add(o)
    for fb_ent in book2fb.values():
        fb_entities.add(fb_ent)

    # 4. Build node info list node_info.json
    nodes = []

    # User nodes
    for uid in user_ids:
        nodes.append({"id": uid, "name": uid, "description": ""})

    # Book nodes: include all books appearing in ratings
    for asin in all_books_from_ratings:
        # Get Freebase entity ID (if mapping exists)
        fb_ent = book2fb.get(asin)
        # Title: prioritize metadata, then title from ab2fb, finally asin
        meta = book_meta.get(asin, {})
        title = meta.get('title') or book_titles_from_fb.get(asin, f"Book_{asin}")
        # Description: combine metadata description, Freebase description, and category
        desc_parts = []
        if meta.get('description'):
            desc_parts.append(meta['description'])
        if fb_ent and fb_descs.get(fb_ent):
            desc_parts.append(fb_descs[fb_ent])
        if meta.get('category'):
            desc_parts.append(f"Categories: {meta['category']}")
        desc = ". ".join(desc_parts) if desc_parts else f"Book {asin}"
        nodes.append({"id": f"book_{asin}", "name": title, "description": desc})

    # Freebase entity nodes
    for ent in fb_entities:
        name = fb_names.get(ent, ent)
        desc = fb_descs.get(ent, f"Freebase entity {ent}")
        nodes.append({"id": ent, "name": name, "description": desc})

    # Deduplicate
    unique_nodes = {node["id"]: node for node in nodes}.values()
    with open(ENTITY_INFO_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(unique_nodes), f, ensure_ascii=False, indent=2)
    print(f"Saved node_info.json with {len(unique_nodes)} nodes")

    # 5. Build edge list (including inverse edges)
    edges = []
    relations = set()

    # User-item interaction edges (using positive triples)
    for uid, mid, _ in ratings_pos:
        edges.append((uid, "interact", mid, "head"))
        edges.append((mid, "interact_inv", uid, "tail"))
        relations.add("interact")
        relations.add("interact_inv")

    # Item-entity link edges (only for books that have Freebase mapping)
    for asin, fb_ent in book2fb.items():
        book_node = f"book_{asin}"
        edges.append((book_node, "has_entity", fb_ent, "head"))
        edges.append((fb_ent, "belongs_to", book_node, "tail"))
        relations.add("has_entity")
        relations.add("belongs_to")

    # Freebase entity-entity edges (from subgraph)
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

    # 7. Split by timestamp into training/validation/test sets
    ratings_ts = load_ratings_with_ts(RATINGS_FILE, MIN_RATING)
    # Sort by timestamp
    ratings_ts.sort(key=lambda x: x[3])
    n = len(ratings_ts)
    train_end = int(0.8 * n)
    valid_end = int(0.9 * n)
    train = ratings_ts[:train_end]
    valid = ratings_ts[train_end:valid_end]
    test = ratings_ts[valid_end:]

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