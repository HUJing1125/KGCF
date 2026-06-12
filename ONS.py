# neighbor_sampler_batch.py
"""
Online Neighbor Sampling Module (Supports Batch Sampling)

- Load semantic vectors from HDF5 files

- Construct a graph from train.txt

- Implement semantic-structural co-sampling, supporting batch processing
"""

import h5py
import numpy as np
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InMemoryGraph:
    def __init__(self):
        self.outgoing: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        self.incoming: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        self.relations: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)

    def add_edge(self, src: str, dst: str, relation: str, direction: str = "head"):
        self.outgoing[src].append((dst, relation, direction))
        self.incoming[dst].append((src, relation, "tail" if direction == "head" else "head"))
        self.relations[(src, dst)].append((relation, direction))

    def get_neighbors(self, node: str, direction: str = "both") -> List[Tuple[str, str, str]]:
        neigh = []
        if direction in ('out', 'both'):
            neigh.extend(self.outgoing.get(node, []))
        if direction in ('in', 'both'):
            neigh.extend(self.incoming.get(node, []))
        return neigh

    def get_relations(self, node1: str, node2: str) -> List[Tuple[str, str]]:
        return self.relations.get((node1, node2), [])

    def get_all_nodes(self):
        return set(self.outgoing.keys()) | set(self.incoming.keys())


def load_embeddings_from_h5(h5_path: str) -> Dict[str, np.ndarray]:
    with h5py.File(h5_path, 'r') as f:
        ids = [x.decode('utf-8') for x in f['ids'][:]]
        embeddings = f['embeddings'][:]
    return {eid: embeddings[i] for i, eid in enumerate(ids)}

def build_graph_from_edges_file(edge_file: str) -> InMemoryGraph:
    """
    Read the `graph_edges.txt` file (4 columns: head, relation, tail, direction) and construct a directed graph.

    The direction field is either 'head' or 'tail', indicating the role of this edge in the original triple.

    Automatically add a reverse edge (inverting the direction) for each edge.。
    """
    graph = InMemoryGraph()
    with open(edge_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 4:
                continue
            head, rel, tail, direction = parts
            graph.add_edge(head, tail, rel, direction)
            inv_dir = 'tail' if direction == 'head' else 'head'
            graph.add_edge(tail, head, f"inv_{rel}", inv_dir)
    print(f"Built graph from {edge_file}: {len(graph.outgoing)} nodes, {sum(len(v) for v in graph.outgoing.values())} edges")
    return graph

def build_graph_from_triples(triple_path: str) -> InMemoryGraph:
    graph = InMemoryGraph()
    with open(triple_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            head, rel, tail = [p.strip() for p in parts]
            graph.add_edge(head, tail, rel, direction="head")
            graph.add_edge(tail, head, f"inv_{rel}", direction="tail")
    return graph


class BatchNeighborSampler:
    """
    Online neighbor sampler for batch sampling
    """
    def __init__(
        self,
        graph: InMemoryGraph,
        embeddings: Dict[str, np.ndarray],
        lambda_weight: float = 0.5,
        T: int = 50,
        walk_times: int = 20,
        walk_length: int = 5,
        cache_random_walk: bool = True
    ):
        self.graph = graph
        self.embeddings = embeddings
        self.lambda_weight = lambda_weight
        self.T = T
        self.walk_times = walk_times
        self.walk_length = walk_length
        self.cache_random_walk = cache_random_walk
        self._rw_cache: Dict[str, Dict[str, float]] = {}

    def _random_walk_importance(self, start_node: str) -> Dict[str, float]:
        if start_node in self._rw_cache and self.cache_random_walk:
            return self._rw_cache[start_node]
        if start_node not in self.graph.outgoing and start_node not in self.graph.incoming:
            return {}
        visit_count = defaultdict(int)
        for _ in range(self.walk_times):
            current = start_node
            for _ in range(self.walk_length):
                neighbors = self.graph.get_neighbors(current, direction='both')
                if not neighbors:
                    break
                next_node, _, _ = random.choice(neighbors)
                visit_count[next_node] += 1
                current = next_node
        total = sum(visit_count.values())
        if total == 0:
            importance = {}
        else:
            importance = {node: cnt / total for node, cnt in visit_count.items()}
        if self.cache_random_walk:
            self._rw_cache[start_node] = importance
        return importance

    def sample_single(self, target_node: str) -> List[Tuple[str, str, str]]:
        I_struct = self._random_walk_importance(target_node)
        candidates = list(I_struct.keys())
        if not candidates:
            return []

        target_emb = self.embeddings.get(target_node)
        if target_emb is None:
            sorted_candidates = sorted(candidates, key=lambda x: I_struct.get(x, 0), reverse=True)
            selected = sorted_candidates[:self.T]
        else:
            sim_scores = {}
            for v in candidates:
                v_emb = self.embeddings.get(v)
                if v_emb is not None:
                    sim_scores[v] = float(np.dot(target_emb, v_emb))
                else:
                    sim_scores[v] = 0.0
            combined = {}
            for v in candidates:
                combined[v] = self.lambda_weight * I_struct.get(v, 0.0) + (1 - self.lambda_weight) * sim_scores.get(v, 0.0)
            sorted_candidates = sorted(candidates, key=lambda x: combined[x], reverse=True)
            selected = sorted_candidates[:self.T]

        neighbors = []
        for v in selected:
            rels = self.graph.get_relations(target_node, v)
            if not rels:

                rev_rels = self.graph.get_relations(v, target_node)
                if rev_rels:
                    for r, d in rev_rels:
                        new_d = "tail" if d == "head" else "head"
                        neighbors.append((v, r, new_d))

                continue
            for r, d in rels:
                neighbors.append((v, r, d))
        seen = set()
        unique_neighbors = []
        for n, r, d in neighbors:
            if n not in seen:
                seen.add(n)
                unique_neighbors.append((n, r, d))
        return unique_neighbors[:self.T]

    def sample_batch(self, target_nodes: List[str]) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Batch sampling of multiple target nodes
        """
        results = {}
        for node in target_nodes:
            results[node] = self.sample_single(node)
        return results

    def precompute_importance_for_nodes(self, nodes: List[str]):
        for node in nodes:
            self._random_walk_importance(node)


if __name__ == "__main__":

    H5_PATH = "./semantic_embeddings/fb15k_embeddings.h5"
    TRIPLE_PATH = "./FB15k237/train.txt"



    embeddings_dict = load_embeddings_from_h5(H5_PATH)



    graph = build_graph_from_triples(TRIPLE_PATH)
    all_nodes = list(graph.get_all_nodes())

    sampler = BatchNeighborSampler(
        graph=graph,
        embeddings=embeddings_dict,
        lambda_weight=0.5,
        T=20,
        walk_times=10,
        walk_length=3,
        cache_random_walk=True
    )


    sample_nodes = all_nodes[:100]
    sampler.precompute_importance_for_nodes(sample_nodes)


    batch_results = sampler.sample_batch(sample_nodes)
    for node, neighs in list(batch_results.items())[:3]:
        print(f"\n node {node} sampling neighbor：")
        for n, r, d in neighs[:5]:
            print(f"  {n}  {r} ({d})")