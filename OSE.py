# offline_semantic_encoder.py
"""
- Reading node information from node_info.json

- Uses Sentence-BERT to encode text descriptions (description first, fallback to name)

- Saves in HDF5 format for online sampling and model training.
"""

import os
import json
import numpy as np
import h5py
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OfflineSemanticEncoder:


    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 128,
        normalize: bool = True,
        output_dir: str = "./semantic_embeddings"
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.output_dir = output_dir

        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initialize encoder: model={model_name}, device={self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded successfully, embedded dimensions: {self.embedding_dim}")

        os.makedirs(output_dir, exist_ok=True)
        self.embeddings: Dict[str, np.ndarray] = {}

    @staticmethod
    def load_nodes_from_json(json_path: str) -> List[Dict]:
        with open(json_path, 'r', encoding='utf-8') as f:
            nodes = json.load(f)
        logger.info(f" {json_path} load {len(nodes)} nodes")
        return nodes

    def encode_nodes_from_json(self, json_path: str, text_fallback: str = "name"):
        """
        Read nodes from a JSON file, encode them, and store them in an internal dictionary.


        """
        nodes = self.load_nodes_from_json(json_path)

        texts = []
        node_ids = []
        for node in nodes:
            node_id = str(node["id"])
            desc = node.get("description", "")
            name = node.get("name", "")
            # 优先使用 description，若缺失或为空则使用 name
            text = desc.strip() if desc and desc.strip() else name
            if not text:
                logger.warning(f"node {node_id} no description no name")
                text = ""
            texts.append(text)
            node_ids.append(node_id)

        # 批量编码
        logger.info(f"Start encoding {len(texts)} nodes.")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # 存储
        for nid, emb in zip(node_ids, embeddings):
            self.embeddings[nid] = emb

        logger.info(f"Encoding complete， {len(self.embeddings)} vectors")
        return self.embeddings

    def save(self, name: str = "entity_embeddings"):
        if not self.embeddings:
            logger.warning("No data")
            return

        node_ids = list(self.embeddings.keys())
        embeddings_array = np.array([self.embeddings[nid] for nid in node_ids])
        save_path = os.path.join(self.output_dir, f"{name}.h5")

        with h5py.File(save_path, "w") as hf:
            dt = h5py.special_dtype(vlen=str)
            hf.create_dataset("ids", data=node_ids, dtype=dt)
            hf.create_dataset("embeddings", data=embeddings_array, dtype='float32')

        metadata = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "total_nodes": len(self.embeddings),
            "normalized": self.normalize
        }
        with open(os.path.join(self.output_dir, f"{name}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Save {len(node_ids)} embedding to {save_path}")

    def load(self, name: str = "entity_embeddings"):
        load_path = os.path.join(self.output_dir, f"{name}.h5")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"File not exist: {load_path}")

        with h5py.File(load_path, "r") as hf:
            ids = [id_.decode('utf-8').strip() for id_ in hf["ids"][:]]
            embeddings = hf["embeddings"][:]

        self.embeddings = {nid: embeddings[i] for i, nid in enumerate(ids)}
        logger.info(f"Load {len(self.embeddings)} embeddings")
        return self.embeddings

    def get_embedding(self, node_id) -> Optional[np.ndarray]:
        return self.embeddings.get(str(node_id))

    def get_embeddings_batch(self, node_ids) -> np.ndarray:
        return np.array([self.embeddings[str(nid)] for nid in node_ids if str(nid) in self.embeddings])

    def cosine_similarity(self, node_id_1, node_id_2) -> float:
        emb1 = self.get_embedding(node_id_1)
        emb2 = self.get_embedding(node_id_2)
        if emb1 is None or emb2 is None:
            return 0.0
        return float(np.dot(emb1, emb2))


if __name__ == "__main__":
    encoder = OfflineSemanticEncoder(batch_size=64, output_dir="./semantic_embeddings")


    encoder.encode_nodes_from_json("./data/ckg_ml1m/node_info.json", text_fallback="name")

    encoder.save("ml_embeddings")

