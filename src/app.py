from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_config
from src.dataset.temporal_dataset import build_temporal_graph_dataset
from src.graph.graph_compose import concat_edges, compute_z_from_edges
from src.inference.blockwise_scoring import blockwise_topk_dot_product
from src.quantization.dynamic_int8 import prepare_model_for_dynamic_int8
from src.training.runner import init_models
from src.dataset.movie_metadata import load_ml100k_movie_metadata

def _parse_movie_id(raw_item_id: object) -> int | None:
    if raw_item_id is None:
        return None

    text = str(raw_item_id)

    if text.startswith("i_"):
        text = text[2:]

    try:
        return int(text)
    except ValueError:
        return None


@dataclass
class ServingState:
    loaded: bool = False
    device: torch.device = torch.device("cpu")
    num_items: int = 0
    item_offset: int = 0
    item_embeddings_small: torch.Tensor | None = None
    user_embeddings_small: torch.Tensor | None = None
    item_embeddings_big: torch.Tensor | None = None
    user_embeddings_big: torch.Tensor | None = None

    item_node_to_raw_id: dict[int, str] | None = None
    movie_metadata: dict[str, dict[str, Any]] | None = None


class LoadRequest(BaseModel):
    checkpoint_path: str = Field(..., description="Path to model checkpoint (.ckpt/.pt)")
    config_path: str = Field(..., description="Path to model yaml config")
    quantize_int8: bool = Field(default=False, description="Apply dynamic int8 quantization to encoder/compressor")


class RecommendRequest(BaseModel):
    user_id: int
    k: int = Field(default=20, ge=1)
    item_block_size: int = Field(default=1024, ge=1)
    branch: str = Field(default="small")


class RecommenderService:
    def __init__(self) -> None:
        self.state = ServingState()

    def load(self, checkpoint_path: str, config_path: str, quantize_int8: bool = False) -> dict[str, Any]:
        cfg = load_config(config_path)
        dataset = build_temporal_graph_dataset(cfg)

        raw_item_to_internal = dataset.item_map
        internal_item_to_raw = {
            internal_id: raw_item_id
            for raw_item_id, internal_id in raw_item_to_internal.items()
        }

        item_node_to_raw_id = {
            dataset.item_offset + internal_id: raw_item_id
            for internal_id, raw_item_id in internal_item_to_raw.items()
        }

        movie_metadata = load_ml100k_movie_metadata("data/u.item")

        node_emb, encoder, compressor = init_models(cfg, dataset.num_nodes)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        node_emb.load_state_dict({k.replace("node_emb.", "", 1): v for k, v in state_dict.items() if k.startswith("node_emb.")})
        encoder.load_state_dict({k.replace("encoder.", "", 1): v for k, v in state_dict.items() if k.startswith("encoder.")})
        compressor.load_state_dict({k.replace("compressor.", "", 1): v for k, v in state_dict.items() if k.startswith("compressor.")})

        if quantize_int8:
            encoder = prepare_model_for_dynamic_int8(encoder, inplace=False)
            compressor = prepare_model_for_dynamic_int8(compressor, inplace=False)

        node_emb.eval()
        encoder.eval()
        compressor.eval()

        edge_list = []
        for sid in sorted(dataset.mp_by_sid.keys()):
            df = dataset.mp_by_sid[sid]
            src = torch.tensor(df["from"].to_numpy(), dtype=torch.long)
            dst = torch.tensor(df["to"].to_numpy(), dtype=torch.long)
            edge_list.append((src, dst))

        edge_src, edge_dst = concat_edges(edge_list)

        with torch.no_grad():
            z_big, z_small = compute_z_from_edges(
                edge_src=edge_src,
                edge_dst=edge_dst,
                num_nodes=dataset.num_nodes,
                encoder=encoder,
                compressor=compressor,
                node_emb=node_emb,
                device=torch.device("cpu"),
            )

        item_ids_global = torch.arange(
            dataset.item_offset,
            dataset.item_offset + dataset.num_items,
            dtype=torch.long,
        )
        user_ids_global = torch.arange(dataset.item_offset, dtype=torch.long)

        self.state.item_embeddings_small = z_small[item_ids_global].contiguous().cpu()
        self.state.user_embeddings_small = z_small[user_ids_global].contiguous().cpu()

        self.state.item_embeddings_big = z_big[item_ids_global].contiguous().cpu()
        self.state.user_embeddings_big = z_big[user_ids_global].contiguous().cpu()
        self.state.num_items = int(dataset.num_items)
        self.state.item_offset = int(dataset.item_offset)

        self.state.item_node_to_raw_id = item_node_to_raw_id
        self.state.movie_metadata = movie_metadata

        self.state.loaded = True

        return {
            "status": "ok",
            "num_items": self.state.num_items,
            "num_users": int(user_ids_global.numel()),
            "quantize_int8": quantize_int8,
            "checkpoint": str(Path(checkpoint_path)),
            "config": str(Path(config_path)),
        }

    def recommend(
            self,
            user_id: int,
            k: int = 20,
            item_block_size: int = 1024,
            branch: str = "small",
    ) -> dict[str, Any]:
        if not self.state.loaded:
            raise HTTPException(status_code=400, detail="Model is not loaded. Call /load first.")

        branch = branch.lower().strip()

        if branch == "small":
            user_embeddings = self.state.user_embeddings_small
            item_embeddings = self.state.item_embeddings_small
        elif branch == "big":
            user_embeddings = self.state.user_embeddings_big
            item_embeddings = self.state.item_embeddings_big
        else:
            raise HTTPException(status_code=400, detail="Unknown branch. Use 'small' or 'big'.")

        if user_embeddings is None or item_embeddings is None:
            raise HTTPException(status_code=400, detail="Model embeddings are not loaded.")

        if user_id < 0 or user_id >= user_embeddings.shape[0]:
            raise HTTPException(status_code=404, detail=f"Unknown user_id={user_id}")

        k = min(k, self.state.num_items)

        user_vec = user_embeddings[user_id: user_id + 1]

        scores, indices = blockwise_topk_dot_product(
            user_embeddings=user_vec,
            item_embeddings=item_embeddings,
            k=k,
            item_block_size=item_block_size,
            largest=True,
            sorted=True,
        )

        rec_item_ids = (indices[0] + self.state.item_offset).tolist()
        rec_scores = scores[0].tolist()

        recommendations = []

        for item_id, score in zip(rec_item_ids, rec_scores):
            raw_movie_id = None
            movie_id = None
            movie_info = {}

            if self.state.item_node_to_raw_id is not None:
                raw_movie_id = self.state.item_node_to_raw_id.get(int(item_id))
                movie_id = _parse_movie_id(raw_movie_id)

            if (
                    movie_id is not None
                    and self.state.movie_metadata is not None
            ):
                movie_info = self.state.movie_metadata.get(str(movie_id), {})

            recommendations.append(
                {
                    "item_id": int(item_id),
                    "raw_item_id": raw_movie_id,
                    "movie_id": movie_id,
                    "title": movie_info.get("title"),
                    "release_date": movie_info.get("release_date"),
                    "genres": movie_info.get("genres", []),
                    "imdb_url": movie_info.get("imdb_url"),
                    "score": float(score),
                }
            )

        return {
            "user_id": user_id,
            "k": k,
            "branch": branch,
            "recommendations": recommendations,
        }


service = RecommenderService()
app = FastAPI(title="Dynamic GNN Recsys API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": service.state.loaded}


@app.post("/load")
def load_model(payload: LoadRequest) -> dict[str, Any]:
    return service.load(
        checkpoint_path=payload.checkpoint_path,
        config_path=payload.config_path,
        quantize_int8=payload.quantize_int8,
    )


@app.post("/recommend")
def recommend(payload: RecommendRequest) -> dict[str, Any]:
    return service.recommend(
        user_id=payload.user_id,
        k=payload.k,
        item_block_size=payload.item_block_size,
        branch=payload.branch,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FastAPI app for recommender inference")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("src.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
