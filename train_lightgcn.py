import argparse
import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


RATING_THRESHOLD = 3.5


def parse_genres(value):
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    return [
        genre["name"]
        for genre in parsed
        if isinstance(genre, dict) and isinstance(genre.get("name"), str)
    ]


def load_joined_ratings(data_dir):
    data_dir = Path(data_dir)
    ratings = pd.read_csv(data_dir / "ratings_small.csv")
    links = pd.read_csv(data_dir / "links_small.csv")
    metadata = pd.read_csv(data_dir / "movies_metadata.csv", low_memory=False)

    metadata["tmdbId"] = pd.to_numeric(metadata["id"], errors="coerce")
    invalid_metadata_ids = int(metadata["tmdbId"].isna().sum())
    metadata = metadata.dropna(subset=["tmdbId"]).copy()
    metadata["tmdbId"] = metadata["tmdbId"].astype("int64")
    duplicate_metadata_ids = int(metadata["tmdbId"].duplicated().sum())
    metadata = metadata.drop_duplicates(subset=["tmdbId"], keep="first")
    metadata = metadata[
        ["tmdbId", "title", "genres", "overview", "vote_average"]
    ]

    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
    missing_tmdb_links = int(links["tmdbId"].isna().sum())
    links = links.dropna(subset=["movieId", "tmdbId"]).copy()
    links["movieId"] = links["movieId"].astype("int64")
    links["tmdbId"] = links["tmdbId"].astype("int64")
    if links["movieId"].duplicated().any():
        raise ValueError("links_small.csv contains duplicate movieId values")

    movie_info = links.merge(
        metadata,
        on="tmdbId",
        how="left",
        validate="many_to_one",
        indicator="_metadata_merge",
    )
    movie_info["metadata_matched"] = movie_info["_metadata_merge"].eq("both")
    movie_info = movie_info.drop(columns=["_metadata_merge"])

    joined = ratings.merge(
        movie_info[
            [
                "movieId",
                "tmdbId",
                "title",
                "genres",
                "overview",
                "vote_average",
                "metadata_matched",
            ]
        ],
        on="movieId",
        how="left",
        validate="many_to_one",
    )
    joined["metadata_matched"] = joined["metadata_matched"].eq(True)
    joined["genre_list"] = joined["genres"].apply(parse_genres).apply(
        lambda genres: json.dumps(genres, ensure_ascii=False)
    )
    joined = joined.drop(columns=["genres"])

    stats = {
        "ratings_rows": int(len(ratings)),
        "ratings_with_metadata": int(joined["metadata_matched"].sum()),
        "ratings_without_metadata": int((~joined["metadata_matched"]).sum()),
        "missing_tmdb_links": missing_tmdb_links,
        "invalid_metadata_id_rows": invalid_metadata_ids,
        "duplicate_metadata_id_rows": duplicate_metadata_ids,
    }
    return joined, stats


def leave_one_out_split(user_indices, item_indices, seed=42):
    user_indices = np.asarray(user_indices, dtype=np.int64)
    item_indices = np.asarray(item_indices, dtype=np.int64)
    if len(user_indices) != len(item_indices):
        raise ValueError("user_indices and item_indices must have equal lengths")
    if not len(user_indices):
        raise ValueError("Cannot split an empty interaction set")

    rng = np.random.default_rng(seed)
    item_counts = np.bincount(item_indices)
    validation_indices = []
    for user_index in np.unique(user_indices):
        candidates = np.flatnonzero(user_indices == user_index)
        if len(candidates) < 2:
            continue
        for candidate in rng.permutation(candidates):
            item_index = item_indices[candidate]
            if item_counts[item_index] > 1:
                validation_indices.append(int(candidate))
                item_counts[item_index] -= 1
                break

    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    if not len(validation_indices):
        raise ValueError("Unable to create a validation split")
    is_validation = np.zeros(len(user_indices), dtype=bool)
    is_validation[validation_indices] = True
    return np.flatnonzero(~is_validation), validation_indices


def build_normalized_adjacency(n_users, n_items, user_indices, item_indices, device):
    users = torch.as_tensor(user_indices, dtype=torch.long, device=device)
    items = torch.as_tensor(item_indices, dtype=torch.long, device=device) + n_users
    rows = torch.cat((users, items))
    columns = torch.cat((items, users))
    indices = torch.stack((rows, columns))
    degrees = torch.bincount(rows, minlength=n_users + n_items).to(torch.float32)
    inverse_sqrt_degrees = torch.zeros_like(degrees)
    nonzero = degrees > 0
    inverse_sqrt_degrees[nonzero] = degrees[nonzero].rsqrt()
    values = inverse_sqrt_degrees[rows] * inverse_sqrt_degrees[columns]
    return torch.sparse_coo_tensor(
        indices, values, (n_users + n_items, n_users + n_items), device=device
    ).coalesce()


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def propagate(self, adjacency):
        embeddings = torch.cat(
            (self.user_embedding.weight, self.item_embedding.weight), dim=0
        )
        layer_embeddings = [embeddings]
        for _ in range(self.n_layers):
            embeddings = torch.sparse.mm(adjacency, embeddings)
            layer_embeddings.append(embeddings)
        embeddings = torch.stack(layer_embeddings, dim=0).mean(dim=0)
        return embeddings[: self.n_users], embeddings[self.n_users :]


def sample_negative_items(user_indices, n_items, positive_items_by_user, rng):
    negatives = rng.integers(0, n_items, size=len(user_indices), dtype=np.int64)
    for position, user_index in enumerate(user_indices):
        seen_items = positive_items_by_user[int(user_index)]
        if len(seen_items) >= n_items:
            raise ValueError(f"User {user_index} has no unseen item for negative sampling")
        while int(negatives[position]) in seen_items:
            negatives[position] = rng.integers(0, n_items)
    return negatives


def evaluate(model, adjacency, user_indices, item_indices, seen_items_by_user, k=10):
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model.propagate(adjacency)
        hits = 0
        ndcg = 0.0
        evaluated_users = 0
        for user_index, target_item in zip(user_indices, item_indices):
            scores = user_embeddings[user_index] @ item_embeddings.T
            blocked_items = seen_items_by_user[int(user_index)].difference(
                {int(target_item)}
            )
            if blocked_items:
                scores[list(blocked_items)] = -torch.inf
            top_items = torch.topk(scores, k=min(k, len(scores))).indices.tolist()
            evaluated_users += 1
            if int(target_item) in top_items:
                hits += 1
                rank = top_items.index(int(target_item)) + 1
                ndcg += 1.0 / np.log2(rank + 1)
    if not evaluated_users:
        return {"recall_at_10": 0.0, "ndcg_at_10": 0.0}
    return {
        "recall_at_10": hits / evaluated_users,
        "ndcg_at_10": ndcg / evaluated_users,
    }


def recommend(model, adjacency, user_ids, movie_ids, seen_items_by_user, item_details, k=10):
    model.eval()
    recommendations = []
    with torch.no_grad():
        user_embeddings, item_embeddings = model.propagate(adjacency)
        for user_index, user_id in enumerate(user_ids):
            scores = user_embeddings[user_index] @ item_embeddings.T
            blocked_items = seen_items_by_user[user_index]
            if blocked_items:
                scores[list(blocked_items)] = -torch.inf
            available = int(torch.isfinite(scores).sum().item())
            if not available:
                continue
            top_items = torch.topk(scores, k=min(k, available)).indices.tolist()
            for rank, item_index in enumerate(top_items, start=1):
                movie_id = int(movie_ids[item_index])
                details = item_details[movie_id]
                recommendations.append(
                    {
                        "userId": int(user_id),
                        "rank": rank,
                        "movieId": movie_id,
                        "title": details["title"],
                        "genre_list": details["genre_list"],
                        "score": float(scores[item_index].item()),
                    }
                )
    return pd.DataFrame(
        recommendations,
        columns=["userId", "rank", "movieId", "title", "genre_list", "score"],
    )


def train(args):
    processed_dir = Path(args.output_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    joined, stats = load_joined_ratings(args.data_dir)
    joined.drop(columns=["overview"]).to_csv(
        processed_dir / "ratings_small_joined.csv", index=False
    )
    movie_catalog = (
        joined[joined["metadata_matched"]]
        .drop_duplicates("movieId")
        [
            [
                "movieId",
                "tmdbId",
                "title",
                "genre_list",
                "overview",
                "vote_average",
                "metadata_matched",
            ]
        ]
    )
    movie_catalog.to_csv(processed_dir / "movie_catalog.csv", index=False)

    positive = joined[
        joined["metadata_matched"] & (joined["rating"] >= RATING_THRESHOLD)
    ].copy()
    positive = positive.dropna(subset=["title"])
    positive = positive[positive["title"].astype(str).str.strip().ne("")]
    if positive.duplicated(["userId", "movieId"]).any():
        raise ValueError("Positive interactions contain duplicate userId/movieId pairs")
    if positive.empty:
        raise ValueError("No mapped ratings meet the positive rating threshold")

    user_ids = sorted(int(value) for value in positive["userId"].unique())
    movie_ids = sorted(int(value) for value in positive["movieId"].unique())
    user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    movie_to_index = {movie_id: index for index, movie_id in enumerate(movie_ids)}
    positive_user_indices = positive["userId"].map(user_to_index).to_numpy(dtype=np.int64)
    positive_item_indices = positive["movieId"].map(movie_to_index).to_numpy(dtype=np.int64)
    train_indices, validation_indices = leave_one_out_split(
        positive_user_indices, positive_item_indices, args.seed
    )

    train_users = positive_user_indices[train_indices]
    train_items = positive_item_indices[train_indices]
    validation_users = positive_user_indices[validation_indices]
    validation_items = positive_item_indices[validation_indices]
    device = torch.device(args.device)
    adjacency = build_normalized_adjacency(
        len(user_ids), len(movie_ids), train_users, train_items, device
    )

    positive_items_by_user = {index: set() for index in range(len(user_ids))}
    for user_index, item_index in zip(positive_user_indices, positive_item_indices):
        positive_items_by_user[int(user_index)].add(int(item_index))

    seen_items_by_user = {index: set() for index in range(len(user_ids))}
    mapped_ratings = joined[
        joined["metadata_matched"] & joined["movieId"].isin(movie_to_index)
    ]
    for user_id, movie_id in mapped_ratings[["userId", "movieId"]].itertuples(
        index=False, name=None
    ):
        user_index = user_to_index.get(int(user_id))
        item_index = movie_to_index.get(int(movie_id))
        if user_index is not None and item_index is not None:
            seen_items_by_user[user_index].add(item_index)

    model = LightGCN(
        len(user_ids), len(movie_ids), args.embedding_dim, args.layers
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_users_tensor = torch.as_tensor(train_users, dtype=torch.long, device=device)
    train_items_tensor = torch.as_tensor(train_items, dtype=torch.long, device=device)
    rng = np.random.default_rng(args.seed)
    history = []
    best_state = None
    best_epoch = 0
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        negative_items = sample_negative_items(
            train_users, len(movie_ids), positive_items_by_user, rng
        )
        negative_items_tensor = torch.as_tensor(
            negative_items, dtype=torch.long, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        user_embeddings, item_embeddings = model.propagate(adjacency)
        positive_scores = (
            user_embeddings[train_users_tensor] * item_embeddings[train_items_tensor]
        ).sum(dim=1)
        negative_scores = (
            user_embeddings[train_users_tensor] * item_embeddings[negative_items_tensor]
        ).sum(dim=1)
        bpr_loss = F.softplus(negative_scores - positive_scores).mean()
        regularization = (
            model.user_embedding(train_users_tensor).pow(2).sum(dim=1)
            + model.item_embedding(train_items_tensor).pow(2).sum(dim=1)
            + model.item_embedding(negative_items_tensor).pow(2).sum(dim=1)
        ).mean()
        loss = bpr_loss + args.regularization * regularization
        loss.backward()
        optimizer.step()

        validation_metrics = evaluate(
            model,
            adjacency,
            validation_users,
            validation_items,
            seen_items_by_user,
        )
        epoch_metrics = {
            "epoch": epoch,
            "loss": float(loss.item()),
            **validation_metrics,
        }
        history.append(epoch_metrics)
        print(
            "epoch {epoch:02d}/{total} loss={loss:.4f} "
            "recall@10={recall:.4f} ndcg@10={ndcg:.4f}".format(
                epoch=epoch,
                total=args.epochs,
                loss=epoch_metrics["loss"],
                recall=epoch_metrics["recall_at_10"],
                ndcg=epoch_metrics["ndcg_at_10"],
            )
        )
        score = (validation_metrics["ndcg_at_10"], validation_metrics["recall_at_10"])
        best_score = (
            (best_metrics["ndcg_at_10"], best_metrics["recall_at_10"])
            if best_metrics
            else (-1.0, -1.0)
        )
        if score > best_score:
            best_epoch = epoch
            best_metrics = validation_metrics
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    checkpoint_path = processed_dir / "lightgcn_checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "user_ids": user_ids,
            "movie_ids": movie_ids,
            "embedding_dim": args.embedding_dim,
            "layers": args.layers,
            "rating_threshold": RATING_THRESHOLD,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "validation_metrics": best_metrics,
        },
        checkpoint_path,
    )

    item_columns = [
        "movieId",
        "tmdbId",
        "title",
        "genre_list",
        "overview",
        "vote_average",
    ]
    item_metadata = positive.drop_duplicates("movieId")[item_columns].copy()
    item_details = item_metadata.set_index("movieId")[
        ["title", "genre_list"]
    ].to_dict(orient="index")
    item_index = item_metadata
    item_index.insert(0, "item_idx", item_index["movieId"].map(movie_to_index))
    item_index.to_csv(processed_dir / "item_index.csv", index=False)
    user_index = pd.DataFrame(
        {"user_idx": range(len(user_ids)), "userId": user_ids}
    )
    user_index.to_csv(processed_dir / "user_index.csv", index=False)

    recommendations = recommend(
        model,
        adjacency,
        user_ids,
        movie_ids,
        seen_items_by_user,
        item_details,
    )
    recommendations.to_csv(
        processed_dir / "recommendations_top10.csv", index=False
    )

    stats.update(
        {
            "positive_interactions": int(len(positive)),
            "train_interactions": int(len(train_indices)),
            "validation_interactions": int(len(validation_indices)),
            "users": len(user_ids),
            "items": len(movie_ids),
            "rating_threshold": RATING_THRESHOLD,
            "best_epoch": best_epoch,
            "best_validation_metrics": best_metrics,
            "config": {
                "embedding_dim": args.embedding_dim,
                "layers": args.layers,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "regularization": args.regularization,
                "seed": args.seed,
                "device": str(device),
                "loss": "BPR",
            },
            "epoch_history": history,
            "artifacts": {
                "joined_ratings": "ratings_small_joined.csv",
                "movie_catalog": "movie_catalog.csv",
                "checkpoint": "lightgcn_checkpoint.pt",
                "user_index": "user_index.csv",
                "item_index": "item_index.csv",
                "recommendations": "recommendations_top10.csv",
            },
        }
    )
    with (processed_dir / "training_metrics.json").open(
        "w", encoding="utf-8"
    ) as metrics_file:
        json.dump(stats, metrics_file, ensure_ascii=False, indent=2)
    print(
        f"Saved {checkpoint_path}; {len(recommendations):,} recommendations "
        f"for {recommendations['userId'].nunique() if not recommendations.empty else 0} users."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Join MovieLens ratings_small with TMDB metadata and train LightGCN."
    )
    parser.add_argument(
        "--data-dir",
        default="data/the-movies-dataset",
        help="Directory containing the raw CSV files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for joined data and model outputs",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if arguments.embedding_dim < 1 or arguments.layers < 1:
        raise ValueError("--embedding-dim and --layers must be at least 1")
    if arguments.learning_rate <= 0 or arguments.regularization < 0:
        raise ValueError("--learning-rate must be positive and regularization non-negative")
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)
    cpu_threads = max(1, min(4, os.cpu_count() or 1))
    torch.set_num_threads(cpu_threads)
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    train(arguments)
