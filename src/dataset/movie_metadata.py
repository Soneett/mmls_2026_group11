from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


GENRE_COLUMNS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def load_ml100k_movie_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        return {}

    columns = [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        *GENRE_COLUMNS,
    ]

    df = pd.read_csv(
        path,
        sep="|",
        names=columns,
        encoding="latin-1",
    )

    metadata: dict[str, dict[str, Any]] = {}

    for _, row in df.iterrows():
        movie_id = str(row["movie_id"])

        genres = [
            genre
            for genre in GENRE_COLUMNS
            if int(row[genre]) == 1
        ]

        metadata[movie_id] = {
            "movie_id": int(row["movie_id"]),
            "title": row["title"],
            "release_date": None if pd.isna(row["release_date"]) else row["release_date"],
            "imdb_url": None if pd.isna(row["imdb_url"]) else row["imdb_url"],
            "genres": genres,
        }

    return metadata