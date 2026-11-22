import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_lightgcn import (
    build_normalized_adjacency,
    leave_one_out_split,
    load_joined_ratings,
    parse_genres,
)


class ParseGenresTests(unittest.TestCase):
    def test_parses_genres_and_ignores_invalid_values(self):
        self.assertEqual(
            parse_genres("[{'id': 28, 'name': 'Action'}, {'id': 35, 'name': 'Comedy'}]"),
            ["Action", "Comedy"],
        )
        self.assertEqual(parse_genres("not-a-list"), [])
        self.assertEqual(parse_genres(None), [])


class JoinRatingsTests(unittest.TestCase):
    def test_joins_via_tmdb_id_and_keeps_unmatched_ratings(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            pd.DataFrame(
                [
                    {"userId": 1, "movieId": 10, "rating": 4.0, "timestamp": 1},
                    {"userId": 1, "movieId": 11, "rating": 2.0, "timestamp": 2},
                ]
            ).to_csv(data_dir / "ratings_small.csv", index=False)
            pd.DataFrame(
                [
                    {"movieId": 10, "imdbId": "0000010", "tmdbId": 100},
                    {"movieId": 11, "imdbId": "0000011", "tmdbId": None},
                ]
            ).to_csv(data_dir / "links_small.csv", index=False)
            pd.DataFrame(
                [
                    {"id": "100", "title": "First title", "genres": "[{'name': 'Drama'}]", "overview": "", "vote_average": 7.0},
                    {"id": "100", "title": "Duplicate title", "genres": "[]", "overview": "", "vote_average": 0.0},
                    {"id": "1997-08-20", "title": "Bad id", "genres": "[]", "overview": "", "vote_average": 0.0},
                ]
            ).to_csv(data_dir / "movies_metadata.csv", index=False)

            joined, stats = load_joined_ratings(data_dir)

        self.assertEqual(len(joined), 2)
        self.assertEqual(joined.loc[0, "title"], "First title")
        self.assertEqual(joined.loc[0, "genre_list"], '["Drama"]')
        self.assertTrue(joined.loc[0, "metadata_matched"])
        self.assertFalse(joined.loc[1, "metadata_matched"])
        self.assertEqual(stats["invalid_metadata_id_rows"], 1)
        self.assertEqual(stats["duplicate_metadata_id_rows"], 1)
        self.assertEqual(stats["missing_tmdb_links"], 1)


class GraphPreparationTests(unittest.TestCase):
    def test_split_keeps_each_item_in_training(self):
        users = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        items = np.array([0, 1, 0, 2, 1, 2], dtype=np.int64)
        train_indices, validation_indices = leave_one_out_split(users, items, seed=7)
        self.assertEqual(len(validation_indices), 3)
        self.assertTrue(set(items[validation_indices]).issubset(set(items[train_indices])))
        self.assertTrue(all(np.sum(users[train_indices] == user) >= 1 for user in users))

    def test_normalized_adjacency_is_symmetric(self):
        adjacency = build_normalized_adjacency(
            2, 2, np.array([0, 1]), np.array([0, 1]), torch.device("cpu")
        ).to_dense()
        self.assertTrue(torch.allclose(adjacency, adjacency.T))
        self.assertEqual(tuple(adjacency.shape), (4, 4))


if __name__ == "__main__":
    unittest.main()
