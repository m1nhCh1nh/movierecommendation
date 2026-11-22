import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

import app as web_app


class MovieWebAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_directory = tempfile.TemporaryDirectory()
        cls.original_db_path = web_app.DB_PATH
        web_app.DB_PATH = Path(cls.temp_directory.name) / "test-movies.db"
        cls.admin_environment = patch.dict(
            os.environ,
            {
                "ENABLE_ADMIN_DEMO": "true",
                "ADMIN_EMAIL": "admin@example.test",
                "ADMIN_PASSWORD": "SecureAdmin1234",
            },
        )
        cls.admin_environment.start()
        cls.client = TestClient(web_app.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        cls.admin_environment.stop()
        web_app.DB_PATH = cls.original_db_path
        cls.temp_directory.cleanup()

    def test_guest_catalog_contains_top_rated_movies_and_genres(self):
        self.client.cookies.clear()
        homepage = self.client.get("/")
        self.assertEqual(homepage.status_code, 200)
        self.assertIn("CineSuggest", homepage.text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        health = self.client.get("/api/health").json()
        self.assertTrue(health["model_ready"])
        self.assertTrue(web_app.recommender.ready)
        self.assertEqual(health["recommendation_model"], "hybrid_gnn_llm")
        self.assertEqual(health["mapped_movies"], 9025)
        response = self.client.get("/api/guest")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["movies"]["items"])
        self.assertLessEqual(len(data["movies"]["items"]), 10)
        first_movie = data["movies"]["items"][0]
        self.assertGreaterEqual(first_movie["rating_count"], 5)
        self.assertGreater(first_movie["rating_average"], 0)
        with web_app.database() as connection:
            expected_popular = connection.execute(
                "SELECT COUNT(*) FROM movies WHERE rating_count >= 5"
            ).fetchone()[0]
            expected_rating = connection.execute(
                "SELECT AVG(rating) FROM dataset_ratings WHERE movie_id = ?",
                (first_movie["movie_id"],),
            ).fetchone()[0]
            expected_rating_count = connection.execute(
                "SELECT COUNT(*) FROM dataset_ratings WHERE movie_id = ?",
                (first_movie["movie_id"],),
            ).fetchone()[0]
        self.assertEqual(data["movies"]["total"], expected_popular)
        self.assertAlmostEqual(first_movie["rating_average"], expected_rating)
        self.assertEqual(first_movie["rating_count"], expected_rating_count)
        filtered = self.client.get(
            f"/api/guest?genre={data['genres'][0]}&page_size=100"
        )
        self.assertEqual(filtered.status_code, 200)
        self.assertLessEqual(len(filtered.json()["movies"]["items"]), 10)
        self.assertTrue(
            all(
                data["genres"][0] in movie["genres"]
                for movie in filtered.json()["movies"]["items"]
            )
        )
        database_page = self.client.get("/api/guest?page=2&page_size=100").json()["movies"]
        self.assertEqual(database_page["page_size"], 10)
        self.assertNotEqual(
            first_movie["movie_id"], database_page["items"][0]["movie_id"]
        )
        with web_app.database() as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM dataset_users").fetchone()[0],
                671,
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM dataset_ratings").fetchone()[0],
                99810,
            )

    def test_registration_profile_recommendations_and_csrf(self):
        self.client.cookies.clear()
        registration = self.client.post(
            "/api/auth/register",
            json={
                "email": "movie-fan@example.test",
                "display_name": "Movie Fan",
                "password": "MovieFanPass123",
            },
        )
        self.assertEqual(registration.status_code, 200)
        account = registration.json()
        self.assertEqual(account["user"]["role"], "user")
        csrf = account["csrf_token"]

        profile = self.client.get("/api/profile")
        self.assertEqual(profile.status_code, 200)
        genre = profile.json()["genres"][0]
        favorite_movie_id = (
            next(iter(web_app.recommender.item_index))
            if web_app.recommender.ready
            else profile.json()["movies"]["items"][0]["movie_id"]
        )
        default_recommendations = self.client.get("/api/recommendations").json()
        self.assertFalse(default_recommendations["personalized"])
        self.assertTrue(default_recommendations["items"])
        self.assertLessEqual(len(default_recommendations["items"]), 10)
        update = self.client.put(
            "/api/profile",
            headers={"X-CSRF-Token": csrf},
            json={
                "display_name": "Movie Fan",
                "favorite_genres": [genre],
                "favorite_movie_ids": [favorite_movie_id],
            },
        )
        self.assertEqual(update.status_code, 200)
        updated_recommendations = update.json()["recommendations"]
        if web_app.recommender.ready:
            self.assertTrue(updated_recommendations["personalized"])
            self.assertEqual(
                updated_recommendations["recommendation_source"], "hybrid_gnn_llm"
            )
            self.assertTrue(
                all(
                    movie["movie_id"] in web_app.recommender.item_index
                    and movie["movie_id"] != favorite_movie_id
                    for movie in updated_recommendations["items"]
                )
            )
        else:
            self.assertFalse(updated_recommendations["personalized"])
            self.assertEqual(
                updated_recommendations["recommendation_source"], "genre_filter"
            )
            self.assertTrue(
                all(genre in movie["genres"] for movie in updated_recommendations["items"])
            )
        personalized = self.client.get("/api/recommendations").json()
        self.assertEqual(
            personalized["recommendation_source"],
            updated_recommendations["recommendation_source"],
        )
        self.assertEqual(personalized["items"], updated_recommendations["items"])
        self.assertLessEqual(len(personalized["items"]), 10)

        missing_csrf = self.client.put(
            "/api/profile",
            json={
                "display_name": "Movie Fan",
                "favorite_genres": [],
                "favorite_movie_ids": [],
            },
        )
        self.assertEqual(missing_csrf.status_code, 403)

    def test_admin_routes_enforce_role_and_manage_movies(self):
        self.client.cookies.clear()
        registration = self.client.post(
            "/api/auth/register",
            json={
                "email": "regular-user@example.test",
                "display_name": "Regular User",
                "password": "RegularUser1234",
            },
        )
        self.assertEqual(registration.status_code, 200)
        denied = self.client.get("/api/admin/users")
        self.assertEqual(denied.status_code, 403)

        self.client.cookies.clear()
        login = self.client.post(
            "/api/auth/login",
            json={
                "email": "admin@example.test",
                "password": "SecureAdmin1234",
            },
        )
        self.assertEqual(login.status_code, 200)
        csrf = login.json()["csrf_token"]
        users = self.client.get("/api/admin/users")
        self.assertEqual(users.status_code, 200)
        self.assertTrue(
            any(user["role"] == "admin" for user in users.json()["items"])
        )

        movie = self.client.get("/api/admin/movies?page_size=1")
        self.assertEqual(movie.status_code, 200)
        self.assertLessEqual(len(movie.json()["items"]), 10)
        original = movie.json()["items"][0]
        update = self.client.patch(
            f"/api/admin/movies/{original['movie_id']}",
            headers={"X-CSRF-Token": csrf},
            json={
                "title": original["title"] + " (test)",
                "genres": original["genres"],
                "overview": original["overview"],
            },
        )
        self.assertEqual(update.status_code, 200)
        restore = self.client.patch(
            f"/api/admin/movies/{original['movie_id']}",
            headers={"X-CSRF-Token": csrf},
            json={
                "title": original["title"],
                "genres": original["genres"],
                "overview": original["overview"],
            },
        )
        self.assertEqual(restore.status_code, 200)

    def test_empty_database_has_no_recommendations_or_ratings(self):
        original_db_path = web_app.DB_PATH
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                web_app.DB_PATH = Path(temporary_directory) / "empty.db"
                web_app.create_schema()
                response = self.client.get("/api/guest")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["movies"]["total"], 0)
                self.assertEqual(response.json()["movies"]["items"], [])
        finally:
            web_app.DB_PATH = original_db_path

    def test_admin_movie_deletion_survives_dataset_sync(self):
        original_db_path = web_app.DB_PATH
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                web_app.DB_PATH = Path(temporary_directory) / "movie-delete.db"
                web_app.create_schema()
                web_app.sync_catalog()
                web_app.initialize_demo_admin()
                self.client.cookies.clear()
                login = self.client.post(
                    "/api/auth/login",
                    json={
                        "email": "admin@example.test",
                        "password": "SecureAdmin1234",
                    },
                )
                self.assertEqual(login.status_code, 200)
                csrf = login.json()["csrf_token"]
                movie = web_app.query_movie_page(page=1, page_size=1)["items"][0]
                deleted = self.client.delete(
                    f"/api/admin/movies/{movie['movie_id']}",
                    headers={"X-CSRF-Token": csrf},
                )
                self.assertEqual(deleted.status_code, 200)
                web_app.sync_catalog()
                self.assertIsNone(web_app.get_movie(movie["movie_id"]))
        finally:
            web_app.DB_PATH = original_db_path
            self.client.cookies.clear()

    def test_chat_refuses_out_of_scope_without_calling_model(self):
        self.client.cookies.clear()
        response = self.client.post(
            "/api/chat", json={"message": "Hôm nay thời tiết thế nào?"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["refused"])
        self.assertEqual(response.json()["movies"], [])

    def test_chat_only_returns_catalog_titles_selected_by_the_model(self):
        self.client.cookies.clear()
        selected_movie = web_app.popular_movies(1)[0]
        selected_movie["reason"] = "Phù hợp với thể loại bạn yêu cầu."
        with patch.object(web_app.recommender, "ready", True), patch.object(
            web_app, "ask_ollama", new=AsyncMock(return_value=[selected_movie])
        ) as ask_model:
            response = self.client.post(
                "/api/chat", json={"message": "Gợi ý phim hành động"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["reply"],
            f"• {selected_movie['title']} — Thể loại: {', '.join(selected_movie['genres'])} "
            f"— Vì sao: {selected_movie['reason']}",
        )
        self.assertEqual(
            [movie["movie_id"] for movie in response.json()["movies"]],
            [selected_movie["movie_id"]],
        )
        ask_model.assert_awaited_once()


class OllamaResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_model_ids_in_the_candidate_catalog_are_returned(self):
        first = {
            "movie_id": 10,
            "title": "Known film",
            "genres": [],
            "overview": "",
            "rating_average": 4.0,
            "rating_count": 10,
        }
        second = {
            "movie_id": 20,
            "title": "Another film",
            "genres": [],
            "overview": "",
            "rating_average": 3.5,
            "rating_count": 5,
        }
        response = Mock()
        response.json.return_value = {
            "message": {
                "content": json.dumps(
                    {
                        "recommendations": [
                            {"movie_id": 10, "reason": "Hợp gu."},
                            {"movie_id": 999999, "reason": "Không có trong catalog."},
                            {"movie_id": "20", "reason": "ID không hợp lệ."},
                            {"movie_id": 20, "reason": "Có trong catalog."},
                        ]
                    }
                )
            }
        }
        mock_client = AsyncMock()
        mock_client.post.return_value = response
        client_context = AsyncMock()
        client_context.__aenter__.return_value = mock_client
        with patch("app.httpx.AsyncClient", return_value=client_context):
            movies = await web_app.ask_ollama("gợi ý phim", [first, second])
        self.assertEqual([movie["movie_id"] for movie in movies], [10, 20])
        self.assertTrue(all(movie["reason"] for movie in movies))
        response.raise_for_status.assert_called_once()


if __name__ == "__main__":
    unittest.main()
