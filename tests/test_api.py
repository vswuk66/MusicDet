import pytest


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self, client):
        """Test that health endpoint returns correct status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "vector_db" in data


class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    def test_login_success(self, client):
        """Test successful login with valid credentials."""
        response = client.post(
            "/token",
            data={"username": "testuser", "password": "testpass"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_credentials(self, client):
        """Test login failure with invalid credentials."""
        response = client.post(
            "/token",
            data={"username": "wronguser", "password": "wrongpass"}
        )
        assert response.status_code == 401

    def test_register_new_user(self, client):
        """Test user registration."""
        import uuid
        unique_username = f"testuser_{uuid.uuid4().hex[:8]}"
        response = client.post(
            "/register",
            json={
                "username": unique_username,
                "email": f"{unique_username}@example.com",
                "password": "securepassword123"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == unique_username

    def test_get_current_user(self, client, auth_headers):
        """Test getting current user info with valid token."""
        if not auth_headers:
            pytest.skip("Authentication not available")
        
        response = client.get("/users/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "username" in data


class TestPredictEndpoint:
    """Tests for the prediction endpoint."""

    def test_predict_no_file(self, client):
        """Test prediction endpoint without file."""
        response = client.post("/predict")
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_file_type(self, client):
        """Test prediction endpoint with invalid file type."""
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an audio file", "text/plain")}
        )
        assert response.status_code == 400


class TestStaticFiles:
    """Tests for static file serving."""

    def test_index_page(self, client):
        """Test that index page is served."""
        response = client.get("/")
        assert response.status_code == 200


class TestEmbeddingEndpoints:
    """Tests for pgvector embedding endpoints."""

    def test_embedding_status_endpoint(self, client):
        response = client.get("/embeddings/status")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "ready" in data
        assert "embedding_dim" in data

    def test_add_embedding_returns_503_when_vector_db_disabled(self, client):
        import app as app_module

        if app_module.vector_store_service.status()["enabled"]:
            pytest.skip("Vector DB is enabled in this environment.")

        dim = app_module.vector_store_service.embedding_dim
        response = client.post(
            "/embeddings",
            json={
                "track_name": "sample.wav",
                "genre": "classical",
                "embedding": [0.0] * dim,
                "metadata": {"test": True},
            },
        )
        assert response.status_code == 503

    def test_search_embeddings_returns_503_when_vector_db_disabled(self, client):
        import app as app_module

        if app_module.vector_store_service.status()["enabled"]:
            pytest.skip("Vector DB is enabled in this environment.")

        dim = app_module.vector_store_service.embedding_dim
        response = client.post(
            "/embeddings/search",
            json={"embedding": [0.0] * dim, "top_k": 3},
        )
        assert response.status_code == 503
