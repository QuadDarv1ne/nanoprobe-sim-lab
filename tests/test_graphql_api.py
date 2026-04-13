"""
Тесты для GraphQL API (api/routes/graphql.py и api/graphql_schema.py)

Покрытие:
- GET /graphql/schema — получение схемы
- Schema types, queries, mutations existence
- Error handling
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

TEST_DB = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_PATH"] = TEST_DB

from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Фикстура: HTTP клиент"""
    with TestClient(app) as test_client:
        yield test_client
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except Exception:
            pass


class TestGraphQLSchema:
    """Тесты схемы GraphQL"""

    def test_graphql_schema_endpoint(self, client):
        """Тест получения схемы GraphQL"""
        response = client.get("/api/v1/graphql/schema")

        assert response.status_code == 200
        data = response.json()
        assert "schema" in data
        assert "types" in data
        assert "queries" in data
        assert "mutations" in data
        assert len(data["types"]) > 0

    def test_graphql_schema_types(self, client):
        """Тест типов в схеме"""
        response = client.get("/api/v1/graphql/schema")

        assert response.status_code == 200
        data = response.json()
        expected_types = ["Scan", "Simulation", "Image", "DefectAnalysis", "SurfaceComparison"]
        for expected_type in expected_types:
            assert expected_type in data["types"]

    def test_graphql_schema_queries(self, client):
        """Тест доступных запросов в схеме"""
        response = client.get("/api/v1/graphql/schema")

        assert response.status_code == 200
        data = response.json()
        expected_queries = ["scans", "scan", "simulations", "images", "stats"]
        for expected_query in expected_queries:
            assert any(expected_query in q for q in data["queries"])

    def test_graphql_schema_mutations(self, client):
        """Тест доступных мутаций в схеме"""
        response = client.get("/api/v1/graphql/schema")

        assert response.status_code == 200
        data = response.json()
        expected_mutations = ["createScan"]
        for expected_mutation in expected_mutations:
            assert any(expected_mutation in m for m in data["mutations"])


class TestGraphQLEndpoint:
    """Тесты GraphQL endpoint"""

    def test_graphql_post_empty_query(self, client):
        """Тест пустого запроса"""
        response = client.post("/api/v1/graphql", json={"query": ""})

        assert response.status_code in [200, 400, 422]

    def test_graphql_post_invalid_json(self, client):
        """Тест некорректного JSON"""
        response = client.post(
            "/api/v1/graphql",
            content="not json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code in [400, 422]


class TestGraphQLValidation:
    """Тесты валидации"""

    def test_graphql_schema_exists(self):
        """Тест существования схемы"""
        from api.graphql_schema import schema

        assert schema is not None
        assert hasattr(schema, "execute")

    def test_graphql_query_class_exists(self):
        """Тест существования Query класса"""
        from api.graphql_schema import Query

        assert Query is not None
        assert hasattr(Query, "scans")
        assert hasattr(Query, "stats")

    def test_graphql_mutation_class_exists(self):
        """Тест существования Mutation класса"""
        from api.graphql_schema import Mutation

        assert Mutation is not None
        assert hasattr(Mutation, "create_scan")

    def test_graphql_types_exist(self):
        """Тест существования типов"""
        from api.graphql_schema import (
            DashboardStats,
            DefectAnalysis,
            Image,
            Scan,
            Simulation,
            SurfaceComparison,
        )

        # Все типы должны быть импортированы
        assert Scan is not None
        assert Simulation is not None
        assert Image is not None
        assert DashboardStats is not None
        assert DefectAnalysis is not None
        assert SurfaceComparison is not None
