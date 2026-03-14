"""
GraphQL API routes для Nanoprobe Sim Lab
"""

from fastapi import APIRouter
from typing import Optional, Dict, Any

from api.error_handlers import DatabaseError

router = APIRouter(prefix="/graphql", tags=["GraphQL"])


@router.post(
    "",
    summary="GraphQL endpoint",
    description="Выполнение GraphQL запросов",
)
async def graphql_query(
    query: str,
    variables: Optional[Dict[str, Any]] = None,
    operation_name: Optional[str] = None
):
    """
    GraphQL endpoint для выполнения запросов

    Примеры запросов:

    ```graphql
    query {
        stats {
            totalScans
            totalSimulations
            totalImages
            activeSimulations
        }
    }
    ```

    ```graphql
    query {
        scans(limit: 10) {
            id
            scanType
            timestamp
            surfaceType
        }
    }
    ```

    ```graphql
    mutation {
        createScan(scanType: "spm", width: 100, height: 100) {
            id
            scanType
            timestamp
        }
    }
    ```
    """
    from api.graphql_schema import schema

    try:
        result = await schema.execute(
            query,
            variable_values=variables,
            operation_name=operation_name
        )

        if result.errors:
            return {
                "data": result.data,
                "errors": [str(e) for e in result.errors],
                "success": False
            }

        return {
            "data": result.data,
            "errors": None,
            "success": True
        }

    except Exception as e:
        raise DatabaseError(f"GraphQL execution error: {str(e)}")


@router.get(
    "/schema",
    summary="GraphQL Schema",
    description="Получить схему GraphQL API",
)
async def get_graphql_schema():
    """Получить схему GraphQL API"""
    from api.graphql_schema import schema
    return {
        "schema": str(schema),
        "types": [
            "Scan",
            "Simulation",
            "Image",
            "DefectAnalysis",
            "SurfaceComparison",
            "DashboardStats",
        ],
        "queries": [
            "scans(limit)",
            "scan(scanId)",
            "simulations(limit)",
            "images(limit)",
            "stats()",
        ],
        "mutations": [
            "createScan(scanType, surfaceType, width, height)",
        ]
    }
