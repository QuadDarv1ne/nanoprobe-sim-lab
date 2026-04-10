# API Routes for Nanoprobe Sim Lab


# Lazy imports for faster loading
def __getattr__(name):
    import importlib

    _modules = {
        "auth": "api.routes.auth",
        "scans": "api.routes.scans",
        "simulations": "api.routes.simulations",
        "analysis": "api.routes.analysis",
        "comparison": "api.routes.comparison",
        "reports": "api.routes.reports",
        "admin": "api.routes.admin",
        "dashboard": "api.routes.dashboard",
        "sstv": "api.routes.sstv",
        "graphql": "api.routes.graphql",
        "ml_analysis": "api.routes.ml_analysis",
        "external_services": "api.routes.external_services",
        "nasa": "api.routes.nasa",
        "monitoring": "api.routes.monitoring",
        "weather": "api.routes.weather",
    }

    if name in _modules:
        return importlib.import_module(_modules[name])

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "auth",
    "scans",
    "simulations",
    "analysis",
    "comparison",
    "reports",
    "admin",
    "dashboard",
    "sstv",
    "graphql",
    "ml_analysis",
    "external_services",
    "nasa",
    "monitoring",
    "weather",
]
