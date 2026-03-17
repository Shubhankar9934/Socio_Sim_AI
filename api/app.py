"""
FastAPI application: CORS, lifespan, route registration.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import agents, analytics, calibration, discovery, evaluation, population, simulation, survey
from api.routes import websocket as ws_routes


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: optionally generate a default population. Shutdown: nothing."""
    # Optionally pre-generate 50 agents for quick testing (skip if you want POST /population/generate only)
    # from api.state import agents_store
    # if not agents_store:
    #     from api.routes.population import generate_population_endpoint
    #     from api.schemas import GeneratePopulationRequest
    #     generate_population_endpoint(GeneratePopulationRequest(n=50, method="bayesian"))
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="JADU",
        description="Synthetic society simulation platform – survey orchestration, analytics, evaluation",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(population.router)
    app.include_router(agents.router)
    app.include_router(survey.router)
    app.include_router(simulation.router)
    app.include_router(analytics.router)
    app.include_router(evaluation.router)
    app.include_router(discovery.router)
    app.include_router(calibration.router)
    app.include_router(ws_routes.router)
    return app


app = create_app()
