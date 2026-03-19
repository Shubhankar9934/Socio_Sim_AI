"""Pydantic settings loaded from environment."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings from env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_agent_model: str = Field(
        default="gpt-4o-mini",
        description="Model for agent reasoning",
    )
    openai_judge_model: str = Field(
        default="gpt-4o",
        description="Model for LLM-as-judge evaluation",
    )

    # Domain
    domain_id: str = Field(
        default="dubai",
        description="Domain/city config to load from data/domains/{domain_id}/",
    )
    demographics_path: str = Field(
        default="",
        description="Override path for demographics JSON (empty = use domain_id)",
    )

    # Simulation
    population_size: int = Field(default=500, ge=10, le=100_000)
    max_concurrent_llm_calls: int = Field(default=20, ge=1, le=100)
    simulation_days_default: int = Field(default=30, ge=1, le=365)

    # Archetype compression
    archetype_count: int = Field(default=80, ge=10, le=200)
    use_archetypes_above_agents: int = Field(default=500, ge=100)

    # LLM generation diversity
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_top_p: float = Field(default=0.9, ge=0.0, le=1.0)

    # Validation
    population_realism_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    drift_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # ChromaDB
    chroma_persist_dir: str = Field(
        default="",
        description="Directory for ChromaDB persistence; empty = in-memory",
    )

    # Multi-question survey settings
    max_survey_questions: int = Field(default=50, ge=1, le=200)
    summarize_memory_every: int = Field(
        default=5, ge=1, le=50,
        description="Compress agent memory every N survey rounds",
    )
    social_influence_between_rounds: bool = Field(
        default=True,
        description="Run social diffusion between survey rounds",
    )
    jsonl_output_dir: str = Field(
        default="data/sessions",
        description="Directory for JSONL streaming response files",
    )

    # Buffered JSONL writer
    jsonl_buffer_size: int = Field(
        default=100, ge=1, le=10_000,
        description="Flush JSONL buffer to disk every N records",
    )

    # Memory summarization
    max_summary_length: int = Field(
        default=200, ge=50, le=2000,
        description="Hard character cap on agent dialogue_summary",
    )

    # Social influence
    social_neighbor_sample_k: int = Field(
        default=15, ge=1, le=100,
        description="Max neighbors sampled per agent during diffusion",
    )

    # Archetype execution
    archetype_noise_std: float = Field(
        default=0.05, ge=0.0, le=0.5,
        description="Gaussian noise std when expanding archetype distributions",
    )
    narrative_budget: float = Field(
        default=0.20, ge=0.0, le=1.0,
        description="Fraction of non-representative agents that receive LLM narratives",
    )

    # Event-driven scheduler
    event_batch_size: int = Field(
        default=100, ge=10, le=1000,
        description="Max events processed per scheduler tick",
    )

    # Archetype aggregation (Risk 1)
    archetype_aggregation: str = Field(
        default="median",
        description="Aggregation method for archetype states: median | trimmed_mean | mean",
    )

    # Narrative templates per archetype (Risk 2)
    narrative_templates_per_archetype: int = Field(
        default=3, ge=1, le=10,
        description="Number of narrative variants generated per archetype",
    )

    # Periodic reclustering (Risk 3)
    recluster_every: int = Field(
        default=10, ge=1, le=100,
        description="Rebuild archetypes every N survey rounds",
    )

    # Columnar state store threshold (Risk 4)
    vectorize_state_above: int = Field(
        default=1000, ge=100, le=1_000_000,
        description="Enable StateMatrix when population exceeds this size",
    )

    # Memory eviction (Risk 5)
    max_last_answers: int = Field(
        default=10, ge=1, le=100,
        description="Max answer entries kept in AgentState.last_answers",
    )
    max_structured_memory_keys: int = Field(
        default=20, ge=1, le=200,
        description="Max semantic keys in AgentState.structured_memory",
    )

    # Time-based JSONL flushing
    jsonl_flush_interval: float = Field(
        default=5.0, ge=0.1, le=300.0,
        description="Seconds between time-based JSONL buffer flushes",
    )

    # World feedback loop
    enable_world_feedback: bool = Field(
        default=False,
        description="Enable environment dynamics / world feedback loop",
    )
    max_feedback_events_per_step: int = Field(
        default=5, ge=1, le=50,
        description="Max world feedback events injected per timestep",
    )

    # JSONL file rotation
    jsonl_max_file_size_mb: float = Field(
        default=100.0, ge=1.0, le=10_000.0,
        description="Rotate JSONL files when they exceed this size (MB)",
    )
    jsonl_max_file_age_hours: float = Field(
        default=1.0, ge=0.01, le=168.0,
        description="Rotate JSONL files when they exceed this age (hours)",
    )

    # --- Bias engine ---
    bias_gamma_floor: float = Field(default=0.05, ge=0.0, le=1.0)
    bias_gamma_ceiling: float = Field(default=0.95, ge=0.0, le=1.0)
    bias_epsilon_base: float = Field(default=0.05, ge=0.0, le=1.0)
    bias_epsilon_floor: float = Field(default=0.01, ge=0.0, le=1.0)
    bias_epsilon_ceiling: float = Field(default=0.3, ge=0.0, le=1.0)
    calcification_rate: float = Field(
        default=0.001, ge=0.0, le=0.1,
        description="Per-day calcification increment for agent rigidity",
    )

    # --- Media ecosystem ---
    media_prior_weight: float = Field(default=0.70, ge=0.0, le=1.0)
    media_influence_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    social_influence_weight: float = Field(default=0.15, ge=0.0, le=1.0)
    attention_sharpness_k: float = Field(default=5.0, ge=0.0, le=20.0)
    attention_sharpness_p: float = Field(default=2.0, ge=0.5, le=5.0)
    attention_entropy_floor: float = Field(default=0.05, ge=0.0, le=0.5)
    alignment_beta: float = Field(default=0.15, ge=0.0, le=1.0)

    # --- Cascade detection ---
    activation_decay: float = Field(default=0.85, ge=0.0, le=1.0)
    activation_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    cascade_min_size: int = Field(default=200, ge=10, le=100_000)
    cascade_min_fraction: float = Field(default=0.005, ge=0.0, le=1.0)
    cascade_min_density: float = Field(default=0.01, ge=0.0, le=1.0)
    fatigue_factor: float = Field(default=0.3, ge=0.0, le=1.0)
    cooldown_days: int = Field(default=5, ge=0, le=100)
    cooldown_decay: float = Field(default=0.1, ge=0.0, le=1.0)
    outrage_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    validation_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    social_lambda: float = Field(default=0.2, ge=0.0, le=1.0)

    # --- Survey cognition (narrative hidden state, social, belief) ---
    # Emergence test preset (run 20 agents, share % neutral, % balanced phrases, 5 extreme responses):
    #   belief_nonlinearity = 6, social_damping = 0.2, entropy_floor_epsilon = 0.005,
    #   neutral_penalty_when_competing_above = 0.3
    survey_social_warmup_steps: int = Field(
        default=1, ge=0, le=10,
        description="Social diffusion steps before each survey (1 = light nudge, 3 = legacy)",
    )
    social_damping: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Multiply social factor weight by this to avoid double-counting with diffusion",
    )
    belief_nonlinearity: float = Field(
        default=0.0, ge=0.0, le=20.0,
        description="Sigmoid steepness for belief_score (0 = linear, e.g. 6 = nonlinear)",
    )
    response_variability_std: float = Field(
        default=0.0, ge=0.0, le=0.2,
        description="Extra Gaussian jitter on raw_scores before softmax (0 = off)",
    )
    entropy_floor_epsilon: float = Field(
        default=0.0, ge=0.0, le=0.1,
        description="Add epsilon to each prob and renormalize to avoid extreme determinism (0 = off)",
    )
    media_weight_survey: float = Field(
        default=0.05, ge=0.0, le=0.5,
        description="Weight for real-time media belief update at survey time (when current_events provided)",
    )
    neutral_penalty_when_competing_above: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="When > 0, downweight neutral option if second-best prob > 0.4 (0 = off, e.g. 0.3 for emergence test)",
    )

    # --- Research layer ---
    research_api_provider: str = Field(
        default="tavily", description="Web search provider: tavily | serpapi | none",
    )
    research_cache_path: str = Field(
        default="data/research_cache.json",
        description="Path for research context cache",
    )


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
