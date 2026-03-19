"""
Pydantic request/response models for API.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --- Population ---
class GeneratePopulationRequest(BaseModel):
    n: int = Field(default=500, ge=10, le=10000)
    method: str = Field(default="bayesian", description="monte_carlo | bayesian | ipf")
    seed: Optional[int] = None
    id_prefix: str = "DXB"


# --- Agents ---
class AgentSummary(BaseModel):
    agent_id: str
    age: str
    nationality: str
    income: str
    location: str
    occupation: str


class AgentDetail(BaseModel):
    agent_id: str
    persona: Dict[str, Any]
    state: Optional[Dict[str, Any]] = None


# --- Survey ---
class SurveyRequest(BaseModel):
    question: str
    question_id: Optional[str] = ""
    use_archetypes: bool = False
    options: Optional[List[str]] = None  # If empty or None, treated as open_text
    current_events: Optional[List[Dict[str, Any]]] = None  # Real-time media: temp_beliefs at survey time


class AgentDemographics(BaseModel):
    age_group: str = ""
    nationality: str = ""
    income_band: str = ""
    location: str = ""
    occupation: str = ""
    household_size: str = ""
    family_children: int = 0
    has_spouse: bool = False


class AgentLifestyle(BaseModel):
    cuisine_preference: str = ""
    diet: str = ""
    hobby: str = ""
    work_schedule: str = ""
    health_focus: str = ""
    commute_method: str = ""


class SurveyResponseItem(BaseModel):
    agent_id: str
    answer: str
    sampled_option: Optional[str] = None
    distribution: Optional[Dict[str, float]] = None
    demographics: Optional[AgentDemographics] = None
    lifestyle: Optional[AgentLifestyle] = None
    error: Optional[str] = None


class SurveyResult(BaseModel):
    survey_id: str
    question: str
    responses: List[SurveyResponseItem]
    n_total: int


# --- Multi-Question Survey ---
class SurveyQuestionItem(BaseModel):
    question: str
    question_id: str = ""
    options: Optional[List[str]] = None


class MultiSurveyRequest(BaseModel):
    questions: List[SurveyQuestionItem]
    use_archetypes: bool = False
    social_influence_between_rounds: bool = True
    summarize_every: int = Field(default=5, ge=1, le=50, description="Summarize agent memory every N rounds")


class MultiSurveyProgress(BaseModel):
    session_id: str
    current_round: int
    total_rounds: int
    status: str = Field(description="running | completed | failed")
    completed_questions: List[str] = Field(default_factory=list)


class RoundResultItem(BaseModel):
    round_idx: int
    question: str
    question_id: str
    responses: List[SurveyResponseItem]
    n_total: int
    elapsed_seconds: float = 0.0


class SurveySessionResult(BaseModel):
    session_id: str
    questions: List[str]
    rounds: List[RoundResultItem] = Field(default_factory=list)
    total_responses: int = 0
    elapsed_seconds: float = 0.0
    status: str = "completed"


# --- Simulation ---
class SimulateRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=365)


class EventInjectRequest(BaseModel):
    day: int = Field(ge=0, description="Day on which the event triggers")
    type: str = Field(description="Event type: price_change | policy | infrastructure | market | new_service | new_metro_station")
    payload: Dict[str, Any] = Field(default_factory=dict)
    district: Optional[str] = None


# --- Scenario ---
class ScenarioEventRequest(BaseModel):
    day: int = Field(ge=0)
    type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    district: Optional[str] = None


class ScenarioRunRequest(BaseModel):
    name: str = "unnamed"
    days: int = Field(default=30, ge=1)
    seed: Optional[int] = None
    events: List[ScenarioEventRequest] = Field(default_factory=list)


class ScenarioCompareRequest(BaseModel):
    scenario_a: ScenarioRunRequest
    scenario_b: ScenarioRunRequest


class ScenarioWithSurveyRequest(BaseModel):
    scenario: ScenarioRunRequest
    questions: List[str] = Field(..., min_length=1)


class ScenarioCompareWithSurveyRequest(BaseModel):
    scenario_a: ScenarioRunRequest
    scenario_b: ScenarioRunRequest
    questions: List[str] = Field(..., min_length=1)


# --- Analytics ---
class AnalyticsResponse(BaseModel):
    survey_id: str
    segment_by: str
    aggregated: Dict[str, Dict[str, float]]
    insights: List[str] = []


# --- Evaluation ---
class EvaluateRequest(BaseModel):
    run_judge: bool = False
    judge_sample: Optional[int] = 20
    realism_threshold: float = 0.85
    drift_threshold: float = 0.3
    run_similarity: bool = True
    similarity_threshold: float = 0.9


class DashboardMetrics(BaseModel):
    duplicate_narrative_rate: float = Field(description="Target < 0.05")
    persona_realism_score: float = Field(description="Target > 0.9")
    distribution_similarity: float = Field(description="Target > 0.85")
    consistency_score: float = Field(description="Target > 0.9")
    drift_rate: float = Field(description="Fraction of agents that drifted")
    mean_judge_score: float = Field(default=0.0, description="Average LLM judge score (1-5)")


class EvaluationReportResponse(BaseModel):
    population_realism: Dict[str, Any]
    drift: Dict[str, Any]
    consistency_score: float
    distribution_validation: Optional[Dict[str, Any]] = None
    narrative_similarity: Optional[Dict[str, Any]] = None
    llm_judge: Optional[Dict[str, Any]] = None
    dashboard: Optional[DashboardMetrics] = None
    quantitative_metrics: Optional[Dict[str, Any]] = None
    summary: Dict[str, Any]
