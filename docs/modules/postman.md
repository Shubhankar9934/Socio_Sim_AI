# Postman Module

API testing with Postman collections and environment.

## Files

| File | Description |
|------|-------------|
| `Socio_Sim_AI_Surveys.postman_collection.json` | Postman collection with requests for population generation, single-question surveys, multi-survey, analytics, evaluation. |
| `Socio_Sim_AI.postman_environment.json` | Environment variables (e.g. base_url, api_key). |
| `README.md` | Setup and usage: start API, import collection and environment, run requests. |

## Usage

1. Start API: `python main.py run`
2. Import into Postman: Collection + Environment
3. One question at a time: POST /survey with question, question_id, use_archetypes
4. Multi-survey: POST /survey/multi for batch; poll /session/{id}/progress
