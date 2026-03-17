# Testing Surveys with Postman

Test JADU surveys: **one question → get response → next question → get response**.

## Setup

1. **Start the API server**:
   ```bash
   python main.py run
   ```
   API runs at `http://localhost:8000`.

2. **Import into Postman**
   - Collection: `Socio_Sim_AI_Surveys.postman_collection.json`
   - Environment (optional): `Socio_Sim_AI.postman_environment.json`
   - File → Import → select both → choose "JADU - Local" environment

## One Question at a Time (Recommended)

| Step | Request | What it does |
|------|---------|--------------|
| 0 | **Setup - Generate Population** | Create 50 agents. Run once. |
| 1 | **Question 1** | Ask → get response immediately |
| 2 | **Question 2** | Ask → get response |
| 3... | **Question 3, 4, 5...** | Same pattern |

Each **POST /survey** returns the full response right away. No polling.

### Request body (one question)

```json
{
  "question": "How often do you use delivery? Options: Never, Rarely, Often",
  "question_id": "q1",
  "use_archetypes": false
}
```

To add more questions: duplicate a Question request, change the text and `question_id`.

## Multi-Survey (Optional - Batch Mode)

The **Multi-Survey** folder sends all questions at once; results come via polling. Use **POST /survey** (one at a time) for your flow.

## API Docs

Open `http://localhost:8000/docs` for Swagger UI.
