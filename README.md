# LLM Genetic Algorithm API Service

API service for optimizing prompts using genetic algorithms.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set required environment variables:
```bash
export OPENAI_API_KEY=your_api_key
export ANTHROPIC_API_KEY=your_api_key  # if used
```

3. Run the API server:
```bash
python api_server.py
```

For production deployment:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```

## API Endpoints

### Health Check
- `GET /health`: Check if the server is running

### Tone Styles
- `GET /api/tones`: Get list of available tone styles

### Optimize Prompts
- `POST /api/optimize`: Start optimization process
  - Request body:
    ```json
    {
      "tone_style": "professional",
      "example_texts": ["example1", "example2"],
      "neutral_examples": ["neutral1", "neutral2"],
      "pop_size": 20,
      "generations": 5
    }
    ```
  - Response:
    ```json
    {
      "task_id": "unique-task-id",
      "status": "running"
    }
    ```

### Check Optimization Status
- `GET /api/optimize/<task_id>`: Check status of optimization
  - Response:
    ```json
    {
      "status": "completed",
      "prompt": "optimized prompt",
      "stats": { ... }
    }
    ```

### Transform Text
- `POST /api/transform`: Transform text using optimized prompt
  - Request body:
    ```json
    {
      "text": "text to transform",
      "prompt": "optimized prompt"
    }
    ```
  - Response:
    ```json
    {
      "transformed_text": "transformed text"
    }
    ```

### Evaluate Transformation
- `POST /api/evaluate`: Evaluate text transformation
  - Request body:
    ```json
    {
      "original": "original text",
      "transformed": "transformed text",
      "target_style": "professional"
    }
    ```
  - Response:
    ```json
    {
      "score": 0.85,
      "explanation": "Explanation of the score"
    }
    ```

### Clear Cache
- `POST /api/cache/clear`: Clear the prompt cache 