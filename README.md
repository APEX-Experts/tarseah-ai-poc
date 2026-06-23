# Tarseah AI POC Service

FastAPI-based Artificial Intelligence integration layer for the **Tarseah** project Proof-of-Concept. This service handles document analysis, prompt engineering, memory management, and automated proposal section generation using Google Gemini and Groq models.

---

## Features

- **Context Initializer**: Preprocesses RFPs, company profiles, and bid specifications.
- **Section Generation**: Generates targeted sections of proposals (e.g., Cover Letters, Executive Summaries) in Arabic/English.
- **Session-Based Chat**: Stateful chat assistant with memory-window clipping (k=3 window logic).
- **File Processing**: Native text extraction support for PDFs, DOCX files, and Markdown.

---

## Local Development Setup

### 1. Prerequisites

- **Python 3.10 to 3.12** installed locally.
- A terminal with standard Unix utilities (`bash`, `curl`) or Git Bash on Windows.

### 2. Environment Configuration

The service relies on external LLM services. Create a local `.env` file by copying the sample template:

```bash
cp .env.examples .env
```

Open `.env` and fill in the required API keys:

- **`GOOGLE_API_KEY`**: Required for live Gemini models (`gemini-flash-latest`). Get one from [Google AI Studio](https://aistudio.google.com/).
- **`GROQ_API_KEY`**: Required for Groq endpoints.
- **`GROQ_MODEL`**: Model name to use on Groq (default: `openai/gpt-oss-20b`).
- **`GROQ_MAX_OUTPUT_TOKENS`**: Max tokens allowed per response (default: `16384`).

### 3. Install Dependencies (Virtual Environment)

It is recommended to run the app in a dedicated virtual environment.

**On Linux/macOS:**

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies (Note: file name is singular)
pip install -r src/requirement.txt
```

**On Windows (Command Prompt / PowerShell):**

```cmd
:: Create the virtual environment
python -m venv .venv

:: Activate the virtual environment
.venv\Scripts\activate

:: Install dependencies
pip install -r src/requirement.txt
```

### 4. Running the Server Locally

Since internal controller modules reference packages relative to the `src` directory, you must run Uvicorn with the Python path pointing to `src`.

#### Option A: Running from the root directory (Recommended)

```bash
PYTHONPATH=src uvicorn main:app --reload --port 9676
```

#### Option B: Running from inside the `src` directory

```bash
cd src
uvicorn main:app --reload --port 9676
```

The API docs will be available at [http://127.0.0.1:9676/docs](http://127.0.0.1:9676/docs).

---

## Running the Test Suite

Unit and integration tests are located in the `tests/` directory. Pytest requires `PYTHONPATH=src` to resolve modules correctly.

```bash
# Run all tests
PYTHONPATH=src pytest

# Run a specific test file with verbose output
PYTHONPATH=src pytest tests/test_proposal_api.py -v
```

---

## Local Setup with Docker (Alternative)

If you have Docker installed locally, you can build and run the service without setting up a python environment:

```bash
# Build and start the container
docker compose up -d --build

# View container logs
docker compose logs -f

# Stop the container
docker compose down
```

---

## API Endpoints Overview

| Method | Endpoint                                            | Description                                 |
| :----- | :-------------------------------------------------- | :------------------------------------------ |
| `GET`  | `/`                                                 | Service health check                        |
| `GET`  | `/test-langchain`                                   | Basic LangChain execution test              |
| `POST` | `/chat`                                             | Context-aware chat with memory retention    |
| `POST` | `/proposals/initialize/{project_id}`                | Upload RFP/assets and build project context |
| `POST` | `/proposals/generate/{project_id}/{section}`        | Synchronously generate a proposal section   |
| `POST` | `/proposals/generate/{project_id}/{section}/stream` | Stream (SSE) proposal section generation    |

---

## Deployment (VPS / Production)

For deploying this service in a production environment (Virtual Private Server), Nginx reverse proxy configuration, SSL setup, and log rotations:

👉 Refer to the **[VPS Deployment Guide (DEPLOYMENT.md)](file:///home/amr-mohamed27/trseah/poc/ai/DEPLOYMENT.md)**.
