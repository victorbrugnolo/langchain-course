# LangChain Course

A comprehensive hands-on course exploring LangChain framework capabilities, from fundamentals to advanced topics including vector databases, RAG systems, and AI agents.

## Overview

This repository contains practical examples and implementations demonstrating various LangChain features and patterns. The project progresses from basic chat model interactions to complex RAG (Retrieval-Augmented Generation) systems with vector search capabilities.

## Features

- **Fundamentals**: Chat models, prompt templates, and basic LangChain operations
- **Chains & Processing**: Pipeline creation, decorators, summarization, and map-reduce patterns
- **Memory Management**: Conversation history storage and sliding window techniques
- **Agents & Tools**: ReAct agents with custom and Hub-based prompts
- **Vector Databases**: Document ingestion, embeddings, and similarity search using pgvector
- **RAG System**: Complete retrieval-augmented generation pipeline with PostgreSQL

## Project Structure

```
.
├── fundamentals/
│   ├── hello_world.py                    # Basic chat model initialization
│   ├── init_chat_model.py                # Chat model setup
│   ├── prompt_template.py                # Prompt template examples
│   └── chat_prompt_template.py           # Chat-specific prompts
│
├── chains-and-processing/
│   ├── starting_with_chains.py           # Chain fundamentals
│   ├── chains_with_decorators.py         # Decorator-based chains
│   ├── runnable_lambdas.py               # Lambda runnables
│   ├── processing_pipeline.py            # Pipeline examples
│   ├── summarize.py                      # Text summarization
│   ├── summarize_with_map_reduce.py      # Distributed summarization
│   └── pipeline_summarize.py             # Pipeline-based summarization
│
├── memory-management/
│   ├── history_storage.py                # Conversation history persistence
│   └── history_based_on_sliding_window.py # Sliding window memory
│
├── agents-and-tools/
│   ├── agent_react_and_tools.py          # ReAct agent with custom tools
│   └── agent_react_using_prompt_hub.py   # ReAct agent with Hub prompts
│
├── loaders-and-vectorized-databases/
│   ├── load_using_web_base_loader.py     # Web content loading
│   ├── load_pdf.py                       # PDF document loading
│   ├── ingestion_pgvector.py             # Document ingestion to pgvector
│   └── search_vector.py                  # Vector similarity search
│
└── docker-compose.yaml                    # PostgreSQL with pgvector setup
```

## Prerequisites

- Python 3.13
- Pipenv
- Docker and Docker Compose (for vector database)
- OpenAI API key
- Google Gemini API key (optional)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd langchain-course
```

2. Install dependencies using Pipenv:
```bash
pipenv install
```

3. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```
OPENAI_API_KEY=your_openapi_key_here
GOOGLE_API_KEY=your_google_gemini_api_key_here
OPEN_AI_MODEL=text-embedding-3-small
PGVECTOR_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PGVECTOR_COLLECTION=gpt5_collection
```

4. Start the PostgreSQL database with pgvector:
```bash
docker-compose up -d
```

The Docker setup includes:
- PostgreSQL 17 with pgvector extension
- Automatic vector extension bootstrapping
- Persistent data storage
- Health checks

## Usage

Activate the Pipenv environment:
```bash
pipenv shell
```

Run any example script:
```bash
# Fundamentals
python fundamentals/hello_world.py

# Chains and processing
python chains-and-processing/chains_with_decorators.py

# Memory management
python memory-management/history_storage.py

# Agents
python agents-and-tools/agent_react_and_tools.py

# Vector database operations
python loaders-and-vectorized-databases/ingestion_pgvector.py
python loaders-and-vectorized-databases/search_vector.py
```

## Key Concepts

### Chains
LangChain chains allow you to combine multiple components into sequential processing pipelines. Examples include:
- Basic chains with prompt templates
- Decorator-based chain creation
- Runnable lambdas for custom transformations

### Memory Management
Maintain conversation context with:
- In-memory chat history storage
- Sliding window for managing token limits
- Session-based conversation tracking

### Agents & Tools
ReAct (Reasoning + Acting) agents that:
- Use tools to gather information
- Execute multi-step reasoning
- Handle custom and pre-built prompts from LangChain Hub

### Vector Databases & RAG
Complete RAG implementation featuring:
- Document loading from web and PDF sources
- Text embedding generation (OpenAI embeddings)
- Vector storage in PostgreSQL with pgvector
- Similarity search for relevant document retrieval

## Dependencies

Main packages:
- `langchain` - Core LangChain framework
- `langchain-openai` - OpenAI integrations
- `langchain-google-genai` - Google Gemini integrations
- `langchain-community` - Community integrations
- `langchain-postgres` - PostgreSQL vector store
- `python-dotenv` - Environment variable management
- `beautifulsoup4` - Web scraping for loaders
- `pypdf` - PDF document processing

## Development Timeline

Based on git history, this course was developed with the following progression:

1. **Fundamentals** - Chat model initialization and prompt templates
2. **Chains** - Pipeline creation and processing patterns
3. **Processing** - Summarization and map-reduce implementations
4. **Agents** - ReAct pattern and tool integration
5. **Memory** - Conversation history and window management
6. **Loaders** - Web and PDF document ingestion
7. **Vector Database** - pgvector setup and integration
8. **RAG System** - Document ingestion and similarity search

## License

This project is for educational purposes.

## Contributing

This is a learning project. Feel free to fork and experiment with the examples.
