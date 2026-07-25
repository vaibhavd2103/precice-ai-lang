# Codex Agent Brief for `precice-ai`

## Purpose

This file explains the project context and the next product goal for any Codex agent
working in this repository.

The goal is to evolve `precice-ai` into a LangGraph-based assistant for the preCICE
ecosystem that users can run locally, configure with an LLM provider API key, and use
to:

- understand the preCICE project and docs,
- read and write files inside a selected working directory,
- inspect local simulation/configuration files,
- answer preCICE-related questions,
- help users create or edit project files safely.

## Current Repository State

This repository already contains a working first version of a local preCICE assistant.

### Existing stack

- Python package: `precice-ai`
- Agent runtime: `langgraph`
- LLM client: `langchain-openai`
- Web server: `fastapi`
- Vector store: `chromadb`
- Embeddings: `sentence-transformers`
- Frontend: single static page in `static/index.html`

### Existing entry points

- CLI entry: `precice-ai`
- Web app server: `precice_ai/server.py`
- Agent graph: `precice_ai/graph.py`
- Tools: `precice_ai/tools.py`
- Ingestion pipeline: `precice_ai/ingest.py`
- Session state: `precice_ai/conversation.py`
- Settings: `precice_ai/config.py`

### Existing behavior

The app already supports:

- browser-based chat,
- session creation,
- working-directory selection,
- local file reads and writes inside the chosen directory,
- preCICE docs/forum search through a Chroma-backed knowledge base,
- live forum search,
- `precice-config.xml` validation through `precice-tools check`,
- streaming responses over SSE.

## New Product Direction

The next version should be framed as a reusable LangGraph agent for the preCICE package,
not only as a hardcoded OpenRouter demo.

Users should be able to provide:

- an OpenRouter API key, or
- another compatible provider API key/platform configuration,

and then use the agent with minimal setup.

## Primary Objective

Build a provider-flexible preCICE assistant that can:

1. accept LLM credentials/configuration from the user,
2. initialize a supported chat model,
3. use tools to inspect and modify project files,
4. use RAG over preCICE documentation and forum content,
5. answer questions about both:
   - preCICE in general,
   - the user's selected project specifically.

## Functional Requirements

### 1. Provider-flexible LLM setup

The agent should not be tightly coupled to OpenRouter only.

Target behavior:

- support OpenRouter by default,
- allow other OpenAI-compatible providers when possible,
- keep configuration simple and environment-driven,
- avoid forcing users to change code just to switch providers.

Minimum expected configuration shape:

- `LLM_PROVIDER`
- `LLM_API_KEY`
- `LLM_MODEL`
- `LLM_BASE_URL` when needed

OpenRouter should remain a first-class default, but the code should be organized so
additional providers are easy to add.

### 2. Safe file-system access

The agent must be able to:

- list project files,
- read project files,
- write project files,
- stay confined to the selected working directory.

Non-negotiable rule:

- no file access outside the chosen project root.

### 3. preCICE knowledge and project understanding

The agent should combine:

- general preCICE knowledge from ingested docs/forum sources,
- project-specific knowledge from local files selected by the user.

It should be able to answer questions such as:

- what this `precice-config.xml` is doing,
- why a coupling setup may fail,
- which participant, mesh, mapping, or data definitions are missing,
- what files in the project are relevant to a user question,
- how to modify a file to achieve a requested preCICE behavior.

### 4. Guided file editing

When asked to create or modify files, the agent should:

- inspect the relevant files first,
- explain the intended change,
- write the update only inside the allowed workspace,
- preserve existing project structure when possible.

### 5. Local-first developer experience

The system should remain lightweight:

- no database requirement,
- no heavy frontend framework requirement,
- no unnecessary infrastructure.

The current local-first architecture is a strength and should be preserved unless a
clear need appears.

## Non-Goals

Unless explicitly requested, do not redesign this project into:

- a cloud-hosted SaaS,
- a multi-tenant production platform,
- a database-heavy architecture,
- a React/Vue frontend rewrite,
- a generic agent unrelated to preCICE.

This project should stay focused on preCICE workflows.

## Recommended Design Direction

### LLM abstraction

Introduce a small model-factory layer so the graph does not directly construct one
hardcoded `ChatOpenAI` client.

Prefer a structure like:

- `config.py`: provider/model/base-url/api-key settings
- `llm.py` or equivalent: build the configured chat model
- `graph.py`: consume the prepared model and bind tools

### Tooling

Keep tools as plain, well-scoped functions.

Important tool categories:

- documentation retrieval,
- project file listing,
- project file reading,
- project file writing,
- config validation,
- optional future project summarization/indexing helpers.

### Prompts

The system prompt should consistently instruct the agent to:

- use tools when project context is needed,
- cite sources when answering from docs/forum content,
- avoid guessing when file inspection is required,
- confirm intent before destructive edits,
- stay within the selected workspace.

## Acceptance Criteria

The intended solution is successful when:

1. a user can start the app locally,
2. provide an API key and model/provider config,
3. select a project folder,
4. ask questions about preCICE or their local project,
5. get grounded answers based on docs, forum, and file contents,
6. request file creation or modification safely inside the project folder.

## Important Notes for Future Codex Work

- Prefer extending the current architecture over replacing it.
- Preserve the LangGraph tool-loop design already present in the repo.
- Keep the security boundary around `working_dir`.
- Keep the code simple and easy for local users to run.
- Favor minimal, focused changes over broad rewrites.

## Key Files to Read First

- `README.md`
- `CLAUDE.md`
- `pyproject.toml`
- `precice_ai/config.py`
- `precice_ai/graph.py`
- `precice_ai/tools.py`
- `precice_ai/server.py`
- `precice_ai/ingest.py`

## Summary

This repository already has a strong base implementation. The next step is to turn it
into a cleaner, provider-flexible LangGraph agent for preCICE users, while preserving
the current local-first, tool-enabled, safe file access model.
