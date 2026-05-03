# aipet-llm

AI pet companion project.

## Getting started

```bash
uv sync
uv run python main.py
```

## Building this project

1. Edit [docs/prd.md](docs/prd.md) — describe what you want to build
2. Ask Claude: *"Read the PRD and generate an implementation plan"*
3. Claude writes [docs/plan.md](docs/plan.md) with parallelisable task phases
4. Run each phase's tasks with parallel agents
