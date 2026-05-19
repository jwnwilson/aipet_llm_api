# LLM API — Admin UI

React + TypeScript admin interface for the LLM API service. Provides model management, training run monitoring, and user administration protected by Auth0.

## Prerequisites

- Node 22+
- LLM API backend running on `http://localhost:8000`

## Setup

```bash
cp .env.local.example .env.local
# Edit .env.local and fill in your Auth0 values:
#   VITE_AUTH0_DOMAIN
#   VITE_AUTH0_CLIENT_ID
#   VITE_AUTH0_AUDIENCE
```

## Development

```bash
npm install
npm run dev
```

The dev server starts at `http://localhost:5173`. The API must be running on port 8000.

## Tests

```bash
npm test
```

## Production build

```bash
npm run build
# Output written to dist/
```
