#!/usr/bin/env bash
set -euo pipefail

echo "=== Smoke Tests ==="

# 1. Verify aipet-llm health endpoint responds 200
echo "-- Checking /health endpoint..."
kubectl exec deploy/aipet-llm -- curl -sf http://localhost:8000/health
echo ""

# 2. Verify Alembic migrations created the expected tables
echo "-- Checking database tables..."
TABLES=$(kubectl exec aipet-db-0 -- psql -U aipet -d aipet -t -c \
  "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename;")
echo "$TABLES"

for table in alembic_version training_models training_runs; do
  if ! echo "$TABLES" | grep -q "$table"; then
    echo "ERROR: Expected table '$table' not found in database"
    exit 1
  fi
done

echo "=== Smoke tests passed ==="
