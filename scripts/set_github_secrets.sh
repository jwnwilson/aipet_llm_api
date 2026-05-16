#!/usr/bin/env bash
# Pushes all required GitHub Actions secrets from a local .env file.
# Usage: ./scripts/set_github_secrets.sh [--repo owner/repo] [--env path/to/.env]
set -euo pipefail

REPO=""
ENV_FILE=".env"

while [[ $# -gt 0 ]]; do
  case $1 in
    --repo) REPO="$2"; shift 2 ;;
    --env)  ENV_FILE="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: env file '$ENV_FILE' not found. Copy .env.example to .env and fill in values." >&2
  exit 1
fi

REPO_FLAG=""
[[ -n "$REPO" ]] && REPO_FLAG="--repo $REPO"

# Map: GitHub secret name -> .env variable name
declare -A SECRETS=(
  [AWS_S3_BUCKET]=AWS_S3_BUCKET
  [AIPET_AWS_ACCESS_KEY_ID]=AWS_ACCESS_KEY_ID
  [AIPET_AWS_SECRET_ACCESS_KEY]=AWS_SECRET_ACCESS_KEY
  [AUTH0_DOMAIN]=AUTH0_DOMAIN
  [AUTH0_AUDIENCE]=AUTH0_AUDIENCE
  [AUTH0_CLIENT_ID]=AUTH0_CLIENT_ID
  [CORS_ORIGINS]=CORS_ORIGINS
)

set_secret() {
  local gh_name="$1"
  local env_name="$2"
  local value
  value=$(grep -E "^${env_name}=" "$ENV_FILE" | cut -d= -f2- | tr -d '"'"'" | head -1)
  if [[ -z "$value" ]]; then
    echo "  SKIP  $gh_name ($env_name not set in $ENV_FILE)"
    return
  fi
  # shellcheck disable=SC2086
  echo "$value" | gh secret set "$gh_name" $REPO_FLAG
  echo "  SET   $gh_name"
}

echo "Setting GitHub Actions secrets from $ENV_FILE..."
echo ""
for gh_name in "${!SECRETS[@]}"; do
  set_secret "$gh_name" "${SECRETS[$gh_name]}"
done

echo ""
echo "Done. Verify with: gh secret list ${REPO_FLAG}"
echo ""
echo "Note: AWS_ROLE_ARN and KUBE_CONFIG must be set separately — see README.md."
