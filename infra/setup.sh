#!/usr/bin/env bash
# First-time CI/CD setup for EPIC-3.
# Run from the repo root: bash infra/setup.sh <github_repo>
# Example: bash infra/setup.sh myorg/aipet-llm
#
# Prerequisites:
#   - AWS CLI configured (aws configure or AWS_* env vars)
#   - GitHub CLI authenticated (gh auth login)
#   - Terraform >= 1.6 installed
#   - kubectl configured against your k3s cluster

set -euo pipefail

GITHUB_REPO="${1:?Usage: $0 <owner/repo>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAFORM_DIR="$SCRIPT_DIR/terraform"
K8S_DIR="$SCRIPT_DIR/k8s"

echo "=== 1. Provision AWS resources ==="
cd "$TERRAFORM_DIR"
terraform init -input=false
terraform apply -input=false -auto-approve \
  -var="github_repo=$GITHUB_REPO"

ROLE_ARN=$(terraform output -raw github_actions_role_arn)
REPO_URL=$(terraform output -raw repository_url)

echo "=== 2. Set GitHub Actions secrets ==="
gh secret set AWS_ROLE_ARN --repo "$GITHUB_REPO" --body "$ROLE_ARN"
gh secret set KUBECONFIG   --repo "$GITHUB_REPO" --body "$(kubectl config view --raw | base64)"

echo "=== 3. Patch ECR URL into k8s manifests ==="
cd "$K8S_DIR"

# Safe in-place replacement; .bak files are removed after.
for file in deployment.yaml temporal-worker-deployment.yaml; do
  sed -i.bak "s|<ECR_REPOSITORY_URL>|$REPO_URL|g" "$file"
  rm -f "${file}.bak"
done

echo "=== 4. Apply k8s manifests ==="
kubectl apply -R -f "$K8S_DIR/"

echo ""
echo "Done. ECR URL : $REPO_URL"
echo "IAM role ARN  : $ROLE_ARN"
echo ""
echo "Next: Install the self-hosted runner — see docs/cicd-runner-setup.md"
