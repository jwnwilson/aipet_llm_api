# TypeScript Frontend Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the existing `llm-ui` React/Vite/TypeScript app into this repo as `ui/`, deploy it to S3 + CloudFront at `llm.jwnwilson.co.uk` via GitHub Actions.

**Architecture:** The UI lives in `ui/` as a self-contained Vite app alongside the Python backend. In dev, Vite proxies `/api/*` to the local FastAPI server. In production, GitHub Actions builds and syncs static files to a private S3 bucket served by CloudFront. The Terraform in `infra/terraform/` is extended with reusable modules copied from the `aipet` repo.

**Tech Stack:** React 19, Vite 8, TypeScript 6, Tailwind CSS 4, Auth0, Terraform (AWS S3 + CloudFront + ACM + Route53), GitHub Actions OIDC

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `ui/` | Create | Entire frontend app |
| `ui/vite.config.ts` | Modify | Add dev proxy for `/api` |
| `ui/.env.local.example` | Modify | Update audience URL |
| `ui/index.html` | Modify | Update page title |
| `infra/terraform/modules/s3_static/main.tf` | Create | S3 bucket + CloudFront distribution |
| `infra/terraform/modules/s3_static/variables.tf` | Create | Module inputs |
| `infra/terraform/modules/s3_static/outputs.tf` | Create | Bucket name, CloudFront domain/ARN/ID |
| `infra/terraform/modules/acm/main.tf` | Create | ACM certificate + DNS validation |
| `infra/terraform/modules/acm/variables.tf` | Create | Module inputs |
| `infra/terraform/modules/acm/outputs.tf` | Create | Certificate ARN |
| `infra/terraform/modules/iam/variables.tf` | Modify | Add `ui_bucket_arn`, `ui_distribution_arn` |
| `infra/terraform/modules/iam/main.tf` | Modify | Add UI S3+CloudFront deploy policy |
| `infra/terraform/modules/dns/variables.tf` | Modify | Add `ui_cf_domain` |
| `infra/terraform/modules/dns/main.tf` | Modify | Add CNAME for `llm.jwnwilson.co.uk` |
| `infra/terraform/modules/dns/outputs.tf` | Modify | Add `ui_fqdn` |
| `infra/terraform/main.tf` | Modify | Add `acm_ui`, `s3_ui` modules; wire IAM + DNS |
| `infra/terraform/outputs.tf` | Modify | Add `ui_bucket_name`, `ui_distribution_id`, `ui_fqdn` |
| `.github/workflows/deploy-ui.yml` | Create | Build UI, sync to S3, invalidate CloudFront |

---

## Task 1: Copy `ui/` source from aipet

**Files:**
- Create: `ui/` (entire directory)

- [ ] **Step 1: Copy llm-ui source**

```bash
rsync -av \
  --exclude='node_modules' \
  --exclude='.turbo' \
  --exclude='dist' \
  --exclude='.env.local' \
  /Users/noel/projects/aipet/apps/llm-ui/ \
  ui/
```

- [ ] **Step 2: Update page title in `ui/index.html`**

Find this line:
```html
    <title>aipet_llm_ui</title>
```

Replace with:
```html
    <title>LLM API</title>
```

- [ ] **Step 3: Install dependencies and verify the app builds**

```bash
cd ui && npm install && npm run build
```

Expected: `dist/` directory created with no TypeScript errors.

- [ ] **Step 4: Commit**

```bash
git add ui/
git commit -m "feat: add llm-ui as ui/ subdirectory"
```

---

## Task 2: Configure dev proxy and update env example

**Files:**
- Modify: `ui/vite.config.ts`
- Modify: `ui/.env.local.example`

- [ ] **Step 1: Add dev proxy to `ui/vite.config.ts`**

Replace the entire file with:

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

- [ ] **Step 2: Update `ui/.env.local.example`**

Replace the entire file with:

```
# Copy to .env.local and fill in values from your Auth0 tenant.
# Application: Single Page Application
VITE_AUTH0_DOMAIN=your-tenant.auth0.com
VITE_AUTH0_CLIENT_ID=your-client-id
# API Identifier from Auth0 APIs dashboard
VITE_AUTH0_AUDIENCE=https://api.llm.jwnwilson.co.uk
# Base URL for the LLM API (default: empty — dev proxy handles /api routing)
VITE_API_URL=
```

- [ ] **Step 3: Verify build still passes**

```bash
cd ui && npm run build
```

Expected: exits 0 with no errors.

- [ ] **Step 4: Commit**

```bash
git add ui/vite.config.ts ui/.env.local.example
git commit -m "feat: add vite dev proxy for /api and update env example"
```

---

## Task 3: Copy Terraform modules from aipet

**Files:**
- Create: `infra/terraform/modules/s3_static/main.tf`
- Create: `infra/terraform/modules/s3_static/variables.tf`
- Create: `infra/terraform/modules/s3_static/outputs.tf`
- Create: `infra/terraform/modules/acm/main.tf`
- Create: `infra/terraform/modules/acm/variables.tf`
- Create: `infra/terraform/modules/acm/outputs.tf`

- [ ] **Step 1: Copy the `s3_static` module**

```bash
cp -r /Users/noel/projects/aipet/infra/terraform/modules/s3_static \
      infra/terraform/modules/s3_static
```

- [ ] **Step 2: Copy the `acm` module**

```bash
cp -r /Users/noel/projects/aipet/infra/terraform/modules/acm \
      infra/terraform/modules/acm
```

- [ ] **Step 3: Verify module files are present**

```bash
ls infra/terraform/modules/s3_static/
ls infra/terraform/modules/acm/
```

Expected for each: `main.tf  outputs.tf  variables.tf`

- [ ] **Step 4: Commit**

```bash
git add infra/terraform/modules/s3_static/ infra/terraform/modules/acm/
git commit -m "feat: add s3_static and acm terraform modules"
```

---

## Task 4: Update IAM module to allow UI deployment

**Files:**
- Modify: `infra/terraform/modules/iam/variables.tf`
- Modify: `infra/terraform/modules/iam/main.tf`

- [ ] **Step 1: Add variables to `infra/terraform/modules/iam/variables.tf`**

Append to the end of the file:

```hcl

variable "ui_bucket_arn" {
  description = "ARN of the S3 bucket hosting the UI static files"
  type        = string
  default     = ""
}

variable "ui_distribution_arn" {
  description = "ARN of the CloudFront distribution serving the UI"
  type        = string
  default     = ""
}
```

- [ ] **Step 2: Add UI deploy policy to `infra/terraform/modules/iam/main.tf`**

Append to the end of the file (after the last resource):

```hcl

data "aws_iam_policy_document" "ui_deploy" {
  count = var.ui_bucket_arn != "" ? 1 : 0

  statement {
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [var.ui_bucket_arn]
  }

  statement {
    effect    = "Allow"
    actions   = ["s3:PutObject", "s3:DeleteObject"]
    resources = ["${var.ui_bucket_arn}/*"]
  }

  statement {
    effect    = "Allow"
    actions   = ["cloudfront:CreateInvalidation"]
    resources = [var.ui_distribution_arn]
  }
}

resource "aws_iam_policy" "ui_deploy" {
  count  = var.ui_bucket_arn != "" ? 1 : 0
  name   = "${var.repo_name}-ui-deploy"
  policy = data.aws_iam_policy_document.ui_deploy[0].json
}

resource "aws_iam_role_policy_attachment" "ui_deploy" {
  count      = var.ui_bucket_arn != "" ? 1 : 0
  role       = aws_iam_role.github_actions.name
  policy_arn = aws_iam_policy.ui_deploy[0].arn
}
```

- [ ] **Step 3: Validate Terraform syntax**

```bash
cd infra/terraform && terraform validate
```

Expected: `Success! The configuration is valid.`

- [ ] **Step 4: Commit**

```bash
git add infra/terraform/modules/iam/
git commit -m "feat: add UI S3+CloudFront deploy permissions to IAM module"
```

---

## Task 5: Update DNS module for `llm.jwnwilson.co.uk`

**Files:**
- Modify: `infra/terraform/modules/dns/variables.tf`
- Modify: `infra/terraform/modules/dns/main.tf`
- Modify: `infra/terraform/modules/dns/outputs.tf`

- [ ] **Step 1: Add `ui_cf_domain` variable to `infra/terraform/modules/dns/variables.tf`**

Append to the end of the file:

```hcl

variable "ui_cf_domain" {
  description = "CloudFront domain name for the UI (e.g. d1234abcd.cloudfront.net)"
  type        = string
  default     = ""
}
```

- [ ] **Step 2: Add CNAME record to `infra/terraform/modules/dns/main.tf`**

Append to the end of the file:

```hcl

resource "aws_route53_record" "llm_ui" {
  count   = var.ui_cf_domain != "" ? 1 : 0
  zone_id = data.aws_route53_zone.zone.zone_id
  name    = "llm.${trimsuffix(var.zone_name, ".")}"
  type    = "CNAME"
  ttl     = 300
  records = [var.ui_cf_domain]
}
```

- [ ] **Step 3: Add `ui_fqdn` output to `infra/terraform/modules/dns/outputs.tf`**

Append to the end of the file:

```hcl

output "ui_fqdn" {
  description = "FQDN of the llm-ui DNS record"
  value       = length(aws_route53_record.llm_ui) > 0 ? aws_route53_record.llm_ui[0].fqdn : ""
}
```

- [ ] **Step 4: Validate Terraform syntax**

```bash
cd infra/terraform && terraform validate
```

Expected: `Success! The configuration is valid.`

- [ ] **Step 5: Commit**

```bash
git add infra/terraform/modules/dns/
git commit -m "feat: add llm.jwnwilson.co.uk CNAME to DNS module"
```

---

## Task 6: Wire modules together in `main.tf` and `outputs.tf`

**Files:**
- Modify: `infra/terraform/main.tf`
- Modify: `infra/terraform/outputs.tf`

- [ ] **Step 1: Update `infra/terraform/main.tf`**

Replace the entire file with:

```hcl
provider "aws" {
  region = var.aws_region
}

module "ecr" {
  source                = "./modules/ecr"
  repo_name             = var.repo_name
  image_retention_count = var.image_retention_count
}

module "ecr_temporal_ui" {
  source                = "./modules/ecr"
  repo_name             = "aipet-temporal-ui"
  image_retention_count = var.image_retention_count
}

module "acm_ui" {
  source = "./modules/acm"
  domain = "llm.jwnwilson.co.uk"
}

module "s3_ui" {
  source              = "./modules/s3_static"
  name                = "llm-api-ui"
  domain              = "llm.jwnwilson.co.uk"
  acm_certificate_arn = module.acm_ui.certificate_arn
}

module "iam" {
  source                     = "./modules/iam"
  repo_name                  = var.repo_name
  github_repo                = var.github_repo
  s3_bucket                  = var.s3_bucket
  ecr_push_policy_arn        = module.ecr.ecr_push_policy_arn
  extra_ecr_push_policy_arns = [module.ecr_temporal_ui.ecr_push_policy_arn]
  ecr_pull_repo_arns         = [module.ecr.repository_arn, module.ecr_temporal_ui.repository_arn]
  ui_bucket_arn              = module.s3_ui.bucket_arn
  ui_distribution_arn        = module.s3_ui.distribution_arn
}

module "dns" {
  source        = "./modules/dns"
  vps_ip        = var.vps_ip
  ui_cf_domain  = module.s3_ui.cloudfront_domain
}
```

- [ ] **Step 2: Add UI outputs to `infra/terraform/outputs.tf`**

Append to the end of the file:

```hcl

output "ui_bucket_name" {
  description = "S3 bucket for the UI — set as UI_BUCKET GitHub secret after apply"
  value       = module.s3_ui.bucket_name
}

output "ui_distribution_id" {
  description = "CloudFront distribution ID for the UI — set as UI_CF_DISTRIBUTION_ID GitHub secret after apply"
  value       = module.s3_ui.distribution_id
}

output "ui_fqdn" {
  description = "Public URL for the UI"
  value       = module.dns.ui_fqdn
}
```

- [ ] **Step 3: Validate Terraform**

```bash
cd infra/terraform && terraform validate
```

Expected: `Success! The configuration is valid.`

- [ ] **Step 4: Run `terraform plan` to preview changes (read-only, no apply)**

```bash
cd infra/terraform && terraform plan
```

Expected: plan shows new resources — `module.acm_ui`, `module.s3_ui`, additions to `module.iam` and `module.dns`. No unexpected destroys.

- [ ] **Step 5: Commit**

```bash
git add infra/terraform/main.tf infra/terraform/outputs.tf
git commit -m "feat: wire acm_ui, s3_ui, UI IAM permissions and DNS into terraform"
```

---

## Task 7: Create GitHub Actions deploy workflow

**Files:**
- Create: `.github/workflows/deploy-ui.yml`

- [ ] **Step 1: Create `.github/workflows/deploy-ui.yml`**

```yaml
name: Deploy UI

on:
  workflow_run:
    workflows: [Test]
    types: [completed]
    branches: [main]
  workflow_dispatch:

permissions:
  id-token: write
  contents: read

env:
  AWS_REGION: us-east-1

jobs:
  deploy:
    if: ${{ github.event_name == 'workflow_dispatch' || github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Node
        uses: actions/setup-node@v4
        with:
          node-version: '22'
          cache: 'npm'
          cache-dependency-path: ui/package-lock.json

      - name: Install dependencies
        working-directory: ui
        run: npm ci

      - name: Build
        working-directory: ui
        env:
          VITE_AUTH0_DOMAIN: ${{ secrets.VITE_AUTH0_DOMAIN }}
          VITE_AUTH0_CLIENT_ID: ${{ secrets.VITE_AUTH0_CLIENT_ID }}
          VITE_AUTH0_AUDIENCE: ${{ secrets.VITE_AUTH0_AUDIENCE }}
          VITE_API_URL: ${{ secrets.VITE_API_URL }}
        run: npm run build

      - name: Configure AWS credentials via OIDC
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Sync to S3
        run: aws s3 sync ui/dist/ s3://${{ secrets.UI_BUCKET }} --delete

      - name: Invalidate CloudFront cache
        run: |
          aws cloudfront create-invalidation \
            --distribution-id ${{ secrets.UI_CF_DISTRIBUTION_ID }} \
            --paths "/*"
```

- [ ] **Step 2: Verify YAML is valid**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/deploy-ui.yml'))" && echo "valid"
```

Expected: `valid`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/deploy-ui.yml
git commit -m "feat: add deploy-ui GitHub Actions workflow"
```

---

## Task 8: Apply Terraform and set GitHub secrets

This task requires AWS credentials with sufficient permissions to run `terraform apply`.

- [ ] **Step 1: Apply Terraform**

```bash
cd infra/terraform && terraform apply
```

Review the plan output and type `yes` when prompted. This will create:
- ACM certificate for `llm.jwnwilson.co.uk` (takes ~2 min for DNS validation)
- S3 bucket `llm-api-ui`
- CloudFront distribution (takes ~10 min to deploy globally)
- IAM policy for UI deployment
- Route53 CNAME record

- [ ] **Step 2: Read the outputs**

```bash
cd infra/terraform && terraform output
```

Note the values of `ui_bucket_name` and `ui_distribution_id`.

- [ ] **Step 3: Set GitHub repository secrets**

In the GitHub repo settings → Secrets and variables → Actions, add:

| Secret | Value (from terraform output) |
|--------|-------------------------------|
| `UI_BUCKET` | value of `ui_bucket_name` |
| `UI_CF_DISTRIBUTION_ID` | value of `ui_distribution_id` |
| `VITE_AUTH0_DOMAIN` | your Auth0 tenant domain |
| `VITE_AUTH0_CLIENT_ID` | your Auth0 SPA client ID |
| `VITE_AUTH0_AUDIENCE` | `https://api.llm.jwnwilson.co.uk` |
| `VITE_API_URL` | `https://aipet-llm-api.jwnwilson.co.uk` |

- [ ] **Step 4: Trigger the deploy workflow manually to verify**

```bash
gh workflow run deploy-ui.yml
```

Then monitor:

```bash
gh run watch
```

Expected: workflow completes successfully. Visit `https://llm.jwnwilson.co.uk` in a browser — the login page should load.
