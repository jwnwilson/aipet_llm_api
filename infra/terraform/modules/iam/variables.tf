variable "repo_name" {
  description = "Project name — used to prefix IAM resource names"
  type        = string
}

variable "github_repo" {
  description = "GitHub repository in owner/name format — scopes the OIDC trust to main-branch pushes"
  type        = string
}

variable "s3_bucket" {
  description = "S3 bucket name used to store models"
  type        = string
}

variable "ecr_push_policy_arn" {
  description = "ARN of the ECR push IAM policy — attached to the GitHub Actions role"
  type        = string
}

variable "tf_state_bucket" {
  description = "S3 bucket name used for Terraform remote state — grants GitHub Actions read/write access"
  type        = string
  default     = "aipet-llm-terraform-state"
}
