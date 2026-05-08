variable "aws_region" {
  description = "AWS region for the ECR repository"
  type        = string
  default     = "us-east-1"
}

variable "repo_name" {
  description = "ECR repository name"
  type        = string
  default     = "aipet-llm"
}

variable "image_retention_count" {
  description = "Number of tagged images to retain before expiring older ones"
  type        = number
  default     = 10
}

variable "github_repo" {
  description = "GitHub repository in owner/name format — scopes the OIDC trust to main-branch pushes (e.g. myorg/aipet-llm)"
  type        = string
}
