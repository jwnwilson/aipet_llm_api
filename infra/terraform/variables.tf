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

variable "s3_bucket" {
  description = "S3 bucket name used to store models — grants the GitHub Actions role read access"
  type        = string
  default     = "aipet-jwn"
}

variable "vps_ip" {
  description = "Public IP of the VPS / inlets exit node — used for the aipet-llm-api DNS A record"
  type        = string
  default     = "165.22.115.52"
}
