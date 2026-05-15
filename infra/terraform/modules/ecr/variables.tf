variable "repo_name" {
  description = "ECR repository name"
  type        = string
}

variable "image_retention_count" {
  description = "Number of tagged images to retain before expiring older ones"
  type        = number
  default     = 10
}
