output "repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.this.repository_url
}

output "repository_arn" {
  description = "ECR repository ARN"
  value       = aws_ecr_repository.this.arn
}

output "ecr_push_policy_arn" {
  description = "IAM policy ARN granting ECR push"
  value       = aws_iam_policy.ecr_push.arn
}

output "aws_region" {
  description = "AWS region of the repository (for docker login command)"
  value       = aws_ecr_repository.this.registry_id
}
