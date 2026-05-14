output "repository_url" {
  description = "ECR repository URL — use this as the image in infra/k8s/deployment.yaml"
  value       = aws_ecr_repository.aipet_llm.repository_url
}

output "repository_arn" {
  description = "ECR repository ARN"
  value       = aws_ecr_repository.aipet_llm.arn
}

output "ecr_push_policy_arn" {
  description = "IAM policy ARN granting ECR push — attach to your CI/CD role"
  value       = aws_iam_policy.ecr_push.arn
}

output "docker_login_command" {
  description = "Command to authenticate Docker with ECR before pushing"
  value       = "aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${aws_ecr_repository.aipet_llm.repository_url}"
}

output "github_actions_role_arn" {
  description = "IAM role ARN for GitHub Actions OIDC — set this as the AWS_ROLE_ARN repository secret"
  value       = aws_iam_role.github_actions.arn
}

output "aipet_aws_access_key_id" {
  description = "Access key ID for the aipet app IAM user — set as AIPET_AWS_ACCESS_KEY_ID secret"
  value       = aws_iam_access_key.aipet.id
}

output "aipet_aws_secret_access_key" {
  description = "Secret access key for the aipet app IAM user — set as AIPET_AWS_SECRET_ACCESS_KEY secret"
  value       = aws_iam_access_key.aipet.secret
  sensitive   = true
}
