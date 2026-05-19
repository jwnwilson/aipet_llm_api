output "repository_url" {
  description = "ECR repository URL — use this as the image in infra/k8s/deployment.yaml"
  value       = module.ecr.repository_url
}

output "repository_arn" {
  description = "ECR repository ARN"
  value       = module.ecr.repository_arn
}

output "ecr_push_policy_arn" {
  description = "IAM policy ARN granting ECR push — attach to your CI/CD role"
  value       = module.ecr.ecr_push_policy_arn
}

output "docker_login_command" {
  description = "Command to authenticate Docker with ECR before pushing"
  value       = "aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${module.ecr.repository_url}"
}

output "github_actions_role_arn" {
  description = "IAM role ARN for GitHub Actions OIDC — set this as the AWS_ROLE_ARN repository secret"
  value       = module.iam.github_actions_role_arn
}

output "aipet_aws_access_key_id" {
  description = "Access key ID for the aipet app IAM user — set as AIPET_AWS_ACCESS_KEY_ID secret"
  value       = module.iam.aipet_aws_access_key_id
}

output "aipet_aws_secret_access_key" {
  description = "Secret access key for the aipet app IAM user — set as AIPET_AWS_SECRET_ACCESS_KEY secret"
  value       = module.iam.aipet_aws_secret_access_key
  sensitive   = true
}

output "aipet_llm_api_fqdn" {
  description = "DNS name for the aipet LLM API"
  value       = module.dns.fqdn
}

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
