output "github_actions_role_arn" {
  description = "IAM role ARN for GitHub Actions OIDC"
  value       = aws_iam_role.github_actions.arn
}

output "aipet_aws_access_key_id" {
  description = "Access key ID for the aipet app IAM user"
  value       = aws_iam_access_key.aipet.id
}

output "aipet_aws_secret_access_key" {
  description = "Secret access key for the aipet app IAM user"
  value       = aws_iam_access_key.aipet.secret
  sensitive   = true
}
