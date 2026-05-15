# State migration: flat resources → module addresses.
# These blocks tell Terraform to rename existing state entries rather than
# destroy and recreate resources. Safe to remove once all teammates have
# run `terraform apply` after this change.

# ECR
moved {
  from = aws_ecr_repository.aipet_llm
  to   = module.ecr.aws_ecr_repository.this
}

moved {
  from = aws_ecr_lifecycle_policy.aipet_llm
  to   = module.ecr.aws_ecr_lifecycle_policy.this
}

moved {
  from = aws_iam_policy.ecr_push
  to   = module.ecr.aws_iam_policy.ecr_push
}

# IAM — GitHub Actions
moved {
  from = aws_iam_openid_connect_provider.github
  to   = module.iam.aws_iam_openid_connect_provider.github
}

moved {
  from = aws_iam_role.github_actions
  to   = module.iam.aws_iam_role.github_actions
}

moved {
  from = aws_iam_role_policy_attachment.github_actions_ecr
  to   = module.iam.aws_iam_role_policy_attachment.github_actions_ecr
}

moved {
  from = aws_iam_policy.s3_model_read
  to   = module.iam.aws_iam_policy.s3_model_read
}

moved {
  from = aws_iam_role_policy_attachment.github_actions_s3
  to   = module.iam.aws_iam_role_policy_attachment.github_actions_s3
}

# IAM — aipet app user
moved {
  from = aws_iam_user.aipet
  to   = module.iam.aws_iam_user.aipet
}

moved {
  from = aws_iam_policy.aipet_s3
  to   = module.iam.aws_iam_policy.aipet_s3
}

moved {
  from = aws_iam_user_policy_attachment.aipet_s3
  to   = module.iam.aws_iam_user_policy_attachment.aipet_s3
}

moved {
  from = aws_iam_access_key.aipet
  to   = module.iam.aws_iam_access_key.aipet
}

# DNS
moved {
  from = aws_route53_record.aipet_llm_api
  to   = module.dns.aws_route53_record.aipet_llm_api
}
