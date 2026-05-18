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

module "iam" {
  source                     = "./modules/iam"
  repo_name                  = var.repo_name
  github_repo                = var.github_repo
  s3_bucket                  = var.s3_bucket
  ecr_push_policy_arn        = module.ecr.ecr_push_policy_arn
  extra_ecr_push_policy_arns = [module.ecr_temporal_ui.ecr_push_policy_arn]
}

module "dns" {
  source  = "./modules/dns"
  vps_ip  = var.vps_ip
}
