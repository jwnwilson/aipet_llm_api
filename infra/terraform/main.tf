provider "aws" {
  region = var.aws_region
}

module "ecr" {
  source                = "./modules/ecr"
  repo_name             = var.repo_name
  image_retention_count = var.image_retention_count
}

module "iam" {
  source              = "./modules/iam"
  repo_name           = var.repo_name
  github_repo         = var.github_repo
  s3_bucket           = var.s3_bucket
  ecr_push_policy_arn = module.ecr.ecr_push_policy_arn
}

module "dns" {
  source  = "./modules/dns"
  vps_ip  = var.vps_ip
}
