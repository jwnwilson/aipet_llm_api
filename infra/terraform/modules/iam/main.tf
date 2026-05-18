# GitHub Actions OIDC — allows the workflow to authenticate to AWS without
# storing long-lived access keys as secrets. Scoped to pushes on main only.

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]

  # GitHub's OIDC thumbprints — these are stable but check
  # https://github.blog/changelog/ if you see auth failures after a GitHub cert rotation.
  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1",
    "1c58a3a8518e8759bf075b76b750d4f2df264fcd",
  ]
}

data "aws_iam_policy_document" "github_actions_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values = [
        "repo:${var.github_repo}:ref:refs/heads/main",
        "repo:${var.github_repo}:pull_request",
      ]
    }
  }
}

resource "aws_iam_role" "github_actions" {
  name               = "${var.repo_name}-github-actions"
  assume_role_policy = data.aws_iam_policy_document.github_actions_assume.json
}

resource "aws_iam_role_policy_attachment" "github_actions_ecr" {
  role       = aws_iam_role.github_actions.name
  policy_arn = var.ecr_push_policy_arn
}

resource "aws_iam_role_policy_attachment" "github_actions_ecr_extra" {
  for_each   = toset(var.extra_ecr_push_policy_arns)
  role       = aws_iam_role.github_actions.name
  policy_arn = each.value
}

data "aws_iam_policy_document" "s3_model_read" {
  statement {
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:HeadObject", "s3:ListBucket"]
    resources = [
      "arn:aws:s3:::${var.s3_bucket}",
      "arn:aws:s3:::${var.s3_bucket}/*",
    ]
  }
}

resource "aws_iam_policy" "s3_model_read" {
  name   = "${var.repo_name}-s3-model-read"
  policy = data.aws_iam_policy_document.s3_model_read.json
}

resource "aws_iam_role_policy_attachment" "github_actions_s3" {
  role       = aws_iam_role.github_actions.name
  policy_arn = aws_iam_policy.s3_model_read.arn
}

data "aws_iam_policy_document" "terraform_state" {
  statement {
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = ["arn:aws:s3:::${var.tf_state_bucket}"]
  }
  statement {
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["arn:aws:s3:::${var.tf_state_bucket}/terraform.tfstate"]
  }
  statement {
    effect    = "Allow"
    actions   = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem"]
    resources = ["arn:aws:dynamodb:*:*:table/aipet-llm-terraform-locks"]
  }
}

resource "aws_iam_policy" "terraform_state" {
  name   = "${var.repo_name}-terraform-state"
  policy = data.aws_iam_policy_document.terraform_state.json
}

resource "aws_iam_role_policy_attachment" "github_actions_terraform_state" {
  role       = aws_iam_role.github_actions.name
  policy_arn = aws_iam_policy.terraform_state.arn
}

# IAM user for the aipet application (RPi cluster + any non-OIDC workload).
# Scoped to S3 read/write on the project bucket only.

resource "aws_iam_user" "aipet" {
  name = "${var.repo_name}-app"
}

data "aws_iam_policy_document" "aipet_s3" {
  statement {
    effect  = "Allow"
    actions = ["s3:ListBucket"]
    resources = ["arn:aws:s3:::${var.s3_bucket}"]
  }

  statement {
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["arn:aws:s3:::${var.s3_bucket}/*"]
  }
}

resource "aws_iam_policy" "aipet_s3" {
  name   = "${var.repo_name}-app-s3"
  policy = data.aws_iam_policy_document.aipet_s3.json
}

resource "aws_iam_user_policy_attachment" "aipet_s3" {
  user       = aws_iam_user.aipet.name
  policy_arn = aws_iam_policy.aipet_s3.arn
}

resource "aws_iam_access_key" "aipet" {
  user = aws_iam_user.aipet.name
}
