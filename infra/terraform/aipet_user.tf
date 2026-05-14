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
