output "bucket_name" {
  description = "S3 bucket name"
  value       = aws_s3_bucket.this.id
}

output "bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.this.arn
}

output "cloudfront_domain" {
  description = "CloudFront distribution domain name (for DNS CNAME)"
  value       = aws_cloudfront_distribution.this.domain_name
}

output "distribution_id" {
  description = "CloudFront distribution ID (for cache invalidation in CI)"
  value       = aws_cloudfront_distribution.this.id
}

output "distribution_arn" {
  description = "CloudFront distribution ARN (for IAM policy)"
  value       = aws_cloudfront_distribution.this.arn
}
