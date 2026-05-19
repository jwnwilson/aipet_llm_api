variable "name" {
  description = "Short name for this static site — used as the S3 bucket name and in resource names"
  type        = string
}

variable "domain" {
  description = "Full domain name served by this CloudFront distribution (e.g. aipet-v2.jwnwilson.co.uk)"
  type        = string
}

variable "acm_certificate_arn" {
  description = "ARN of the ACM certificate in us-east-1 covering this domain (e.g. wildcard *.jwnwilson.co.uk)"
  type        = string
}
