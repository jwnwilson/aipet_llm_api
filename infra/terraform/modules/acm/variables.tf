variable "domain" {
  description = "Domain name to issue the ACM certificate for"
  type        = string
}

variable "zone_name" {
  description = "Route 53 hosted zone name (trailing dot required, e.g. jwnwilson.co.uk.)"
  type        = string
  default     = "jwnwilson.co.uk."
}
