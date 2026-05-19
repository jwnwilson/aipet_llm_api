variable "zone_name" {
  description = "Route 53 hosted zone name (e.g. jwnwilson.co.uk.)"
  type        = string
  default     = "jwnwilson.co.uk."
}

variable "vps_ip" {
  description = "Public IP of the VPS / inlets exit node"
  type        = string
}

variable "ui_cf_domain" {
  description = "CloudFront domain name for the UI (e.g. d1234abcd.cloudfront.net)"
  type        = string
  default     = ""
}
