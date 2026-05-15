data "aws_route53_zone" "jwnwilson" {
  name         = "jwnwilson.co.uk."
  private_zone = false
}

resource "aws_route53_record" "aipet_llm_api" {
  zone_id = data.aws_route53_zone.jwnwilson.zone_id
  name    = "aipet-llm-api.jwnwilson.co.uk"
  type    = "A"
  ttl     = 300
  records = [var.vps_ip]
}
