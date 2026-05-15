data "aws_route53_zone" "zone" {
  name         = var.zone_name
  private_zone = false
}

resource "aws_route53_record" "aipet_llm_api" {
  zone_id = data.aws_route53_zone.zone.zone_id
  name    = "aipet-llm-api.${trimsuffix(var.zone_name, ".")}"
  type    = "A"
  ttl     = 300
  records = [var.vps_ip]
}
