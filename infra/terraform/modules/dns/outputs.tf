output "fqdn" {
  description = "FQDN of the aipet LLM API DNS record"
  value       = aws_route53_record.aipet_llm_api.fqdn
}
