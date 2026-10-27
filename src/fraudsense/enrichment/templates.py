"""Fixed, pre-approved simulated customer challenge templates."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CustomerTemplate:
    template_id: str
    clause_id: str
    channel: str
    message: str


TEMPLATES = {
    "TMPL_CHALLENGE_UPI_01": CustomerTemplate(
        "TMPL_CHALLENGE_UPI_01",
        "MD-3.4.2",
        "SMS",
        "SIMULATED: FraudSense needs you to verify a simulated payment of ₹{amount:.2f}. No real payment is affected. [MD-3.4.2]",
    )
}


def render_template(template_id: str, amount: float) -> str:
    """Renders a known template only; runtime-generated customer text is disallowed."""
    if template_id not in TEMPLATES:
        raise ValueError(f"Unknown approved template: {template_id}")
    return TEMPLATES[template_id].message.format(amount=amount)
