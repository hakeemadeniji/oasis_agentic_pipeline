from typing import Tuple


class MedicalEthicistAgent:
    """
    Agent 6: Medical Ethicist (Guardrail Agent).
    Enforces rigid clinical constraints, identifies multi-modal metric
    contradictions, and mitigates machine learning hallucination risks.
    """

    def __init__(self, confidence_floor: float = 65.0):
        self.confidence_floor = confidence_floor

    def audit_diagnostic_proposal(
        self, predicted_class: str, confidence: float, mmse_score: float, atrophy_velocity: float
    ) -> Tuple[bool, str]:
        """Audits decision states to ensure safety, clinical consistency, and protocol alignment."""

        # Rule 1: High Uncertainty Attenuation
        if confidence < self.confidence_floor:
            return (
                True,
                f"REJECTED: Sub-threshold confidence score ({confidence:.2f}%). Minimum required: {self.confidence_floor}%.",
            )

        # Rule 2: Cognitive vs Spatial Contradiction Guardrail
        # A patient with an MMSE > 27 mathematically cannot clinically map to Moderate Dementia
        if mmse_score >= 27.0 and predicted_class in ["Mild Dementia", "Moderate Dementia"]:
            return (
                True,
                f"REJECTED: Severe Cross-Modal Variance. Model suggested '{predicted_class}' but patient displays near-perfect cognitive function (MMSE: {mmse_score:.1f}).",
            )

        # Rule 3: Asymptomatic Silent Structural Degradation Alert
        # High brain tissue loss with normal cognitive scores suggests an active underlying compensation mechanism
        if mmse_score >= 28.0 and atrophy_velocity > 2.0:
            return (
                False,
                "APPROVED WITH WARNING: Patient is cognitively intact, but displays high-risk structural brain tissue loss (>2.0%/yr). High potential for clinical conversion.",
            )

        # Rule 4: Critical Failure Check
        if mmse_score <= 12.0 and predicted_class == "Non Demented":
            return (
                True,
                f"REJECTED: Dangerous Type-II Error Risk. Model classified brain as completely clear, but cognitive scores indicate advanced structural impairment (MMSE: {mmse_score:.1f}).",
            )

        return (
            False,
            "VERIFIED: Comprehensive data streams are mathematically and clinically aligned.",
        )


if __name__ == "__main__":
    ethicist = MedicalEthicistAgent()
    print("[*] Simulating Ethicist Guardrail Audit against cross-modal contradiction...")

    is_flagged, alert_msg = ethicist.audit_diagnostic_proposal(
        predicted_class="Moderate Dementia", confidence=89.5, mmse_score=29.0, atrophy_velocity=0.4
    )
    print(f"Audit Status - Flagged: {is_flagged} | Reason: {alert_msg}")
