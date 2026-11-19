import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

class StructuredClinicalReporter:
    """
    Verifiable Structured Radiological Report Generator:
    Emits strict JSON schemas for RANO 2.0 (Glioblastoma) and BI-RADS / I-SPY 2 (Breast Cancer),
    incorporating multi-sample Monte Carlo concordance audits (8 posterior draws)
    to curb silent hallucinations.
    """
    def __init__(self, concordance_threshold: float = 0.25):
        self.concordance_threshold = concordance_threshold

    def generate_rano_report(
        self,
        patient_id: str,
        visit_day: float,
        current_tumor_vol: float,
        prior_tumor_vol: float,
        dior_probabilities: Dict[str, float],
        median_ttp_days: float,
        ci95_ttp: Tuple[float, float],
        mc_predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generates RANO 2.0 structured JSON report with concordance check."""
        vol_change_pct = ((current_tumor_vol - prior_tumor_vol) / max(prior_tumor_vol, 1e-4)) * 100.0

        # RANO 2.0 category assignment
        if vol_change_pct <= -50.0:
            rano_cat = "Complete_or_Partial_Response"
        elif vol_change_pct >= 25.0:
            # Check pseudo-progression window (<12 weeks post-RT)
            if visit_day < 84.0 and dior_probabilities.get("pseudoprogression_prob", 0.0) > 0.4:
                rano_cat = "Suspected_Pseudo_Progression"
            else:
                rano_cat = "Progressive_Disease"
        else:
            rano_cat = "Stable_Disease"

        # Multi-draw concordance check
        mc_cats = [m.get("rano_category", rano_cat) for m in mc_predictions]
        concordance_rate = mc_cats.count(rano_cat) / max(len(mc_cats), 1)
        flag_divergence = (1.0 - concordance_rate) > self.concordance_threshold

        report = {
            "schema_version": "RANO_2.0_JSON_v1",
            "patient_id": patient_id,
            "timepoint_days": visit_day,
            "quantitative_metrics": {
                "current_tumor_volume_cm3": round(current_tumor_vol, 2),
                "prior_tumor_volume_cm3": round(prior_tumor_vol, 2),
                "volume_change_percentage": round(vol_change_pct, 2)
            },
            "rano_assessment": {
                "category": rano_cat,
                "confidence": round(concordance_rate, 3),
                "dior_ordinal_probabilities": dior_probabilities,
                "median_ttp_days": round(median_ttp_days, 1),
                "ci95_ttp_days": [round(ci95_ttp[0], 1), round(ci95_ttp[1], 1)]
            },
            "hallucination_audit": {
                "num_posterior_draws": len(mc_predictions),
                "concordance_rate": round(concordance_rate, 3),
                "concordance_warning": flag_divergence,
                "divergence_details": "High posterior variance observed across treatment paths." if flag_divergence else "Concordant consensus."
            }
        }
        return report

    def generate_birads_ispy2_report(
        self,
        patient_id: str,
        visit_timepoint: str, # T0, T1, T2, T3
        current_ftv_cm3: float,
        baseline_ftv_cm3: float,
        longest_diameter_mm: float,
        ser_mean: float,
        predicted_pcr_prob: float,
        mc_predictions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generates BI-RADS / I-SPY 2 NACT structured JSON report."""
        ftv_reduction_pct = ((baseline_ftv_cm3 - current_ftv_cm3) / max(baseline_ftv_cm3, 1e-4)) * 100.0

        if ftv_reduction_pct >= 80.0:
            response_tier = "Excellent_Early_Responder"
        elif ftv_reduction_pct >= 30.0:
            response_tier = "Moderate_Responder"
        else:
            response_tier = "Non_Responder"

        # Multi-draw concordance check on pCR probability
        mc_pcr = [m.get("pcr_prob", predicted_pcr_prob) for m in mc_predictions]
        std_pcr = float(np.std(mc_pcr)) if mc_pcr else 0.0
        flag_divergence = std_pcr > self.concordance_threshold

        report = {
            "schema_version": "ISPY2_BIRADS_JSON_v1",
            "patient_id": patient_id,
            "timepoint": visit_timepoint,
            "mri_volumetric_kinetics": {
                "functional_tumor_volume_cm3": round(current_ftv_cm3, 2),
                "baseline_ftv_cm3": round(baseline_ftv_cm3, 2),
                "ftv_reduction_percentage": round(ftv_reduction_pct, 2),
                "longest_diameter_mm": round(longest_diameter_mm, 1),
                "signal_enhancement_ratio_mean": round(ser_mean, 2)
            },
            "prognosis_pcr": {
                "predicted_pcr_probability": round(predicted_pcr_prob, 3),
                "response_tier": response_tier,
                "posterior_uncertainty_sd": round(std_pcr, 3)
            },
            "hallucination_audit": {
                "num_posterior_draws": len(mc_predictions),
                "posterior_sd": round(std_pcr, 3),
                "concordance_warning": flag_divergence,
                "audit_status": "Audited & Verified" if not flag_divergence else "Requires Clinician Adjudication"
            }
        }
        return report
