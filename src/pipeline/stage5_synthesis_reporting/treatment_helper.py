import numpy as np
import torch
from typing import Dict, Any, List, Optional
from src.pipeline.stage5_synthesis_reporting.counterfactual import CounterfactualSimulator

class AgenticTreatmentHelper:
    """
    Agentic Treatment-Choice Helper:
    Simulates posterior counterfactual trajectory draws across guideline-eligible regimens
    and ranks interventions by expected benefit minus uncertainty penalty:
      Score(I_m) = E[TTP(I_m)] - lambda_unc * SD[TTP(I_m)]
    """
    def __init__(
        self,
        domain: str = "gbm",
        simulator: Optional[CounterfactualSimulator] = None,
        num_posterior_draws: int = 16,
        lambda_unc: float = 0.5
    ):
        self.domain = domain
        self.simulator = simulator if simulator is not None else CounterfactualSimulator(domain=domain)
        self.num_draws = num_posterior_draws
        self.lambda_unc = lambda_unc

    def get_candidate_regimens(self) -> Dict[str, List[Dict[str, Any]]]:
        """Returns standard guideline-eligible candidate schedules."""
        if self.domain == "gbm":
            return {
                "Standard_RT_60Gy_TMZ": [
                    {"t": float(i * 1.4), "type": "radiotherapy", "dose_gy": 2.0, "tmz_dose": 1.0}
                    for i in range(30) # 60 Gy across 30 fractions
                ],
                "Hypofractionated_RT_40Gy_TMZ": [
                    {"t": float(i * 1.4), "type": "radiotherapy", "dose_gy": 2.67, "tmz_dose": 1.0}
                    for i in range(15) # 40 Gy across 15 fractions
                ],
                "ShortCourse_RT_25Gy": [
                    {"t": float(i * 1.4), "type": "radiotherapy", "dose_gy": 5.0, "tmz_dose": 0.0}
                    for i in range(5)  # 25 Gy across 5 fractions
                ]
            }
        else:
            # Breast I-SPY 2 arms
            return {
                "Paclitaxel_Ganetespib_AC": [
                    {"t": float(i * 7.0), "type": "chemotherapy", "drug_vec": torch.tensor([1.0, 1.0] + [0.0]*14)}
                    for i in range(12) # T0-T2: 12 weeks
                ] + [
                    {"t": float(84.0 + i * 7.0), "type": "chemotherapy", "drug_vec": torch.tensor([0.0, 0.0, 1.0] + [0.0]*13)}
                    for i in range(12) # T2-T3: 12 weeks AC
                ],
                "Paclitaxel_MK2206_AC": [
                    {"t": float(i * 7.0), "type": "chemotherapy", "drug_vec": torch.tensor([1.0, 0.0, 1.0] + [0.0]*13)}
                    for i in range(12)
                ] + [
                    {"t": float(84.0 + i * 7.0), "type": "chemotherapy", "drug_vec": torch.tensor([0.0, 0.0, 1.0] + [0.0]*13)}
                    for i in range(12)
                ],
                "Paclitaxel_Control_AC": [
                    {"t": float(i * 7.0), "type": "chemotherapy", "drug_vec": torch.tensor([1.0] + [0.0]*15)}
                    for i in range(12)
                ] + [
                    {"t": float(84.0 + i * 7.0), "type": "chemotherapy", "drug_vec": torch.tensor([0.0, 0.0, 1.0] + [0.0]*13)}
                    for i in range(12)
                ]
            }

    def rank_interventions(
        self,
        u_d_0: torch.Tensor,
        v_sys_0: torch.Tensor,
        z_a_0: torch.Tensor,
        z_s_ref: torch.Tensor,
        sigma2: torch.Tensor,
        horizon_days: float = 180.0
    ) -> List[Dict[str, Any]]:
        """
        Simulates M posterior draws per candidate regimen and computes
        uncertainty-penalized rankings.
        """
        regimens = self.get_candidate_regimens()
        rankings = []

        std_d = torch.sqrt(torch.clamp(sigma2[:, :64], min=1e-6))
        std_p = torch.sqrt(torch.clamp(sigma2[:, 64:96], min=1e-6))

        for reg_name, schedule in regimens.items():
            draw_ttp = []
            draw_final_vols = []

            for m in range(self.num_draws):
                # Perturb initial state by posterior variance
                eps_d = torch.randn_like(u_d_0) * 0.05
                u_d_pert = torch.clamp(u_d_0 + eps_d, min=0.0)
                v_sys_pert = v_sys_0.clone()
                v_sys_pert[:, :64] += torch.randn_like(std_d) * std_d
                v_sys_pert[:, 64:96] += torch.randn_like(std_p) * std_p

                sim = self.simulator.simulate_regimen(
                    u_d_0=u_d_pert,
                    v_sys_0=v_sys_pert,
                    z_a_0=z_a_0,
                    z_s_ref=z_s_ref,
                    candidate_schedule=schedule,
                    horizon_days=horizon_days,
                    eval_interval_days=30.0
                )

                vols = sim["tumor_volumes"]
                draw_final_vols.append(vols[-1])
                
                # Estimate TTP (days until volume exceeds baseline * 1.25, or horizon)
                baseline_vol = vols[0]
                ttp = horizon_days
                for idx, v in enumerate(vols):
                    if idx > 0 and v > baseline_vol * 1.25:
                        ttp = sim["timestamps"][idx]
                        break
                draw_ttp.append(ttp)

            mean_ttp = float(np.mean(draw_ttp))
            std_ttp = float(np.std(draw_ttp))
            mean_final_vol = float(np.mean(draw_final_vols))
            std_final_vol = float(np.std(draw_final_vols))

            # Utility score
            utility_score = mean_ttp - self.lambda_unc * std_ttp

            rankings.append({
                "regimen": reg_name,
                "utility_score": utility_score,
                "mean_ttp_days": mean_ttp,
                "std_ttp_days": std_ttp,
                "ci95_ttp": (max(0.0, mean_ttp - 1.96 * std_ttp), mean_ttp + 1.96 * std_ttp),
                "mean_final_volume": mean_final_vol,
                "std_final_volume": std_final_vol,
                "num_draws": self.num_draws
            })

        # Sort descending by utility score
        rankings.sort(key=lambda x: x["utility_score"], reverse=True)
        return rankings
