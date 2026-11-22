"""Stage 5 — Synthesis and Reporting.

Counterfactual trajectory generation, anatomical warping, latent flow
refinement, agentic treatment ranking, structured VLM clinical reporting,
and EU AI Act audit logging.

Key classes:
    AnatomicalSpatialWarper: Darcy-driven 3D anatomy warping.
    LatentRectifiedFlowRefiner: Straight-line flow refinement for sharp volumes.
    CounterfactualSimulator: Runs candidate intervention trajectories.
    AgenticTreatmentHelper: Ranks regimens by uncertainty-penalized TTP.
    StructuredClinicalReporter: VLM-powered structured radiology reporting.
    VLMReportGenerator: Free-text VLM report generation.
    EUAIActAuditLogger: EU AI Act compliant audit trail.
"""
from src.pipeline.stage5_synthesis_reporting.spatial_warper import AnatomicalSpatialWarper
from src.pipeline.stage5_synthesis_reporting.flow_refiner import LatentRectifiedFlowRefiner
from src.pipeline.stage5_synthesis_reporting.counterfactual import CounterfactualSimulator
from src.pipeline.stage5_synthesis_reporting.treatment_helper import AgenticTreatmentHelper
from src.pipeline.stage5_synthesis_reporting.structured_vlm import StructuredClinicalReporter
from src.pipeline.stage5_synthesis_reporting.vlm_reporter import VLMReportGenerator
from src.pipeline.stage5_synthesis_reporting.eu_ai_act_logger import EUAIActAuditLogger

__all__ = [
    "AnatomicalSpatialWarper",
    "LatentRectifiedFlowRefiner",
    "CounterfactualSimulator",
    "AgenticTreatmentHelper",
    "StructuredClinicalReporter",
    "VLMReportGenerator",
    "EUAIActAuditLogger"
]
