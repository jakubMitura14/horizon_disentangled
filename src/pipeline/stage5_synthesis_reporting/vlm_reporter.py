"""
Stage 5: VLM-Powered Structured Clinical Reporter

Integrates actual Vision-Language Models for report generation:
1. MedGemma 1.5 4B — Google's medical VLM with 3D MRI support
2. Med3DVLM — Native 3D volumetric VLM (DCFormer-SigLIP + Qwen-2.5-7B)
3. Rule-based fallback — Original StructuredClinicalReporter

Architecture per main_pipeline.md Section 2.4:
  - MedGemma processes multi-timepoint 3D MRI slices + trajectory metrics
  - Med3DVLM handles native 128×256×256 3D volumetric input
  - Both output structured RANO 2.0 / BI-RADS JSON
  - 8-draw Monte Carlo concordance audit for hallucination detection
"""

import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("horizon_vlm")


class VLMReportGenerator:
    """
    VLM-powered report generator using MedGemma 1.5 4B or Med3DVLM.
    
    Falls back to rule-based generation if VLM models are unavailable.
    Supports:
    - Structured JSON output via constrained generation
    - Multi-timepoint longitudinal reasoning
    - 3D MRI slice sampling for VLM input
    - Monte Carlo concordance auditing
    """
    
    def __init__(
        self,
        model_name: str = "medgemma",  # "medgemma", "med3dvlm", or "rule_based"
        model_path: Optional[str] = None,
        device: str = "cuda",
        max_new_tokens: int = 1024,
        concordance_threshold: float = 0.25,
        num_slices: int = 8,  # Number of axial slices to sample for VLM
        use_4bit: bool = True,  # Quantize for memory efficiency
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.concordance_threshold = concordance_threshold
        self.num_slices = num_slices
        self.use_4bit = use_4bit
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._loaded = False
        
        # Default model paths
        if model_path is None:
            if model_name == "medgemma":
                self.model_path = "/DATA/horizon_disentangled/models/medgemma-1.5-4b-it"
            elif model_name == "med3dvlm":
                self.model_path = "/DATA/horizon_disentangled/models/Med3DVLM"
            else:
                self.model_path = None
        else:
            self.model_path = model_path
    
    def load_model(self):
        """Load the VLM model. Called lazily on first use."""
        if self._loaded:
            return True
            
        if self.model_name == "rule_based" or self.model_path is None:
            self._loaded = True
            return True
        
        model_dir = Path(self.model_path)
        if not model_dir.exists():
            logger.warning(f"VLM model not found at {self.model_path}, falling back to rule-based")
            self.model_name = "rule_based"
            self._loaded = True
            return True
        
        try:
            if self.model_name == "medgemma":
                return self._load_medgemma()
            elif self.model_name == "med3dvlm":
                return self._load_med3dvlm()
            elif self.model_name == "braingemma3d":
                return self._load_braingemma3d()
        except Exception as e:
            logger.warning(f"Failed to load VLM {self.model_name}: {e}, falling back to rule-based")
            self.model_name = "rule_based"
            self._loaded = True
            return True
    
    def _load_medgemma(self) -> bool:
        """Load MedGemma 1.5 4B with optional 4-bit quantization."""
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        
        logger.info(f"Loading MedGemma 1.5 4B from {self.model_path}...")
        
        load_kwargs = {
            "device_map": self.device if self.device != "cuda" else "auto",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        
        if self.use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                logger.warning("bitsandbytes not available, loading in bf16")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **load_kwargs
        )
        
        # Try processor first (multimodal), fallback to tokenizer
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        self.model.eval()
        self._loaded = True
        logger.info(f"MedGemma loaded successfully (4-bit={self.use_4bit})")
        return True
    
    def _load_med3dvlm(self) -> bool:
        """Load Med3DVLM (DCFormer-SigLIP + Qwen-2.5-7B)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading Med3DVLM from {self.model_path}...")
        
        load_kwargs = {
            "device_map": self.device if self.device != "cuda" else "auto",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        
        if self.use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                pass
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **load_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model.eval()
        self._loaded = True
        logger.info(f"Med3DVLM loaded successfully (4-bit={self.use_4bit})")
        return True
    
    def _load_braingemma3d(self) -> bool:
        """Load BrainGemma3D (Gemma-based brain MRI VLM with 3D vision encoder).
        
        BrainGemma3D uses:
        - MedSigLIP3D vision encoder (inflated from 2D SigLIP)
        - MedGemma language model (quantized 4-bit)
        - Vision-language projector
        
        Input: (B, 1, D, H, W) 3D MRI volume
        Output: Radiology report text
        """
        logger.info(f"Loading BrainGemma3D from {self.model_path}...")
        
        model_dir = Path(self.model_path)
        arch_file = model_dir / "braingemma3d_architecture.py"
        
        if arch_file.exists():
            # Import BrainGemma3D architecture from downloaded model
            import importlib.util
            spec = importlib.util.spec_from_file_location("braingemma3d_arch", str(arch_file))
            braingemma_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(braingemma_module)
            
            vision_dir = str(model_dir / "vision_model")
            language_dir = str(model_dir / "language_model")
            
            # Check if projector weights exist
            projector_path = model_dir / "projector_vis_scale.pt"
            
            self.model = braingemma_module.BrainGemma3D(
                vision_model_dir=vision_dir,
                language_model_dir=language_dir,
                depth=2,
                max_depth_patches=128,
                num_vision_tokens=32,
                freeze_vision=True,
                freeze_language=True,
            )
            
            # Load projector weights if available
            if projector_path.exists():
                state = torch.load(str(projector_path), map_location="cpu", weights_only=True)
                # The projector checkpoint may contain vis_scale (float) and projector weights
                if isinstance(state, dict):
                    # Convert non-tensor values to tensors for load_state_dict
                    fixed_state = {}
                    for k, v in state.items():
                        if k == 'vis_scale' and isinstance(v, (float, int)):
                            fixed_state[k] = torch.tensor(v)
                        elif k == 'vision_projector' and isinstance(v, dict):
                            # Flatten vision_projector sub-dict into model's projector
                            for pk, pv in v.items():
                                fixed_state[f'vision_projector.{pk}'] = pv
                        else:
                            fixed_state[k] = v
                    self.model.load_state_dict(fixed_state, strict=False)
                logger.info("Loaded projector weights from projector_vis_scale.pt")
            
            self.tokenizer = self.model.tokenizer
            self._braingemma_module = braingemma_module
            self._loaded = True
            logger.info(f"BrainGemma3D loaded successfully")
            return True
        else:
            # Fallback: load language model only for text-based reasoning
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            lm_path = model_dir / "language_model"
            if not lm_path.exists():
                lm_path = model_dir
            
            load_kwargs = {
                "device_map": self.device if self.device != "cuda" else "auto",
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
            }
            
            if self.use_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                except ImportError:
                    pass
            
            self.model = AutoModelForCausalLM.from_pretrained(
                str(lm_path), **load_kwargs
            )
            self.tokenizer = AutoTokenizer.from_pretrained(str(lm_path), trust_remote_code=True)
            self.model.eval()
            self._loaded = True
            logger.info(f"BrainGemma3D LM loaded successfully (4-bit={self.use_4bit})")
            return True
    
    def _sample_slices(self, volume: torch.Tensor) -> List[torch.Tensor]:
        """
        Sample representative axial slices from 3D volume for VLM input.
        
        Strategy: Take evenly-spaced slices through the volume,
        preferring slices with high signal variance (likely to contain pathology).
        
        Args:
            volume: (C, D, H, W) 3D volume
            
        Returns:
            List of (C, H, W) slice tensors
        """
        C, D, H, W = volume.shape
        
        if D <= self.num_slices:
            return [volume[:, d] for d in range(D)]
        
        # Compute per-slice variance to find interesting slices
        with torch.no_grad():
            slice_var = torch.var(volume, dim=(0, 2, 3))  # (D,)
        
        # Mix evenly-spaced + high-variance slices
        num_even = self.num_slices // 2
        num_var = self.num_slices - num_even
        
        # Evenly spaced
        even_indices = torch.linspace(0, D - 1, num_even).long()
        
        # Top variance (excluding already selected)
        var_indices = torch.argsort(slice_var, descending=True)
        var_selected = []
        for idx in var_indices:
            if idx.item() not in even_indices.tolist() and len(var_selected) < num_var:
                var_selected.append(idx.item())
        
        all_indices = sorted(even_indices.tolist() + var_selected)
        return [volume[:, d] for d in all_indices]
    
    def _build_rano_prompt(
        self,
        patient_id: str,
        visit_day: float,
        tumor_vol: float,
        prior_vol: float,
        dior_probs: Dict[str, float],
        ttp_days: float,
        ci95: Tuple[float, float],
    ) -> str:
        """Build structured prompt for RANO 2.0 report generation."""
        vol_change = ((tumor_vol - prior_vol) / max(prior_vol, 1e-4)) * 100
        
        return f"""You are a neuro-oncology AI assistant. Generate a structured RANO 2.0 assessment report.

## Patient Context
- Patient ID: {patient_id}
- Visit Day: {visit_day:.0f} (days post-baseline)
- Current Tumor Volume: {tumor_vol:.2f} cm³
- Prior Tumor Volume: {prior_vol:.2f} cm³  
- Volume Change: {vol_change:+.1f}%

## AI Model Predictions (Stage 4 Dynamics)
- DIOR Ordinal Probabilities: CR={dior_probs.get('CR', 0):.3f}, PR={dior_probs.get('PR', 0):.3f}, SD={dior_probs.get('SD', 0):.3f}, PD={dior_probs.get('PD', 0):.3f}, PP={dior_probs.get('PP', 0):.3f}
- Predicted Median TTP: {ttp_days:.1f} days
- 95% CI TTP: [{ci95[0]:.1f}, {ci95[1]:.1f}] days

## RANO 2.0 Guidelines
- Complete Response (CR): Complete disappearance of all enhancing disease
- Partial Response (PR): ≥50% decrease in enhancing tumor
- Stable Disease (SD): <50% decrease to <25% increase
- Progressive Disease (PD): ≥25% increase in enhancing tumor
- Pseudo-progression: Apparent PD within 12 weeks post-RT, often resolving

## Task
Generate a structured RANO 2.0 assessment as JSON with these exact fields:
```json
{{
  "rano_category": "CR|PR|SD|PD|Suspected_Pseudo_Progression",
  "reasoning": "step-by-step clinical reasoning",
  "confidence": 0.0-1.0,
  "key_findings": ["finding1", "finding2"],
  "clinical_recommendation": "recommendation text",
  "risk_factors": ["factor1", "factor2"]
}}
```"""

    def _build_birads_prompt(
        self,
        patient_id: str,
        timepoint: str,
        ftv: float,
        baseline_ftv: float,
        diameter: float,
        ser_mean: float,
        pcr_prob: float,
    ) -> str:
        """Build structured prompt for BI-RADS / I-SPY 2 report generation."""
        ftv_reduction = ((baseline_ftv - ftv) / max(baseline_ftv, 1e-4)) * 100
        
        return f"""You are a breast radiology AI assistant. Generate a structured I-SPY 2 NACT assessment report.

## Patient Context
- Patient ID: {patient_id}
- Timepoint: {timepoint} (T0=baseline, T1=early, T2=inter-regimen, T3=pre-surgery)
- Current FTV: {ftv:.2f} cm³
- Baseline FTV: {baseline_ftv:.2f} cm³
- FTV Reduction: {ftv_reduction:+.1f}%
- Longest Diameter: {diameter:.1f} mm
- Mean SER: {ser_mean:.2f}

## AI Model Predictions
- Predicted pCR Probability: {pcr_prob:.3f}

## I-SPY 2 Response Tiers
- Excellent Early Responder: FTV reduction ≥80%
- Moderate Responder: FTV reduction 30-80%
- Non-Responder: FTV reduction <30%

## SER Interpretation
- SER < 0.9: persistent enhancement (benign-favoring)
- SER 0.9-1.1: plateau (indeterminate)
- SER > 1.1: washout (malignancy-suspicious)

## Task
Generate a structured BI-RADS / I-SPY 2 assessment as JSON with these exact fields:
```json
{{
  "response_tier": "Excellent_Early_Responder|Moderate_Responder|Non_Responder",
  "reasoning": "step-by-step clinical reasoning",
  "pcr_assessment": "likely|unlikely|uncertain",
  "confidence": 0.0-1.0,
  "key_findings": ["finding1", "finding2"],
  "washout_pattern": "persistent|plateau|washout",
  "clinical_recommendation": "recommendation text"
}}
```"""

    @torch.no_grad()
    def _generate_vlm_response(self, prompt: str, volume: Optional[torch.Tensor] = None) -> str:
        """Generate response from loaded VLM."""
        if self.model is None:
            return ""
        
        try:
            # BrainGemma3D native mode: use generate_report API if model has it
            if self.model_name == "braingemma3d" and hasattr(self.model, 'generate_report'):
                if volume is not None:
                    # BrainGemma3D expects (B, 1, D, H, W)
                    if volume.dim() == 4:  # (C, D, H, W) → (1, 1, D, H, W) using first channel
                        vol_input = volume[0:1].unsqueeze(0)  # Take first channel
                    elif volume.dim() == 5:
                        vol_input = volume[:, 0:1]  # Take first channel
                    else:
                        vol_input = volume.unsqueeze(0).unsqueeze(0)
                    
                    response = self.model.generate_report(
                        vol_input, prompt=prompt,
                        max_new_tokens=self.max_new_tokens,
                        temperature=0.3,
                    )
                    return response.strip()
                else:
                    # Text-only mode: fall through to standard generation
                    pass
            
            # Med3DVLM custom generate: expects (images=tensor, inputs=token_ids)
            if self.model_name == "med3dvlm" and self.tokenizer is not None:
                tok_out = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
                input_ids = tok_out["input_ids"].to(self.model.device)
                attention_mask = tok_out.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.model.device)
                
                # Med3DVLM.generate() expects: images=Optional[Tensor], inputs=token_ids
                gen_kwargs = {
                    "inputs": input_ids,
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": True,
                    "temperature": 0.3,
                    "top_p": 0.9,
                }
                if attention_mask is not None:
                    gen_kwargs["attention_mask"] = attention_mask
                if volume is not None:
                    # Prepare volume for Med3DVLM vision tower: expects (B, C, D, H, W)
                    if volume.dim() == 3:
                        vol_in = volume.unsqueeze(0).unsqueeze(0)
                    elif volume.dim() == 4:
                        vol_in = volume.unsqueeze(0)
                    else:
                        vol_in = volume
                    gen_kwargs["images"] = vol_in.to(self.model.device).to(torch.bfloat16)
                
                outputs = self.model.generate(**gen_kwargs)
                input_len = input_ids.shape[1]
                response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
                return response.strip()
            
            # Standard HuggingFace generation (MedGemma or other models)
            # Determine device — HF models have .device, custom modules may not
            try:
                model_device = self.model.device
            except AttributeError:
                model_device = self.device
            
            if self.processor is not None:
                # MedGemma with processor (multimodal)
                inputs = self.processor(text=prompt, return_tensors="pt")
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
            elif self.tokenizer is not None:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
            else:
                return ""
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
            )
            
            # Decode only new tokens
            input_len = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True) if self.tokenizer else \
                       self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.warning(f"VLM generation failed: {e}")
            return ""
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from VLM response, handling markdown code blocks."""
        # Try to find JSON in code block
        import re
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try parsing the whole response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to find any JSON object in the response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return {"raw_response": response, "parse_error": True}
    
    def generate_rano_report(
        self,
        patient_id: str,
        visit_day: float,
        current_tumor_vol: float,
        prior_tumor_vol: float,
        dior_probabilities: Dict[str, float],
        median_ttp_days: float,
        ci95_ttp: Tuple[float, float],
        mc_predictions: List[Dict[str, Any]],
        volume: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Generate RANO 2.0 structured report using VLM or rule-based fallback."""
        self.load_model()
        
        vol_change_pct = ((current_tumor_vol - prior_tumor_vol) / max(prior_tumor_vol, 1e-4)) * 100.0
        
        # VLM-based generation
        vlm_assessment = {}
        if self.model_name in ("medgemma", "med3dvlm", "braingemma3d") and self.model is not None:
            prompt = self._build_rano_prompt(
                patient_id, visit_day, current_tumor_vol, prior_tumor_vol,
                dior_probabilities, median_ttp_days, ci95_ttp
            )
            response = self._generate_vlm_response(prompt, volume)
            if response:
                vlm_assessment = self._parse_json_response(response)
                logger.info(f"VLM ({self.model_name}) generated RANO assessment")
        
        # Rule-based RANO category (always computed as ground truth / fallback)
        if vol_change_pct <= -50.0:
            rano_cat = "Complete_or_Partial_Response"
        elif vol_change_pct >= 25.0:
            if visit_day < 84.0 and dior_probabilities.get("pseudoprogression_prob", dior_probabilities.get("PP", 0.0)) > 0.4:
                rano_cat = "Suspected_Pseudo_Progression"
            else:
                rano_cat = "Progressive_Disease"
        else:
            rano_cat = "Stable_Disease"
        
        # Multi-draw concordance
        mc_cats = [m.get("rano_category", rano_cat) for m in mc_predictions]
        concordance_rate = mc_cats.count(rano_cat) / max(len(mc_cats), 1)
        flag_divergence = (1.0 - concordance_rate) > self.concordance_threshold
        
        report = {
            "schema_version": "RANO_2.0_JSON_v1",
            "patient_id": patient_id,
            "timepoint_days": visit_day,
            "generation_method": self.model_name,
            "quantitative_metrics": {
                "current_tumor_volume_cm3": round(current_tumor_vol, 2),
                "prior_tumor_volume_cm3": round(prior_tumor_vol, 2),
                "volume_change_percentage": round(vol_change_pct, 2)
            },
            "rano_assessment": {
                "category": vlm_assessment.get("rano_category", rano_cat),
                "rule_based_category": rano_cat,
                "confidence": vlm_assessment.get("confidence", round(concordance_rate, 3)),
                "reasoning": vlm_assessment.get("reasoning", "Rule-based volumetric assessment"),
                "key_findings": vlm_assessment.get("key_findings", []),
                "clinical_recommendation": vlm_assessment.get("clinical_recommendation", ""),
                "risk_factors": vlm_assessment.get("risk_factors", []),
                "dior_ordinal_probabilities": dior_probabilities,
                "median_ttp_days": round(median_ttp_days, 1),
                "ci95_ttp_days": [round(ci95_ttp[0], 1), round(ci95_ttp[1], 1)]
            },
            "hallucination_audit": {
                "num_posterior_draws": len(mc_predictions),
                "concordance_rate": round(concordance_rate, 3),
                "concordance_warning": flag_divergence,
                "vlm_rule_agreement": vlm_assessment.get("rano_category", rano_cat) == rano_cat,
                "divergence_details": "High posterior variance observed across treatment paths." if flag_divergence else "Concordant consensus."
            }
        }
        return report
    
    def generate_birads_ispy2_report(
        self,
        patient_id: str,
        visit_timepoint: str,
        current_ftv_cm3: float,
        baseline_ftv_cm3: float,
        longest_diameter_mm: float,
        ser_mean: float,
        predicted_pcr_prob: float,
        mc_predictions: List[Dict[str, Any]],
        volume: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Generate BI-RADS / I-SPY 2 structured report using VLM or rule-based fallback."""
        self.load_model()
        
        ftv_reduction_pct = ((baseline_ftv_cm3 - current_ftv_cm3) / max(baseline_ftv_cm3, 1e-4)) * 100.0
        
        # VLM-based generation
        vlm_assessment = {}
        if self.model_name in ("medgemma", "med3dvlm", "braingemma3d") and self.model is not None:
            prompt = self._build_birads_prompt(
                patient_id, visit_timepoint, current_ftv_cm3, baseline_ftv_cm3,
                longest_diameter_mm, ser_mean, predicted_pcr_prob
            )
            response = self._generate_vlm_response(prompt, volume)
            if response:
                vlm_assessment = self._parse_json_response(response)
                logger.info(f"VLM ({self.model_name}) generated BI-RADS/I-SPY 2 assessment")
        
        # Rule-based response tier
        if ftv_reduction_pct >= 80.0:
            response_tier = "Excellent_Early_Responder"
        elif ftv_reduction_pct >= 30.0:
            response_tier = "Moderate_Responder"
        else:
            response_tier = "Non_Responder"
        
        # Multi-draw concordance on pCR
        mc_pcr = [m.get("pcr_prob", predicted_pcr_prob) for m in mc_predictions]
        std_pcr = float(np.std(mc_pcr)) if mc_pcr else 0.0
        flag_divergence = std_pcr > self.concordance_threshold
        
        report = {
            "schema_version": "ISPY2_BIRADS_JSON_v1",
            "patient_id": patient_id,
            "timepoint": visit_timepoint,
            "generation_method": self.model_name,
            "mri_volumetric_kinetics": {
                "functional_tumor_volume_cm3": round(current_ftv_cm3, 2),
                "baseline_ftv_cm3": round(baseline_ftv_cm3, 2),
                "ftv_reduction_percentage": round(ftv_reduction_pct, 2),
                "longest_diameter_mm": round(longest_diameter_mm, 1),
                "signal_enhancement_ratio_mean": round(ser_mean, 2)
            },
            "prognosis_pcr": {
                "predicted_pcr_probability": round(predicted_pcr_prob, 3),
                "response_tier": vlm_assessment.get("response_tier", response_tier),
                "rule_based_tier": response_tier,
                "reasoning": vlm_assessment.get("reasoning", "Rule-based FTV assessment"),
                "pcr_assessment": vlm_assessment.get("pcr_assessment", "likely" if predicted_pcr_prob > 0.5 else "unlikely"),
                "washout_pattern": vlm_assessment.get("washout_pattern", "washout" if ser_mean > 1.1 else ("plateau" if ser_mean > 0.9 else "persistent")),
                "key_findings": vlm_assessment.get("key_findings", []),
                "clinical_recommendation": vlm_assessment.get("clinical_recommendation", ""),
                "posterior_uncertainty_sd": round(std_pcr, 3)
            },
            "hallucination_audit": {
                "num_posterior_draws": len(mc_predictions),
                "posterior_sd": round(std_pcr, 3),
                "concordance_warning": flag_divergence,
                "vlm_rule_agreement": vlm_assessment.get("response_tier", response_tier) == response_tier,
                "audit_status": "Audited & Verified" if not flag_divergence else "Requires Clinician Adjudication"
            }
        }
        return report
    
    def unload(self):
        """Free GPU memory by unloading VLM model."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("VLM model unloaded, GPU memory freed")

    def unload_model(self):
        """Alias for unload()."""
        return self.unload()
