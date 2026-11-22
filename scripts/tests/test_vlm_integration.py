#!/usr/bin/env python
"""
Test VLM Integration with downloaded models.
Runs inference with BrainGemma3D and/or Med3DVLM on synthetic data.

Usage:
    cd /DATA/horizon_disentangled
    LD_LIBRARY_PATH=/DATA/conda/envs/nukl-to-ct/lib:$LD_LIBRARY_PATH \
    /DATA/conda/envs/nukl-to-ct/bin/python scripts/test_vlm_integration.py
"""
import os
import sys
import json
import torch
import logging

sys.path.insert(0, "/DATA/horizon_disentangled")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("vlm_test")

from src.pipeline.stage5_synthesis_reporting.vlm_reporter import VLMReportGenerator


def test_rule_based():
    """Test rule-based reporting (always works)."""
    logger.info("=" * 60)
    logger.info("Test 1: Rule-Based RANO Report")
    logger.info("=" * 60)
    
    reporter = VLMReportGenerator(model_name='rule_based')
    
    report = reporter.generate_rano_report(
        patient_id='TEST_GBM_001', visit_day=90.0,
        current_tumor_vol=25.3, prior_tumor_vol=28.1,
        dior_probabilities={'CR': 0.05, 'PR': 0.15, 'SD': 0.60, 'PD': 0.15, 'PP': 0.05},
        median_ttp_days=120.0, ci95_ttp=(80.0, 160.0),
        mc_predictions=[{'rano_category': 'Stable_Disease'} for _ in range(8)]
    )
    logger.info(f"Method: {report['generation_method']}")
    logger.info(f"RANO: {report['rano_assessment']['category']}")
    logger.info(f"Confidence: {report['rano_assessment']['confidence']}")
    logger.info(json.dumps(report, indent=2, default=str)[:500])
    
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Rule-Based BI-RADS Report")
    logger.info("=" * 60)
    
    report2 = reporter.generate_birads_ispy2_report(
        patient_id='TEST_BREAST_001', visit_timepoint='T1',
        current_ftv_cm3=15.2, baseline_ftv_cm3=42.0,
        longest_diameter_mm=35.0, ser_mean=1.4,
        predicted_pcr_prob=0.68,
        mc_predictions=[{'pcr_prob': 0.65 + i * 0.01} for i in range(8)]
    )
    logger.info(f"Method: {report2['generation_method']}")
    logger.info(f"Response Tier: {report2['prognosis_pcr']['response_tier']}")
    logger.info(f"Washout: {report2['prognosis_pcr']['washout_pattern']}")
    
    return True


def test_vlm_model(model_name, model_path):
    """Test actual VLM inference."""
    logger.info("=" * 60)
    logger.info(f"Test: {model_name} VLM Inference")
    logger.info(f"Path: {model_path}")
    logger.info("=" * 60)
    
    if not os.path.exists(model_path):
        logger.warning(f"Model not found at {model_path}, skipping")
        return False
    
    # Check for actual model files
    has_model = any(
        f.endswith('.safetensors') or f.endswith('.bin') or f.endswith('.pt')
        for f in os.listdir(model_path) if os.path.isfile(os.path.join(model_path, f))
    )
    # Also check subdirectories
    for subdir in os.listdir(model_path):
        subpath = os.path.join(model_path, subdir)
        if os.path.isdir(subpath):
            has_model = has_model or any(
                f.endswith('.safetensors') or f.endswith('.bin') or f.endswith('.pt')
                for f in os.listdir(subpath) if os.path.isfile(os.path.join(subpath, f))
            )
    
    if not has_model:
        logger.warning(f"No model files found in {model_path}")
        return False
    
    logger.info("Loading VLM model (4-bit quantized)...")
    reporter = VLMReportGenerator(
        model_name=model_name,
        model_path=model_path,
        device='cuda',
        use_4bit=True,
        max_new_tokens=512,
    )
    
    try:
        success = reporter.load_model()
        if not success or reporter.model_name == 'rule_based':
            logger.warning(f"Failed to load {model_name}, fell back to rule-based")
            return False
        
        logger.info(f"Model loaded successfully on {reporter.device}")
        
        # Test RANO report
        logger.info("\nGenerating RANO 2.0 report...")
        report = reporter.generate_rano_report(
            patient_id='VLM_TEST_GBM', visit_day=90.0,
            current_tumor_vol=25.3, prior_tumor_vol=28.1,
            dior_probabilities={'CR': 0.05, 'PR': 0.15, 'SD': 0.60, 'PD': 0.15, 'PP': 0.05},
            median_ttp_days=120.0, ci95_ttp=(80.0, 160.0),
            mc_predictions=[{'rano_category': 'Stable_Disease'} for _ in range(8)]
        )
        
        logger.info(f"  Method: {report['generation_method']}")
        logger.info(f"  RANO category: {report['rano_assessment']['category']}")
        logger.info(f"  Reasoning: {report['rano_assessment'].get('reasoning', 'N/A')[:200]}")
        logger.info(f"  VLM-Rule agreement: {report['hallucination_audit']['vlm_rule_agreement']}")
        
        # Test BI-RADS report
        logger.info("\nGenerating BI-RADS report...")
        report2 = reporter.generate_birads_ispy2_report(
            patient_id='VLM_TEST_BREAST', visit_timepoint='T1',
            current_ftv_cm3=15.2, baseline_ftv_cm3=42.0,
            longest_diameter_mm=35.0, ser_mean=1.4,
            predicted_pcr_prob=0.68,
            mc_predictions=[{'pcr_prob': 0.65 + i * 0.01} for i in range(8)]
        )
        
        logger.info(f"  Method: {report2['generation_method']}")
        logger.info(f"  Response tier: {report2['prognosis_pcr']['response_tier']}")
        logger.info(f"  pCR assessment: {report2['prognosis_pcr'].get('pcr_assessment', 'N/A')}")
        
        # Measure GPU memory
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"\n  Peak GPU memory: {mem_gb:.2f} GB")
        
        # Unload to free memory
        reporter.unload()
        logger.info(f"  VLM unloaded, GPU memory freed")
        
        return True
        
    except Exception as e:
        logger.error(f"VLM test failed: {e}")
        import traceback
        traceback.print_exc()
        reporter.unload()
        return False


def test_merlin_contrastive():
    """Test Merlin-style contrastive learning components."""
    logger.info("=" * 60)
    logger.info("Test: Merlin-Style Contrastive Learning")
    logger.info("=" * 60)
    
    from src.pipeline.stage1_supervisors.losses import (
        MerlinInfoNCELoss, ClinicalTextEncoder, MerlinMultiTaskLoss
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test InfoNCE
    infonce = MerlinInfoNCELoss(visual_dim=768, text_dim=256).to(device)
    v = torch.randn(8, 768, device=device)
    t = torch.randn(8, 256, device=device)
    loss, metrics = infonce(v, t)
    logger.info(f"  InfoNCE loss: {loss.item():.4f}")
    logger.info(f"  I2T accuracy: {metrics['accuracy_i2t']:.2f}")
    logger.info(f"  T2I accuracy: {metrics['accuracy_t2i']:.2f}")
    logger.info(f"  Temperature: {metrics['temperature']:.4f}")
    
    # Test ClinicalTextEncoder
    text_enc = ClinicalTextEncoder(num_clinical_tokens=32, out_dim=256).to(device)
    clinical = torch.randn(8, 16, device=device)
    text_emb = text_enc(clinical)
    logger.info(f"  Text encoder output: {text_emb.shape}")
    logger.info(f"  Text encoder params: {sum(p.numel() for p in text_enc.parameters()):,}")
    
    # Test MerlinMultiTaskLoss
    merlin = MerlinMultiTaskLoss(visual_dim=768, text_dim=256, num_phenotype_labels=32).to(device)
    phenotype = torch.randint(0, 2, (8, 32), device=device).float()
    loss_m, metrics_m = merlin(v, text_emb, phenotype_targets=phenotype)
    logger.info(f"  Merlin total loss: {loss_m.item():.4f}")
    logger.info(f"  BCE phenotype loss: {metrics_m['loss_bce_phenotype']:.4f}")
    
    # Backward pass
    loss_m.backward()
    logger.info("  ✓ Backward pass successful")
    
    total_params = sum(p.numel() for p in infonce.parameters()) + \
                   sum(p.numel() for p in text_enc.parameters()) + \
                   sum(p.numel() for p in merlin.parameters())
    logger.info(f"  Total contrastive params: {total_params:,}")
    
    return True


if __name__ == "__main__":
    results = {}
    
    # Test 1: Rule-based reporting
    results['rule_based'] = test_rule_based()
    
    # Test 2: Merlin contrastive
    results['merlin_contrastive'] = test_merlin_contrastive()
    
    # Test 3: BrainGemma3D VLM
    results['braingemma3d'] = test_vlm_model(
        'braingemma3d',
        '/DATA/horizon_disentangled/models/BrainGemma3D'
    )
    
    # Test 4: Med3DVLM
    results['med3dvlm'] = test_vlm_model(
        'med3dvlm',
        '/DATA/horizon_disentangled/models/Med3DVLM-Qwen-2.5-7B'
    )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VLM INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED/SKIPPED"
        logger.info(f"  {name}: {status}")
    
    all_passed = all(results.values())
    logger.info(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED/SKIPPED'}")
