#!/usr/bin/env python
"""
Test script to verify GSM8K analysis setup.

This script performs basic sanity checks to ensure all dependencies
and components are properly configured.
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ Transformers: {e}")
        return False
    
    try:
        import datasets
        print(f"  ✓ Datasets {datasets.__version__}")
    except ImportError as e:
        print(f"  ✗ Datasets: {e}")
        return False
    
    try:
        import lm_eval
        print(f"  ✓ lm-eval {lm_eval.__version__}")
    except ImportError as e:
        print(f"  ✗ lm-eval: {e}")
        return False
    
    try:
        import pandas
        print(f"  ✓ Pandas {pandas.__version__}")
    except ImportError as e:
        print(f"  ✗ Pandas: {e}")
        return False
    
    return True


def test_alphaquant_imports():
    """Test AlphaQuant module imports."""
    print("\nTesting AlphaQuant imports...")
    
    try:
        from alphaquant.gptq.quantize import gptq_quantize_model
        print("  ✓ alphaquant.gptq.quantize")
    except ImportError as e:
        print(f"  ✗ alphaquant.gptq.quantize: {e}")
        return False
    
    try:
        from alphaquant.gptq.data_utils import CalibrationDataLoader
        print("  ✓ alphaquant.gptq.data_utils")
    except ImportError as e:
        print(f"  ✗ alphaquant.gptq.data_utils: {e}")
        return False
    
    try:
        from alphaquant.utils.replacement import apply_layer_wise_quantization
        print("  ✓ alphaquant.utils.replacement")
    except ImportError as e:
        print(f"  ✗ alphaquant.utils.replacement: {e}")
        return False
    
    try:
        from alphaquant.utils.eval_utils import make_table
        print("  ✓ alphaquant.utils.eval_utils")
    except ImportError as e:
        print(f"  ✗ alphaquant.utils.eval_utils: {e}")
        return False
    
    try:
        from alphaquant.quantizers.mxfp4 import MXFP4Quantizer
        print("  ✓ alphaquant.quantizers.mxfp4")
    except ImportError as e:
        print(f"  ✗ alphaquant.quantizers.mxfp4: {e}")
        return False
    
    return True


def test_gsm8k_imports():
    """Test GSM8K analysis module imports."""
    print("\nTesting GSM8K analysis imports...")
    
    try:
        from gsm8k_analysis.data_utils import GSM8KCalibrationDataLoader
        print("  ✓ gsm8k_analysis.data_utils")
    except ImportError as e:
        print(f"  ✗ gsm8k_analysis.data_utils: {e}")
        return False
    
    return True


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    - Device count: {torch.cuda.device_count()}")
            print(f"    - Current device: {torch.cuda.current_device()}")
            print(f"    - Device name: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("  ⚠ CUDA not available (will use CPU)")
            return True
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False


def test_dataset_loading():
    """Test GSM8K dataset loading."""
    print("\nTesting GSM8K dataset loading...")
    
    try:
        from datasets import load_dataset
        
        # Try to load a small sample
        dataset = load_dataset('gsm8k', 'main', split='train[:5]')
        print(f"  ✓ Loaded {len(dataset)} samples")
        
        # Check data structure
        if len(dataset) > 0:
            sample = dataset[0]
            if 'question' in sample and 'answer' in sample:
                print("  ✓ Dataset structure correct")
                return True
            else:
                print("  ✗ Unexpected dataset structure")
                return False
        
        return True
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        return False


def test_directory_structure():
    """Test that required directories exist."""
    print("\nTesting directory structure...")
    
    base_dir = Path(__file__).parent
    
    required_files = [
        "__init__.py",
        "data_utils.py",
        "eval_baseline.py",
        "quantize_mxfp4.py",
        "quantize_gptq.py",
        "run_pipeline.py",
        "run_pipeline.sh",
        "analyze_results.py",
        "README.md",
        "QUICKSTART.md",
        "SUMMARY.md"
    ]
    
    all_exist = True
    for filename in required_files:
        filepath = base_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} not found")
            all_exist = False
    
    return all_exist


def test_scripts_syntax():
    """Test that all Python scripts have valid syntax."""
    print("\nTesting script syntax...")
    
    base_dir = Path(__file__).parent
    
    scripts = [
        "data_utils.py",
        "eval_baseline.py",
        "quantize_mxfp4.py",
        "quantize_gptq.py",
        "run_pipeline.py",
        "analyze_results.py",
        "example_usage.py"
    ]
    
    all_valid = True
    for script in scripts:
        filepath = base_dir / script
        try:
            with open(filepath, 'r') as f:
                compile(f.read(), str(filepath), 'exec')
            print(f"  ✓ {script}")
        except SyntaxError as e:
            print(f"  ✗ {script}: {e}")
            all_valid = False
    
    return all_valid


def main():
    """Run all tests."""
    print("=" * 60)
    print("GSM8K Analysis Setup Test")
    print("=" * 60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("AlphaQuant Imports", test_alphaquant_imports),
        ("GSM8K Analysis Imports", test_gsm8k_imports),
        ("CUDA", test_cuda),
        ("Dataset Loading", test_dataset_loading),
        ("Directory Structure", test_directory_structure),
        ("Script Syntax", test_scripts_syntax)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Setup is ready.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please fix before running pipeline.")
        return 1


if __name__ == '__main__':
    exit(main())

