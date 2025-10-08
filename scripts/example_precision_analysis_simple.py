#!/usr/bin/env python
"""
Simple example: Run precision analysis on a small model for quick testing.
This uses a smaller model (Llama 3.2 1B) for faster execution.
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run command and print output in real-time."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    if process.returncode != 0:
        print(f"\n✗ Command failed with return code {process.returncode}")
        sys.exit(1)
    
    print(f"\n✓ Command completed successfully")


def main():
    print("\n" + "="*80)
    print("PRECISION ANALYSIS - QUICK START EXAMPLE")
    print("="*80)
    print("\nThis example uses Llama-3.2-1B for fast execution.")
    print("For Llama-3.1-8B, update the model name below.\n")
    
    # Configuration
    model = "meta-llama/Llama-3.2-1B"  # Change to "meta-llama/Llama-3.1-8B" for 8B
    device = "cpu"  # Change to "cuda" if GPU available
    load_dtype = "fp32"  # Change to "bf16" for faster loading on GPU
    precisions = "fp32,fp16,bf16"
    output_dir = "./results/precision_example"
    
    print(f"Settings:")
    print(f"  Model: {model}")
    print(f"  Device: {device}")
    print(f"  Precisions: {precisions}")
    print(f"  Output: {output_dir}")
    print()
    
    # Determine script path
    script_dir = Path(__file__).parent
    analysis_script = script_dir / "analyze_precision_alpha_hill.py"
    
    if not analysis_script.exists():
        print(f"✗ Error: Script not found at {analysis_script}")
        sys.exit(1)
    
    # Example 1: Quick test with first 3 layers only
    print("\n" + "="*80)
    print("EXAMPLE 1: Quick test (first 3 layers only)")
    print("="*80)
    
    cmd1 = [
        sys.executable,
        str(analysis_script),
        "--model", model,
        "--device", device,
        "--load-dtype", load_dtype,
        "--precisions", precisions,
        "--filter-layers", ".*layers\\.[0-2]\\..*",  # Only first 3 layers
        "--output-dir", f"{output_dir}/quick_test",
        "--max-layers-per-plot", "0",  # All in one plot
        "--log-level", "INFO"
    ]
    
    run_command(cmd1)
    
    # Example 2: Attention layers only
    print("\n" + "="*80)
    print("EXAMPLE 2: Attention layers analysis")
    print("="*80)
    
    cmd2 = [
        sys.executable,
        str(analysis_script),
        "--model", model,
        "--device", device,
        "--load-dtype", load_dtype,
        "--precisions", precisions,
        "--filter-layers", ".*(q_proj|k_proj|v_proj|o_proj).*",
        "--output-dir", f"{output_dir}/attention",
        "--max-layers-per-plot", "20",
        "--log-level", "INFO"
    ]
    
    run_command(cmd2)
    
    # Summary
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE!")
    print("="*80)
    
    output_path = Path(output_dir)
    print(f"\nResults saved to: {output_path.absolute()}")
    print("\nGenerated files:")
    
    for subdir in ["quick_test", "attention"]:
        subdir_path = output_path / subdir
        if subdir_path.exists():
            print(f"\n  {subdir}/")
            csv_file = subdir_path / "precision_alpha_hill_results.csv"
            if csv_file.exists():
                print(f"    ✓ {csv_file.name}")
            summary_file = subdir_path / "precision_summary_statistics.csv"
            if summary_file.exists():
                print(f"    ✓ {summary_file.name}")
            plots_dir = subdir_path / "plots"
            if plots_dir.exists():
                plot_count = len(list(plots_dir.glob("*.png")))
                print(f"    ✓ plots/ ({plot_count} files)")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. View CSV files for numerical results:")
    print(f"   - {output_dir}/quick_test/precision_alpha_hill_results.csv")
    print(f"   - {output_dir}/attention/precision_alpha_hill_results.csv")
    
    print("\n2. Check visualizations:")
    print(f"   - {output_dir}/quick_test/plots/")
    print(f"   - {output_dir}/attention/plots/")
    
    print("\n3. To run on full Llama-3.1-8B model:")
    print("   - Change model to 'meta-llama/Llama-3.1-8B'")
    print("   - Set device to 'cuda' (if available)")
    print("   - Set load_dtype to 'bf16' for faster loading")
    print("   - Remove --filter-layers to analyze all layers")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

