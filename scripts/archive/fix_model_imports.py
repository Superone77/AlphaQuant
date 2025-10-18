#!/usr/bin/env python3
"""
Fix imports in model files copied from transformers.
Changes relative imports (from ...) to absolute imports (from transformers...).

Usage:
    python scripts/fix_model_imports.py
"""

import re
from pathlib import Path

# Import mapping for common transformers imports
IMPORT_REPLACEMENTS = [
    (r'^from \.\.\.activations import', 'from transformers.activations import'),
    (r'^from \.\.\.cache_utils import', 'from transformers.cache_utils import'),
    (r'^from \.\.\.generation import', 'from transformers.generation import'),
    (r'^from \.\.\.integrations import', 'from transformers.integrations import'),
    (r'^from \.\.\.masking_utils import', 'from transformers.masking_utils import'),
    (r'^from \.\.\.modeling_layers import', 'from transformers.modeling_layers import'),
    (r'^from \.\.\.modeling_outputs import', 'from transformers.modeling_outputs import'),
    (r'^from \.\.\.modeling_rope_utils import', 'from transformers.modeling_rope_utils import'),
    (r'^from \.\.\.modeling_utils import', 'from transformers.modeling_utils import'),
    (r'^from \.\.\.modeling_attn_mask_utils import', 'from transformers.modeling_attn_mask_utils import'),
    (r'^from \.\.\.processing_utils import', 'from transformers.processing_utils import'),
    (r'^from \.\.\.utils import', 'from transformers.utils import'),
    (r'^from \.\.\.utils\.generic import', 'from transformers.utils.generic import'),
    (r'^from \.\.\.configuration_utils import', 'from transformers.configuration_utils import'),
]

def fix_imports_in_file(filepath):
    """Fix imports in a single file."""
    print(f"Processing: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    lines = content.split('\n')
    fixed_lines = []
    changes = 0
    
    for line in lines:
        fixed_line = line
        for pattern, replacement in IMPORT_REPLACEMENTS:
            if re.match(pattern, line):
                fixed_line = re.sub(pattern, replacement, line)
                if fixed_line != line:
                    changes += 1
                    print(f"  Fixed: {line.strip()} -> {fixed_line.strip()}")
                break
        fixed_lines.append(fixed_line)
    
    if changes > 0:
        content = '\n'.join(fixed_lines)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Made {changes} changes to {filepath.name}")
    else:
        print(f"  ✓ No changes needed for {filepath.name}")
    
    return changes

def main():
    """Main function."""
    # Get script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / 'models'
    
    if not models_dir.exists():
        print(f"models directory not found at {models_dir}!")
        print("Please run this script from the project root.")
        return 1
    
    total_changes = 0
    files_processed = 0
    
    # Process all modeling and configuration files
    for model_file in models_dir.rglob('*.py'):
        if model_file.name in ['__init__.py']:
            continue  # Skip __init__.py files
        
        if 'modeling_' in model_file.name or 'configuration_' in model_file.name:
            files_processed += 1
            changes = fix_imports_in_file(model_file)
            total_changes += changes
    
    print("\n" + "=" * 60)
    print(f"Summary: Processed {files_processed} files")
    print(f"Total import fixes: {total_changes}")
    print("=" * 60)
    
    return 0

if __name__ == '__main__':
    exit(main())

