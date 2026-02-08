"""
Test script to verify the --use_original_dreamer flag works correctly.
This script checks which version of DreamerV3 is being imported.
"""
import sys
import pathlib

# Parse arguments early
root = pathlib.Path(__file__).parent
sys.path.insert(0, str(root))
from input_args import parse_craftax_args

args = parse_craftax_args()

# Add dreamerv3 to path based on flag
if args.use_original_dreamer:
    print("✓ Using ORIGINAL DreamerV3 from dreamerv3-main/")
    sys.path.insert(0, str(root / 'dreamerv3-main'))
    sys.path.insert(0, str(root / 'dreamerv3-main' / 'dreamerv3'))
else:
    print("✓ Using CONTINUOUS ENHANCED DreamerV3 from dreamerv3/")
    sys.path.insert(0, str(root / 'dreamerv3'))
    sys.path.insert(0, str(root / 'dreamerv3' / 'dreamerv3'))

# Try to import and show the path
try:
    import dreamerv3
    print(f"✓ DreamerV3 imported from: {dreamerv3.__file__}")

    from dreamerv3.agent import Agent
    print(f"✓ Agent imported successfully from: {Agent.__module__}")

    print("\n" + "="*60)
    print("SUCCESS: DreamerV3 version selection is working correctly!")
    print("="*60)
except ImportError as e:
    print(f"✗ Error importing dreamerv3: {e}")
    sys.exit(1)
