#!/usr/bin/env python3
"""Export trained PyTorch models to ONNX format.

This script exports CubeMaster color classification models (MLP, Shallow CNN, MobileNet)
from PyTorch checkpoint files (.pth/.pt) to ONNX format for deployment.

Usage:
    # Export with auto-detection of model type from checkpoint
    python scripts/export_to_onnx.py --checkpoint models/shallow_cnn/best.pt
    
    # Specify model type explicitly
    python scripts/export_to_onnx.py --checkpoint models/mlp/best.pt --model mlp
    
    # Custom output path
    python scripts/export_to_onnx.py --checkpoint models/shallow_cnn/best.pt --output models/onnx/model.onnx
    
    # Custom input size (default: 50x50)
    python scripts/export_to_onnx.py --checkpoint models/shallow_cnn/best.pt --input-size 40 40
    
    # Skip validation
    python scripts/export_to_onnx.py --checkpoint models/shallow_cnn/best.pt --no-validate
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cubemaster.models import MODEL_REGISTRY, MLPClassifier, ShallowCNNClassifier, MobileNetV3Classifier

# Try to import ONNX dependencies
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export CubeMaster PyTorch models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint file (.pth or .pt)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "shallow_cnn", "mobilenet"],
        default=None,
        help="Model architecture (auto-detected from checkpoint if not specified)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX file path (default: same directory as checkpoint)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[50, 50],
        metavar=("HEIGHT", "WIDTH"),
        help="Input image size (default: 50 50)",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=11,
        help="ONNX opset version (default: 11)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip ONNX model validation",
    )
    parser.add_argument(
        "--no-test-inference",
        action="store_true",
        help="Skip inference test with ONNX Runtime",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    return parser.parse_args()


def detect_model_type(checkpoint: dict) -> Optional[str]:
    """Detect model type from checkpoint metadata or state dict keys.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        
    Returns:
        Model type string or None if not detected
    """
    # Check for explicit model name in checkpoint
    if "model_name" in checkpoint:
        model_name = checkpoint["model_name"]
        if model_name in MODEL_REGISTRY:
            return model_name
    
    # Check config
    if "config" in checkpoint:
        config = checkpoint["config"]
        if isinstance(config, dict) and "model" in config:
            model_name = config["model"].get("name")
            if model_name in MODEL_REGISTRY:
                return model_name
    
    # Try to detect from state dict structure
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    
    if isinstance(state_dict, dict):
        keys = list(state_dict.keys())
        
        # MLP has fc layers named fc1, fc2, etc.
        if any("fc1" in k for k in keys) and not any("conv" in k.lower() for k in keys):
            return "mlp"
        
        # Shallow CNN has conv layers
        if any("conv1" in k for k in keys) and any("conv2" in k for k in keys):
            if not any("backbone" in k for k in keys):
                return "shallow_cnn"
        
        # MobileNet has backbone
        if any("backbone" in k for k in keys):
            return "mobilenet"
    
    return None


def build_model(model_type: str, checkpoint: dict, input_size: Tuple[int, int]) -> torch.nn.Module:
    """Build model from checkpoint.
    
    Args:
        model_type: Model architecture name
        checkpoint: Loaded checkpoint dictionary
        input_size: Input image size (height, width)
        
    Returns:
        Loaded PyTorch model
    """
    # Get model class
    model_cls = MODEL_REGISTRY.get(model_type)
    if model_cls is None:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Extract model kwargs from checkpoint config
    kwargs = {
        "num_classes": 6,
        "input_size": input_size,
    }

    # Get config from checkpoint if available
    config = checkpoint.get("config", {})
    model_config = config.get("model", {}) if isinstance(config, dict) else {}

    # Model-specific parameters
    if model_type == "mlp":
        kwargs["hidden_dims"] = model_config.get("hidden_dims", [256, 128])
        kwargs["dropout_rate"] = model_config.get("dropout_rate", 0.3)
    elif model_type == "shallow_cnn":
        kwargs["dropout_rate"] = model_config.get("dropout_rate", 0.5)
    elif model_type == "mobilenet":
        kwargs["pretrained"] = False  # Don't download weights, we're loading from checkpoint
        kwargs["freeze_backbone"] = model_config.get("freeze_backbone", False)

    # Create model
    model = model_cls(**kwargs)

    # Load state dict
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict)

    return model


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    input_size: Tuple[int, int],
    opset_version: int = 11,
    verbose: bool = False,
) -> Path:
    """Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        output_path: Output ONNX file path
        input_size: Input image size (height, width)
        opset_version: ONNX opset version
        verbose: Print verbose output

    Returns:
        Path to exported ONNX model
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set model to eval mode
    model.eval()
    model.cpu()

    # Create dummy input (batch, channels, height, width)
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        verbose=verbose,
        dynamo=False,  # Use legacy exporter for compatibility
    )

    return output_path


def validate_onnx_model(onnx_path: Path) -> bool:
    """Validate ONNX model structure.

    Args:
        onnx_path: Path to ONNX model

    Returns:
        True if valid, False otherwise
    """
    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        return True
    except Exception as e:
        print(f"❌ ONNX validation failed: {e}")
        return False


def test_onnx_inference(
    onnx_path: Path,
    pytorch_model: torch.nn.Module,
    input_size: Tuple[int, int],
    verbose: bool = False,
) -> bool:
    """Test ONNX model inference and compare with PyTorch.

    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model for comparison
        input_size: Input image size
        verbose: Print verbose output

    Returns:
        True if outputs match, False otherwise
    """
    # Create test input
    test_input = torch.randn(1, 3, input_size[0], input_size[1])

    # PyTorch inference
    pytorch_model.eval()
    pytorch_model.cpu()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()

    # ONNX Runtime inference
    ort_session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # Compare outputs
    max_diff = np.abs(pytorch_output - ort_output).max()
    mean_diff = np.abs(pytorch_output - ort_output).mean()

    if verbose:
        print(f"   PyTorch output shape: {pytorch_output.shape}")
        print(f"   ONNX output shape: {ort_output.shape}")
        print(f"   Max difference: {max_diff:.6e}")
        print(f"   Mean difference: {mean_diff:.6e}")

    # Tolerance for floating point comparison
    tolerance = 1e-5
    if max_diff < tolerance:
        return True
    else:
        print(f"⚠️  Output difference ({max_diff:.6e}) exceeds tolerance ({tolerance})")
        return max_diff < 1e-3  # Still acceptable if within 1e-3


def main():
    """Main export entry point."""
    args = parse_args()

    # Check ONNX dependencies
    if not ONNX_AVAILABLE:
        print("❌ Error: ONNX dependencies not installed.")
        print("   Install with: pip install onnx onnxruntime")
        sys.exit(1)

    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"CubeMaster ONNX Export")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load checkpoint
    print(f"\n📦 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Detect or use specified model type
    model_type = args.model
    if model_type is None:
        model_type = detect_model_type(checkpoint)
        if model_type is None:
            print("❌ Error: Could not detect model type from checkpoint.")
            print("   Please specify --model explicitly.")
            sys.exit(1)
        print(f"   Auto-detected model type: {model_type}")
    else:
        print(f"   Model type: {model_type}")

    # Input size
    input_size = tuple(args.input_size)
    print(f"   Input size: {input_size[0]}x{input_size[1]}")

    # Build model
    print(f"\n🔧 Building {model_type} model...")
    try:
        model = build_model(model_type, checkpoint, input_size)
        params = model.count_parameters()
        print(f"   Parameters: {params['total']:,} total, {params['trainable']:,} trainable")
    except Exception as e:
        print(f"❌ Error building model: {e}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = checkpoint_path.parent / f"{checkpoint_path.stem}.onnx"

    print(f"\n📤 Exporting to ONNX...")
    print(f"   Output: {output_path}")
    print(f"   Opset version: {args.opset_version}")

    try:
        export_to_onnx(model, output_path, input_size, args.opset_version, args.verbose)
        print(f"   ✅ Export successful")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)

    # Get file size
    size_bytes = output_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"   File size: {size_mb:.2f} MB ({size_bytes:,} bytes)")

    # Validate ONNX model
    if not args.no_validate:
        print(f"\n🔍 Validating ONNX model...")
        if validate_onnx_model(output_path):
            print(f"   ✅ Model structure valid")
        else:
            print(f"   ❌ Validation failed")
            sys.exit(1)

    # Test inference
    if not args.no_test_inference:
        print(f"\n🧪 Testing inference...")
        if test_onnx_inference(output_path, model, input_size, args.verbose):
            print(f"   ✅ Inference test passed")
        else:
            print(f"   ⚠️  Inference test had differences (may still be usable)")

    print(f"\n{'='*60}")
    print(f"✅ Export complete: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

