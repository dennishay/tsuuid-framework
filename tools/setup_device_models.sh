#!/bin/bash
# Copy Core ML models to the iOS app's App Group container on a connected device.
#
# Usage: ./tools/setup_device_models.sh
#
# This copies LaBSE-full.mlpackage and CLIP-full.mlpackage to the shared
# App Group container (group.com.tsuuid.768) on the connected iPhone.
# Run this ONCE after first installing the app on the device.
#
# The models are too large to bundle in the app binary (942MB + 176MB).
# Instead, the app expects them in the App Group container at:
#   group.com.tsuuid.768/models/LaBSE-full.mlmodelc
#   group.com.tsuuid.768/models/CLIP-full.mlmodelc
#
# Core ML compiles .mlpackage → .mlmodelc automatically at build time,
# but for sideloading we need to compile them first.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"

echo "Compiling Core ML models..."

# Compile .mlpackage → .mlmodelc
if [ -d "$MODELS_DIR/LaBSE-full.mlpackage" ]; then
    xcrun coremlcompiler compile "$MODELS_DIR/LaBSE-full.mlpackage" "$MODELS_DIR/"
    echo "  LaBSE compiled → LaBSE-full.mlmodelc"
else
    echo "  ERROR: $MODELS_DIR/LaBSE-full.mlpackage not found"
    echo "  Run: /tmp/coreml-py311/bin/python3 tools/convert_labse_coreml.py"
    exit 1
fi

if [ -d "$MODELS_DIR/CLIP-full.mlpackage" ]; then
    xcrun coremlcompiler compile "$MODELS_DIR/CLIP-full.mlpackage" "$MODELS_DIR/"
    echo "  CLIP compiled → CLIP-full.mlmodelc"
else
    echo "  WARNING: CLIP model not found, skipping"
fi

echo ""
echo "Models compiled. To copy to device:"
echo "  1. Open TSUUID768.xcodeproj in Xcode"
echo "  2. Run the app on your iPhone"
echo "  3. In Xcode: Window → Devices and Simulators"
echo "  4. Select your iPhone → TSUUID768 → Download Container"
echo "  5. Copy models/ into AppData/Library/group.com.tsuuid.768/models/"
echo "  6. Replace Container on device"
echo ""
echo "Or add models as On-Demand Resources in a future update."
