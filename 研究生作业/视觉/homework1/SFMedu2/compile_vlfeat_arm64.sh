#!/bin/bash
# Script to compile VLFeat for ARM64 using Makefile

echo "=========================================="
echo "Compiling VLFeat for ARM64 (mexmaca64)"
echo "=========================================="
echo ""

cd "$(dirname "$0")/matchSIFT/vlfeat"

# Check if Makefile exists
if [ ! -f "Makefile" ]; then
    echo "ERROR: Makefile not found!"
    exit 1
fi

# Clean previous builds for maca64
echo "Cleaning previous builds..."
make ARCH=maca64 archclean 2>/dev/null || true

# Build for ARM64 (maca64)
echo ""
echo "Building VLFeat for ARM64..."
echo "This may take several minutes..."
echo ""

make ARCH=maca64

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Compilation successful!"
    echo "=========================================="
    echo ""
    echo "MEX files should be in: toolbox/mex/mexmaca64/"
    echo ""
    echo "You can now run SFMedu2.m in MATLAB"
else
    echo ""
    echo "=========================================="
    echo "✗ Compilation failed!"
    echo "=========================================="
    echo ""
    echo "Please check the error messages above."
    exit 1
fi

