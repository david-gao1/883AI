#!/bin/bash
# Script to start MATLAB under Rosetta 2 (x86_64 mode) for VLFeat compatibility

echo "=========================================="
echo "Starting MATLAB under Rosetta 2 (x86_64)"
echo "=========================================="
echo ""

# Common MATLAB installation paths
MATLAB_PATHS=(
    "/Applications/MATLAB_R2025a.app/bin/matlab"
    "/Applications/MATLAB_R2024b.app/bin/matlab"
    "/Applications/MATLAB_R2024a.app/bin/matlab"
    "/Applications/MATLAB_R2023b.app/bin/matlab"
    "/Applications/MATLAB_R2023a.app/bin/matlab"
    "/Applications/MATLAB_R2022b.app/bin/matlab"
    "/Applications/MATLAB_R2022a.app/bin/matlab"
    "/Applications/MATLAB_R2021b.app/bin/matlab"
    "/Applications/MATLAB_R2021a.app/bin/matlab"
)

# Try to find MATLAB
MATLAB_CMD=""
for path in "${MATLAB_PATHS[@]}"; do
    if [ -f "$path" ]; then
        MATLAB_CMD="$path"
        echo "Found MATLAB at: $MATLAB_CMD"
        break
    fi
done

# If not found, try to find any MATLAB installation
if [ -z "$MATLAB_CMD" ]; then
    echo "Searching for MATLAB installation..."
    MATLAB_CMD=$(find /Applications -name "matlab" -type f 2>/dev/null | grep -E "MATLAB.*app/bin/matlab" | head -1)
    if [ -n "$MATLAB_CMD" ]; then
        echo "Found MATLAB at: $MATLAB_CMD"
    fi
fi

# If still not found, prompt user
if [ -z "$MATLAB_CMD" ]; then
    echo ""
    echo "ERROR: Could not find MATLAB installation."
    echo "Please provide the full path to MATLAB:"
    read -p "MATLAB path: " MATLAB_CMD
    if [ ! -f "$MATLAB_CMD" ]; then
        echo "ERROR: File not found: $MATLAB_CMD"
        exit 1
    fi
fi

echo ""
echo "Starting MATLAB in x86_64 mode..."
echo ""

# Start MATLAB under Rosetta 2
arch -x86_64 "$MATLAB_CMD"

