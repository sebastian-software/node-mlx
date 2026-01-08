#!/bin/bash
# Copy xcodebuild artifacts to the expected locations

set -e

# Find the most recent DerivedData directory for this project
DERIVED_DATA_BASE="$HOME/Library/Developer/Xcode/DerivedData"
BUILD_DIR=$(find "$DERIVED_DATA_BASE" -maxdepth 1 -type d -name "swift-*" -exec stat -f '%m %N' {} \; 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

if [ -z "$BUILD_DIR" ]; then
  echo "Error: Could not find DerivedData directory for swift project"
  exit 1
fi

BUILD_DIR="${BUILD_DIR}/Build/Products/Release"

echo "Copying build artifacts from: ${BUILD_DIR}"

# Create target directories
mkdir -p .build/release
mkdir -p ../node-mlx/swift

# Copy the framework
if [ -d "${BUILD_DIR}/PackageFrameworks/NodeMLX.framework" ]; then
  rm -rf .build/release/NodeMLX.framework
  cp -R "${BUILD_DIR}/PackageFrameworks/NodeMLX.framework" .build/release/
  echo "✓ Copied NodeMLX.framework to .build/release/"

  # Copy metallib bundle INTO the framework's Resources
  if [ -d "${BUILD_DIR}/mlx-swift_Cmlx.bundle" ]; then
    cp -R "${BUILD_DIR}/mlx-swift_Cmlx.bundle" ".build/release/NodeMLX.framework/Versions/A/Resources/"
    echo "✓ Embedded mlx-swift_Cmlx.bundle in Framework"
  fi
else
  echo "Error: NodeMLX.framework not found at ${BUILD_DIR}/PackageFrameworks/"
  exit 1
fi

# Create dylib symlink
ln -sf "NodeMLX.framework/Versions/A/NodeMLX" .build/release/libNodeMLX.dylib
echo "✓ Created libNodeMLX.dylib symlink"

# Copy metallib bundle
if [ -d "${BUILD_DIR}/mlx-swift_Cmlx.bundle" ]; then
  rm -rf .build/release/mlx-swift_Cmlx.bundle
  cp -R "${BUILD_DIR}/mlx-swift_Cmlx.bundle" .build/release/
  echo "✓ Copied mlx-swift_Cmlx.bundle to .build/release/"
else
  echo "Warning: mlx-swift_Cmlx.bundle not found"
fi

# Copy artifacts to node-mlx/swift/ for npm publishing
echo ""
echo "Copying to packages/node-mlx/swift/ for npm publishing..."

# Copy dylib (actual file, not symlink)
if [ -f ".build/release/NodeMLX.framework/Versions/A/NodeMLX" ]; then
  cp ".build/release/NodeMLX.framework/Versions/A/NodeMLX" ../node-mlx/swift/libNodeMLX.dylib
  echo "✓ Copied libNodeMLX.dylib"
fi

# Copy metallib bundle
if [ -d ".build/release/mlx-swift_Cmlx.bundle" ]; then
  rm -rf ../node-mlx/swift/mlx-swift_Cmlx.bundle
  cp -R ".build/release/mlx-swift_Cmlx.bundle" ../node-mlx/swift/
  echo "✓ Copied mlx-swift_Cmlx.bundle"
fi

echo ""
echo "Build artifacts copied successfully!"
