#!/bin/bash
# Copy xcodebuild artifacts to the expected locations

set -e

# Use known DerivedData path
BUILD_DIR="/Users/sebastian/Library/Developer/Xcode/DerivedData/swift-gkebfpwoaxkhlaccfkirsnabbjnh/Build/Products/Release"

echo "Copying build artifacts from: ${BUILD_DIR}"

# Create target directories
mkdir -p .build/release
mkdir -p ../node-mlx/swift

# Copy the framework
if [ -d "${BUILD_DIR}/PackageFrameworks/NodeMLX.framework" ]; then
  cp -R "${BUILD_DIR}/PackageFrameworks/NodeMLX.framework" .build/release/
  echo "✓ Copied NodeMLX.framework to .build/release/"
fi

# Create dylib symlink
ln -sf "NodeMLX.framework/Versions/A/NodeMLX" .build/release/libNodeMLX.dylib
echo "✓ Created libNodeMLX.dylib symlink"

# Copy metallib bundle
if [ -d "${BUILD_DIR}/mlx-swift_Cmlx.bundle" ]; then
  cp -R "${BUILD_DIR}/mlx-swift_Cmlx.bundle" .build/release/
  echo "✓ Copied mlx-swift_Cmlx.bundle to .build/release/"
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
