#!/bin/bash
# Copy xcodebuild artifacts to the expected locations

set -e

# Use known DerivedData path
BUILD_DIR="/Users/sebastian/Library/Developer/Xcode/DerivedData/swift-gkebfpwoaxkhlaccfkirsnabbjnh/Build/Products/Release"

echo "Copying build artifacts from: ${BUILD_DIR}"

# Create target directory
mkdir -p .build/release

# Copy the framework
if [ -d "${BUILD_DIR}/PackageFrameworks/NodeMLX.framework" ]; then
  cp -R "${BUILD_DIR}/PackageFrameworks/NodeMLX.framework" .build/release/
  echo "✓ Copied NodeMLX.framework"
fi

# Create dylib symlink
ln -sf "NodeMLX.framework/Versions/A/NodeMLX" .build/release/libNodeMLX.dylib
echo "✓ Created libNodeMLX.dylib symlink"

# Copy metallib bundle
if [ -d "${BUILD_DIR}/mlx-swift_Cmlx.bundle" ]; then
  cp -R "${BUILD_DIR}/mlx-swift_Cmlx.bundle" .build/release/
  echo "✓ Copied mlx-swift_Cmlx.bundle"
fi

# Copy metallib to project root for runtime access (2 levels up from packages/swift)
if [ -f "${BUILD_DIR}/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" ]; then
  cp "${BUILD_DIR}/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" ../../default.metallib
  echo "✓ Copied default.metallib to project root"
fi

echo "Build artifacts copied successfully!"
