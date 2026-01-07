#!/bin/bash
# Copy xcodebuild artifacts to the expected locations

set -e

DERIVED_DATA=$(xcodebuild -showBuildSettings -scheme NodeMLX 2>/dev/null | grep "BUILD_DIR" | head -1 | awk '{print $3}')
BUILD_DIR="${DERIVED_DATA}/Release"

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

# Copy metallib to project root for runtime access
if [ -f "${BUILD_DIR}/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" ]; then
  cp "${BUILD_DIR}/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" ../default.metallib
  echo "✓ Copied default.metallib to project root"
fi

echo "Build artifacts copied successfully!"

