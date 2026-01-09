#!/bin/bash
# Copy build artifacts to the expected locations
# Works with Swift CLI (swift build -c release)

set -e

# Swift CLI build directory
BUILD_DIR=".build/arm64-apple-macosx/release"

if [ ! -f "${BUILD_DIR}/libNodeMLX.dylib" ]; then
  echo "Error: libNodeMLX.dylib not found. Run 'swift build -c release' first."
  exit 1
fi

echo "Copying build artifacts from: ${BUILD_DIR}"

# Create target directories
mkdir -p ../node-mlx/swift

# Copy dylib and strip debug symbols (removes local paths from errors)
cp "${BUILD_DIR}/libNodeMLX.dylib" ../node-mlx/swift/
strip -x ../node-mlx/swift/libNodeMLX.dylib 2>/dev/null || true
echo "✓ Copied and stripped libNodeMLX.dylib"

# Copy metallib bundle
if [ -d "${BUILD_DIR}/mlx-swift_Cmlx.bundle" ]; then
  rm -rf ../node-mlx/swift/mlx-swift_Cmlx.bundle
  cp -R "${BUILD_DIR}/mlx-swift_Cmlx.bundle" ../node-mlx/swift/
  echo "✓ Copied mlx-swift_Cmlx.bundle"

  # Also copy metallib directly to the binary directory for MLX to find
  cp "${BUILD_DIR}/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" ../node-mlx/swift/mlx.metallib
  echo "✓ Copied mlx.metallib"
else
  echo "Warning: mlx-swift_Cmlx.bundle not found"
fi

# Also copy metallib to the .build/release directory for development
mkdir -p .build/release
cp "${BUILD_DIR}/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" .build/release/mlx.metallib 2>/dev/null || true
cp -R "${BUILD_DIR}/mlx-swift_Cmlx.bundle" .build/release/ 2>/dev/null || true
ln -sf "../arm64-apple-macosx/release/libNodeMLX.dylib" .build/release/libNodeMLX.dylib 2>/dev/null || true

echo ""
echo "Build artifacts copied successfully!"
