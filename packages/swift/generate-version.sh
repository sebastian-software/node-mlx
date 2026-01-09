#!/bin/bash
# Generate Version.swift from package.json
# This ensures Swift version matches the npm package version

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_JSON="$SCRIPT_DIR/../node-mlx/package.json"
VERSION_SWIFT="$SCRIPT_DIR/Sources/NodeMLX/Version.swift"

# Extract version from package.json
if [ -f "$PACKAGE_JSON" ]; then
  VERSION=$(grep '"version"' "$PACKAGE_JSON" | head -1 | sed 's/.*"version": *"\([^"]*\)".*/\1/')
else
  echo "Warning: package.json not found, using fallback version"
  VERSION="0.0.0"
fi

echo "Generating Version.swift with version: $VERSION"

cat > "$VERSION_SWIFT" << EOF
//
//  Version.swift
//  NodeMLX
//
//  AUTO-GENERATED - DO NOT EDIT
//  This file is regenerated during build from package.json
//

/// Package version (synchronized with package.json)
public let NODE_MLX_VERSION = "$VERSION"
EOF

echo "✓ Generated $VERSION_SWIFT"
