# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.x.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public GitHub issue
2. Email the maintainer directly or use GitHub's private vulnerability reporting
3. Include detailed information about the vulnerability
4. Allow reasonable time for a fix before public disclosure

## Security Considerations

### Model Downloads

- Models are downloaded from Hugging Face over HTTPS
- Downloaded files are stored in `~/.cache/huggingface/`
- No authentication tokens are stored or transmitted by node-mlx

### Local Processing

- All inference happens locally on your device
- No data is sent to external servers
- The library works fully offline after model download

### Swift CLI

- The Swift CLI is compiled from source during installation
- Source code is available for review in `swift/Sources/`
- Uses official Apple MLX libraries
