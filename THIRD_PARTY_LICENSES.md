# Third-Party Licenses

This project uses the following third-party libraries:

## Apple MLX

**License:** MIT License  
**Copyright:** (c) 2023 ml-explore  
**Repository:** https://github.com/ml-explore/mlx

```
MIT License

Copyright (c) 2023 ml-explore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## mlx-swift

**License:** MIT License  
**Copyright:** (c) 2023 ml-explore  
**Repository:** https://github.com/ml-explore/mlx-swift

Same MIT License as above.

## mlx-swift-lm

**License:** MIT License  
**Copyright:** (c) 2024 ml-explore  
**Repository:** https://github.com/ml-explore/mlx-swift-lm

Same MIT License as above.

---

## Usage Notes

This project (`node-mlx`) uses `mlx-swift-lm` as a runtime dependency for model loading, tokenization, and generation. The native binding (`swift/Sources/NodeMLX/`) interfaces with `mlx-swift-lm` which in turn uses Apple's MLX framework.

The code in `tools/hf2swift/` generates Swift model definitions from HuggingFace Transformers Python code. This generated code follows the same architectural patterns as `mlx-swift-lm` models but is independently generated from upstream Python sources.

All three dependencies (MLX, mlx-swift, mlx-swift-lm) are developed by Apple's [ml-explore](https://github.com/ml-explore) team and released under the MIT License.
