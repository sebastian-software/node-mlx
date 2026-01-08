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

## swift-transformers

**License:** Apache License 2.0
**Copyright:** (c) 2023 Hugging Face
**Repository:** https://github.com/huggingface/swift-transformers

Used for tokenization (AutoTokenizer) and HuggingFace Hub integration.

---

## Code Attribution

### mlx-swift-lm Patterns

Parts of the `NodeMLXCore` implementation are based on patterns from [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) (MIT License, ml-explore):

- `KVCache.swift` - Key-value cache implementations
- `StringOrNumber.swift` - JSON config parsing utilities
- `RoPEUtils.swift` - Rotary position embedding implementations
- `AttentionUtils.swift` - Attention with cache routing
- `Generate.swift` - Sampling strategies

These files contain attribution comments referencing the original source.

---

## Usage Notes

This project (`node-mlx`) provides native Node.js bindings for Apple's MLX framework. The Swift code in `swift/Sources/NodeMLXCore/` implements LLM functionality independently, with some utility code adapted from `mlx-swift-lm`.

The code in `tools/hf2swift/generator.py` generates Swift model definitions from HuggingFace Transformers Python code. This generated code follows architectural patterns similar to `mlx-swift-lm` models.

All MLX-related dependencies are developed by Apple's [ml-explore](https://github.com/ml-explore) team and released under the MIT License.
