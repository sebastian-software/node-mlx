/**
 * MLP component generator
 */

import type { ModelFeatures } from "../features.js"

export function generateMlp(
  modelName: string,
  configClass: string,
  features: ModelFeatures
): string {
  const activation =
    features.activation === "geluApproximate"
      ? "geluApproximate"
      : features.activation === "silu"
        ? "silu"
        : "gelu"

  return `// MARK: - MLP

class ${modelName}MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: ${configClass}) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(${activation}(gateProj(x)) * upProj(x))
    }
}`
}
