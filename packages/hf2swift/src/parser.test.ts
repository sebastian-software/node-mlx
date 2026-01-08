import { describe, it, expect } from "vitest"
import { HFModelParser } from "./parser.js"

describe("HFModelParser", () => {
  describe("parse", () => {
    it("parses a simple module class", () => {
      const source = `
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
`
      const parser = new HFModelParser("qwen2")
      const modules = parser.parse(source)

      expect(modules).toHaveLength(1)
      expect(modules[0].name).toBe("Qwen2MLP")
      expect(modules[0].baseClasses).toContain("nn.Module")
      expect(modules[0].attributes).toHaveLength(3)

      const gateProj = modules[0].attributes.find((a) => a.name === "gate_proj")
      expect(gateProj).toBeDefined()
      expect(gateProj?.moduleType).toBe("Linear")
      expect(gateProj?.swiftName).toBe("gateProj")
    })

    it("parses attention module with multiple projections", () => {
      const source = `
class Qwen2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        pass
`
      const parser = new HFModelParser("qwen2")
      const modules = parser.parse(source)

      expect(modules).toHaveLength(1)
      expect(modules[0].attributes).toHaveLength(4)

      const projNames = modules[0].attributes.map((a) => a.swiftName)
      expect(projNames).toContain("qProj")
      expect(projNames).toContain("kProj")
      expect(projNames).toContain("vProj")
      expect(projNames).toContain("oProj")
    })

    it("parses forward method arguments", () => {
      const source = `
class TestModule(nn.Module):
    def __init__(self, config):
        pass

    def forward(self, x, attention_mask=None, cache=None):
        pass
`
      const parser = new HFModelParser("test")
      const modules = parser.parse(source)

      expect(modules[0].methods).toHaveLength(1)
      expect(modules[0].methods[0].swiftName).toBe("callAsFunction")
      expect(modules[0].methods[0].args).toHaveLength(3)
      expect(modules[0].methods[0].args[0].name).toBe("x")
    })

    it("parses modules with RMSNorm", () => {
      const source = `
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, mask=None, cache=None):
        pass
`
      const parser = new HFModelParser("test")
      const modules = parser.parse(source)

      expect(modules[0].attributes).toHaveLength(2)

      const inputNorm = modules[0].attributes.find((a) => a.name === "input_layernorm")
      expect(inputNorm?.moduleType).toBe("RMSNorm")
      expect(inputNorm?.swiftName).toBe("inputLayernorm")
    })

    it("handles multiple classes", () => {
      const source = `
class ModuleA(nn.Module):
    def __init__(self, config):
        self.layer = nn.Linear(10, 10)
    def forward(self, x):
        return x

class ModuleB(nn.Module):
    def __init__(self, config):
        self.embed = nn.Embedding(1000, 512)
    def forward(self, x):
        return x
`
      const parser = new HFModelParser("test")
      const modules = parser.parse(source)

      expect(modules).toHaveLength(2)
      expect(modules[0].name).toBe("ModuleA")
      expect(modules[1].name).toBe("ModuleB")
    })

    it("skips non-module classes", () => {
      const source = `
class Config:
    hidden_size: int = 768

class MyModel(nn.Module):
    def __init__(self, config):
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        return x
`
      const parser = new HFModelParser("test")
      const modules = parser.parse(source)

      expect(modules).toHaveLength(1)
      expect(modules[0].name).toBe("MyModel")
    })

    it("handles classes inheriting from other parsed classes", () => {
      const source = `
class BaseAttention(nn.Module):
    def __init__(self, config):
        self.proj = nn.Linear(10, 10)
    def forward(self, x):
        return x

class SpecialAttention(BaseAttention):
    def __init__(self, config):
        super().__init__(config)
        self.extra = nn.Linear(10, 10)
    def forward(self, x):
        return x
`
      const parser = new HFModelParser("test")
      const modules = parser.parse(source)

      expect(modules).toHaveLength(2)
      expect(modules[1].name).toBe("SpecialAttention")
      expect(modules[1].baseClasses).toContain("BaseAttention")
    })

    it("extracts properties from simple assignments", () => {
      const source = `
class Module(nn.Module):
    def __init__(self, config):
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.proj = nn.Linear(10, 10)
    def forward(self, x):
        return x
`
      const parser = new HFModelParser("test")
      const modules = parser.parse(source)

      // Should have 1 attribute (proj) and 2 properties
      expect(modules[0].attributes).toHaveLength(1)
      expect(modules[0].properties).toHaveLength(2)
      expect(modules[0].properties[0].swiftName).toBe("numHeads")
    })
  })
})
