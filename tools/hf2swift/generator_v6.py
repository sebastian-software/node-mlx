#!/usr/bin/env python3
"""
hf2swift v6 - HuggingFace to MLX Swift Code Generator

Key improvements over v5:
- Uses PARSED modules from Python source instead of fixed templates
- Correctly handles model-specific features (q_norm, k_norm, Laurel blocks, etc.)
- Generates modules in dependency order
- Supports nn.Embedding subclasses and custom layer types

Usage:
    python generator_v6.py --model gemma3n --source modeling_gemma3n.py --config mlx-community/gemma-3n-E4B-it-lm-4bit
"""

import ast
import re
import json
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

# =============================================================================
# NAMING CONVENTIONS
# =============================================================================

def to_camel(name: str) -> str:
    """Convert snake_case to camelCase"""
    parts = name.split('_')
    return parts[0] + ''.join(p.capitalize() for p in parts[1:])


def to_pascal(name: str) -> str:
    """Convert snake_case to PascalCase"""
    if name.lower() == 'gpt_oss':
        return 'GptOSS'
    parts = name.replace('-', '_').split('_')
    return ''.join(p.capitalize() for p in parts)


# =============================================================================
# KNOWN MODULE MAPPINGS
# =============================================================================

NN_MODULES = {
    "nn.Linear": ("Linear", ["in_features", "out_features", "bias"]),
    "nn.Embedding": ("Embedding", ["num_embeddings", "embedding_dim"]),
    "nn.LayerNorm": ("LayerNorm", ["normalized_shape", "eps"]),
    "RMSNorm": ("RMSNorm", ["dimensions", "eps"]),
    "nn.RMSNorm": ("RMSNorm", ["dimensions", "eps"]),
    "nn.Dropout": (None, []),
    "nn.ModuleList": ("Array", []),
    "nn.Conv1d": ("Conv1d", ["in_channels", "out_channels", "kernel_size"]),
    "nn.Conv2d": ("Conv2d", ["in_channels", "out_channels", "kernel_size"]),
}


# =============================================================================
# EXPRESSION CONVERSIONS
# =============================================================================

EXPR_CONVERSIONS = [
    (r"getattr\((\w+),\s*['\"](\w+)['\"],\s*([^)]+)\)", r"\1.\2 ?? \3"),
    (r"getattr\((\w+),\s*['\"](\w+)['\"]\)", r"\1.\2"),
    (r"isinstance\((\w+),\s*(\w+)\)", r"\1 is \2"),
    (r"\bint\(([^)]+)\)", r"Int(\1)"),
    (r"\bfloat\(([^)]+)\)", r"Float(\1)"),
    (r"\band\b", "&&"),
    (r"\bor\b", "||"),
    (r"\bnot\b", "!"),
    (r"\bTrue\b", "true"),
    (r"\bFalse\b", "false"),
    (r"\bNone\b", "nil"),
    (r" is nil", " == nil"),
    (r" is not nil", " != nil"),
    (r"\bself\.", ""),
    (r"\.view\(", ".reshaped(["),
    (r"\.reshape\(", ".reshaped(["),
    (r"\.transpose\((\d+),\s*(\d+)\)", r".transposed(\1, \2)"),
    (r"\.contiguous\(\)", ""),
    (r"\.unsqueeze\((\d+)\)", r".expandedDimensions(axis: \1)"),
    (r"\.squeeze\((\d+)\)", r".squeezed(axis: \1)"),
    (r"'([^']*)'", r'"\1"'),
]


def convert_expr(expr: str) -> str:
    result = expr
    for pattern, replacement in EXPR_CONVERSIONS:
        result = re.sub(pattern, replacement, result)
    return result


# =============================================================================
# TYPE INFERENCE
# =============================================================================

def infer_swift_type(value: Any) -> Tuple[str, bool]:
    if value is None:
        return ("Any?", True)
    if isinstance(value, bool):
        return ("Bool", True)
    if isinstance(value, int):
        return ("Int", True)
    if isinstance(value, float):
        return ("Float", True)
    if isinstance(value, str):
        return ("String", True)
    if isinstance(value, list):
        if not value:
            return ("[Any]", False)
        first = value[0]
        if isinstance(first, int):
            return ("[Int]", True)
        elif isinstance(first, float):
            return ("[Float]", True)
        elif isinstance(first, str):
            return ("[String]", True)
        return ("[Any]", False)
    return ("Any", False)


# =============================================================================
# CONFIG GENERATION
# =============================================================================

@dataclass
class ConfigField:
    name: str
    swift_name: str
    swift_type: str
    default: Optional[str]
    optional: bool
    coding_key: str


def generate_config_from_json(config_json: Dict[str, Any], model_name: str) -> str:
    """Generate Swift Configuration struct from config.json"""
    important_fields = {
        'hidden_size', 'num_hidden_layers', 'num_attention_heads',
        'intermediate_size', 'vocab_size', 'model_type'
    }
    skip_fields = {
        'auto_map', 'quantization', 'rope_scaling', 'quantization_config',
        'task_specific_params', 'id2label', 'label2id',
        'vision_config', 'audio_config'
    }
    default_fields = {
        'attention_bias': False,
        'rms_norm_eps': 1e-6,
        'num_key_value_heads': None,
    }

    for key, default_value in default_fields.items():
        if key not in config_json:
            config_json[key] = default_value

    is_vlm = 'text_config' in config_json and isinstance(config_json['text_config'], dict)
    text_config = config_json.get('text_config', {}) if is_vlm else {}

    fields: List[ConfigField] = []

    if is_vlm:
        for key, value in text_config.items():
            if key.startswith('_') or key in skip_fields:
                continue
            swift_name = to_camel(key)
            swift_type, is_codable = infer_swift_type(value)
            if not is_codable:
                continue
            optional = key not in important_fields
            if optional and not swift_type.endswith('?'):
                swift_type += '?'
            fields.append(ConfigField(
                name=key, swift_name=swift_name, swift_type=swift_type,
                default=None, optional=optional, coding_key=key
            ))

    for key, value in config_json.items():
        if key.startswith('_') or key in ('architectures', 'transformers_version', 'torch_dtype'):
            continue
        if key in skip_fields or (is_vlm and key == 'text_config'):
            continue
        if any(f.name == key for f in fields):
            continue

        swift_name = to_camel(key)
        swift_type, is_codable = infer_swift_type(value)
        if not is_codable:
            continue

        optional = value is None or key not in important_fields
        if optional and not swift_type.endswith('?'):
            swift_type += '?'

        fields.append(ConfigField(
            name=key, swift_name=swift_name, swift_type=swift_type,
            default=None, optional=optional, coding_key=key
        ))

    class_name = f"{to_pascal(model_name)}Configuration"
    protocol = "Decodable, Sendable" if is_vlm else "Codable, Sendable"
    lines = [f"public struct {class_name}: {protocol} {{"]

    for f in fields:
        lines.append(f"    public var {f.swift_name}: {f.swift_type}")

    lines.append("")

    if is_vlm:
        text_config_fields = {to_camel(k) for k in text_config.keys() if not k.startswith('_')}
        lines.append("    enum CodingKeys: String, CodingKey {")
        lines.append('        case textConfig = "text_config"')
        for f in fields:
            if f.swift_name not in text_config_fields:
                lines.append(f'        case {f.swift_name} = "{f.name}"')
        lines.append("    }")
        lines.append("")
        lines.append("    enum TextConfigCodingKeys: String, CodingKey {")
        for f in fields:
            if f.swift_name in text_config_fields:
                lines.append(f'        case {f.swift_name} = "{f.name}"')
        lines.append("    }")
        lines.append("")
        lines.append("    public init(from decoder: Swift.Decoder) throws {")
        lines.append("        let container = try decoder.container(keyedBy: CodingKeys.self)")
        lines.append("        let textContainer = try container.nestedContainer(keyedBy: TextConfigCodingKeys.self, forKey: .textConfig)")
        lines.append("")
        for f in fields:
            base_type = f.swift_type.rstrip('?')
            if f.swift_name in text_config_fields:
                if f.optional or f.swift_type.endswith('?'):
                    lines.append(f"        self.{f.swift_name} = try textContainer.decodeIfPresent({base_type}.self, forKey: .{f.swift_name})")
                else:
                    lines.append(f"        self.{f.swift_name} = try textContainer.decode({base_type}.self, forKey: .{f.swift_name})")
            else:
                if f.optional or f.swift_type.endswith('?'):
                    lines.append(f"        self.{f.swift_name} = try container.decodeIfPresent({base_type}.self, forKey: .{f.swift_name})")
                else:
                    lines.append(f"        self.{f.swift_name} = try container.decode({base_type}.self, forKey: .{f.swift_name})")
        lines.append("    }")
    else:
        lines.append("    enum CodingKeys: String, CodingKey {")
        for f in fields:
            lines.append(f'        case {f.swift_name} = "{f.name}"')
        lines.append("    }")

    lines.append("")
    has_head_dim = any(f.name == 'head_dim' for f in fields)
    if not has_head_dim:
        lines.append("    public var headDim: Int {")
        lines.append("        hiddenSize / numAttentionHeads")
        lines.append("    }")

    lines.append("}")
    return "\n".join(lines)


# =============================================================================
# AST PARSER
# =============================================================================

@dataclass
class ModuleAttribute:
    name: str
    swift_name: str
    module_type: Optional[str]
    init_args: List[str]
    key: str
    is_parameter: bool = False


@dataclass
class ModuleProperty:
    name: str
    swift_name: str
    swift_type: str
    init_expr: str


@dataclass
class ModuleMethod:
    name: str
    swift_name: str
    args: List[Tuple[str, str]]
    body: List[str]
    return_type: str = "MLXArray"


@dataclass
class ParsedModule:
    name: str
    swift_name: str
    attributes: List[ModuleAttribute] = field(default_factory=list)
    methods: List[ModuleMethod] = field(default_factory=list)
    properties: List[ModuleProperty] = field(default_factory=list)
    base_classes: List[str] = field(default_factory=list)


class HFModelParser(ast.NodeVisitor):
    """Parse HuggingFace model Python code"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.modules: List[ParsedModule] = []
        self._current: Optional[ParsedModule] = None
        self._all_class_names: Set[str] = set()

    def parse(self, source: str) -> List[ParsedModule]:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self._all_class_names.add(node.name)
        self.visit(tree)
        return self.modules

    def visit_ClassDef(self, node):
        base_names = [self._base_name(b) for b in node.bases]
        recognized_bases = {
            'nn.Module', 'Module', 'PreTrainedModel',
            'nn.Embedding', 'nn.Linear', 'nn.LayerNorm',
            'GradientCheckpointingLayer', 'GenerationMixin'
        }
        is_module = any(
            b in recognized_bases or 'Model' in b or 'Layer' in b or 'Block' in b or b in self._all_class_names
            for b in base_names
        )
        if not is_module:
            return

        self._current = ParsedModule(
            name=node.name,
            swift_name=node.name,
            base_classes=base_names
        )

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name == "__init__":
                    self._parse_init(item)
                elif item.name == "forward":
                    self._parse_forward(item)

        self.modules.append(self._current)
        self._current = None

    def _base_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._base_name(node.value)}.{node.attr}"
        return ""

    def _parse_init(self, node):
        existing = set()
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and \
                       isinstance(target.value, ast.Name) and \
                       target.value.id == "self":
                        name = target.attr
                        if name not in existing:
                            existing.add(name)
                            self._extract_attr(name, stmt.value)

    def _extract_attr(self, name: str, value):
        if not isinstance(value, ast.Call):
            if isinstance(value, (ast.Name, ast.Attribute, ast.BinOp, ast.Constant)):
                swift_type = self._infer_type(value)
                self._current.properties.append(ModuleProperty(
                    name=name, swift_name=to_camel(name),
                    swift_type=swift_type, init_expr=convert_expr(ast.unparse(value))
                ))
            return

        func_name = self._call_name(value)
        if func_name not in NN_MODULES and func_name not in self._all_class_names:
            swift_type = self._infer_type(value)
            self._current.properties.append(ModuleProperty(
                name=name, swift_name=to_camel(name),
                swift_type=swift_type, init_expr=convert_expr(ast.unparse(value))
            ))
            return

        module_info = NN_MODULES.get(func_name)
        if module_info:
            swift_type, _ = module_info
            if swift_type is None:
                return
        else:
            swift_type = func_name

        init_args = []
        for arg in value.args:
            init_args.append(convert_expr(ast.unparse(arg)))
        for kw in value.keywords:
            if kw.arg:
                init_args.append(f"{to_camel(kw.arg)}: {convert_expr(ast.unparse(kw.value))}")

        self._current.attributes.append(ModuleAttribute(
            name=name, swift_name=to_camel(name),
            module_type=swift_type, init_args=init_args, key=name
        ))

    def _call_name(self, node) -> str:
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            return node.func.attr
        elif isinstance(node.func, ast.Name):
            return node.func.id
        return ""

    def _parse_forward(self, node):
        args = []
        for arg in node.args.args[1:]:
            arg_type = "MLXArray"
            if arg.annotation:
                ann = ast.unparse(arg.annotation)
                if "Optional" in ann:
                    arg_type = "MLXArray?"
            args.append((arg.arg, arg_type))

        self._current.methods.append(ModuleMethod(
            name="forward", swift_name="callAsFunction",
            args=args, body=[]
        ))

    def _infer_type(self, node) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "Bool"
            elif isinstance(node.value, int):
                return "Int"
            elif isinstance(node.value, float):
                return "Float"
        elif isinstance(node, ast.BinOp):
            return "Int" if isinstance(node.op, ast.Div) else "Float"
        return "Any"


# =============================================================================
# SWIFT CODE GENERATOR v6
# =============================================================================

class SwiftGenerator:
    def __init__(self, model_name: str):
        self.model_name = to_pascal(model_name)
        self.model_name_lower = model_name.lower().replace('-', '').replace('_', '')
        self.parsed_modules: Dict[str, ParsedModule] = {}
        self.config_class = f"{self.model_name}Configuration"

    def generate(self, modules: List[ParsedModule], config_json: Optional[Dict] = None) -> str:
        for m in modules:
            self.parsed_modules[m.name] = m

        lines = [
            "//",
            f"//  {self.model_name}.swift",
            "//  Auto-generated by hf2swift v6",
            "//",
            "//  Uses parsed modules from HuggingFace Transformers source.",
            "//  Based on patterns from mlx-swift-lm (MIT License, ml-explore).",
            "//",
            "",
            "import Foundation",
            "import MLX",
            "import MLXFast",
            "import MLXNN",
            "",
        ]

        if config_json:
            lines.append("// MARK: - Configuration")
            lines.append("")
            lines.append(generate_config_from_json(config_json, self.model_name))
            lines.append("")

        lines.extend(self._gen_helpers())
        lines.append("")

        # Sort modules by dependency
        ordered = self._sort_by_dependency(modules)

        generated = set()
        for m in ordered:
            if m.name in generated:
                continue
            # Skip wrapper classes, output types, and audio modules (for text-only LM)
            skip_patterns = [
                'ForCausalLM', 'ForConditionalGeneration', 'PreTrainedModel',
                'OutputWithPast', 'GenerationMixin',
                'Audio',  # Skip audio modules for text-only
                'Multimodal',  # Skip multimodal modules
            ]
            if any(x in m.name for x in skip_patterns):
                continue
            # Skip the multimodal Gemma3nModel (keep only our LLMModel wrapper)
            if m.name == f'{self.model_name}Model':
                continue
            generated.add(m.name)
            lines.append(f"// MARK: - {m.swift_name}")
            lines.append("")
            lines.extend(self._gen_module(m))
            lines.append("")

        # Generate model wrapper
        lines.extend(self._gen_model_wrapper(modules))

        return "\n".join(lines)

    def _sort_by_dependency(self, modules: List[ParsedModule]) -> List[ParsedModule]:
        """Sort modules so dependencies come first"""
        deps = defaultdict(set)
        for m in modules:
            for attr in m.attributes:
                if attr.module_type and attr.module_type in self.parsed_modules:
                    deps[m.name].add(attr.module_type)

        ordered = []
        visited = set()

        def visit(name):
            if name in visited:
                return
            visited.add(name)
            for dep in deps.get(name, []):
                visit(dep)
            if name in self.parsed_modules:
                ordered.append(self.parsed_modules[name])

        for m in modules:
            visit(m.name)
        return ordered

    def _gen_helpers(self) -> List[str]:
        return [
            "// MARK: - Helper Functions",
            "",
            f"private func {self.model_name_lower}ApplyRotaryPosEmb(",
            "    _ q: MLXArray,",
            "    _ k: MLXArray,",
            "    cos: MLXArray,",
            "    sin: MLXArray",
            ") -> (MLXArray, MLXArray) {",
            f"    let qEmbed = (q * cos) + ({self.model_name_lower}RotateHalf(q) * sin)",
            f"    let kEmbed = (k * cos) + ({self.model_name_lower}RotateHalf(k) * sin)",
            "    return (qEmbed, kEmbed)",
            "}",
            "",
            f"private func {self.model_name_lower}RotateHalf(_ x: MLXArray) -> MLXArray {{",
            "    let halfDim = x.dim(-1) / 2",
            "    let x1 = x[.ellipsis, ..<halfDim]",
            "    let x2 = x[.ellipsis, halfDim...]",
            "    return concatenated([-x2, x1], axis: -1)",
            "}",
        ]

    def _gen_module(self, m: ParsedModule) -> List[str]:
        """Generate a Swift module class"""
        # Special handling for different module types
        if 'RMSNorm' in m.name:
            return self._gen_rms_norm(m)
        if 'Embedding' in m.name and 'nn.Embedding' in m.base_classes:
            return self._gen_scaled_embedding(m)
        if 'Attention' in m.name:
            return self._gen_attention(m)
        if 'MLP' in m.name:
            return self._gen_mlp(m)
        if 'DecoderLayer' in m.name:
            return self._gen_decoder_layer(m)
        if 'RotaryEmbedding' in m.name:
            return self._gen_rotary_embedding(m)
        return self._gen_generic_module(m)

    def _gen_rms_norm(self, m: ParsedModule) -> List[str]:
        """Generate RMSNorm class"""
        return [
            f"class {m.swift_name}: Module {{",
            "    let eps: Float",
            "    let weight: MLXArray",
            "    let withScale: Bool",
            "",
            f"    init(dimensions: Int, eps: Float = 1e-6, withScale: Bool = true) {{",
            "        self.eps = eps",
            "        self.withScale = withScale",
            "        self.weight = withScale ? MLXArray.ones([dimensions]) : MLXArray.ones([dimensions])",
            "    }",
            "",
            "    func callAsFunction(_ x: MLXArray) -> MLXArray {",
            "        let variance = x.pow(2).mean(axis: -1, keepDims: true)",
            "        let normalized = x * rsqrt(variance + eps)",
            "        return withScale ? normalized * weight : normalized",
            "    }",
            "}",
        ]

    def _gen_scaled_embedding(self, m: ParsedModule) -> List[str]:
        """Generate scaled word embedding (inherits from Embedding)"""
        return [
            f"class {m.swift_name}: Embedding {{",
            "    let embedScale: Float",
            "",
            f"    init(embeddingCount: Int, dimensions: Int, embedScale: Float = 1.0) {{",
            "        self.embedScale = embedScale",
            "        super.init(embeddingCount: embeddingCount, dimensions: dimensions)",
            "    }",
            "",
            "    override func callAsFunction(_ x: MLXArray) -> MLXArray {",
            "        return super.callAsFunction(x) * embedScale",
            "    }",
            "}",
        ]

    def _gen_attention(self, m: ParsedModule) -> List[str]:
        """Generate attention module with all parsed attributes"""
        lines = [f"class {m.swift_name}: Module {{"]

        # Module attributes
        for attr in m.attributes:
            if attr.module_type == "Array":
                lines.append(f"    var {attr.swift_name}: [Module] = []")
            else:
                lines.append(f'    @ModuleInfo(key: "{attr.key}") var {attr.swift_name}: {attr.module_type}')

        # Computed properties needed for attention
        lines.extend([
            "",
            "    let numHeads: Int",
            "    let numKVHeads: Int",
            "    let headDim: Int",
            "    let scale: Float",
            "",
            f"    init(_ config: {self.config_class}) {{",
            "        self.numHeads = config.numAttentionHeads",
            "        self.numKVHeads = config.numKeyValueHeads ?? config.numAttentionHeads",
            "        self.headDim = config.headDim ?? (config.hiddenSize / config.numAttentionHeads)",
            "        if let scalar = config.queryPreAttnScalar {",
            "            self.scale = 1.0 / sqrt(Float(scalar))",
            "        } else {",
            "            self.scale = 1.0 / sqrt(Float(headDim))",
            "        }",
            "",
            "        let hiddenSize = config.hiddenSize",
            "        let kvDim = numKVHeads * headDim",
            "        let qDim = numHeads * headDim",
        ])

        # Initialize projections
        if any(a.swift_name == 'qProj' for a in m.attributes):
            lines.append("        self._qProj.wrappedValue = Linear(hiddenSize, qDim, bias: false)")
        if any(a.swift_name == 'kProj' for a in m.attributes):
            lines.append("        self._kProj.wrappedValue = Linear(hiddenSize, kvDim, bias: false)")
        if any(a.swift_name == 'vProj' for a in m.attributes):
            lines.append("        self._vProj.wrappedValue = Linear(hiddenSize, kvDim, bias: false)")
        if any(a.swift_name == 'oProj' for a in m.attributes):
            lines.append("        self._oProj.wrappedValue = Linear(qDim, hiddenSize, bias: false)")

        # Initialize norms if present - use the correct RMSNorm type
        rms_norm_type = f"{self.model_name}RMSNorm"
        if any(a.swift_name == 'qNorm' for a in m.attributes):
            lines.append(f"        self._qNorm.wrappedValue = {rms_norm_type}(dimensions: headDim, eps: config.rmsNormEps ?? 1e-6)")
        if any(a.swift_name == 'kNorm' for a in m.attributes):
            lines.append(f"        self._kNorm.wrappedValue = {rms_norm_type}(dimensions: headDim, eps: config.rmsNormEps ?? 1e-6)")
        if any(a.swift_name == 'vNorm' for a in m.attributes):
            lines.append(f"        self._vNorm.wrappedValue = {rms_norm_type}(dimensions: headDim, eps: config.rmsNormEps ?? 1e-6, withScale: false)")

        lines.extend([
            "    }",
            "",
            "    func callAsFunction(",
            "        _ x: MLXArray,",
            "        mask: MLXArray? = nil,",
            "        cache: KVCache? = nil",
            "    ) -> MLXArray {",
            "        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))",
            "",
            "        var queries = qProj(x).reshaped([B, L, numHeads, headDim])",
            "        var keys = kProj(x).reshaped([B, L, numKVHeads, headDim])",
            "        var values = vProj(x).reshaped([B, L, numKVHeads, headDim])",
        ])

        # Apply norms if present
        if any(a.swift_name == 'qNorm' for a in m.attributes):
            lines.append("        queries = qNorm(queries)")
        if any(a.swift_name == 'kNorm' for a in m.attributes):
            lines.append("        keys = kNorm(keys)")
        if any(a.swift_name == 'vNorm' for a in m.attributes):
            lines.append("        values = vNorm(values)")

        lines.extend([
            "",
            "        queries = queries.transposed(0, 2, 1, 3)",
            "        keys = keys.transposed(0, 2, 1, 3)",
            "        values = values.transposed(0, 2, 1, 3)",
            "",
            "        // KV cache update",
            "        if var cache = cache {",
            "            (keys, values) = cache.update(keys: keys, values: values)",
            "        }",
            "",
            "        // Expand KV heads if needed",
            "        if numKVHeads < numHeads {",
            "            let repeats = numHeads / numKVHeads",
            "            keys = MLXArray.repeated(keys, count: repeats, axis: 1)",
            "            values = MLXArray.repeated(values, count: repeats, axis: 1)",
            "        }",
            "",
            "        var scores = matmul(queries, keys.transposed(0, 1, 3, 2)) * scale",
            "        if let mask = mask {",
            "            scores = scores + mask",
            "        }",
            "        let weights = softmax(scores, axis: -1)",
            "        let output = matmul(weights, values)",
            "            .transposed(0, 2, 1, 3)",
            "            .reshaped([B, L, -1])",
            "",
            "        return oProj(output)",
            "    }",
            "}",
        ])
        return lines

    def _gen_mlp(self, m: ParsedModule) -> List[str]:
        """Generate MLP module"""
        lines = [f"class {m.swift_name}: Module {{"]

        for attr in m.attributes:
            lines.append(f'    @ModuleInfo(key: "{attr.key}") var {attr.swift_name}: {attr.module_type}')

        lines.extend([
            "",
            f"    init(_ config: {self.config_class}) {{",
            "        let hiddenSize = config.hiddenSize",
            "        let intermediateSize = config.intermediateSize",
        ])

        if any(a.swift_name == 'gateProj' for a in m.attributes):
            lines.append("        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)")
        if any(a.swift_name == 'upProj' for a in m.attributes):
            lines.append("        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)")
        if any(a.swift_name == 'downProj' for a in m.attributes):
            lines.append("        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)")

        lines.extend([
            "    }",
            "",
            "    func callAsFunction(_ x: MLXArray) -> MLXArray {",
        ])

        if any(a.swift_name == 'gateProj' for a in m.attributes):
            lines.append("        return downProj(gelu(gateProj(x)) * upProj(x))")
        else:
            lines.append("        return x")

        lines.extend([
            "    }",
            "}",
        ])
        return lines

    def _gen_decoder_layer(self, m: ParsedModule) -> List[str]:
        """Generate decoder layer with attention + MLP"""
        lines = [f"class {m.swift_name}: Module {{"]

        for attr in m.attributes:
            if attr.module_type == "Array":
                continue
            lines.append(f'    @ModuleInfo(key: "{attr.key}") var {attr.swift_name}: {attr.module_type}')

        lines.extend([
            "",
            f"    init(_ config: {self.config_class}, layerIdx: Int = 0) {{",
        ])

        # Initialize attention
        attention_name = [a for a in m.attributes if 'Attention' in (a.module_type or '')]
        if attention_name:
            attn = attention_name[0]
            lines.append(f"        self._{attn.swift_name}.wrappedValue = {attn.module_type}(config)")

        # Initialize MLP
        mlp_name = [a for a in m.attributes if 'MLP' in (a.module_type or '')]
        if mlp_name:
            mlp = mlp_name[0]
            lines.append(f"        self._{mlp.swift_name}.wrappedValue = {mlp.module_type}(config)")

        # Initialize norms - use model-specific RMSNorm
        rms_norm_type = f"{self.model_name}RMSNorm"
        for attr in m.attributes:
            if 'Norm' in (attr.module_type or '') and attr.module_type:
                lines.append(f"        self._{attr.swift_name}.wrappedValue = {rms_norm_type}(dimensions: config.hiddenSize, eps: config.rmsNormEps ?? 1e-6)")

        # Initialize other modules
        for attr in m.attributes:
            if 'LaurelBlock' in (attr.module_type or '') or 'AltUp' in (attr.module_type or ''):
                lines.append(f"        self._{attr.swift_name}.wrappedValue = {attr.module_type}(config)")

        lines.extend([
            "    }",
            "",
            "    func callAsFunction(",
            "        _ x: MLXArray,",
            "        mask: MLXArray? = nil,",
            "        cache: KVCache? = nil",
            "    ) -> MLXArray {",
            "        var h = x",
        ])

        # Pre-attention norm
        pre_attn_norm = [a for a in m.attributes if a.swift_name in ('inputLayernorm', 'preAttnNorm', 'preFfnNorm')]
        if pre_attn_norm:
            lines.append(f"        let normed = {pre_attn_norm[0].swift_name}(h)")
        else:
            lines.append("        let normed = h")

        # Attention
        if attention_name:
            lines.append(f"        let attnOut = {attention_name[0].swift_name}(normed, mask: mask, cache: cache)")
            lines.append("        h = h + attnOut")

        # Post-attention norm
        post_attn_norm = [a for a in m.attributes if a.swift_name in ('postAttentionLayernorm', 'postAttnNorm')]
        if post_attn_norm:
            lines.append(f"        let postNormed = {post_attn_norm[0].swift_name}(h)")
        else:
            lines.append("        let postNormed = h")

        # MLP
        if mlp_name:
            lines.append(f"        let mlpOut = {mlp_name[0].swift_name}(postNormed)")
            lines.append("        h = h + mlpOut")

        lines.extend([
            "",
            "        return h",
            "    }",
            "}",
        ])
        return lines

    def _gen_rotary_embedding(self, m: ParsedModule) -> List[str]:
        """Generate rotary position embedding"""
        return [
            f"class {m.swift_name}: Module {{",
            "    let dim: Int",
            "    let maxPositionEmbeddings: Int",
            "    let base: Float",
            "",
            f"    init(_ config: {self.config_class}) {{",
            "        self.dim = config.headDim ?? (config.hiddenSize / config.numAttentionHeads)",
            "        self.maxPositionEmbeddings = config.maxPositionEmbeddings ?? 8192",
            "        self.base = config.ropeTheta ?? 10000.0",
            "    }",
            "",
            "    func callAsFunction(_ x: MLXArray, offset: Int = 0) -> (MLXArray, MLXArray) {",
            "        let seqLen = x.dim(1)",
            "        let freqs = MLXArray(stride(from: 0, to: Float(dim), by: 2).map { pow(base, -$0 / Float(dim)) })",
            "        let positions = MLXArray((offset..<(offset + seqLen)).map { Float($0) })",
            "        let angles = positions.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)",
            "        return (cos(angles), sin(angles))",
            "    }",
            "}",
        ]

    def _gen_generic_module(self, m: ParsedModule) -> List[str]:
        """Generate a generic module"""
        lines = [f"class {m.swift_name}: Module {{"]

        for attr in m.attributes:
            if attr.module_type == "Array":
                lines.append(f"    var {attr.swift_name}: [Module] = []")
            else:
                lines.append(f'    @ModuleInfo(key: "{attr.key}") var {attr.swift_name}: {attr.module_type}')

        # Skip properties that would cause init issues - we don't need them in Swift
        # since we use config directly
        skip_props = {'config', 'layerIdx', 'paddingIdx', 'vocabSize', 'gradientCheckpointing',
                      'hiddenSize', 'hiddenSizePerLayerInput', 'training'}

        lines.extend([
            "",
            f"    init(_ config: {self.config_class}) {{",
        ])

        for attr in m.attributes:
            if attr.module_type == "Linear":
                lines.append(f"        self._{attr.swift_name}.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)")
            elif attr.module_type == "Embedding":
                lines.append(f"        self._{attr.swift_name}.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)")
            elif 'RMSNorm' in (attr.module_type or ''):
                lines.append(f"        self._{attr.swift_name}.wrappedValue = {self.model_name}RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps ?? 1e-6)")
            elif 'ScaledWordEmbedding' in (attr.module_type or ''):
                # Special init for scaled embeddings
                lines.append(f"        self._{attr.swift_name}.wrappedValue = {attr.module_type}(")
                lines.append(f"            embeddingCount: config.vocabSize,")
                lines.append(f"            dimensions: config.hiddenSize,")
                lines.append(f"            embedScale: sqrt(Float(config.hiddenSize))")
                lines.append(f"        )")
            elif 'RotaryEmbedding' in (attr.module_type or ''):
                # RotaryEmbedding takes config
                lines.append(f"        self._{attr.swift_name}.wrappedValue = {attr.module_type}(config)")
            elif attr.module_type in self.parsed_modules:
                lines.append(f"        self._{attr.swift_name}.wrappedValue = {attr.module_type}(config)")

        lines.extend([
            "    }",
            "",
            "    func callAsFunction(_ x: MLXArray) -> MLXArray {",
            "        return x",
            "    }",
            "}",
        ])
        return lines

    def _gen_model_wrapper(self, modules: List[ParsedModule]) -> List[str]:
        """Generate the top-level model wrapper"""
        # Find the main text model
        text_model = None
        for m in modules:
            if 'TextModel' in m.name and 'ForCausalLM' not in m.name:
                text_model = m
                break

        # Find decoder layer name
        decoder_layer = None
        for m in modules:
            if 'DecoderLayer' in m.name:
                decoder_layer = m.name
                break

        return [
            f"// MARK: - {self.model_name}ModelInner",
            "",
            f"class {self.model_name}ModelInner: Module {{",
            f'    @ModuleInfo(key: "embed_tokens") var embedTokens: {self.model_name}TextScaledWordEmbedding',
            f'    @ModuleInfo(key: "layers") var layers: [{decoder_layer or "Module"}]',
            f'    @ModuleInfo(key: "norm") var norm: {self.model_name}RMSNorm',
            "",
            f"    init(_ config: {self.config_class}) {{",
            f"        self._embedTokens.wrappedValue = {self.model_name}TextScaledWordEmbedding(",
            "            embeddingCount: config.vocabSize,",
            "            dimensions: config.hiddenSize,",
            "            embedScale: config.hiddenSize > 0 ? sqrt(Float(config.hiddenSize)) : 1.0",
            "        )",
            f"        self._layers.wrappedValue = (0..<config.numHiddenLayers).map {{ {decoder_layer or 'Module'}(config, layerIdx: $0) }}",
            f"        self._norm.wrappedValue = {self.model_name}RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps ?? 1e-6)",
            "    }",
            "",
            "    func callAsFunction(_ inputIds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {",
            "        var h = embedTokens(inputIds)",
            "        for (i, layer) in layers.enumerated() {",
            "            let layerCache = cache?[i]",
            "            h = layer(h, mask: nil, cache: layerCache)",
            "        }",
            "        return norm(h)",
            "    }",
            "}",
            "",
            f"// MARK: - {self.model_name}Model",
            "",
            f"public class {self.model_name}Model: Module, LLMModel {{",
            "    public let vocabularySize: Int",
            "    public let numLayers: Int",
            "",
            f'    @ModuleInfo(key: "model") var model: {self.model_name}ModelInner',
            f'    @ModuleInfo(key: "lm_head") var lmHead: Linear',
            f"    private let config: {self.config_class}",
            "",
            f"    public init(_ config: {self.config_class}) {{",
            "        self.config = config",
            "        self.vocabularySize = config.vocabSize",
            "        self.numLayers = config.numHiddenLayers",
            f"        self._model.wrappedValue = {self.model_name}ModelInner(config)",
            "        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)",
            "    }",
            "",
            "    // LLMModel protocol requirement",
            "    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {",
            "        let h = model(inputIds, cache: nil)",
            "        return lmHead(h)",
            "    }",
            "",
            "    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {",
            "        var result: [String: MLXArray] = [:]",
            "        for (key, value) in weights {",
            "            var newKey = key",
            '            if newKey.hasPrefix("model.language_model.") {',
            '                newKey = String(newKey.dropFirst("model.language_model.".count))',
            '            } else if newKey.hasPrefix("language_model.") {',
            '                newKey = String(newKey.dropFirst("language_model.".count))',
            "            }",
            "            result[newKey] = value",
            "        }",
            "        return result",
            "    }",
            "}",
        ]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='HuggingFace to MLX Swift Generator v6')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--config', help='HuggingFace model ID for config.json')
    parser.add_argument('--output', help='Output Swift file path')
    parser.add_argument('--source', help='Python source file path')
    args = parser.parse_args()

    source = ""
    if args.source:
        with open(args.source) as f:
            source = f.read()

    parser_obj = HFModelParser(args.model)
    modules = parser_obj.parse(source) if source else []

    print(f"Parsed {len(modules)} modules:")
    for m in modules:
        print(f"  - {m.name}: {len(m.attributes)} attrs, bases={m.base_classes}")

    config_json = None
    if args.config:
        import urllib.request
        url = f"https://huggingface.co/{args.config}/raw/main/config.json"
        try:
            with urllib.request.urlopen(url) as response:
                config_json = json.loads(response.read().decode())
        except Exception as e:
            print(f"Warning: Could not load config: {e}")

    generator = SwiftGenerator(args.model)
    code = generator.generate(modules, config_json)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(code)
        print(f"Generated: {args.output}")
    else:
        print(code)


if __name__ == "__main__":
    main()
