#!/usr/bin/env python3
"""
HuggingFace Transformers → MLX Swift Generator v4

Goal: Generate ~90%+ usable Swift code to replace mlx-swift-lm dependency.

Features:
- Complete Configuration struct from config.json schema
- All nn.Module detection including norms and embeddings
- RoPE and helper function generation
- Full model wrapper (XXXModel, XXXForCausalLM)
- Weight key mapping for sanitization
"""

import ast
import re
import argparse
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Set
import urllib.request


# =============================================================================
# TYPE MAPPINGS
# =============================================================================

# Python type → Swift type
SWIFT_TYPES = {
    "int": "Int",
    "float": "Float", 
    "bool": "Bool",
    "str": "String",
    "None": "nil",
    "NoneType": "nil",
    "list": "[Any]",
    "dict": "[String: Any]",
}

# nn.Module → Swift Module class
NN_MODULES = {
    "nn.Linear": ("Linear", ["in_features", "out_features", "bias"]),
    "nn.Embedding": ("Embedding", ["num_embeddings", "embedding_dim"]),
    "nn.LayerNorm": ("LayerNorm", ["normalized_shape", "eps"]),
    "RMSNorm": ("RMSNorm", ["dimensions", "eps"]),
    "nn.RMSNorm": ("RMSNorm", ["dimensions", "eps"]),
    "nn.Dropout": (None, []),  # Skip
    "nn.ModuleList": ("Array", []),
    "nn.Conv1d": ("Conv1d", ["in_channels", "out_channels", "kernel_size"]),
    "nn.Conv2d": ("Conv2d", ["in_channels", "out_channels", "kernel_size"]),
}

# Expression conversions (Python → Swift)
EXPR_CONVERSIONS = [
    # Tensor operations
    (r'\.transpose\((\d+),\s*(\d+)\)', r'.transposed(\1, \2)'),
    (r'\.reshape\(([^)]+)\)', r'.reshaped([\1])'),
    (r'\.view\(([^)]+)\)', r'.reshaped([\1])'),
    (r'\.unsqueeze\((\d+)\)', r'.expandedDimensions(axis: \1)'),
    (r'\.squeeze\((\d+)\)', r'.squeezed(axis: \1)'),
    (r'\.contiguous\(\)', ''),
    (r'\.float\(\)', '.asType(.float32)'),
    (r'\.half\(\)', '.asType(.float16)'),
    (r'\.bfloat16\(\)', '.asType(.bfloat16)'),
    (r'\.to\([^)]+\)', ''),
    (r'\.type_as\((\w+)\)', r'.asType(\1.dtype)'),
    (r'\.size\((\d+)\)', r'.dim(\1)'),
    (r'\.shape\[(\d+)\]', r'.dim(\1)'),
    (r'\.shape\[:-1\]', r'.shape.dropLast()'),
    (r'\.split\(([^,]+),\s*dim=(-?\d+)\)', r'.split(parts: \1, axis: \2)'),
    (r'\.chunk\((\d+),\s*dim=(-?\d+)\)', r'.split(parts: \1, axis: \2)'),
    (r'\.pow\((\d+)\)', r'.power(\1)'),
    (r'\.mean\((-?\d+)[^)]*\)', r'.mean(axis: \1)'),
    (r'\.sum\((-?\d+)[^)]*\)', r'.sum(axis: \1)'),
    
    # Torch functions
    (r'torch\.matmul\(([^,]+),\s*([^)]+)\)', r'matmul(\1, \2)'),
    (r'torch\.cat\(\[([^\]]+)\],\s*dim=(-?\d+)\)', r'concatenated([\1], axis: \2)'),
    (r'torch\.cat\(\(([^)]+)\),\s*dim=(-?\d+)\)', r'concatenated([\1], axis: \2)'),
    (r'torch\.stack\(\[([^\]]+)\],\s*dim=(-?\d+)\)', r'stacked([\1], axis: \2)'),
    (r'torch\.sqrt\(([^)]+)\)', r'sqrt(\1)'),
    (r'torch\.rsqrt\(([^)]+)\)', r'rsqrt(\1)'),
    (r'torch\.tanh\(([^)]+)\)', r'tanh(\1)'),
    (r'torch\.exp\(([^)]+)\)', r'exp(\1)'),
    (r'torch\.sin\(([^)]+)\)', r'sin(\1)'),
    (r'torch\.cos\(([^)]+)\)', r'cos(\1)'),
    (r'torch\.where\(([^,]+),\s*([^,]+),\s*([^)]+)\)', r'where(\1, \2, \3)'),
    (r'torch\.ones\(([^)]+)\)', r'MLXArray.ones([\1])'),
    (r'torch\.zeros\(([^)]+)\)', r'MLXArray.zeros([\1])'),
    (r'torch\.arange\(([^,)]+)\)', r'MLXArray(0..<Int(\1))'),
    (r'torch\.arange\(([^,]+),\s*([^,)]+)\)', r'MLXArray(Int(\1)..<Int(\2))'),
    (r'torch\.triu\(([^,]+),\s*diagonal=(\d+)\)', r'triu(\1, k: \2)'),
    (r'torch\.tril\(([^,]+)\)', r'tril(\1)'),
    (r'torch\.bmm\(([^,]+),\s*([^)]+)\)', r'matmul(\1, \2)'),
    (r'torch\.einsum\([^)]+\)', r'/* einsum - manual conversion needed */'),
    
    # F functions  
    (r'F\.gelu\(([^)]+)\)', r'gelu(\1)'),
    (r'F\.relu\(([^)]+)\)', r'relu(\1)'),
    (r'F\.silu\(([^)]+)\)', r'silu(\1)'),
    (r'F\.softmax\(([^,]+),\s*dim=(-?\d+)\)', r'softmax(\1, axis: \2)'),
    (r'F\.scaled_dot_product_attention\(([^)]+)\)', r'scaledDotProductAttention(\1)'),
    (r'F\.linear\(([^,]+),\s*([^,)]+)[^)]*\)', r'matmul(\1, \2.T)'),
    
    # Operators
    (r'(\w+)\s*@\s*(\w+)', r'matmul(\1, \2)'),
    
    # Python → Swift constants
    (r'\bNone\b', 'nil'),
    (r'\bTrue\b', 'true'),
    (r'\bFalse\b', 'false'),
    (r'\bself\.', ''),
]


def to_camel(name: str) -> str:
    """snake_case → camelCase"""
    parts = name.split('_')
    return parts[0].lower() + ''.join(p.title() for p in parts[1:])


def to_pascal(name: str) -> str:
    """snake_case → PascalCase"""
    return ''.join(p.title() for p in name.split('_'))


def convert_expr(py_expr: str) -> str:
    """Convert Python expression to Swift"""
    result = py_expr
    for pattern, replacement in EXPR_CONVERSIONS:
        result = re.sub(pattern, replacement, result)
    # Convert remaining snake_case identifiers
    result = re.sub(r'\b([a-z]+)_([a-z_]+)\b', lambda m: to_camel(m.group(0)), result)
    return result


# =============================================================================
# CONFIG GENERATOR
# =============================================================================

@dataclass
class ConfigField:
    name: str
    swift_name: str
    swift_type: str
    default: Optional[str] = None
    optional: bool = False
    coding_key: Optional[str] = None


def infer_swift_type(value: Any) -> str:
    """Infer Swift type from Python value"""
    if value is None:
        return "Any?"
    elif isinstance(value, bool):
        return "Bool"
    elif isinstance(value, int):
        return "Int"
    elif isinstance(value, float):
        return "Float"
    elif isinstance(value, str):
        return "String"
    elif isinstance(value, list):
        if not value:
            return "[Any]"
        inner = infer_swift_type(value[0])
        return f"[{inner}]"
    elif isinstance(value, dict):
        return "[String: AnyCodable]"
    return "Any"


def generate_config_from_json(config_json: Dict[str, Any], model_name: str) -> str:
    """Generate Swift Configuration struct from config.json"""
    
    # Common fields to include
    important_fields = {
        'hidden_size', 'num_hidden_layers', 'num_attention_heads', 
        'num_key_value_heads', 'intermediate_size', 'vocab_size',
        'max_position_embeddings', 'rms_norm_eps', 'rope_theta',
        'head_dim', 'hidden_act', 'tie_word_embeddings', 'attention_bias',
        'sliding_window', 'rope_scaling', 'bos_token_id', 'eos_token_id',
        'model_type'
    }
    
    fields: List[ConfigField] = []
    
    for key, value in config_json.items():
        if key.startswith('_') or key in ('architectures', 'transformers_version', 'torch_dtype'):
            continue
            
        swift_name = to_camel(key)
        swift_type = infer_swift_type(value)
        
        # Make optional types explicit
        optional = value is None or key not in important_fields
        if optional and not swift_type.endswith('?'):
            swift_type += '?'
        
        default = None
        if value is None:
            default = 'nil'
        elif isinstance(value, bool):
            default = 'true' if value else 'false'
        elif isinstance(value, (int, float)):
            default = str(value)
        elif isinstance(value, str):
            default = f'"{value}"'
        
        fields.append(ConfigField(
            name=key,
            swift_name=swift_name,
            swift_type=swift_type,
            default=default,
            optional=optional,
            coding_key=key
        ))
    
    # Generate Swift code
    class_name = f"{to_pascal(model_name)}Configuration"
    lines = [
        f"public struct {class_name}: Codable, Sendable {{",
    ]
    
    # Properties
    for f in fields:
        if f.default and not f.optional:
            lines.append(f"    public var {f.swift_name}: {f.swift_type}")
        else:
            lines.append(f"    public let {f.swift_name}: {f.swift_type}")
    
    lines.append("")
    
    # CodingKeys
    lines.append("    enum CodingKeys: String, CodingKey {")
    for f in fields:
        lines.append(f'        case {f.swift_name} = "{f.name}"')
    lines.append("    }")
    
    lines.append("")
    
    # Computed properties for common patterns
    lines.append("    // Computed properties")
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
    module_type: Optional[str]  # "Linear", "RMSNorm", etc.
    init_args: List[str]  # Arguments for initialization
    key: str  # For @ModuleInfo(key: "...")
    is_parameter: bool = False  # For @ParameterInfo


@dataclass
class ModuleMethod:
    name: str
    swift_name: str
    args: List[Tuple[str, str]]  # [(name, type), ...]
    body: List[str]
    return_type: str = "MLXArray"


@dataclass
class ParsedModule:
    name: str
    swift_name: str
    attributes: List[ModuleAttribute] = field(default_factory=list)
    methods: List[ModuleMethod] = field(default_factory=list)
    properties: List[Tuple[str, str, str]] = field(default_factory=list)  # (name, type, init)
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
        
        # First pass: collect all class names
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self._all_class_names.add(node.name)
        
        # Second pass: parse modules
        self.visit(tree)
        return self.modules
    
    def visit_ClassDef(self, node):
        bases = [self._base_name(b) for b in node.bases]
        
        # Only parse nn.Module subclasses
        if not any(b in bases for b in ["nn.Module", "PreTrainedModel", "Module"]):
            self.generic_visit(node)
            return
        
        self._current = ParsedModule(
            name=node.name,
            swift_name=node.name,
            base_classes=bases
        )
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name == "__init__":
                    self._parse_init(item)
                elif item.name == "forward":
                    self._parse_forward(item)
                elif not item.name.startswith("_"):
                    self._parse_method(item)
        
        if self._current.attributes or self._current.methods:
            self.modules.append(self._current)
        self._current = None
        
        self.generic_visit(node)
    
    def _base_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._base_name(node.value)}.{node.attr}"
        return ""
    
    def _parse_init(self, node):
        """Parse __init__ to extract nn.Module attributes"""
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and \
                       isinstance(target.value, ast.Name) and \
                       target.value.id == "self":
                        self._extract_attr(target.attr, stmt.value)
    
    def _extract_attr(self, name: str, value):
        """Extract attribute from assignment"""
        if not isinstance(value, ast.Call):
            # Check if it's a simple property assignment
            if isinstance(value, (ast.Name, ast.Attribute, ast.BinOp, ast.Constant)):
                swift_type = self._infer_type(value)
                init_expr = convert_expr(ast.unparse(value))
                self._current.properties.append((to_camel(name), swift_type, init_expr))
            return
        
        func_name = self._call_name(value)
        
        # Skip non-module calls
        if func_name not in NN_MODULES and func_name not in self._all_class_names:
            # Might be a property
            swift_type = self._infer_type(value)
            init_expr = convert_expr(ast.unparse(value))
            self._current.properties.append((to_camel(name), swift_type, init_expr))
            return
        
        # Check if it's an nn.Module
        module_info = NN_MODULES.get(func_name)
        if module_info:
            swift_type, _ = module_info
            if swift_type is None:
                return  # Skip (e.g., Dropout)
        else:
            # Custom module class
            swift_type = func_name
        
        # Extract initialization arguments
        init_args = []
        for arg in value.args:
            init_args.append(convert_expr(ast.unparse(arg)))
        for kw in value.keywords:
            if kw.arg:
                swift_key = to_camel(kw.arg)
                swift_val = convert_expr(ast.unparse(kw.value))
                init_args.append(f"{swift_key}: {swift_val}")
        
        self._current.attributes.append(ModuleAttribute(
            name=name,
            swift_name=to_camel(name),
            module_type=swift_type,
            init_args=init_args,
            key=name
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
        """Parse forward method"""
        args = []
        for arg in node.args.args[1:]:  # Skip self
            arg_type = "MLXArray"
            if arg.annotation:
                ann = ast.unparse(arg.annotation)
                if "Optional" in ann:
                    arg_type = "MLXArray?"
                elif "Tuple" in ann:
                    arg_type = "(MLXArray, MLXArray)"
            args.append((arg.arg, arg_type))
        
        body_lines = []
        for stmt in node.body:
            lines = self._convert_stmt(stmt)
            body_lines.extend(lines)
        
        self._current.methods.append(ModuleMethod(
            name="forward",
            swift_name="callAsFunction",
            args=args,
            body=body_lines
        ))
    
    def _parse_method(self, node):
        """Parse helper method"""
        args = []
        for arg in node.args.args[1:]:
            args.append((arg.arg, "MLXArray"))
        
        body_lines = []
        for stmt in node.body:
            lines = self._convert_stmt(stmt)
            body_lines.extend(lines)
        
        self._current.methods.append(ModuleMethod(
            name=node.name,
            swift_name=to_camel(node.name),
            args=args,
            body=body_lines
        ))
    
    def _convert_stmt(self, stmt) -> List[str]:
        """Convert Python statement to Swift"""
        lines = []
        
        if isinstance(stmt, ast.Return):
            if stmt.value:
                val = convert_expr(ast.unparse(stmt.value))
                lines.append(f"return {val}")
            else:
                lines.append("return")
        
        elif isinstance(stmt, ast.Assign):
            targets = [ast.unparse(t).replace("self.", "") for t in stmt.targets]
            value = convert_expr(ast.unparse(stmt.value))
            for target in targets:
                swift_target = to_camel(target)
                # Detect tuple unpacking
                if "," in target:
                    parts = [to_camel(p.strip()) for p in target.split(",")]
                    lines.append(f"let ({', '.join(parts)}) = {value}")
                else:
                    lines.append(f"let {swift_target} = {value}")
        
        elif isinstance(stmt, ast.AugAssign):
            target = to_camel(ast.unparse(stmt.target).replace("self.", ""))
            value = convert_expr(ast.unparse(stmt.value))
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(type(stmt.op), "?")
            lines.append(f"{target} = {target} {op} {value}")
        
        elif isinstance(stmt, ast.If):
            test = convert_expr(ast.unparse(stmt.test))
            lines.append(f"if {test} {{")
            for s in stmt.body:
                for l in self._convert_stmt(s):
                    lines.append(f"    {l}")
            if stmt.orelse:
                lines.append("} else {")
                for s in stmt.orelse:
                    for l in self._convert_stmt(s):
                        lines.append(f"    {l}")
            lines.append("}")
        
        elif isinstance(stmt, ast.For):
            target = to_camel(ast.unparse(stmt.target))
            iter_expr = convert_expr(ast.unparse(stmt.iter))
            lines.append(f"for {target} in {iter_expr} {{")
            for s in stmt.body:
                for l in self._convert_stmt(s):
                    lines.append(f"    {l}")
            lines.append("}")
        
        elif isinstance(stmt, ast.With):
            lines.append("// with block - may need manual conversion")
            for s in stmt.body:
                for l in self._convert_stmt(s):
                    lines.append(l)
        
        elif isinstance(stmt, ast.Expr):
            if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                pass  # Skip docstrings
            else:
                expr = convert_expr(ast.unparse(stmt.value))
                lines.append(f"_ = {expr}")
        
        elif isinstance(stmt, ast.Pass):
            pass
        
        else:
            # Fallback: comment out
            code = ast.unparse(stmt)[:80]
            lines.append(f"// TODO: {code}")
        
        return lines
    
    def _infer_type(self, node) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return "Int"
            elif isinstance(node.value, float):
                return "Float"
            elif isinstance(node.value, bool):
                return "Bool"
        elif isinstance(node, ast.Name):
            return "Any"
        elif isinstance(node, ast.BinOp):
            return "Int"  # Usually arithmetic
        return "Any"


# =============================================================================
# SWIFT CODE GENERATOR
# =============================================================================

class SwiftGenerator:
    def __init__(self, model_name: str):
        self.model_name = to_pascal(model_name)
        self.indent = "    "
    
    def generate(self, modules: List[ParsedModule], config_json: Optional[Dict] = None) -> str:
        lines = [
            "//",
            f"//  {self.model_name}.swift",
            "//  Auto-generated by hf2swift v4",
            "//",
            "//  Review and adjust before use!",
            "//",
            "",
            "import Foundation",
            "import MLX",
            "import MLXFast",
            "import MLXNN",
            "",
        ]
        
        # Generate Configuration
        if config_json:
            lines.append("// MARK: - Configuration")
            lines.append("")
            lines.append(generate_config_from_json(config_json, self.model_name))
            lines.append("")
        
        # Generate helper functions
        lines.extend(self._gen_helpers())
        lines.append("")
        
        # Generate modules
        lines.append("// MARK: - Model Components")
        lines.append("")
        
        for module in modules:
            lines.extend(self._gen_module(module))
            lines.append("")
        
        # Generate TransformerBlock, ModelInner, and top-level Model
        lines.extend(self._gen_transformer_block())
        lines.append("")
        lines.extend(self._gen_model_inner())
        lines.append("")
        lines.extend(self._gen_top_level_model(config_json))
        
        return "\n".join(lines)
    
    def _gen_transformer_block(self) -> List[str]:
        """Generate TransformerBlock combining Attention + MLP + Norms"""
        return [
            f"// MARK: - {self.model_name}TransformerBlock",
            "",
            f"class {self.model_name}TransformerBlock: Module {{",
            f'    @ModuleInfo(key: "self_attn") var attention: {self.model_name}Attention',
            f"    let mlp: {self.model_name}MLP",
            f'    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm',
            f'    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm',
            "",
            f"    init(_ config: {self.model_name}Configuration) {{",
            f"        self._attention.wrappedValue = {self.model_name}Attention(config)",
            f"        self.mlp = {self.model_name}MLP(config)",
            f"        self._inputLayerNorm.wrappedValue = RMSNorm(",
            f"            dimensions: config.hiddenSize, eps: config.rmsNormEps)",
            f"        self._postAttentionLayerNorm.wrappedValue = RMSNorm(",
            f"            dimensions: config.hiddenSize, eps: config.rmsNormEps)",
            "    }",
            "",
            "    func callAsFunction(",
            "        _ x: MLXArray,",
            "        mask: MLXArray? = nil,",
            "        cache: KVCache? = nil",
            "    ) -> MLXArray {",
            "        // Self-attention with residual",
            "        let residual = x",
            "        let h = inputLayerNorm(x)",
            "        let attnOut = attention(h, mask: mask, cache: cache)",
            "        let h = residual + attnOut",
            "",
            "        // MLP with residual",
            "        let residual = h",
            "        let h = postAttentionLayerNorm(h)",
            "        let mlpOut = mlp(h)",
            "        return residual + mlpOut",
            "    }",
            "}",
        ]
    
    def _gen_model_inner(self) -> List[str]:
        """Generate ModelInner with embeddings, layers, and final norm"""
        return [
            f"// MARK: - {self.model_name}ModelInner",
            "",
            f"public class {self.model_name}ModelInner: Module {{",
            f'    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding',
            f"    let layers: [{self.model_name}TransformerBlock]",
            f'    @ModuleInfo(key: "norm") var norm: RMSNorm',
            "",
            f"    public init(_ config: {self.model_name}Configuration) {{",
            f"        self._embedTokens.wrappedValue = Embedding(",
            f"            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)",
            "",
            f"        self.layers = (0..<config.numHiddenLayers).map {{ _ in",
            f"            {self.model_name}TransformerBlock(config)",
            "        }",
            "",
            f"        self._norm.wrappedValue = RMSNorm(",
            f"            dimensions: config.hiddenSize, eps: config.rmsNormEps)",
            "    }",
            "",
            "    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {",
            "        var h = embedTokens(inputs)",
            "",
            "        for (i, layer) in layers.enumerated() {",
            "            h = layer(h, mask: nil, cache: cache?[i])",
            "        }",
            "",
            "        return norm(h)",
            "    }",
            "}",
        ]
    
    def _gen_top_level_model(self, config_json: Optional[Dict]) -> List[str]:
        """Generate top-level Model with LLMModel protocol"""
        tie_embeddings = config_json.get("tie_word_embeddings", False) if config_json else False
        
        return [
            f"// MARK: - {self.model_name}Model",
            "",
            f"public class {self.model_name}Model: Module, LLMModel {{",
            f"    public let vocabularySize: Int",
            f"    public let kvHeads: [Int]",
            "",
            f"    public let model: {self.model_name}ModelInner",
            f"    private let config: {self.model_name}Configuration",
            "",
            f'    @ModuleInfo(key: "lm_head") var lmHead: Linear?',
            "",
            f"    public init(_ config: {self.model_name}Configuration) {{",
            f"        self.vocabularySize = config.vocabSize",
            f"        self.kvHeads = (0..<config.numHiddenLayers).map {{ _ in config.numKeyValueHeads }}",
            f"        self.model = {self.model_name}ModelInner(config)",
            f"        self.config = config",
            "",
            f"        if !config.tieWordEmbeddings {{",
            f"            self._lmHead.wrappedValue = Linear(",
            f"                config.hiddenSize, config.vocabSize, bias: false)",
            "        }",
            "    }",
            "",
            "    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {",
            "        let out = model(inputs, cache: cache)",
            "        if config.tieWordEmbeddings {",
            "            return model.embedTokens.asLinear(out)",
            "        } else if let lmHead {",
            "            return lmHead(out)",
            "        } else {",
            '            fatalError("Model config error: No lm_head or tied embeddings")',
            "        }",
            "    }",
            "",
            "    // MARK: - LLMModel Protocol",
            "",
            "    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {",
            "        weights  // Override if weight key mapping needed",
            "    }",
            "}",
        ]
    
    def _gen_helpers(self) -> List[str]:
        """Generate common helper functions"""
        return [
            "// MARK: - Helper Functions",
            "",
            "/// Apply rotary position embeddings",
            "func applyRotaryPosEmb(",
            "    _ q: MLXArray,",
            "    _ k: MLXArray,",
            "    _ cos: MLXArray,",
            "    _ sin: MLXArray",
            ") -> (MLXArray, MLXArray) {",
            "    let qEmbed = (q * cos) + (rotateHalf(q) * sin)",
            "    let kEmbed = (k * cos) + (rotateHalf(k) * sin)",
            "    return (qEmbed, kEmbed)",
            "}",
            "",
            "/// Rotate half of the hidden dims",
            "func rotateHalf(_ x: MLXArray) -> MLXArray {",
            "    let half = x.dim(-1) / 2",
            "    let x1 = x[..., ..<half]",
            "    let x2 = x[..., half...]",
            "    return concatenated([-x2, x1], axis: -1)",
            "}",
        ]
    
    def _gen_module(self, module: ParsedModule) -> List[str]:
        lines = [
            f"// MARK: - {module.name}",
            "",
            f"class {module.swift_name}: Module {{"
        ]
        
        # Properties (non-module)
        for name, typ, _ in module.properties:
            if typ in ("Int", "Float", "Bool"):
                lines.append(f"{self.indent}let {name}: {typ}")
        
        # Module attributes with @ModuleInfo
        for attr in module.attributes:
            if attr.module_type:
                lines.append(f'{self.indent}@ModuleInfo(key: "{attr.key}") var {attr.swift_name}: {attr.module_type}')
        
        lines.append("")
        
        # Init
        lines.append(f"{self.indent}init(_ config: {self.model_name}Configuration) {{")
        
        # Init properties
        for name, typ, init_expr in module.properties:
            if typ in ("Int", "Float", "Bool"):
                lines.append(f"{self.indent}{self.indent}self.{name} = {init_expr}")
        
        if module.properties:
            lines.append("")
        
        # Init modules
        for attr in module.attributes:
            if attr.module_type:
                args_str = ", ".join(attr.init_args) if attr.init_args else "/* TODO */"
                lines.append(f"{self.indent}{self.indent}self._{attr.swift_name}.wrappedValue = {attr.module_type}({args_str})")
        
        lines.append(f"{self.indent}}}")
        lines.append("")
        
        # Methods
        for method in module.methods:
            args_str = ", ".join([f"_ {to_camel(n)}: {t}" for n, t in method.args])
            lines.append(f"{self.indent}func {method.swift_name}({args_str}) -> {method.return_type} {{")
            
            for line in method.body:
                lines.append(f"{self.indent}{self.indent}{line}")
            
            if not method.body or not any(l.strip().startswith("return") for l in method.body):
                lines.append(f'{self.indent}{self.indent}fatalError("Not implemented")')
            
            lines.append(f"{self.indent}}}")
            lines.append("")
        
        lines.append("}")
        return lines


# =============================================================================
# MAIN
# =============================================================================

def fetch_model_code(model_name: str) -> Optional[str]:
    """Fetch model code from HuggingFace Transformers"""
    url = f"https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/{model_name}/modeling_{model_name}.py"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return r.read().decode()
    except Exception as e:
        print(f"Error fetching {model_name}: {e}")
        return None


def fetch_config_json(model_id: str) -> Optional[Dict]:
    """Fetch config.json from HuggingFace Hub"""
    url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read().decode())
    except:
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate MLX Swift code from HuggingFace models (v4)")
    parser.add_argument("--model", "-m", type=str, help="HuggingFace model name (e.g., llama, phi3)")
    parser.add_argument("--config", "-c", type=str, help="HuggingFace model ID for config (e.g., microsoft/phi-4)")
    parser.add_argument("--file", "-f", type=str, help="Local Python file path")
    parser.add_argument("--output", "-o", type=str, help="Output Swift file")
    args = parser.parse_args()
    
    if not args.model and not args.file:
        parser.print_help()
        return 1
    
    # Get source code
    if args.model:
        source = fetch_model_code(args.model)
        if not source:
            return 1
        model_name = args.model
    else:
        with open(args.file) as f:
            source = f.read()
        model_name = Path(args.file).stem.replace("modeling_", "")
    
    # Get config.json
    config_json = None
    if args.config:
        config_json = fetch_config_json(args.config)
        if config_json:
            print(f"✓ Fetched config from {args.config}")
    
    # Parse
    hf_parser = HFModelParser(model_name)
    modules = hf_parser.parse(source)
    
    # Generate
    generator = SwiftGenerator(model_name)
    swift_code = generator.generate(modules, config_json)
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(swift_code)
        print(f"✓ Generated: {args.output}")
    else:
        print(swift_code)
    
    # Summary
    print(f"\n// {'═' * 50}")
    print(f"// Model: {to_pascal(model_name)}")
    print(f"// Modules: {len(modules)}")
    for m in modules:
        attrs = len([a for a in m.attributes if a.module_type])
        props = len(m.properties)
        print(f"//   • {m.name}: {attrs} modules, {props} props, {len(m.methods)} methods")
    print(f"// {'═' * 50}")


if __name__ == "__main__":
    exit(main() or 0)

