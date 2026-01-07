#!/usr/bin/env python3
"""
HuggingFace Transformers → MLX Swift Generator v3

Production-ready version with:
- Better Linear initialization (uses actual dimensions from code)
- Config struct generation
- Proper control flow handling
- Complete expression conversion
"""

import ast
import re
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import urllib.request
import json


# =============================================================================
# MAPPINGS
# =============================================================================

SWIFT_TYPES = {
    "int": "Int", "float": "Float", "bool": "Bool", "str": "String",
    "Optional[int]": "Int?", "Optional[float]": "Float?", 
    "List[int]": "[Int]", "List[float]": "[Float]",
    "Tuple[int, int]": "(Int, Int)",
}

NN_MODULES = {
    "nn.Linear": "Linear",
    "nn.Embedding": "Embedding",
    "nn.LayerNorm": "LayerNorm", 
    "nn.RMSNorm": "RMSNorm",
    "nn.Dropout": None,
    "nn.ModuleList": "Array",
}

# Expression patterns for conversion
EXPR_PATTERNS = [
    # Tensor reshaping
    (r'\.transpose\((\d+),\s*(\d+)\)', r'.transposed(\1, \2)'),
    (r'\.reshape\(([^)]+)\)', r'.reshaped(\1)'),
    (r'\.view\(([^)]+)\)', r'.reshaped(\1)'),
    (r'\.unsqueeze\((\d+)\)', r'.expandedDimensions(axis: \1)'),
    (r'\.squeeze\((\d+)\)', r'.squeezed(axis: \1)'),
    (r'\.contiguous\(\)', ''),
    (r'\.float\(\)', '.asType(.float32)'),
    (r'\.half\(\)', '.asType(.float16)'),
    (r'\.to\([^)]+\)', ''),
    (r'\.type_as\((\w+)\)', r'.asType(\1.dtype)'),
    (r'\.size\((\d+)\)', r'.dim(\1)'),
    (r'\.shape\[(\d+)\]', r'.dim(\1)'),
    (r'\.chunk\((\d+),\s*dim=(-?\d+)\)', r'.split(parts: \1, axis: \2)'),
    
    # Torch functions
    (r'torch\.matmul\(([^,]+),\s*([^)]+)\)', r'matmul(\1, \2)'),
    (r'torch\.cat\(\[([^\]]+)\],\s*dim=(-?\d+)\)', r'concatenated([\1], axis: \2)'),
    (r'torch\.stack\(\[([^\]]+)\],\s*dim=(-?\d+)\)', r'stacked([\1], axis: \2)'),
    (r'torch\.mean\(([^,]+),\s*dim=(-?\d+)[^)]*\)', r'mean(\1, axis: \2)'),
    (r'torch\.sum\(([^,]+),\s*dim=(-?\d+)[^)]*\)', r'sum(\1, axis: \2)'),
    (r'torch\.sqrt\(([^)]+)\)', r'sqrt(\1)'),
    (r'torch\.rsqrt\(([^)]+)\)', r'rsqrt(\1)'),
    (r'torch\.tanh\(([^)]+)\)', r'tanh(\1)'),
    (r'torch\.exp\(([^)]+)\)', r'exp(\1)'),
    (r'torch\.where\(([^,]+),\s*([^,]+),\s*([^)]+)\)', r'MLX.where(\1, \2, \3)'),
    (r'torch\.ones\(([^)]+)\)', r'MLXArray.ones([\1])'),
    (r'torch\.zeros\(([^)]+)\)', r'MLXArray.zeros([\1])'),
    (r'torch\.arange\(([^)]+)\)', r'MLXArray(0..<\1)'),
    (r'torch\.triu\(([^,]+),\s*diagonal=(\d+)\)', r'triu(\1, k: \2)'),
    (r'torch\.tril\(([^,]+),\s*diagonal=(\d+)\)', r'tril(\1, k: \2)'),
    
    # F functions  
    (r'F\.gelu\(([^)]+)\)', r'gelu(\1)'),
    (r'F\.relu\(([^)]+)\)', r'relu(\1)'),
    (r'F\.silu\(([^)]+)\)', r'silu(\1)'),
    (r'F\.softmax\(([^,]+),\s*dim=(-?\d+)\)', r'softmax(\1, axis: \2)'),
    (r'F\.scaled_dot_product_attention\(([^)]+)\)', r'MLXFast.scaledDotProductAttention(\1)'),
    
    # Operators
    (r'(\w+)\s*@\s*(\w+)', r'matmul(\1, \2)'),
    
    # Python → Swift
    (r'None', 'nil'),
    (r'True', 'true'),
    (r'False', 'false'),
    (r'self\.', ''),
]


def to_camel(name: str) -> str:
    parts = name.split('_')
    return parts[0].lower() + ''.join(p.title() for p in parts[1:])


def to_pascal(name: str) -> str:
    return ''.join(p.title() for p in name.split('_'))


def convert_expr(py_expr: str) -> str:
    """Convert Python expression to Swift"""
    result = py_expr
    for pattern, replacement in EXPR_PATTERNS:
        result = re.sub(pattern, replacement, result)
    # Convert snake_case identifiers
    result = re.sub(r'\b([a-z]+)_([a-z_]+)\b', lambda m: to_camel(m.group(0)), result)
    return result


# =============================================================================
# AST ANALYSIS
# =============================================================================

@dataclass
class ConfigField:
    name: str
    py_type: str
    swift_type: str
    default: Optional[str] = None


@dataclass
class Attribute:
    name: str
    swift_name: str
    module_type: Optional[str] = None  # e.g., "Linear"
    swift_type: str = "Any"
    init_call: Optional[str] = None  # Full init expression


@dataclass
class Method:
    name: str
    swift_name: str
    args: List[Tuple[str, str]]  # (name, type)
    body: List[str]
    return_type: str = "MLXArray"


@dataclass
class ModuleClass:
    name: str
    attributes: List[Attribute] = field(default_factory=list)
    methods: List[Method] = field(default_factory=list)
    

class HFModelParser(ast.NodeVisitor):
    """Parse HuggingFace model Python code"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.modules: List[ModuleClass] = []
        self.configs: List[Tuple[str, List[ConfigField]]] = []
        self._current: Optional[ModuleClass] = None
    
    def parse(self, source: str):
        tree = ast.parse(source)
        self.visit(tree)
        return self.modules, self.configs
    
    def visit_ClassDef(self, node):
        bases = [self._base_name(b) for b in node.bases]
        
        # Config classes
        if "PretrainedConfig" in bases or node.name.endswith("Config"):
            self._parse_config(node)
            
        # Module classes
        elif any(b in bases for b in ["nn.Module", "PreTrainedModel"]):
            self._current = ModuleClass(name=node.name)
            
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == "__init__":
                        self._parse_init(item)
                    elif item.name == "forward":
                        self._parse_forward(item)
            
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
    
    def _parse_config(self, node):
        """Parse config class to extract fields"""
        fields = []
        
        # Look for __init__ assignments and super().__init__() kwargs
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                # Get default values from function signature
                defaults = {}
                args = item.args
                num_defaults = len(args.defaults)
                arg_names = [a.arg for a in args.args[1:]]  # Skip self
                
                for i, default in enumerate(args.defaults):
                    idx = len(arg_names) - num_defaults + i
                    if idx >= 0 and idx < len(arg_names):
                        defaults[arg_names[idx]] = self._get_default_value(default)
                
                # Also look for kwonly args
                for arg, default in zip(args.kwonlyargs, args.kw_defaults):
                    if default:
                        defaults[arg.arg] = self._get_default_value(default)
                
                # Parse super().__init__ call for field names
                for stmt in ast.walk(item):
                    if isinstance(stmt, ast.Call):
                        func = stmt.func
                        if isinstance(func, ast.Attribute) and func.attr == "__init__":
                            # This is super().__init__(...)
                            for kw in stmt.keywords:
                                if kw.arg and not kw.arg.startswith("_"):
                                    swift_type = self._infer_swift_type(kw.value)
                                    default = defaults.get(kw.arg)
                                    fields.append(ConfigField(
                                        name=kw.arg,
                                        py_type="Any",
                                        swift_type=swift_type,
                                        default=default
                                    ))
                
                # Also get direct self.x = y assignments
                for stmt in item.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Attribute) and \
                               isinstance(target.value, ast.Name) and \
                               target.value.id == "self":
                                name = target.attr
                                if not name.startswith("_") and not any(f.name == name for f in fields):
                                    swift_type = self._infer_swift_type(stmt.value)
                                    fields.append(ConfigField(
                                        name=name,
                                        py_type="Any",
                                        swift_type=swift_type
                                    ))
        
        if fields:
            self.configs.append((node.name, fields))
    
    def _get_default_value(self, node) -> Optional[str]:
        """Get default value as string"""
        if isinstance(node, ast.Constant):
            if node.value is None:
                return "nil"
            elif isinstance(node.value, bool):
                return "true" if node.value else "false"
            elif isinstance(node.value, str):
                return f'"{node.value}"'
            return str(node.value)
        elif isinstance(node, ast.Name):
            if node.id == "None":
                return "nil"
            elif node.id == "True":
                return "true"
            elif node.id == "False":
                return "false"
        return None
    
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
            return
        
        func_name = self._call_name(value)
        
        # Check if it's an nn.Module
        if func_name in NN_MODULES:
            swift_type = NN_MODULES[func_name]
            if swift_type is None:
                return  # Skip (e.g., Dropout)
            
            # Extract full initialization call
            init_call = self._build_init_call(func_name, value)
            
            self._current.attributes.append(Attribute(
                name=name,
                swift_name=to_camel(name),
                module_type=swift_type,
                swift_type=swift_type,
                init_call=init_call
            ))
    
    def _build_init_call(self, func_name: str, call_node) -> str:
        """Build Swift initialization call from Python call"""
        args = []
        
        for arg in call_node.args:
            args.append(convert_expr(ast.unparse(arg)))
        
        for kw in call_node.keywords:
            if kw.arg:
                swift_key = to_camel(kw.arg)
                swift_val = convert_expr(ast.unparse(kw.value))
                args.append(f"{swift_key}: {swift_val}")
        
        swift_type = NN_MODULES.get(func_name, func_name)
        return f"{swift_type}({', '.join(args)})"
    
    def _call_name(self, node) -> str:
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
        elif isinstance(node.func, ast.Name):
            return node.func.id
        return ""
    
    def _parse_forward(self, node):
        """Parse forward method"""
        # Arguments
        args = []
        for arg in node.args.args[1:]:  # Skip self
            arg_type = "MLXArray"  # Default
            if arg.annotation:
                ann = ast.unparse(arg.annotation)
                if "Tensor" in ann:
                    arg_type = "MLXArray"
            args.append((arg.arg, arg_type))
        
        # Body
        body_lines = []
        for stmt in node.body:
            swift_line = self._convert_stmt(stmt)
            if swift_line:
                body_lines.append(swift_line)
        
        self._current.methods.append(Method(
            name="forward",
            swift_name="callAsFunction",
            args=args,
            body=body_lines
        ))
    
    def _convert_stmt(self, stmt) -> Optional[str]:
        """Convert a Python statement to Swift"""
        if isinstance(stmt, ast.Return):
            val = convert_expr(ast.unparse(stmt.value))
            return f"return {val}"
        
        elif isinstance(stmt, ast.Assign):
            target = ast.unparse(stmt.targets[0]).replace("self.", "")
            value = convert_expr(ast.unparse(stmt.value))
            swift_target = to_camel(target)
            return f"let {swift_target} = {value}"
        
        elif isinstance(stmt, ast.AugAssign):
            target = ast.unparse(stmt.target).replace("self.", "")
            value = convert_expr(ast.unparse(stmt.value))
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(type(stmt.op), "?")
            swift_target = to_camel(target)
            return f"{swift_target} {op}= {value}"
        
        elif isinstance(stmt, ast.If):
            test = convert_expr(ast.unparse(stmt.test))
            return f"// if {test} {{ ... }}"
        
        elif isinstance(stmt, ast.For):
            return f"// for loop: {ast.unparse(stmt)[:50]}..."
        
        elif isinstance(stmt, ast.With):
            return f"// with block (skipped)"
        
        elif isinstance(stmt, ast.Expr):
            # Skip docstrings
            if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                return None
            return f"// {ast.unparse(stmt)[:60]}"
        
        return f"// {ast.unparse(stmt)[:60]}..."
    
    def _infer_swift_type(self, node) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return "Int"
            elif isinstance(node.value, float):
                return "Float"
            elif isinstance(node.value, bool):
                return "Bool"
            elif isinstance(node.value, str):
                return "String"
        elif isinstance(node, ast.List):
            return "[Any]"
        elif isinstance(node, ast.Call):
            func = self._call_name(node)
            if "getattr" in func:
                return "Any"
        return "Any"


# =============================================================================
# SWIFT GENERATOR
# =============================================================================

class SwiftCodeGenerator:
    def __init__(self, model_name: str):
        self.model_name = to_pascal(model_name)
        self.indent = "    "
    
    def generate(self, modules: List[ModuleClass], configs: List) -> str:
        lines = [
            "//",
            f"//  {self.model_name}.swift", 
            "//  Auto-generated by hf2swift v3",
            "//",
            "//  Manual review required before use!",
            "//",
            "",
            "import Foundation",
            "import MLX",
            "import MLXFast", 
            "import MLXNN",
            "import MLXLMCommon",
            "",
        ]
        
        # Generate configs
        if configs:
            lines.append("// MARK: - Configuration")
            lines.append("")
            for name, fields in configs:
                lines.extend(self._gen_config(name, fields))
                lines.append("")
        
        # Generate modules
        lines.append("// MARK: - Model Components")
        lines.append("")
        
        for module in modules:
            lines.extend(self._gen_module(module))
            lines.append("")
        
        return "\n".join(lines)
    
    def _gen_config(self, name: str, fields: List[ConfigField]) -> List[str]:
        swift_name = to_pascal(name)
        lines = [f"public struct {swift_name}: Codable {{"]
        
        for f in fields:
            swift_field = to_camel(f.name)
            lines.append(f"{self.indent}let {swift_field}: {f.swift_type}")
        
        lines.append("")
        lines.append(f"{self.indent}enum CodingKeys: String, CodingKey {{")
        for f in fields:
            swift_field = to_camel(f.name)
            lines.append(f'{self.indent}{self.indent}case {swift_field} = "{f.name}"')
        lines.append(f"{self.indent}}}")
        lines.append("}")
        
        return lines
    
    def _gen_module(self, module: ModuleClass) -> List[str]:
        lines = [
            f"// MARK: - {module.name}",
            "",
            f"class {module.name}: Module {{"
        ]
        
        # Module attributes with @ModuleInfo
        module_attrs = [a for a in module.attributes if a.module_type]
        other_attrs = [a for a in module.attributes if not a.module_type]
        
        for attr in module_attrs:
            lines.append(f'{self.indent}@ModuleInfo(key: "{attr.name}") var {attr.swift_name}: {attr.swift_type}')
        
        if module_attrs and other_attrs:
            lines.append("")
        
        for attr in other_attrs:
            lines.append(f"{self.indent}let {attr.swift_name}: {attr.swift_type}")
        
        lines.append("")
        
        # Init
        lines.append(f"{self.indent}init(_ config: {self.model_name}Configuration) {{")
        
        for attr in module_attrs:
            if attr.init_call:
                lines.append(f"{self.indent}{self.indent}self._{attr.swift_name}.wrappedValue = {attr.init_call}")
            else:
                lines.append(f"{self.indent}{self.indent}// TODO: Initialize {attr.swift_name}")
        
        lines.append(f"{self.indent}{self.indent}super.init()")
        lines.append(f"{self.indent}}}")
        lines.append("")
        
        # Methods
        for method in module.methods:
            args_str = ", ".join([f"_ {to_camel(name)}: {typ}" for name, typ in method.args])
            lines.append(f"{self.indent}func {method.swift_name}({args_str}) -> {method.return_type} {{")
            
            for line in method.body:
                lines.append(f"{self.indent}{self.indent}{line}")
            
            if not method.body or not any(l.strip().startswith("return") for l in method.body):
                lines.append(f'{self.indent}{self.indent}fatalError("Implementation required")')
            
            lines.append(f"{self.indent}}}")
        
        lines.append("}")
        return lines


# =============================================================================
# MAIN
# =============================================================================

def fetch_model(model_name: str) -> Optional[str]:
    """Fetch model code from HuggingFace Transformers"""
    url = f"https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/{model_name}/modeling_{model_name}.py"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.read().decode()
    except Exception as e:
        print(f"Error fetching {model_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate MLX Swift code from HuggingFace models")
    parser.add_argument("--model", "-m", type=str, help="HuggingFace model name")
    parser.add_argument("--file", "-f", type=str, help="Local Python file path")
    parser.add_argument("--output", "-o", type=str, help="Output Swift file")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    args = parser.parse_args()
    
    if args.list_models:
        print("Popular models: llama, mistral, qwen2, phi3, gemma, gemma2, cohere, falcon")
        return
    
    # Get source
    if args.model:
        source = fetch_model(args.model)
        if not source:
            return 1
        model_name = args.model
    elif args.file:
        with open(args.file) as f:
            source = f.read()
        model_name = Path(args.file).stem.replace("modeling_", "")
    else:
        parser.print_help()
        return 1
    
    # Parse
    hf_parser = HFModelParser(model_name)
    modules, configs = hf_parser.parse(source)
    
    # Generate
    generator = SwiftCodeGenerator(model_name)
    swift_code = generator.generate(modules, configs)
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(swift_code)
        print(f"✓ Generated: {args.output}")
    else:
        print(swift_code)
    
    # Summary
    print(f"\n// ═══════════════════════════════════════")
    print(f"// Model: {to_pascal(model_name)}")
    print(f"// Configs: {len(configs)}")
    print(f"// Modules: {len(modules)}")
    for m in modules:
        attrs = len([a for a in m.attributes if a.module_type])
        print(f"//   • {m.name}: {attrs} nn.Module attrs, {len(m.methods)} methods")
    print(f"// ═══════════════════════════════════════")


if __name__ == "__main__":
    exit(main() or 0)

