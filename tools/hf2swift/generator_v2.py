#!/usr/bin/env python3
"""
HuggingFace Transformers → MLX Swift Generator v2

Enhanced version with better type inference and forward pass conversion.
"""

import ast
import re
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict


# =============================================================================
# COMPREHENSIVE MAPPINGS
# =============================================================================

# nn.Module types → Swift types
NN_TYPE_MAP = {
    "nn.Linear": ("Linear", "Linear({in_features}, {out_features}, bias: {bias})"),
    "nn.Embedding": ("Embedding", "Embedding(embeddingCount: {num_embeddings}, dimensions: {embedding_dim})"),
    "nn.LayerNorm": ("LayerNorm", "LayerNorm(dimensions: {normalized_shape}, eps: {eps})"),
    "nn.RMSNorm": ("RMSNorm", "RMSNorm(dimensions: {dim}, eps: {eps})"),
    "nn.Dropout": (None, None),  # Skip
    "nn.GELU": ("GELU", "GELU()"),
    "nn.SiLU": ("SiLU", "SiLU()"),
}

# Python expressions → Swift expressions  
EXPR_MAP = {
    # Tensor operations
    r"(\w+)\.transpose\((\d+),\s*(\d+)\)": r"\1.transposed(\2, \3)",
    r"(\w+)\.reshape\((.+)\)": r"\1.reshaped(\2)",
    r"(\w+)\.view\((.+)\)": r"\1.reshaped(\2)",
    r"(\w+)\.unsqueeze\((\d+)\)": r"expandedDimensions(\1, axis: \2)",
    r"(\w+)\.squeeze\((\d+)\)": r"squeezed(\1, axis: \2)",
    r"(\w+)\.contiguous\(\)": r"\1",
    r"(\w+)\.float\(\)": r"\1.asType(.float32)",
    r"(\w+)\.half\(\)": r"\1.asType(.float16)",
    r"(\w+)\.to\((.+)\)": r"\1",  # Device transfer not needed
    r"(\w+)\.type_as\((\w+)\)": r"\1.asType(\2.dtype)",
    
    # Torch functions
    r"torch\.matmul\((.+),\s*(.+)\)": r"matmul(\1, \2)",
    r"torch\.einsum\((.+)\)": r"einsum(\1)",
    r"torch\.cat\(\[(.+)\],\s*dim=(-?\d+)\)": r"concatenated([\1], axis: \2)",
    r"torch\.stack\(\[(.+)\],\s*dim=(-?\d+)\)": r"stacked([\1], axis: \2)",
    r"torch\.mean\((.+),\s*dim=(-?\d+)\)": r"mean(\1, axis: \2)",
    r"torch\.sum\((.+),\s*dim=(-?\d+)\)": r"sum(\1, axis: \2)",
    r"torch\.sqrt\((.+)\)": r"sqrt(\1)",
    r"torch\.tanh\((.+)\)": r"tanh(\1)",
    r"torch\.exp\((.+)\)": r"exp(\1)",
    r"torch\.where\((.+),\s*(.+),\s*(.+)\)": r"MLX.where(\1, \2, \3)",
    r"torch\.ones\((.+)\)": r"MLXArray.ones(\1)",
    r"torch\.zeros\((.+)\)": r"MLXArray.zeros(\1)",
    r"torch\.arange\((.+)\)": r"MLXArray(\1)",
    
    # F functions
    r"F\.gelu\((.+)\)": r"gelu(\1)",
    r"F\.relu\((.+)\)": r"relu(\1)", 
    r"F\.silu\((.+)\)": r"silu(\1)",
    r"F\.softmax\((.+),\s*dim=(-?\d+)\)": r"softmax(\1, axis: \2)",
    r"F\.scaled_dot_product_attention\((.+)\)": r"MLXFast.scaledDotProductAttention(\1)",
    
    # Attribute access
    r"self\.(\w+)": r"\1",  # Remove self.
    r"config\.(\w+)": r"config.\1",  # Keep config.
    
    # Operators
    r"(\w+)\s*@\s*(\w+)": r"matmul(\1, \2)",
    r"(\w+)\.shape\[(\d+)\]": r"\1.dim(\2)",
    r"(\w+)\.size\((\d+)\)": r"\1.dim(\2)",
}

# Python types → Swift types
TYPE_MAP = {
    "int": "Int",
    "float": "Float", 
    "bool": "Bool",
    "str": "String",
    "None": "nil",
    "True": "true",
    "False": "false",
    "Optional[int]": "Int?",
    "Optional[float]": "Float?",
    "List[int]": "[Int]",
    "List[float]": "[Float]",
    "Tuple": "tuple",
}


def to_camel_case(name: str) -> str:
    """snake_case → camelCase"""
    parts = name.split('_')
    return parts[0].lower() + ''.join(p.title() for p in parts[1:])


def to_pascal_case(name: str) -> str:
    """snake_case → PascalCase"""
    return ''.join(p.title() for p in name.split('_'))


def convert_expr(python_expr: str) -> str:
    """Convert a Python expression to Swift"""
    result = python_expr
    
    for pattern, replacement in EXPR_MAP.items():
        result = re.sub(pattern, replacement, result)
    
    # Convert snake_case identifiers to camelCase
    result = re.sub(r'\b([a-z]+)_([a-z_]+)\b', lambda m: to_camel_case(m.group(0)), result)
    
    return result


# =============================================================================
# AST ANALYSIS
# =============================================================================

@dataclass
class Attribute:
    name: str
    swift_name: str
    nn_type: str
    swift_type: str
    init_args: Dict[str, str] = field(default_factory=dict)
    is_module: bool = False


@dataclass
class Method:
    name: str
    args: List[str]
    body_lines: List[str]
    returns: str = "MLXArray"


@dataclass
class ModuleClass:
    name: str
    attributes: List[Attribute] = field(default_factory=list)
    methods: List[Method] = field(default_factory=list)
    config_type: str = "Configuration"


class EnhancedVisitor(ast.NodeVisitor):
    """Enhanced AST visitor with better type inference"""
    
    def __init__(self):
        self.modules: List[ModuleClass] = []
        self.current: Optional[ModuleClass] = None
        
    def visit_ClassDef(self, node):
        bases = [self._base_name(b) for b in node.bases]
        
        if any(b in ["nn.Module", "PreTrainedModel"] for b in bases):
            self.current = ModuleClass(name=node.name)
            
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == "__init__":
                        self._analyze_init(item)
                    elif item.name == "forward":
                        self._analyze_forward(item)
                    elif not item.name.startswith("_"):
                        self._analyze_method(item)
            
            self.modules.append(self.current)
            self.current = None
            
        self.generic_visit(node)
    
    def _base_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._base_name(node.value)}.{node.attr}"
        return ""
    
    def _analyze_init(self, node):
        """Analyze __init__ to extract attributes"""
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and \
                       isinstance(target.value, ast.Name) and \
                       target.value.id == "self":
                        self._extract_attribute(target.attr, stmt.value)
    
    def _extract_attribute(self, name: str, value_node):
        """Extract attribute info from assignment"""
        if isinstance(value_node, ast.Call):
            func_name = self._get_call_name(value_node)
            
            # Check if it's an nn.Module type
            if func_name in NN_TYPE_MAP:
                swift_type, init_template = NN_TYPE_MAP[func_name]
                if swift_type is None:
                    return  # Skip (e.g., Dropout)
                
                # Extract initialization arguments
                init_args = {}
                for kw in value_node.keywords:
                    init_args[kw.arg] = ast.unparse(kw.value)
                for i, arg in enumerate(value_node.args):
                    init_args[f"arg{i}"] = ast.unparse(arg)
                
                attr = Attribute(
                    name=name,
                    swift_name=to_camel_case(name),
                    nn_type=func_name,
                    swift_type=swift_type,
                    init_args=init_args,
                    is_module=True
                )
                self.current.attributes.append(attr)
            else:
                # Regular attribute
                attr = Attribute(
                    name=name,
                    swift_name=to_camel_case(name),
                    nn_type="",
                    swift_type=self._infer_swift_type(value_node),
                    is_module=False
                )
                self.current.attributes.append(attr)
    
    def _get_call_name(self, node) -> str:
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
            return node.func.attr
        elif isinstance(node.func, ast.Name):
            return node.func.id
        return ""
    
    def _infer_swift_type(self, node) -> str:
        """Infer Swift type from Python expression"""
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
        return "Any"
    
    def _analyze_forward(self, node):
        """Analyze forward method"""
        args = [arg.arg for arg in node.args.args[1:]]  # Skip self
        
        body_lines = []
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                body_lines.append(f"return {convert_expr(ast.unparse(stmt.value))}")
            elif isinstance(stmt, ast.Assign):
                target = ast.unparse(stmt.targets[0])
                value = convert_expr(ast.unparse(stmt.value))
                swift_target = to_camel_case(target.replace("self.", ""))
                body_lines.append(f"let {swift_target} = {value}")
            elif isinstance(stmt, ast.AugAssign):
                target = ast.unparse(stmt.target)
                value = convert_expr(ast.unparse(stmt.value))
                op = self._aug_op(stmt.op)
                swift_target = to_camel_case(target.replace("self.", ""))
                body_lines.append(f"{swift_target} {op}= {value}")
            else:
                # Add as comment
                body_lines.append(f"// {ast.unparse(stmt)}")
        
        method = Method(
            name="callAsFunction",
            args=args,
            body_lines=body_lines
        )
        self.current.methods.append(method)
    
    def _analyze_method(self, node):
        """Analyze other methods"""
        args = [arg.arg for arg in node.args.args[1:]]
        
        body_lines = []
        for stmt in node.body:
            body_lines.append(f"// {ast.unparse(stmt)}")
        
        method = Method(
            name=to_camel_case(node.name),
            args=args,
            body_lines=body_lines
        )
        self.current.methods.append(method)
    
    def _aug_op(self, op) -> str:
        ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
        return ops.get(type(op), "?")


# =============================================================================
# SWIFT GENERATOR
# =============================================================================

class SwiftGenerator:
    def __init__(self, model_name: str):
        self.model_name = to_pascal_case(model_name)
        self.indent = "    "
    
    def generate(self, modules: List[ModuleClass]) -> str:
        lines = [
            "//",
            f"//  {self.model_name}.swift",
            "//  Auto-generated by hf2swift v2",
            "//",
            "",
            "import Foundation",
            "import MLX", 
            "import MLXFast",
            "import MLXNN",
            "import MLXLMCommon",
            "",
        ]
        
        for module in modules:
            lines.extend(self._generate_module(module))
            lines.append("")
        
        return "\n".join(lines)
    
    def _generate_module(self, module: ModuleClass) -> List[str]:
        lines = [
            f"// MARK: - {module.name}",
            "",
            f"class {module.name}: Module {{",
        ]
        
        # Module properties (with @ModuleInfo)
        module_attrs = [a for a in module.attributes if a.is_module]
        other_attrs = [a for a in module.attributes if not a.is_module]
        
        for attr in module_attrs:
            lines.append(f'{self.indent}@ModuleInfo(key: "{attr.name}") var {attr.swift_name}: {attr.swift_type}')
        
        if module_attrs and other_attrs:
            lines.append("")
        
        # Regular properties
        for attr in other_attrs:
            lines.append(f"{self.indent}let {attr.swift_name}: {attr.swift_type}")
        
        lines.append("")
        
        # Init
        lines.append(f"{self.indent}init(_ config: {self.model_name}Configuration) {{")
        
        for attr in module_attrs:
            if attr.swift_type == "Linear":
                lines.append(f"{self.indent}{self.indent}self._{attr.swift_name}.wrappedValue = Linear(")
                lines.append(f"{self.indent}{self.indent}{self.indent}config.hiddenSize,")
                lines.append(f"{self.indent}{self.indent}{self.indent}config.hiddenSize,")
                lines.append(f"{self.indent}{self.indent}{self.indent}bias: false")
                lines.append(f"{self.indent}{self.indent})")
        
        lines.append(f"{self.indent}{self.indent}super.init()")
        lines.append(f"{self.indent}}}")
        lines.append("")
        
        # Methods
        for method in module.methods:
            args_str = ", ".join([f"_ {to_camel_case(a)}: MLXArray" for a in method.args])
            lines.append(f"{self.indent}func {method.name}({args_str}) -> {method.returns} {{")
            
            for body_line in method.body_lines:
                lines.append(f"{self.indent}{self.indent}{body_line}")
            
            if not method.body_lines or not method.body_lines[-1].startswith("return"):
                lines.append(f'{self.indent}{self.indent}fatalError("Not fully implemented")')
            
            lines.append(f"{self.indent}}}")
            lines.append("")
        
        lines.append("}")
        return lines


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="HuggingFace model name")
    parser.add_argument("--file", type=str, help="Local file path")
    parser.add_argument("--output", "-o", type=str, help="Output file")
    args = parser.parse_args()
    
    if args.model:
        import urllib.request
        url = f"https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/{args.model}/modeling_{args.model}.py"
        with urllib.request.urlopen(url) as r:
            source = r.read().decode()
        model_name = args.model
    elif args.file:
        with open(args.file) as f:
            source = f.read()
        model_name = Path(args.file).stem.replace("modeling_", "")
    else:
        parser.print_help()
        return
    
    # Parse
    tree = ast.parse(source)
    visitor = EnhancedVisitor()
    visitor.visit(tree)
    
    # Generate
    gen = SwiftGenerator(model_name)
    swift_code = gen.generate(visitor.modules)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(swift_code)
        print(f"✓ Generated {args.output}")
    else:
        print(swift_code)
    
    print(f"\n// Found {len(visitor.modules)} modules:")
    for m in visitor.modules:
        attrs = len([a for a in m.attributes if a.is_module])
        print(f"//   {m.name}: {attrs} nn.Module attributes")


if __name__ == "__main__":
    main()

