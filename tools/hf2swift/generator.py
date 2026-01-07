#!/usr/bin/env python3
"""
HuggingFace Transformers → MLX Swift Generator

Converts PyTorch model definitions from HuggingFace Transformers
to Swift code compatible with Apple's MLX framework.

Usage:
    python generator.py --model gemma
    python generator.py --file path/to/modeling_xyz.py
"""

import ast
import re
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# MAPPING TABLES
# =============================================================================

# PyTorch → MLX type mappings
TYPE_MAPPINGS = {
    "nn.Linear": "Linear",
    "nn.Embedding": "Embedding", 
    "nn.LayerNorm": "LayerNorm",
    "nn.RMSNorm": "RMSNorm",
    "nn.Dropout": None,  # MLX doesn't use dropout in inference
    "nn.ModuleList": "[]",  # Swift array
}

# PyTorch → MLX function mappings
FUNCTION_MAPPINGS = {
    "F.gelu": "gelu",
    "F.relu": "relu",
    "F.silu": "silu",
    "F.softmax": "softmax",
    "F.scaled_dot_product_attention": "MLXFast.scaledDotProductAttention",
    "torch.matmul": "matmul",
    "torch.einsum": "einsum",
    "torch.cat": "concatenated",
    "torch.stack": "stacked",
    "torch.mean": "mean",
    "torch.sum": "sum",
    "torch.sqrt": "sqrt",
    "torch.tanh": "tanh",
    "torch.exp": "exp",
    "torch.where": "MLX.where",
    ".transpose": ".transposed",
    ".reshape": ".reshaped",
    ".view": ".reshaped",
    ".unsqueeze": "expandedDimensions",
    ".squeeze": "squeezed",
    ".contiguous": "",  # Not needed in MLX
}

# Python → Swift naming convention
def to_swift_name(python_name: str) -> str:
    """Convert snake_case to camelCase"""
    components = python_name.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


def to_swift_class_name(python_name: str) -> str:
    """Convert snake_case to PascalCase"""
    return ''.join(x.title() for x in python_name.split('_'))


# =============================================================================
# AST VISITORS
# =============================================================================

@dataclass
class ModuleInfo:
    """Information about a nn.Module class"""
    name: str
    base_class: str
    init_args: list
    attributes: dict  # name -> type
    forward_args: list
    forward_body: str


@dataclass 
class ConfigField:
    """A field in the config class"""
    name: str
    python_type: str
    swift_type: str
    coding_key: str


class ModuleVisitor(ast.NodeVisitor):
    """Extract information from nn.Module classes"""
    
    def __init__(self):
        self.modules = []
        self.current_module = None
        self.configs = []
        
    def visit_ClassDef(self, node):
        # Check if it's a nn.Module subclass
        bases = [self._get_base_name(b) for b in node.bases]
        
        if "nn.Module" in bases or "PreTrainedModel" in bases:
            self.current_module = ModuleInfo(
                name=node.name,
                base_class=bases[0] if bases else "Module",
                init_args=[],
                attributes={},
                forward_args=[],
                forward_body=""
            )
            
            # Visit methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == "__init__":
                        self._process_init(item)
                    elif item.name == "forward":
                        self._process_forward(item)
            
            self.modules.append(self.current_module)
            self.current_module = None
            
        elif "PretrainedConfig" in bases or "Config" in node.name:
            self._process_config(node)
            
        self.generic_visit(node)
    
    def _get_base_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_base_name(node.value)}.{node.attr}"
        return ""
    
    def _process_init(self, node):
        """Process __init__ method"""
        # Get arguments (skip self)
        for arg in node.args.args[1:]:
            self.current_module.init_args.append(arg.arg)
        
        # Find attribute assignments
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and \
                       isinstance(target.value, ast.Name) and \
                       target.value.id == "self":
                        attr_name = target.attr
                        attr_type = self._infer_type(stmt.value)
                        self.current_module.attributes[attr_name] = attr_type
    
    def _process_forward(self, node):
        """Process forward method"""
        # Get arguments (skip self)
        for arg in node.args.args[1:]:
            self.current_module.forward_args.append(arg.arg)
        
        # Convert body to string for now (will be properly converted later)
        self.current_module.forward_body = ast.unparse(node)
    
    def _process_config(self, node):
        """Process config class"""
        fields = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                field = ConfigField(
                    name=item.target.id,
                    python_type=ast.unparse(item.annotation) if item.annotation else "Any",
                    swift_type=self._python_to_swift_type(item.annotation),
                    coding_key=self._to_coding_key(item.target.id)
                )
                fields.append(field)
        self.configs.append((node.name, fields))
    
    def _infer_type(self, node) -> str:
        """Infer the type of an expression"""
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                return f"{func.value.id}.{func.attr}" if isinstance(func.value, ast.Name) else func.attr
            elif isinstance(func, ast.Name):
                return func.id
        return "Unknown"
    
    def _python_to_swift_type(self, node) -> str:
        """Convert Python type annotation to Swift type"""
        if node is None:
            return "Any"
        type_str = ast.unparse(node)
        mappings = {
            "int": "Int",
            "float": "Float",
            "bool": "Bool",
            "str": "String",
            "Optional[int]": "Int?",
            "Optional[float]": "Float?",
            "Optional[bool]": "Bool?",
            "Optional[str]": "String?",
            "List[int]": "[Int]",
            "List[float]": "[Float]",
            "List[str]": "[String]",
        }
        return mappings.get(type_str, type_str)
    
    def _to_coding_key(self, name: str) -> str:
        """Convert Python name to JSON key"""
        return name  # Usually snake_case in JSON


# =============================================================================
# SWIFT CODE GENERATOR  
# =============================================================================

class SwiftGenerator:
    """Generate Swift code from parsed module info"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.indent = "    "
    
    def generate_config(self, name: str, fields: list) -> str:
        """Generate Swift Codable config struct"""
        swift_name = to_swift_class_name(name)
        
        lines = [
            f"public struct {swift_name}: Codable {{",
        ]
        
        # Properties
        for field in fields:
            swift_field_name = to_swift_name(field.name)
            lines.append(f"{self.indent}let {swift_field_name}: {field.swift_type}")
        
        lines.append("")
        
        # CodingKeys enum
        lines.append(f"{self.indent}enum CodingKeys: String, CodingKey {{")
        for field in fields:
            swift_field_name = to_swift_name(field.name)
            if swift_field_name != field.coding_key:
                lines.append(f"{self.indent}{self.indent}case {swift_field_name} = \"{field.coding_key}\"")
            else:
                lines.append(f"{self.indent}{self.indent}case {swift_field_name}")
        lines.append(f"{self.indent}}}")
        
        lines.append("}")
        return "\n".join(lines)
    
    def generate_module(self, module: ModuleInfo) -> str:
        """Generate Swift Module class"""
        lines = [
            f"class {module.name}: Module {{"
        ]
        
        # Generate @ModuleInfo properties
        for attr_name, attr_type in module.attributes.items():
            swift_name = to_swift_name(attr_name)
            swift_type = TYPE_MAPPINGS.get(attr_type, attr_type)
            
            if swift_type is None:  # Skip (e.g., Dropout)
                continue
            if swift_type == "[]":
                continue  # Handle arrays separately
                
            # Clean up the type name
            if "." in swift_type:
                swift_type = swift_type.split(".")[-1]
            
            lines.append(f"{self.indent}@ModuleInfo(key: \"{attr_name}\") var {swift_name}: {swift_type}")
        
        lines.append("")
        
        # Generate init
        config_arg = module.init_args[0] if module.init_args else "config"
        config_type = f"{self.model_name}Configuration"
        
        lines.append(f"{self.indent}init(_ {config_arg}: {config_type}) {{")
        
        # Generate property initializations
        for attr_name, attr_type in module.attributes.items():
            swift_name = to_swift_name(attr_name)
            swift_type = TYPE_MAPPINGS.get(attr_type, attr_type)
            
            if swift_type is None:
                continue
            if swift_type == "[]":
                continue
                
            # Generate initialization based on type
            if swift_type == "Linear":
                lines.append(f"{self.indent}{self.indent}// TODO: Initialize {swift_name}")
                lines.append(f"{self.indent}{self.indent}// _{swift_name}.wrappedValue = Linear(...)")
            elif swift_type == "Embedding":
                lines.append(f"{self.indent}{self.indent}// TODO: Initialize {swift_name}")
                lines.append(f"{self.indent}{self.indent}// _{swift_name}.wrappedValue = Embedding(...)")
        
        lines.append(f"{self.indent}{self.indent}super.init()")
        lines.append(f"{self.indent}}}")
        
        lines.append("")
        
        # Generate callAsFunction (forward)
        forward_args = ", ".join([f"_ {to_swift_name(a)}: MLXArray" for a in module.forward_args])
        lines.append(f"{self.indent}func callAsFunction({forward_args}) -> MLXArray {{")
        lines.append(f"{self.indent}{self.indent}// TODO: Implement forward pass")
        lines.append(f"{self.indent}{self.indent}// Original Python:")
        
        # Add original Python as comments
        for line in module.forward_body.split('\n')[1:]:  # Skip def line
            lines.append(f"{self.indent}{self.indent}// {line.strip()}")
        
        lines.append(f"{self.indent}{self.indent}fatalError(\"Not implemented\")")
        lines.append(f"{self.indent}}}")
        
        lines.append("}")
        return "\n".join(lines)
    
    def generate_file(self, modules: list, configs: list) -> str:
        """Generate complete Swift file"""
        lines = [
            "//",
            f"//  {self.model_name}.swift",
            "//  Auto-generated by hf2swift",
            "//",
            "",
            "import Foundation",
            "import MLX",
            "import MLXFast",
            "import MLXNN",
            "import MLXLMCommon",
            "",
            "// MARK: - Configuration",
            "",
        ]
        
        # Generate configs
        for name, fields in configs:
            lines.append(self.generate_config(name, fields))
            lines.append("")
        
        lines.append("// MARK: - Model Components")
        lines.append("")
        
        # Generate modules
        for module in modules:
            lines.append(self.generate_module(module))
            lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def parse_file(file_path: str) -> tuple:
    """Parse a Python file and extract module info"""
    with open(file_path, 'r') as f:
        source = f.read()
    
    tree = ast.parse(source)
    visitor = ModuleVisitor()
    visitor.visit(tree)
    
    return visitor.modules, visitor.configs


def fetch_from_huggingface(model_name: str) -> str:
    """Fetch model file from HuggingFace transformers repo"""
    import urllib.request
    
    url = f"https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/{model_name}/modeling_{model_name}.py"
    
    try:
        with urllib.request.urlopen(url) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace models to MLX Swift")
    parser.add_argument("--model", type=str, help="HuggingFace model name (e.g., gemma, llama)")
    parser.add_argument("--file", type=str, help="Path to local modeling_*.py file")
    parser.add_argument("--output", type=str, help="Output Swift file path")
    
    args = parser.parse_args()
    
    if args.file:
        # Parse local file
        source_path = args.file
        model_name = Path(source_path).stem.replace("modeling_", "")
        
        modules, configs = parse_file(source_path)
        
    elif args.model:
        # Fetch from HuggingFace
        model_name = args.model
        source = fetch_from_huggingface(model_name)
        
        if source is None:
            print(f"Could not fetch model: {model_name}")
            return 1
        
        # Parse the source
        tree = ast.parse(source)
        visitor = ModuleVisitor()
        visitor.visit(tree)
        
        modules = visitor.modules
        configs = visitor.configs
    else:
        parser.print_help()
        return 1
    
    # Generate Swift code
    generator = SwiftGenerator(to_swift_class_name(model_name))
    swift_code = generator.generate_file(modules, configs)
    
    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(swift_code)
        print(f"Generated: {args.output}")
    else:
        print(swift_code)
    
    # Print summary
    print(f"\n// Summary:")
    print(f"//   Configs: {len(configs)}")
    print(f"//   Modules: {len(modules)}")
    for m in modules:
        print(f"//     - {m.name}: {len(m.attributes)} attributes")
    
    return 0


if __name__ == "__main__":
    exit(main())

