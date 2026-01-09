/**
 * Python AST Parser using py-ast
 *
 * Parses HuggingFace Transformers Python source code and extracts
 * module definitions, attributes, and methods.
 *
 * Note: This file works with dynamic AST data from py-ast which has no TypeScript types.
 * The 'any' type and unsafe member access are intentional for AST traversal.
 */

/* eslint-disable @typescript-eslint/no-unsafe-member-access */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
/* eslint-disable @typescript-eslint/no-unsafe-call */
/* eslint-disable @typescript-eslint/no-unsafe-argument */
/* eslint-disable @typescript-eslint/no-unsafe-return */
/* eslint-disable @typescript-eslint/restrict-template-expressions */
/* eslint-disable @typescript-eslint/prefer-nullish-coalescing */
/* eslint-disable @typescript-eslint/no-unnecessary-condition */

import { parse } from "py-ast"
import type { ParsedModule } from "./types.js"
import { NN_MODULES, EXPR_CONVERSIONS } from "./types.js"
import { toCamel, convertExpr } from "./naming.js"

// Use 'any' for AST nodes since py-ast types are complex and vary
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ASTNode = any

/**
 * HuggingFace Model Parser
 *
 * Parses Python source code and extracts module definitions
 */
export class HFModelParser {
  private modelName: string
  private allClassNames = new Set<string>()

  constructor(modelName: string) {
    this.modelName = modelName
  }

  /**
   * Parse Python source code and extract modules
   */
  parse(source: string): ParsedModule[] {
    // Cast to any since py-ast types are complex and we need flexible access
    const ast = parse(source) as ASTNode
    const modules: ParsedModule[] = []

    // First pass: collect all class names
    this.collectClassNames(ast)

    // Second pass: parse module classes
    if (ast.body) {
      for (const node of ast.body) {
        if (node.name && this.isModuleClass(node)) {
          const parsed = this.parseClass(node)
          if (parsed) {
            modules.push(parsed)
          }
        }
      }
    }

    return modules
  }

  /**
   * Collect all class names in the source
   */
  private collectClassNames(ast: ASTNode): void {
    if (ast.body) {
      for (const node of ast.body) {
        if (node.name) {
          this.allClassNames.add(node.name)
        }
      }
    }
  }

  /**
   * Check if a class is a nn.Module subclass
   */
  private isModuleClass(node: ASTNode): boolean {
    if (!node.bases) {
      return false
    }

    const recognizedBases = new Set([
      "nn.Module",
      "Module",
      "PreTrainedModel",
      "nn.Embedding",
      "nn.Linear",
      "nn.LayerNorm",
      "GradientCheckpointingLayer",
      "GenerationMixin"
    ])

    for (const base of node.bases) {
      const baseName = this.getBaseName(base)
      if (
        recognizedBases.has(baseName) ||
        baseName.includes("Model") ||
        baseName.includes("Layer") ||
        baseName.includes("Block") ||
        this.allClassNames.has(baseName)
      ) {
        return true
      }
    }

    return false
  }

  /**
   * Get the name of a base class
   */
  private getBaseName(node: ASTNode): string {
    if (node.id) {
      return node.id
    }
    if (node.attr && node.value) {
      return `${this.getBaseName(node.value)}.${node.attr}`
    }
    return ""
  }

  /**
   * Parse a class definition into a ParsedModule
   */
  private parseClass(node: ASTNode): ParsedModule | null {
    const nodeName = node.name as string
    const module: ParsedModule = {
      name: nodeName,
      swiftName: nodeName,
      attributes: [],
      methods: [],
      properties: [],
      baseClasses: ((node.bases ?? []) as ASTNode[]).map((b: ASTNode) => this.getBaseName(b))
    }

    if (node.body) {
      for (const item of node.body) {
        if (item.name === "__init__") {
          this.parseInit(item, module)
        } else if (item.name === "forward") {
          this.parseForward(item, module)
        }
      }
    }

    return module
  }

  /**
   * Parse __init__ method to extract attributes
   */
  private parseInit(node: ASTNode, module: ParsedModule): void {
    const existing = new Set<string>()

    this.walkNode(node, (stmt: ASTNode) => {
      // Look for self.x = ... assignments
      if (stmt.targets && stmt.value) {
        for (const target of stmt.targets) {
          if (target.attr && target.value?.id === "self" && !existing.has(target.attr)) {
            existing.add(target.attr)
            this.extractAttribute(target.attr, stmt.value, module)
          }
        }
      }
    })
  }

  /**
   * Walk AST nodes recursively
   */
  private walkNode(node: ASTNode, callback: (node: ASTNode) => void): void {
    callback(node)
    if (node.body) {
      for (const child of node.body) {
        this.walkNode(child, callback)
      }
    }
  }

  /**
   * Extract an attribute from an assignment
   */
  private extractAttribute(name: string, value: ASTNode, module: ParsedModule): void {
    // Check if it's a Call (e.g., nn.Linear(...))
    if (!value.func) {
      // Not a call - treat as property
      const swiftType = this.inferType(value)
      const initExpr = convertExpr(this.unparse(value), EXPR_CONVERSIONS)

      module.properties.push({
        name,
        swiftName: toCamel(name),
        swiftType,
        initExpr
      })
      return
    }

    // It's a Call - get the function name
    const funcName = this.getCallName(value)

    // Check if it's a known nn module or a class in this file
    if (!(funcName in NN_MODULES) && !this.allClassNames.has(funcName)) {
      // Unknown call - treat as property
      const swiftType = this.inferType(value)
      const initExpr = convertExpr(this.unparse(value), EXPR_CONVERSIONS)

      module.properties.push({
        name,
        swiftName: toCamel(name),
        swiftType,
        initExpr
      })
      return
    }

    // Get the Swift type
    let swiftType: string | null
    const moduleInfo = NN_MODULES[funcName]
    if (moduleInfo) {
      swiftType = moduleInfo[0]
      if (swiftType === null) {
        // Skip this module (e.g., Dropout)
        return
      }
    } else {
      swiftType = funcName
    }

    // Extract init args
    const initArgs: string[] = []

    // Positional args
    const args = Array.isArray(value.args) ? value.args : value.args?.args || []
    for (const arg of args) {
      initArgs.push(convertExpr(this.unparse(arg), EXPR_CONVERSIONS))
    }

    // Keyword args
    if (value.keywords) {
      for (const kw of value.keywords) {
        if (kw.arg && kw.value) {
          initArgs.push(
            `${toCamel(kw.arg)}: ${convertExpr(this.unparse(kw.value), EXPR_CONVERSIONS)}`
          )
        }
      }
    }

    module.attributes.push({
      name,
      swiftName: toCamel(name),
      moduleType: swiftType,
      initArgs,
      key: name,
      isParameter: false
    })
  }

  /**
   * Get the name of a function call
   */
  private getCallName(node: ASTNode): string {
    if (!node.func) {
      return ""
    }

    if (node.func.attr && node.func.value?.id) {
      return `${node.func.value.id}.${node.func.attr}`
    }
    if (node.func.attr) {
      return node.func.attr
    }
    if (node.func.id) {
      return node.func.id
    }
    return ""
  }

  /**
   * Parse forward method to extract signature
   */
  private parseForward(node: ASTNode, module: ParsedModule): void {
    const args: { name: string; type: string }[] = []

    // Get arguments (skip 'self')
    const funcArgs = Array.isArray(node.args) ? node.args : node.args?.args || []
    for (const arg of funcArgs.slice(1)) {
      let argType = "MLXArray"
      if (arg.annotation) {
        const ann = this.unparse(arg.annotation)
        if (ann.includes("Optional")) {
          argType = "MLXArray?"
        }
      }
      if (arg.arg) {
        args.push({ name: arg.arg, type: argType })
      }
    }

    module.methods.push({
      name: "forward",
      swiftName: "callAsFunction",
      args,
      body: [],
      returnType: "MLXArray"
    })
  }

  /**
   * Infer Swift type from an AST node
   */
  private inferType(node: ASTNode): string {
    // Check nodeType for constants
    if (node.nodeType === "Constant" || node.value !== undefined) {
      const val = node.value
      if (typeof val === "boolean") {
        return "Bool"
      }
      if (typeof val === "number") {
        return Number.isInteger(val) ? "Int" : "Float"
      }
    }
    return "Any"
  }

  /**
   * Convert AST node back to string representation
   * (simplified - handles common cases)
   */
  private unparse(node: ASTNode): string {
    if (node.id) {
      return node.id
    }
    if (node.attr && node.value) {
      return `${this.unparse(node.value)}.${node.attr}`
    }
    if (typeof node.value === "string") {
      return `"${node.value}"`
    }
    if (typeof node.value === "number" || typeof node.value === "boolean") {
      return String(node.value)
    }
    if (node.func) {
      const funcName = this.getCallName(node)
      const args = Array.isArray(node.args) ? node.args : node.args?.args || []
      const argStrs = args.map((a: ASTNode) => this.unparse(a))
      if (node.keywords) {
        for (const kw of node.keywords) {
          if (kw.arg && kw.value) {
            argStrs.push(`${kw.arg}=${this.unparse(kw.value)}`)
          }
        }
      }
      return `${funcName}(${argStrs.join(", ")})`
    }
    return "unknown"
  }
}
