/**
 * Naming convention utilities for Python → Swift conversion
 */

/**
 * Convert snake_case to camelCase
 */
export function toCamel(name: string): string {
  const parts = name.split("_")
  return parts[0] + parts.slice(1).map(capitalize).join("")
}

/**
 * Convert snake_case to PascalCase
 */
export function toPascal(name: string): string {
  // Special cases - handle both gpt_oss and gptoss
  const lower = name.toLowerCase()
  if (lower === "gpt_oss" || lower === "gptoss" || lower === "gpt-oss") {
    return "GptOSS"
  }

  const parts = name.replace(/-/g, "_").split("_")
  return parts.map(capitalize).join("")
}

/**
 * Capitalize first letter of a string
 */
export function capitalize(s: string): string {
  if (s.length === 0) {
    return s
  }
  return s[0].toUpperCase() + s.slice(1)
}

/**
 * Convert a Python expression to Swift
 */
export function convertExpr(expr: string, conversions: [RegExp, string][]): string {
  let result = expr
  for (const [pattern, replacement] of conversions) {
    result = result.replace(pattern, replacement)
  }
  return result
}
