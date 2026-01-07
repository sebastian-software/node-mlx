import ArgumentParser
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

@main
struct LLMCli: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "llm-cli",
    abstract: "LLM inference CLI powered by MLX",
    subcommands: [Generate.self, Chat.self, Models.self],
    defaultSubcommand: Generate.self
  )
}

// MARK: - Generate Command

struct Generate: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    abstract: "Generate text from a prompt"
  )

  @Option(name: .shortAndLong, help: "Model name or HuggingFace path")
  var model: String = "mlx-community/gemma-3n-E2B-it-4bit"

  @Option(name: .shortAndLong, help: "Input prompt")
  var prompt: String

  @Option(name: .long, help: "Maximum tokens to generate")
  var maxTokens: Int = 256

  @Option(name: .long, help: "Temperature for sampling")
  var temperature: Float = 0.7

  @Option(name: .long, help: "Top-p sampling")
  var topP: Float = 0.9

  @Flag(name: .long, help: "Output as JSON (for Node.js integration)")
  var json: Bool = false

  mutating func run() async throws {
    let isJson = json
    let maxTok = maxTokens
    let userPrompt = prompt

    // Load model
    let modelContainer = try await LLMModelFactory.shared.loadContainer(
      configuration: ModelConfiguration(id: model)
    )

    let generateParameters = GenerateParameters(
      temperature: temperature,
      topP: topP
    )

    var tokens: [String] = []
    var tokenCount = 0

    // Generate
    let result = try await modelContainer.perform { context in
      let input = try await context.processor.prepare(input: .init(prompt: userPrompt))
      return try MLXLMCommon.generate(
        input: input,
        parameters: generateParameters,
        context: context
      ) { (token: Int) -> GenerateDisposition in
        let tokenString = context.tokenizer.decode(tokens: [token])
        if isJson {
          tokens.append(tokenString)
        } else {
          print(tokenString, terminator: "")
          fflush(stdout)
        }
        tokenCount += 1
        return tokenCount < maxTok ? .more : .stop
      }
    }

    if isJson {
      let output = JSONOutput(
        text: tokens.joined(),
        generatedTokens: tokenCount,
        tokensPerSecond: result.tokensPerSecond
      )
      let encoder = JSONEncoder()
      encoder.outputFormatting = .prettyPrinted
      let data = try encoder.encode(output)
      print(String(data: data, encoding: .utf8)!)
    } else {
      print()  // Final newline
      fputs(
        "\n[Generated: \(tokenCount) tokens, Speed: \(String(format: "%.1f", result.tokensPerSecond)) tok/s]\n",
        stderr
      )
    }
  }
}

// MARK: - Chat Command

struct Chat: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    abstract: "Interactive chat mode"
  )

  @Option(name: .shortAndLong, help: "Model name or HuggingFace path")
  var model: String = "mlx-community/gemma-3n-E2B-it-4bit"

  @Option(name: .long, help: "System prompt")
  var system: String?

  @Option(name: .long, help: "Maximum tokens per response")
  var maxTokens: Int = 512

  @Option(name: .long, help: "Temperature for sampling")
  var temperature: Float = 0.7

  mutating func run() async throws {
    let maxTok = maxTokens

    print("Loading model: \(model)...")

    let modelContainer = try await LLMModelFactory.shared.loadContainer(
      configuration: ModelConfiguration(id: model)
    )

    print("Model loaded. Type 'quit' to exit.\n")

    let generateParameters = GenerateParameters(
      temperature: temperature
    )

    var messages: [[String: String]] = []

    if let system = system {
      messages.append(["role": "system", "content": system])
    }

    while true {
      print("> ", terminator: "")
      fflush(stdout)

      guard let line = readLine(), !line.isEmpty else {
        continue
      }

      if line.lowercased() == "quit" || line.lowercased() == "exit" {
        print("Goodbye!")
        break
      }

      messages.append(["role": "user", "content": line])

      var tokenCount = 0
      var responseTokens: [String] = []
      let currentMessages = messages

      let _ = try await modelContainer.perform { context in
        let input = try await context.processor.prepare(input: .init(messages: currentMessages))
        return try MLXLMCommon.generate(
          input: input,
          parameters: generateParameters,
          context: context
        ) { (token: Int) -> GenerateDisposition in
          let tokenString = context.tokenizer.decode(tokens: [token])
          print(tokenString, terminator: "")
          fflush(stdout)
          responseTokens.append(tokenString)
          tokenCount += 1
          return tokenCount < maxTok ? .more : .stop
        }
      }

      print("\n")

      messages.append(["role": "assistant", "content": responseTokens.joined()])
    }
  }
}

// MARK: - Models Command

struct Models: ParsableCommand {
  static let configuration = CommandConfiguration(
    abstract: "List recommended models"
  )

  func run() {
    print(
      """
      Recommended models for node-mlx:

      Gemma 3n (Google):
        mlx-community/gemma-3n-E2B-it-4bit     (2B params, ~1.5GB)
        mlx-community/gemma-3n-E4B-it-4bit     (4B params, ~2.5GB)

      Gemma 3 (Google):
        mlx-community/gemma-3-4b-it-4bit       (4B params, ~2.5GB)
        mlx-community/gemma-3-12b-it-4bit      (12B params, ~7GB)

      Qwen 3 (Alibaba):
        mlx-community/Qwen3-1.7B-4bit          (1.7B params, ~1GB)
        mlx-community/Qwen3-4B-4bit            (4B params, ~2.5GB)

      Phi 4 (Microsoft):
        mlx-community/phi-4-4bit               (14B params, ~8GB)

      Llama 4 (Meta):
        mlx-community/Llama-4-Scout-17B-4bit   (17B params, ~10GB)

      Usage:
        llm-cli generate --model <model-id> --prompt "Hello"
        llm-cli chat --model <model-id>
      """
    )
  }
}

// MARK: - JSON Output

struct JSONOutput: Codable {
  let text: String
  let generatedTokens: Int
  let tokensPerSecond: Double
}
