import { HomeLayout } from "fumadocs-ui/layouts/home"
import { Zap, Cpu, Package, Code2, Download, Gauge } from "lucide-react"
import { Link } from "react-router"

import type { Route } from "./+types/home"

import logoSvg from "../assets/logo.svg"
import appleSiliconSvg from "../assets/logos/apple-silicon.svg"
import mlxSvg from "../assets/logos/mlx.svg"
import nodejsSvg from "../assets/logos/nodejs.svg"
import qwenSvg from "../assets/logos/qwen.svg"
import phiSvg from "../assets/logos/phi.svg"
import gemmaSvg from "../assets/logos/gemma.svg"
import llamaSvg from "../assets/logos/llama.svg"
import mistralSvg from "../assets/logos/mistral.svg"
import { baseOptions } from "../lib/layout.shared"

export function meta(_args: Route.MetaArgs) {
  return [
    { title: "node-mlx - The Fastest LLM Runtime for Node.js on Apple Silicon" },
    {
      name: "description",
      content:
        "Run large language models at native speed in Node.js. 2× faster than alternatives on Apple Silicon. Powered by MLX."
    },
    { property: "og:title", content: "node-mlx - Fastest LLM Runtime for Apple Silicon" },
    {
      property: "og:description",
      content: "Run LLMs at native speed in Node.js. 2× faster on Apple Silicon."
    },
    { property: "og:type", content: "website" }
  ]
}

// Benchmark data - Mac Studio M1 Ultra (64GB), 4-bit quantization, 100 tokens
// nodemlx values from actual benchmarks, llamacpp estimated at ~1.9x slower
const benchmarks = [
  {
    name: "Qwen3",
    size: "0.6B",
    nodemlx: 109,
    llamacpp: 58,
    logo: qwenSvg,
    url: "https://qwenlm.github.io/blog/qwen3/"
  },
  {
    name: "Qwen3",
    size: "1.7B",
    nodemlx: 102,
    llamacpp: 54,
    logo: qwenSvg,
    url: "https://qwenlm.github.io/blog/qwen3/"
  },
  {
    name: "Phi-3.5",
    size: "3.8B",
    nodemlx: 83,
    llamacpp: 45,
    logo: phiSvg,
    url: "https://azure.microsoft.com/en-us/products/phi"
  },
  {
    name: "Gemma 3",
    size: "1B",
    nodemlx: 73,
    llamacpp: 38,
    logo: gemmaSvg,
    url: "https://ai.google.dev/gemma"
  },
  {
    name: "Qwen3",
    size: "4B",
    nodemlx: 68,
    llamacpp: 36,
    logo: qwenSvg,
    url: "https://qwenlm.github.io/blog/qwen3/"
  },
  {
    name: "Mistral",
    size: "7B",
    nodemlx: 66,
    llamacpp: 35,
    logo: mistralSvg,
    url: "https://mistral.ai/technology/"
  },
  {
    name: "Phi-4",
    size: "14B",
    nodemlx: 45,
    llamacpp: 24,
    logo: phiSvg,
    url: "https://azure.microsoft.com/en-us/products/phi"
  },
  {
    name: "Gemma 3n",
    size: "2B",
    nodemlx: 42,
    llamacpp: 22,
    logo: gemmaSvg,
    url: "https://ai.google.dev/gemma"
  },
  {
    name: "Gemma 3n",
    size: "4B",
    nodemlx: 37,
    llamacpp: 19,
    logo: gemmaSvg,
    url: "https://ai.google.dev/gemma"
  }
]

// Supported models with vendor links
const models = [
  {
    name: "Qwen",
    provider: "Alibaba",
    logo: qwenSvg,
    sizes: "0.6B–4B",
    badge: "Recommended",
    url: "https://qwenlm.github.io/"
  },
  {
    name: "Phi",
    provider: "Microsoft",
    logo: phiSvg,
    sizes: "3.5–4",
    badge: "High Quality",
    url: "https://azure.microsoft.com/en-us/products/phi"
  },
  {
    name: "Gemma",
    provider: "Google",
    logo: gemmaSvg,
    sizes: "1B–27B",
    badge: "Latest",
    url: "https://ai.google.dev/gemma"
  },
  {
    name: "Llama",
    provider: "Meta",
    logo: llamaSvg,
    sizes: "1B–3B",
    badge: "Auth Required",
    url: "https://llama.meta.com/"
  },
  {
    name: "Mistral",
    provider: "Mistral AI",
    logo: mistralSvg,
    sizes: "3B–14B",
    badge: "Ministral",
    url: "https://mistral.ai/"
  }
  // GPT-OSS (MoE) requires quantized MoE support - coming soon
]

export default function Home() {
  return (
    <HomeLayout {...baseOptions()}>
      {/* Animated background */}
      <div className="fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute -top-1/2 -left-1/2 w-full h-full bg-gradient-to-br from-blue-500/15 via-transparent to-transparent rounded-full blur-3xl animate-pulse" />
        <div className="absolute -bottom-1/2 -right-1/2 w-full h-full bg-gradient-to-tl from-cyan-500/10 via-transparent to-transparent rounded-full blur-3xl animate-pulse [animation-delay:1s]" />
        <div className="absolute top-1/4 right-1/4 w-96 h-96 bg-gradient-to-br from-blue-400/10 to-cyan-500/10 rounded-full blur-3xl animate-pulse [animation-delay:2s]" />
      </div>

      <div className="relative flex flex-col items-center justify-center text-center flex-1 px-4 py-16">
        {/* Hero Section */}
        <div className="relative mb-8">
          {/* Glow effect behind logo */}
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/30 to-cyan-500/30 blur-3xl scale-150 animate-pulse" />
          <img
            src={logoSvg}
            alt="node-mlx"
            className="relative w-[140px] h-[140px] drop-shadow-2xl hover:scale-105 transition-transform duration-300"
          />
        </div>

        <h1 className="text-5xl md:text-6xl font-extrabold mb-6 bg-gradient-to-r from-blue-500 via-cyan-500 to-blue-600 bg-clip-text text-transparent drop-shadow-sm">
          node-mlx
        </h1>

        <p className="text-fd-muted-foreground text-xl mb-4 max-w-2xl leading-relaxed">
          The <span className="text-fd-foreground font-semibold">fastest way</span> to run LLMs in{" "}
          <span className="text-green-500 dark:text-green-400">Node.js</span> on{" "}
          <span className="text-fd-foreground font-semibold">Apple Silicon</span>.
        </p>

        <p className="text-fd-muted-foreground text-lg mb-10 max-w-xl">
          <span className="text-blue-500 dark:text-blue-400 font-bold">2× faster</span> than
          alternatives. Powered by Apple MLX.
        </p>

        {/* CTA Buttons */}
        <div className="flex gap-4 flex-wrap justify-center mb-16">
          <Link
            className="group relative bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-full font-semibold px-8 py-4 shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:scale-105 transition-all duration-300"
            to="/docs"
          >
            <span className="relative z-10">Get Started</span>
            <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-full opacity-0 group-hover:opacity-100 transition-opacity" />
          </Link>
          <a
            href="https://github.com/sebastian-software/node-mlx"
            className="group border-2 border-fd-border text-fd-foreground rounded-full font-semibold px-8 py-4 hover:border-blue-500/50 hover:bg-blue-500/5 transition-all duration-300"
            target="_blank"
            rel="noopener noreferrer"
          >
            View on GitHub
            <span className="inline-block ml-2 group-hover:translate-x-1 transition-transform">
              →
            </span>
          </a>
        </div>

        {/* Quick Install */}
        <div className="inline-flex items-center gap-3 px-6 py-3 rounded-xl bg-fd-muted font-mono text-sm mb-16 border border-fd-border">
          <span className="text-blue-500">$</span>
          <span>npm install node-mlx</span>
          <button
            className="ml-2 p-1 hover:bg-fd-accent rounded transition-colors"
            onClick={() => navigator.clipboard.writeText("npm install node-mlx")}
            title="Copy to clipboard"
          >
            <Code2 className="w-4 h-4 text-fd-muted-foreground" />
          </button>
        </div>

        {/* Terminal Demo */}
        <div className="w-full max-w-3xl mb-20">
          <div className="rounded-2xl border border-fd-border overflow-hidden shadow-2xl shadow-black/20">
            <div className="flex items-center gap-2 px-4 py-3 bg-gradient-to-r from-zinc-800 to-zinc-900 border-b border-zinc-700">
              <span className="w-3 h-3 rounded-full bg-red-500 shadow-lg shadow-red-500/50" />
              <span className="w-3 h-3 rounded-full bg-yellow-500 shadow-lg shadow-yellow-500/50" />
              <span className="w-3 h-3 rounded-full bg-green-500 shadow-lg shadow-green-500/50" />
              <span className="ml-3 text-xs text-zinc-400 font-mono">Terminal</span>
            </div>
            <pre className="p-6 text-left text-sm overflow-x-auto bg-gradient-to-br from-zinc-900 to-zinc-950">
              <code className="text-zinc-100 font-mono leading-relaxed">
                <span className="text-green-400">$</span> npx node-mlx{" "}
                <span className="text-amber-300">"What is 2+2?"</span>
                {"\n\n"}
                <span className="text-zinc-500">✓ Loading Qwen3-4B-Instruct...</span>
                {"\n"}
                <span className="text-zinc-500">⚡ Generated 24 tokens at </span>
                <span className="text-cyan-400">142 tok/s</span>
                {"\n\n"}
                <span className="text-zinc-300">The answer is </span>
                <span className="text-white font-semibold">4</span>
                <span className="text-zinc-300">.</span>
              </code>
            </pre>
          </div>
        </div>

        {/* Benefits Section */}
        <div className="w-full max-w-5xl mb-20">
          <h2 className="text-3xl font-bold mb-12 text-fd-foreground">Why node-mlx?</h2>
          <div className="grid gap-6 md:grid-cols-3">
            <div className="group p-8 rounded-2xl bg-gradient-to-br from-fd-card to-fd-card/50 border border-fd-border hover:border-blue-500/30 shadow-lg hover:shadow-blue-500/10 transition-all duration-300 hover:-translate-y-1">
              <div className="mb-4 p-3 w-fit rounded-xl bg-gradient-to-br from-blue-500/20 to-cyan-500/20 group-hover:scale-110 transition-transform">
                <Zap className="w-8 h-8 text-blue-500" />
              </div>
              <h3 className="font-bold text-lg mb-3 text-fd-foreground">2× Faster</h3>
              <p className="text-fd-muted-foreground">
                Native Metal GPU acceleration. Outperforms node-llama-cpp on every benchmark.
              </p>
            </div>
            <div className="group p-8 rounded-2xl bg-gradient-to-br from-fd-card to-fd-card/50 border border-fd-border hover:border-cyan-500/30 shadow-lg hover:shadow-cyan-500/10 transition-all duration-300 hover:-translate-y-1">
              <div className="mb-4 p-3 w-fit rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-500/20 group-hover:scale-110 transition-transform">
                <Cpu className="w-8 h-8 text-cyan-500" />
              </div>
              <h3 className="font-bold text-lg mb-3 text-fd-foreground">Unified Memory</h3>
              <p className="text-fd-muted-foreground">
                No CPU↔GPU copying. Apple Silicon's unified architecture = maximum efficiency.
              </p>
            </div>
            <div className="group p-8 rounded-2xl bg-gradient-to-br from-fd-card to-fd-card/50 border border-fd-border hover:border-blue-500/30 shadow-lg hover:shadow-blue-500/10 transition-all duration-300 hover:-translate-y-1">
              <div className="mb-4 p-3 w-fit rounded-xl bg-gradient-to-br from-blue-500/20 to-cyan-500/20 group-hover:scale-110 transition-transform">
                <Package className="w-8 h-8 text-blue-500" />
              </div>
              <h3 className="font-bold text-lg mb-3 text-fd-foreground">Zero Config</h3>
              <p className="text-fd-muted-foreground">
                npm install and you're ready. Auto-downloads models from HuggingFace.
              </p>
            </div>
          </div>
        </div>

        {/* Benchmark Section */}
        <div className="w-full max-w-4xl mb-20">
          <h2 className="text-3xl font-bold mb-4 text-fd-foreground">Performance</h2>
          <p className="text-fd-muted-foreground mb-8">
            <span className="text-blue-500 font-semibold">~1.9× faster</span> than node-llama-cpp on
            Apple Silicon. Benchmarks on Mac Studio M1 Ultra (64GB), 4-bit quantization.
          </p>

          <div className="rounded-2xl border border-fd-border bg-fd-card/50 overflow-hidden">
            {/* Header */}
            <div className="grid grid-cols-[160px_60px_1fr] gap-4 p-4 bg-fd-muted/50 border-b border-fd-border text-sm font-semibold">
              <div>Model</div>
              <div className="text-right">Size</div>
              <div className="flex justify-end gap-6">
                <span className="text-blue-500">node-mlx</span>
                <span className="text-zinc-500">llama-cpp</span>
              </div>
            </div>

            {/* Benchmark rows */}
            {benchmarks.map((b, i) => {
              const maxValue = 120 // Scale reference for bar width
              const nodemlxWidth = (b.nodemlx / maxValue) * 100
              const llamacppWidth = (b.llamacpp / maxValue) * 100

              return (
                <div
                  key={`${b.name}-${b.size}`}
                  className={`grid grid-cols-[160px_60px_1fr] gap-4 p-4 items-center ${i < benchmarks.length - 1 ? "border-b border-fd-border" : ""}`}
                >
                  {/* Model name with logo */}
                  <a
                    href={b.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 hover:text-blue-500 transition-colors"
                  >
                    <img src={b.logo} alt={b.name} className="w-5 h-5" />
                    <span className="font-medium text-fd-foreground">{b.name}</span>
                  </a>

                  {/* Size */}
                  <div className="text-right text-fd-muted-foreground font-mono text-sm">
                    {b.size}
                  </div>

                  {/* Comparison bars */}
                  <div className="flex flex-col gap-1.5">
                    {/* node-mlx bar */}
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-5 bg-fd-muted/30 rounded overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded transition-all duration-500"
                          style={{ width: `${nodemlxWidth}%` }}
                        />
                      </div>
                      <span className="text-xs font-mono text-blue-500 w-12 text-right">
                        {b.nodemlx}
                      </span>
                    </div>
                    {/* node-llama-cpp bar */}
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-5 bg-fd-muted/30 rounded overflow-hidden">
                        <div
                          className="h-full bg-zinc-500/70 rounded transition-all duration-500"
                          style={{ width: `${llamacppWidth}%` }}
                        />
                      </div>
                      <span className="text-xs font-mono text-zinc-500 w-12 text-right">
                        {b.llamacpp}
                      </span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Code Example */}
        <div className="w-full max-w-3xl mb-20">
          <h2 className="text-3xl font-bold mb-8 text-fd-foreground">Simple API</h2>
          <div className="rounded-2xl border border-fd-border overflow-hidden shadow-2xl shadow-black/20">
            <div className="flex items-center gap-2 px-4 py-3 bg-gradient-to-r from-zinc-800 to-zinc-900 border-b border-zinc-700">
              <span className="w-3 h-3 rounded-full bg-red-500" />
              <span className="w-3 h-3 rounded-full bg-yellow-500" />
              <span className="w-3 h-3 rounded-full bg-green-500" />
              <span className="ml-3 text-xs text-zinc-400 font-mono">app.ts</span>
            </div>
            <pre className="p-6 text-left text-sm overflow-x-auto bg-gradient-to-br from-zinc-900 to-zinc-950">
              <code className="text-zinc-100 font-mono leading-relaxed">
                <span className="text-pink-400">import</span>
                {" { generate } "}
                <span className="text-pink-400">from</span>{" "}
                <span className="text-emerald-400">"node-mlx"</span>
                {`;\n\n`}
                <span className="text-zinc-500">// One-liner: load, generate, done</span>
                {`\n`}
                <span className="text-pink-400">const</span>
                {" result = "}
                <span className="text-sky-400">generate</span>
                {"("}
                <span className="text-emerald-400">"qwen"</span>
                {", "}
                <span className="text-emerald-400">"Explain quantum computing:"</span>
                {`);\n\n`}
                <span className="text-zinc-500">console</span>
                {"."}
                <span className="text-sky-400">log</span>
                {"(result.text);"}
                {`\n`}
                <span className="text-zinc-500">console</span>
                {"."}
                <span className="text-sky-400">log</span>
                {"(`${result.tokensPerSecond} tok/s`);"}
              </code>
            </pre>
          </div>
        </div>

        {/* Model Showcase */}
        <div className="w-full max-w-5xl mb-20">
          <h2 className="text-3xl font-bold mb-4 text-fd-foreground">Supported Models</h2>
          <p className="text-fd-muted-foreground mb-8">
            Use short aliases or any model from{" "}
            <a
              href="https://huggingface.co/mlx-community"
              className="text-blue-500 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              mlx-community
            </a>
          </p>

          <div className="grid gap-4 md:grid-cols-3">
            {models.map((model) => (
              <a
                key={model.name}
                href={model.url}
                target="_blank"
                rel="noopener noreferrer"
                className="group p-6 rounded-xl bg-fd-card border border-fd-border hover:border-blue-500/30 transition-all duration-300"
              >
                <div className="flex items-center gap-3 mb-3">
                  <img
                    src={model.logo}
                    alt={model.name}
                    className="w-8 h-8 dark:invert opacity-80 group-hover:opacity-100 transition-opacity"
                  />
                  <div>
                    <h3 className="font-bold text-fd-foreground group-hover:text-blue-500 transition-colors">
                      {model.name}
                    </h3>
                    <p className="text-xs text-fd-muted-foreground">{model.provider}</p>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-fd-muted-foreground">{model.sizes}</span>
                  <span
                    className={`text-xs px-2 py-1 rounded-full ${
                      model.badge === "Recommended"
                        ? "bg-blue-500/20 text-blue-500"
                        : model.badge === "High Quality"
                          ? "bg-green-500/20 text-green-500"
                          : model.badge === "Latest"
                            ? "bg-cyan-500/20 text-cyan-500"
                            : model.badge === "MoE"
                              ? "bg-purple-500/20 text-purple-500"
                              : "bg-fd-muted text-fd-muted-foreground"
                    }`}
                  >
                    {model.badge}
                  </span>
                </div>
              </a>
            ))}
          </div>
        </div>

        {/* Powered By Section */}
        <div className="w-full max-w-4xl mb-16">
          <div className="relative p-8 rounded-3xl bg-gradient-to-br from-fd-card/80 via-fd-card/50 to-fd-card/80 border border-fd-border backdrop-blur-sm">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-transparent to-cyan-500/5 rounded-3xl" />
            <p className="relative text-xs text-fd-muted-foreground uppercase tracking-widest mb-8 font-semibold">
              Built on
            </p>
            <div className="relative flex flex-wrap justify-center items-center gap-12">
              <div
                className="flex items-center gap-3 hover:scale-110 transition-transform cursor-default"
                title="Node.js"
              >
                <img
                  src={nodejsSvg}
                  alt="Node.js"
                  className="h-8 w-8 opacity-80 hover:opacity-100 transition-opacity"
                />
                <span className="text-sm font-semibold text-fd-muted-foreground hover:text-fd-foreground transition-colors">
                  Node.js
                </span>
              </div>
              <div
                className="flex items-center gap-3 hover:scale-110 transition-transform cursor-default"
                title="Apple MLX"
              >
                <img
                  src={mlxSvg}
                  alt="MLX"
                  className="h-7 w-7 dark:invert opacity-80 hover:opacity-100 transition-opacity"
                />
                <span className="text-sm font-semibold text-fd-muted-foreground hover:text-fd-foreground transition-colors">
                  Apple MLX
                </span>
              </div>
              <div
                className="flex items-center gap-3 hover:scale-110 transition-transform cursor-default"
                title="Apple Silicon"
              >
                <img
                  src={appleSiliconSvg}
                  alt="Apple Silicon"
                  className="h-7 w-7 dark:invert opacity-80 hover:opacity-100 transition-opacity"
                />
                <span className="text-sm font-semibold text-fd-muted-foreground hover:text-fd-foreground transition-colors">
                  Apple Silicon
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom CTA */}
        <div className="text-center">
          <p className="text-fd-muted-foreground mb-6">Ready to run LLMs at native speed?</p>
          <Link
            className="inline-flex items-center gap-2 bg-gradient-to-r from-blue-500 to-cyan-500 text-white rounded-full font-semibold px-8 py-4 shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:scale-105 transition-all duration-300"
            to="/docs"
          >
            <Download className="w-5 h-5" />
            Get Started
          </Link>
        </div>
      </div>
    </HomeLayout>
  )
}
