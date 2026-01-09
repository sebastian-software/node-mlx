# Changelog

# [2.0.0](https://github.com/sebastian-software/node-mlx/compare/v1.0.6...v2.0.0) (2026-01-09)

### Bug Fixes

- enable repetition penalty by default (1.1) ([0a65ffe](https://github.com/sebastian-software/node-mlx/commit/0a65ffea9873048e54c8745c6c7274d28ebfae70))
- **gemma3n:** implement KV-sharing for shared attention layers ([6f809de](https://github.com/sebastian-software/node-mlx/commit/6f809de47fd0b90087a32f5aeb736206e1d52fa8))
- **gemma:** enforce chat template for all Gemma models ([cd73f96](https://github.com/sebastian-software/node-mlx/commit/cd73f96d7ea67f412761979fc3bbf6825bdec2bd))
- **hf2swift:** fix RoPE and transposition order in generator ([88dc173](https://github.com/sebastian-software/node-mlx/commit/88dc173cdd5c4b35b4d20aad82941dc591e1ab0d))
- **hf2swift:** use cache.state for KV-sharing instead of sharedKV param ([dd60c24](https://github.com/sebastian-software/node-mlx/commit/dd60c24dd74578de66882095025e7fe057992110))

### Features

- **phi:** add fused QKV and gate-up projection support for Phi3/Phi4 ([296dd8c](https://github.com/sebastian-software/node-mlx/commit/296dd8c062cf823ab0920cf92eb14c2252278a73))
- **qwen3:** add Qwen3 model support with Q/K norms ([a636578](https://github.com/sebastian-software/node-mlx/commit/a63657868a272187b2f8c545e009802a82cb1415))

### BREAKING CHANGES

- **phi:** Phi3 architecture now uses fused projections matching mlx-lm

* Add hasFusedQKV and hasFusedGateUp feature flags to generator
* Generate fused qkv_proj with split for Q/K/V instead of separate projections
* Generate fused gate_up_proj with split for gate/up instead of separate projections
* Always generate config struct even without JSON input
* Add model ID alias resolution in loadModel() for RECOMMENDED_MODELS
* Update Phi models in RECOMMENDED_MODELS to use mlx-community 4bit variants

Tested models:

- mlx-community/Phi-3-mini-4k-instruct-4bit ✓
- mlx-community/Phi-3.5-mini-instruct-4bit ✓
- Qwen/Qwen2.5-1.5B-Instruct ✓
- mlx-community/gemma-3-1b-it-4bit ✓
- mlx-community/gemma-3n-E4B-it-lm-4bit ✓

## [1.0.6](https://github.com/sebastian-software/node-mlx/compare/v1.0.5...v1.0.6) (2026-01-09)

### Bug Fixes

- **gemma3n:** use asLinear for quantized embedding logits ([97d65b8](https://github.com/sebastian-software/node-mlx/commit/97d65b8649c39611dde851bbcba56926a881b618))

## [1.0.5](https://github.com/sebastian-software/node-mlx/compare/v1.0.3...v1.0.5) (2026-01-09)

### Bug Fixes

- auto-retry swift build with clean .build on failure ([e67a217](https://github.com/sebastian-software/node-mlx/commit/e67a2175b2f873bdee13aeb0a912da978e5cb0e3))
- **gemma3n:** use -lm variants (language model only) ([5e6d006](https://github.com/sebastian-software/node-mlx/commit/5e6d006cc093d5c69ce238d3d584ddb03f099807))

### Performance Improvements

- strip debug symbols from dylib to hide local paths ([a08b21c](https://github.com/sebastian-software/node-mlx/commit/a08b21ce26516601e6a7338c468284fc38dbceba))

## [1.0.3](https://github.com/sebastian-software/node-mlx/compare/v1.0.2...v1.0.3) (2026-01-09)

### Bug Fixes

- **gemma3n:** fix weight sanitization for embed_tokens ([165da1d](https://github.com/sebastian-software/node-mlx/commit/165da1dc699046da1c827821f29cbb693943e558))

## [1.0.2](https://github.com/sebastian-software/node-mlx/compare/v1.0.1...v1.0.2) (2026-01-09)

### Bug Fixes

- add gemma-3n to RECOMMENDED_MODELS ([c64306b](https://github.com/sebastian-software/node-mlx/commit/c64306b4b023a85b54a190dd5c1425a075eb7211))

## [1.0.1](https://github.com/sebastian-software/node-mlx/compare/v1.0.0...v1.0.1) (2026-01-09)

### Bug Fixes

- **qwen2:** add attention_bias and mlp_bias support ([0173919](https://github.com/sebastian-software/node-mlx/commit/01739192eb96c5f1a812c9c881be13708a73a05a))

# 1.0.0 (2026-01-09)

### Bug Fixes

- add image token to vlm prompts for correct image processing ([4577eeb](https://github.com/sebastian-software/node-mlx/commit/4577eeb4b022adc0bebdae35beac4222e5e5ecae))
- align CLI header box formatting ([af78cc9](https://github.com/sebastian-software/node-mlx/commit/af78cc9876f2e3ea6031b81e1ab90700cecde357))
- **ci:** add test:coverage task to turbo.json ([9e4679e](https://github.com/sebastian-software/node-mlx/commit/9e4679e17d2b573498a0818fb4bbe7985b6e7215))
- **ci:** correct paths for monorepo structure ([6279896](https://github.com/sebastian-software/node-mlx/commit/6279896b3987d735a5506788078b8ac3c9f97844))
- **ci:** use isPlatformSupported instead of isSupported ([6b60e0a](https://github.com/sebastian-software/node-mlx/commit/6b60e0a47c6e8d79efbca9461aa971ded4a2e6bb))
- correct benchmark system configuration ([031ec92](https://github.com/sebastian-software/node-mlx/commit/031ec922aeb52247e1fe3c96fa91edc646360960))
- correct CLI header box alignment (38 chars) ([ae77135](https://github.com/sebastian-software/node-mlx/commit/ae771352297be44e60da58b16a88c8f83b3e6874))
- **gemma3:** improve config parsing and weight sanitization ([52e4979](https://github.com/sebastian-software/node-mlx/commit/52e4979d192288b3d24a519ca5ca2e263c82cd5c))
- **gemma3n:** add sanitize function for weight prefix removal ([f668de7](https://github.com/sebastian-software/node-mlx/commit/f668de7597bbe0174318e5c1de28b904b80a4b1d))
- **gemma3n:** correct model generation with proper KV cache handling ([89e2e18](https://github.com/sebastian-software/node-mlx/commit/89e2e18fb96e621c1b15add1ee2f240122a103dc))
- **hf2swift:** resolve all ESLint errors ([acec749](https://github.com/sebastian-software/node-mlx/commit/acec7499b3a92fe62882fa3bc912279065929822))
- hide special tokens like <|end|> in output ([5774663](https://github.com/sebastian-software/node-mlx/commit/5774663c7df78481f5553ed57330d083880d691b))
- improve error message when native libraries are missing ([9eada87](https://github.com/sebastian-software/node-mlx/commit/9eada870a1f4a918d74a0e16d343be697457e144))
- improve native library path resolution and bundle loading ([128f6bf](https://github.com/sebastian-software/node-mlx/commit/128f6bfdaff61d734f5ebd40629199dcb24ea4d4))
- insert image token id directly instead of relying on tokenizer ([fb806a5](https://github.com/sebastian-software/node-mlx/commit/fb806a50d44bc215ccc7c3e65167da16397beec1))
- llama model with RoPE and KV cache support ([d6c9d5f](https://github.com/sebastian-software/node-mlx/commit/d6c9d5fbf0b46e5d178d61c35e206bcedfb21142))
- load EOS token from model config.json as fallback ([289d6aa](https://github.com/sebastian-software/node-mlx/commit/289d6aa2800c82ecbd35ab29c20a03cc68f3e1db))
- phi3 fused QKV/MLP projections and RoPE support ([419299e](https://github.com/sebastian-software/node-mlx/commit/419299e4fda8737a09c508337f42348a4dee792a))
- set DYLD_FRAMEWORK_PATH for MLX metallib loading ([7681a8d](https://github.com/sebastian-software/node-mlx/commit/7681a8d8232ccd46b0fec2a7f08b8e8f4a327621))
- **swift:** upgrade to Swift 6.0 tools version ([8898a7b](https://github.com/sebastian-software/node-mlx/commit/8898a7bc30e9ded095c1e5f9c1b2af988e8546d9))
- update recommended models to working ones ([a450514](https://github.com/sebastian-software/node-mlx/commit/a450514547558132c121d002db14e678ba3c3838))
- vlm support for gemma 3 vision models ([553813c](https://github.com/sebastian-software/node-mlx/commit/553813c851242eb939503904d476f63035caa555))

### Code Refactoring

- reorganize to monorepo structure ([13cc85a](https://github.com/sebastian-software/node-mlx/commit/13cc85a2b3d1359008a730a8e161a39578d291a9))
- restructure as proper monorepo with publishable node-mlx package ([6c088f3](https://github.com/sebastian-software/node-mlx/commit/6c088f3adc6efc2e4879551841918364a6d8797f))

### Features

- add benchmark script for node-mlx vs node-llama-cpp ([5e552ce](https://github.com/sebastian-software/node-mlx/commit/5e552ce95f6e125be34b0affa5708c9276dc83ba))
- add chat template support and fix metallib loading ([8c09380](https://github.com/sebastian-software/node-mlx/commit/8c09380532573476e6d442618f180178205a759d))
- add dev script to root package.json ([45fbdf4](https://github.com/sebastian-software/node-mlx/commit/45fbdf40a7ac36c66d1b0ad203d925ad557241d2))
- add Gemma 3 support ([4ea6551](https://github.com/sebastian-software/node-mlx/commit/4ea65512534f712dbdcc2f3f22fe270490dc42c4))
- add Gemma 3 Vision-Language Model (VLM) support ([c45afb5](https://github.com/sebastian-software/node-mlx/commit/c45afb5970badd642dd94bafb3ccd6e564c0b71f))
- add Gemma 3n support via forked mlx-swift-lm ([16ec1c5](https://github.com/sebastian-software/node-mlx/commit/16ec1c5aed30cbd4acb4e03d074b46c4447d56b8)), closes [#46](https://github.com/sebastian-software/node-mlx/issues/46)
- add hf2swift prototype - Python→Swift model generator ([93efaf2](https://github.com/sebastian-software/node-mlx/commit/93efaf2a2660ac85f03c80ba338d235d635a7fda))
- add interactive CLI for node-mlx ([c187a2e](https://github.com/sebastian-software/node-mlx/commit/c187a2e1874c083d415993ab03c5af8583236794))
- add model aliases and improve CLI model list ([3342f38](https://github.com/sebastian-software/node-mlx/commit/3342f3877ef102c01b3407d510cd8e78cfe3c30c))
- add phi-4 to recommended models ([a4bc800](https://github.com/sebastian-software/node-mlx/commit/a4bc8002fa7dc124176015351281a33ff7794ff1))
- add phi3 to recommended models and update defaults ([866d641](https://github.com/sebastian-software/node-mlx/commit/866d64187fe7839bf5b44b6196b397b104771574))
- add repetition_penalty parameter to prevent token repetition ([fe759e0](https://github.com/sebastian-software/node-mlx/commit/fe759e05136198387383a87e7d11876985885297))
- add Turborepo for monorepo task orchestration ([90fad6c](https://github.com/sebastian-software/node-mlx/commit/90fad6cdf061ad4f45aa3a49aa2f82d6fde14235))
- **cli:** add streaming output for real-time token display ([740dc9d](https://github.com/sebastian-software/node-mlx/commit/740dc9d205707c8133be226a127bb249e2f6710a))
- **core:** add model loader, generation, and KV cache ([582a87f](https://github.com/sebastian-software/node-mlx/commit/582a87fff52f07ffbd1c78063cbb9b712bd2972c))
- **core:** add Qwen2 model and fix optional fields ([20be1e3](https://github.com/sebastian-software/node-mlx/commit/20be1e3895d8c4ad1d393554af3faf740f1fafcd))
- **core:** add standalone tokenizer via swift-transformers ([917c2a3](https://github.com/sebastian-software/node-mlx/commit/917c2a36607f8a03b4697aa67fc84dc9a6a2108c))
- **core:** implement LLMEngine integration ([744a1f7](https://github.com/sebastian-software/node-mlx/commit/744a1f72d5f5c4430e08b1340829e089532b2b05))
- **gemma3:** add Gemma 3 1B model support ([3979257](https://github.com/sebastian-software/node-mlx/commit/3979257f4560910543b634125813b26b6f789962))
- **gemma3:** add linear RoPE scaling for 4B+ models ([934822b](https://github.com/sebastian-software/node-mlx/commit/934822b89a191dc25a492a618ffa0d1c9418f9cc))
- **gemma3:** full Gemma 3 4B+ VLM support ([848d284](https://github.com/sebastian-software/node-mlx/commit/848d2847dfedc419e6944a27d487497138ab1ce9))
- **gemma3:** improve 4B config inference from weight shapes ([6f19a8d](https://github.com/sebastian-software/node-mlx/commit/6f19a8d86d59b2306f69247a5e9b74743c7b5f3b))
- **generator:** add v6 generator with parsed module support ([5d2a1ed](https://github.com/sebastian-software/node-mlx/commit/5d2a1eda8f729689819b7165727e615fc53c8468))
- **generator:** add VLM support for nested configs (Gemma3n) ([b8e974b](https://github.com/sebastian-software/node-mlx/commit/b8e974bc39a87517ed42263998bed087ca420356))
- **generator:** hf2swift v5 generates compilable Swift models ([0d3824f](https://github.com/sebastian-software/node-mlx/commit/0d3824f2d1aa318852758120af1d27464684732d))
- **generator:** v5 fixes model name conversion ([149df6b](https://github.com/sebastian-software/node-mlx/commit/149df6bcf15d25546608115db8acf4ea2997f9e5))
- **hf2swift:** add Gemma3n and GPT-OSS generation examples ([3728107](https://github.com/sebastian-software/node-mlx/commit/3728107c4bbbbddc09eb74f6dadebf560f7e25cb))
- **hf2swift:** add Gemma3n feature support with conditionals ([07c5fc7](https://github.com/sebastian-software/node-mlx/commit/07c5fc7933c545edde86f2c535c51abd303b83c9))
- **hf2swift:** complete AltUp model generation for Gemma3n ([1074863](https://github.com/sebastian-software/node-mlx/commit/107486318c33a546a90771fa91911be1d577583e))
- **hf2swift:** generate all model files, fix metallib loading ([68c697a](https://github.com/sebastian-software/node-mlx/commit/68c697a285b70bb2901ed883cd9948b04e42b9c6))
- **hf2swift:** improve generator with better config parsing ([4bf7f5d](https://github.com/sebastian-software/node-mlx/commit/4bf7f5da0cb723e40e5ce5961f4fffb97c6cf1da))
- **hf2swift:** migrate generator from Python to TypeScript ([832bfc6](https://github.com/sebastian-software/node-mlx/commit/832bfc62d166cbb57cfc8ea0671d1e0efc13b067))
- **hf2swift:** production-quality Swift code generation ([3720bd6](https://github.com/sebastian-software/node-mlx/commit/3720bd65fa2592acc773b30f2a0fbc7d9467b590))
- **hf2swift:** replace manual Gemma3n with generated version ([0c289fd](https://github.com/sebastian-software/node-mlx/commit/0c289fd4dc695608e1378c4f1aa8ce959685ce77))
- **hf2swift:** v4 generator with full model generation ([38ffd09](https://github.com/sebastian-software/node-mlx/commit/38ffd0940a8bbc7e8e82a59530ecbb7a7eed772a))
- implement native Swift-to-Node.js binding ([77cd50e](https://github.com/sebastian-software/node-mlx/commit/77cd50e2ed38a5d159bf5cee8be630d302f2d7e8))
- initial commit - LLM inference for Node.js via Apple MLX ([268a3df](https://github.com/sebastian-software/node-mlx/commit/268a3dfcc547fa310a9d99358832c865e2d13909))
- **node-mlx:** add release-it with Angular changelog and GitHub releases ([670d06f](https://github.com/sebastian-software/node-mlx/commit/670d06fc2bd3ffd97615e7b916d7ec748c9301cb))
- prototype pure MLX implementation (roadmap) ([059f11c](https://github.com/sebastian-software/node-mlx/commit/059f11c0f380e6fd5300315df1f87a0e225a1622))
- **swift:** add helper utilities from mlx-swift-lm ([552f62e](https://github.com/sebastian-software/node-mlx/commit/552f62e207f7424eace62da302a376f3c55815a7))
- **swift:** improve model implementations and quantization ([0e4167e](https://github.com/sebastian-software/node-mlx/commit/0e4167ef92b2f9deba4f88911d8e77b8d8dda784))

### Performance Improvements

- add benchmark scripts for MLX models ([b0f4d9e](https://github.com/sebastian-software/node-mlx/commit/b0f4d9e1b020ed3181265bfc7c5450879ead9dbf))
- add GPT-OSS 20B benchmark - MLX 11.4x faster ([8f586a8](https://github.com/sebastian-software/node-mlx/commit/8f586a8629d37675ffc160a26a6a4f4afd4ec5b3))
- add Ministral 8B benchmark - 101 tok/s, 2x faster ([49e4b32](https://github.com/sebastian-software/node-mlx/commit/49e4b32455baf1809be6a2f556f7465a3ac06726))
- add Qwen3 30B MoE benchmark - MLX 60x faster ([758aec1](https://github.com/sebastian-software/node-mlx/commit/758aec15b004775a8265a39f789784034081fc3d))
- add robust benchmark with statistical analysis ([0940373](https://github.com/sebastian-software/node-mlx/commit/09403733c31d2e53d3a5f9d6b5bafc1b7e969cd6))
- **ci:** add Swift build cache for faster CI runs ([fa9a18d](https://github.com/sebastian-software/node-mlx/commit/fa9a18ddab8b583a91c4ff9823fa2e58b24e0401))
- **ci:** improve Swift build cache with separate Xcode cache ([41b124f](https://github.com/sebastian-software/node-mlx/commit/41b124fa873e3c762a336bd9a384281ac64e4abe))
- phi-4 benchmark shows node-mlx 1.9x faster than llama-cpp ([f16131c](https://github.com/sebastian-software/node-mlx/commit/f16131c8d14abd8e269d238f6f62e01cd2b07e71))
- use identical 4-bit quantization for fair benchmark ([0419c45](https://github.com/sebastian-software/node-mlx/commit/0419c4507f91538b6727d8e071d52226b11f2a4e))

### BREAKING CHANGES

- node-mlx is now a workspace package in packages/node-mlx/

Changes:

- Move tsconfig, tsup, vitest configs to packages/node-mlx/
- Create packages/node-mlx/package.json with npm metadata
- Convert root package.json to private workspace root
- Add prebuildify + node-gyp-build for prebuilt binaries
- Update Swift copy script to copy artifacts to node-mlx/swift/
- Update index.ts to load binaries from new paths
- Add eslint config for node-mlx package

Publishing:

- Run 'pnpm prebuildify' to create prebuilds for Node 20/22/24
- Run 'pnpm build:swift' to build Swift and copy to node-mlx/swift/
- Package includes prebuilds/, swift/, and dist/

* Directory structure changed to monorepo layout

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/).
