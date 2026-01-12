# Create Pull Request Description

Generate a concise, benefit-focused PR description in US English.

## Guidelines

### Structure

```markdown
## Summary

[One paragraph explaining WHAT changed and WHY it matters]

## Key Changes

- [Bullet points of significant changes - focus on impact, not implementation details]

## Architecture Decisions

[Only include if there are decisions other contributors should be aware of]

## Breaking Changes

[Only include if there are breaking changes]
```

### Writing Style

- **Be concise**: Every sentence should add value
- **Focus on benefits**: What does this enable? What problem does it solve?
- **Avoid redundancy**: Don't repeat information, don't state the obvious
- **Skip boilerplate**: No "This PR adds...", no test mentions (CI handles that)
- **Use active voice**: "Ports X from Y" not "X was ported from Y"

### What to Include

- Significant architectural changes
- New capabilities or features
- Performance improvements with context
- Migration guidance if needed
- Links to related issues/RFCs

### What to Exclude

- Test coverage details (CI shows this)
- Obvious file changes (reviewers can see the diff)
- Implementation minutiae
- Changelog-style lists of every file touched

## Example

```markdown
## Summary

Switches MLX infrastructure from vendored mlx-swift-lm to direct ports from mlx-lm (Python). This gives us access to the latest model architectures faster, as mlx-lm releases more frequently and has broader model coverage.

## Key Changes

- Direct Python→Swift ports for KVCache, RoPE, and MoE layers
- New `ported/` directory structure with version tracking
- Generator now produces code matching mlx-swift-lm patterns exactly

## Architecture Decisions

**Why port from Python instead of using mlx-swift-lm?**
mlx-lm (Python) is the primary source, updated more frequently, and supports models like Llama 4 MoE that mlx-swift-lm doesn't yet have.

**Directory structure**:

- `generated/models/` - hf2swift generator output
- `ported/` - LLM-assisted ports from Python with git hash tracking
```

## Instructions

1. Analyze the current branch changes using `git log` and `git diff`
2. Read any relevant RFCs or decision documents
3. Generate a PR description following the structure above
4. Keep total length under 500 words
