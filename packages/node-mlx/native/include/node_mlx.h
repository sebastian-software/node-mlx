#ifndef NODE_MLX_H
#define NODE_MLX_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Load a model from HuggingFace ID or local path
// Returns model handle (>0) on success, -1 on error
int32_t node_mlx_load_model(const char* model_id);

// Unload a model from memory
void node_mlx_unload_model(int32_t handle);

// Generate text from a prompt
// Returns JSON string - caller must free with node_mlx_free_string
// JSON format: {"success":bool,"text":string,"tokenCount":int,"tokensPerSecond":float,"error":string}
char* node_mlx_generate(
  int32_t handle,
  const char* prompt,
  int32_t max_tokens,
  float temperature,
  float top_p
);

// Free a string allocated by this library
void node_mlx_free_string(char* str);

// Check if MLX is available (Apple Silicon macOS)
bool node_mlx_is_available(void);

// Get library version - caller must free with node_mlx_free_string
char* node_mlx_version(void);

#ifdef __cplusplus
}
#endif

#endif // NODE_MLX_H
