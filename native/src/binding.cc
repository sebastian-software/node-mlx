#include <napi.h>
#include <dlfcn.h>
#include <string>
#include "../include/node_mlx.h"

// Function pointers for dynamic loading
static void* dylib_handle = nullptr;

typedef int32_t (*LoadModelFn)(const char*);
typedef void (*UnloadModelFn)(int32_t);
typedef char* (*GenerateFn)(int32_t, const char*, int32_t, float, float);
typedef void (*FreeStringFn)(char*);
typedef bool (*IsAvailableFn)(void);
typedef char* (*GetVersionFn)(void);

static LoadModelFn fn_load_model = nullptr;
static UnloadModelFn fn_unload_model = nullptr;
static GenerateFn fn_generate = nullptr;
static FreeStringFn fn_free_string = nullptr;
static IsAvailableFn fn_is_available = nullptr;
static GetVersionFn fn_get_version = nullptr;

// Initialize the library
Napi::Value Initialize(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  
  if (dylib_handle) {
    return Napi::Boolean::New(env, true);
  }
  
  // Get dylib path from argument or use default
  std::string dylibPath;
  
  if (info.Length() > 0 && info[0].IsString()) {
    dylibPath = info[0].As<Napi::String>().Utf8Value();
  } else {
    Napi::Error::New(env, "dylibPath argument required").ThrowAsJavaScriptException();
    return Napi::Boolean::New(env, false);
  }
  
  dylib_handle = dlopen(dylibPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!dylib_handle) {
    std::string error = std::string("Failed to load dylib at ") + dylibPath + ": " + dlerror();
    Napi::Error::New(env, error).ThrowAsJavaScriptException();
    return Napi::Boolean::New(env, false);
  }
  
  // Load function pointers
  fn_load_model = (LoadModelFn)dlsym(dylib_handle, "node_mlx_load_model");
  fn_unload_model = (UnloadModelFn)dlsym(dylib_handle, "node_mlx_unload_model");
  fn_generate = (GenerateFn)dlsym(dylib_handle, "node_mlx_generate");
  fn_free_string = (FreeStringFn)dlsym(dylib_handle, "node_mlx_free_string");
  fn_is_available = (IsAvailableFn)dlsym(dylib_handle, "node_mlx_is_available");
  fn_get_version = (GetVersionFn)dlsym(dylib_handle, "node_mlx_version");
  
  if (!fn_load_model || !fn_generate || !fn_free_string) {
    std::string missing;
    if (!fn_load_model) missing += "node_mlx_load_model ";
    if (!fn_generate) missing += "node_mlx_generate ";
    if (!fn_free_string) missing += "node_mlx_free_string ";
    
    Napi::Error::New(env, "Failed to load functions: " + missing).ThrowAsJavaScriptException();
    dlclose(dylib_handle);
    dylib_handle = nullptr;
    return Napi::Boolean::New(env, false);
  }
  
  return Napi::Boolean::New(env, true);
}

// Load a model
Napi::Value LoadModel(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  
  if (!fn_load_model) {
    Napi::Error::New(env, "Library not initialized. Call initialize() first.").ThrowAsJavaScriptException();
    return env.Null();
  }
  
  if (info.Length() < 1 || !info[0].IsString()) {
    Napi::TypeError::New(env, "Model ID string required").ThrowAsJavaScriptException();
    return env.Null();
  }
  
  std::string modelId = info[0].As<Napi::String>().Utf8Value();
  int32_t handle = fn_load_model(modelId.c_str());
  
  if (handle < 0) {
    Napi::Error::New(env, "Failed to load model: " + modelId).ThrowAsJavaScriptException();
    return env.Null();
  }
  
  return Napi::Number::New(env, handle);
}

// Unload a model
Napi::Value UnloadModel(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  
  if (!fn_unload_model) {
    Napi::Error::New(env, "Library not initialized").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  
  if (info.Length() < 1 || !info[0].IsNumber()) {
    Napi::TypeError::New(env, "Model handle number required").ThrowAsJavaScriptException();
    return env.Undefined();
  }
  
  int32_t handle = info[0].As<Napi::Number>().Int32Value();
  fn_unload_model(handle);
  
  return env.Undefined();
}

// Generate text - returns JSON string
Napi::Value Generate(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  
  if (!fn_generate) {
    Napi::Error::New(env, "Library not initialized").ThrowAsJavaScriptException();
    return env.Null();
  }
  
  if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsString()) {
    Napi::TypeError::New(env, "Usage: generate(handle, prompt, options?)").ThrowAsJavaScriptException();
    return env.Null();
  }
  
  int32_t handle = info[0].As<Napi::Number>().Int32Value();
  std::string prompt = info[1].As<Napi::String>().Utf8Value();
  
  // Default options
  int32_t maxTokens = 256;
  float temperature = 0.7f;
  float topP = 0.9f;
  
  // Parse options object if provided
  if (info.Length() > 2 && info[2].IsObject()) {
    Napi::Object options = info[2].As<Napi::Object>();
    
    if (options.Has("maxTokens")) {
      maxTokens = options.Get("maxTokens").As<Napi::Number>().Int32Value();
    }
    if (options.Has("temperature")) {
      temperature = options.Get("temperature").As<Napi::Number>().FloatValue();
    }
    if (options.Has("topP")) {
      topP = options.Get("topP").As<Napi::Number>().FloatValue();
    }
  }
  
  char* jsonResult = fn_generate(handle, prompt.c_str(), maxTokens, temperature, topP);
  
  if (!jsonResult) {
    Napi::Error::New(env, "Generate returned null").ThrowAsJavaScriptException();
    return env.Null();
  }
  
  std::string jsonStr(jsonResult);
  fn_free_string(jsonResult);
  
  // Return the JSON string - let JavaScript parse it
  return Napi::String::New(env, jsonStr);
}

// Check if MLX is available
Napi::Value IsAvailable(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  
  if (fn_is_available) {
    return Napi::Boolean::New(env, fn_is_available());
  }
  
  // Fallback check
  #if defined(__APPLE__) && defined(__arm64__)
    return Napi::Boolean::New(env, true);
  #else
    return Napi::Boolean::New(env, false);
  #endif
}

// Get version
Napi::Value GetVersion(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  
  if (fn_get_version) {
    char* version = fn_get_version();
    if (version) {
      std::string versionStr(version);
      fn_free_string(version);
      return Napi::String::New(env, versionStr);
    }
  }
  
  return Napi::String::New(env, "0.1.0");
}

// Check if library is initialized
Napi::Value IsInitialized(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return Napi::Boolean::New(env, dylib_handle != nullptr);
}

// Module initialization
Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set("initialize", Napi::Function::New(env, Initialize));
  exports.Set("isInitialized", Napi::Function::New(env, IsInitialized));
  exports.Set("loadModel", Napi::Function::New(env, LoadModel));
  exports.Set("unloadModel", Napi::Function::New(env, UnloadModel));
  exports.Set("generate", Napi::Function::New(env, Generate));
  exports.Set("isAvailable", Napi::Function::New(env, IsAvailable));
  exports.Set("getVersion", Napi::Function::New(env, GetVersion));
  
  return exports;
}

NODE_API_MODULE(node_mlx, Init)
