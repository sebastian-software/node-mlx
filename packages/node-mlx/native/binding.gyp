{
  "targets": [
    {
      "target_name": "node_mlx",
      "sources": ["src/binding.cc"],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "include"
      ],
      "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS"],
      "cflags!": ["-fno-exceptions"],
      "cflags_cc!": ["-fno-exceptions"],
      "conditions": [
        [
          "OS=='mac'",
          {
            "xcode_settings": {
              "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
              "CLANG_CXX_LIBRARY": "libc++",
              "MACOSX_DEPLOYMENT_TARGET": "14.0",
              "OTHER_CFLAGS": ["-arch arm64"],
              "OTHER_LDFLAGS": ["-arch arm64"]
            }
          }
        ]
      ]
    }
  ]
}

