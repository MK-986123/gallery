cmake_minimum_required(VERSION 3.22.1)

project("gallery-llama" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(NOT DEFINED ANDROID_ABI OR NOT ANDROID_ABI STREQUAL "arm64-v8a")
  message(FATAL_ERROR "Gallery llama.cpp backend currently supports arm64-v8a only")
endif()

set(GGML_SYSTEM_ARCH "ARM")
set(GGML_CPU_KLEIDIAI ON CACHE BOOL "" FORCE)
set(GGML_OPENMP ON CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
  llama
  URL "https://github.com/ggml-org/llama.cpp/archive/60081bb2b5b3294165a4d67c5cbeebe74c868014.tar.gz"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(llama)

add_library(
  ai-chat
  SHARED
  "${llama_SOURCE_DIR}/examples/llama.android/lib/src/main/cpp/ai_chat.cpp"
)

target_compile_definitions(
  ai-chat
  PRIVATE
  GGML_SYSTEM_ARCH=${GGML_SYSTEM_ARCH}
  GGML_CPU_KLEIDIAI=$<BOOL:${GGML_CPU_KLEIDIAI}>
  GGML_OPENMP=$<BOOL:${GGML_OPENMP}>
)

target_include_directories(
  ai-chat
  PRIVATE
  "${llama_SOURCE_DIR}"
  "${llama_SOURCE_DIR}/common"
  "${llama_SOURCE_DIR}/include"
  "${llama_SOURCE_DIR}/ggml/include"
  "${llama_SOURCE_DIR}/ggml/src"
)

target_link_libraries(
  ai-chat
  llama
  llama-common
  android
  log
)
