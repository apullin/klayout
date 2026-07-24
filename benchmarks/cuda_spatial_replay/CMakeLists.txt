cmake_minimum_required(VERSION 3.24)

project(klayout_cuda_spatial_replay LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to compile")

add_executable(cuda_spatial_replay spatial_replay.cu)
target_compile_options(cuda_spatial_replay PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
)

set_target_properties(cuda_spatial_replay PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

add_library(klayout_cuda_spatial_backend SHARED
  spatial_backend.cu
  via1_stack_backend.cu
)
target_include_directories(klayout_cuda_spatial_backend PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
)
target_compile_definitions(klayout_cuda_spatial_backend PRIVATE
  KLAYOUT_CUDA_SPATIAL_BACKEND_BUILD
)
target_compile_options(klayout_cuda_spatial_backend PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
)
set_target_properties(klayout_cuda_spatial_backend PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  CXX_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
)

add_executable(cuda_spatial_backend_smoke backend_smoke.cc)
target_include_directories(cuda_spatial_backend_smoke PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
)
target_link_libraries(cuda_spatial_backend_smoke PRIVATE
  klayout_cuda_spatial_backend
)

add_executable(via1_stack_backend_smoke via1_stack_backend_smoke.cc)
target_include_directories(via1_stack_backend_smoke PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
)
target_link_libraries(via1_stack_backend_smoke PRIVATE
  klayout_cuda_spatial_backend
)

add_executable(active3_exact_predicate_test
  active3_exact_predicate.cu
  active3_exact_predicate_test.cc
  ../../src/db/db/dbEdgePairRelations.cc
)
target_include_directories(active3_exact_predicate_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/tl/tl
)
target_compile_options(active3_exact_predicate_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
)
set_target_properties(active3_exact_predicate_test PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

add_executable(m1_width_space_exact_predicate_test
  m1_width_space_exact_predicate.cu
  m1_width_space_exact_predicate_test.cc
  ../../src/db/db/dbEdgePairRelations.cc
)
target_include_directories(m1_width_space_exact_predicate_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/tl/tl
)
target_compile_options(m1_width_space_exact_predicate_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
)
set_target_properties(m1_width_space_exact_predicate_test PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

add_executable(active3_scene_island
  active3_scene_island.cu
)
target_compile_options(active3_scene_island PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
)
set_target_properties(active3_scene_island PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

add_executable(m1_width_space_scene_island
  m1_width_space_scene_island.cu
)
target_compile_options(m1_width_space_scene_island PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
)
set_target_properties(m1_width_space_scene_island PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)
