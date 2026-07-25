cmake_minimum_required(VERSION 3.24)

project(klayout_cuda_spatial_replay LANGUAGES CXX CUDA)

find_package(Threads REQUIRED)

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

add_executable(manhattan_union_replay
  manhattan_union_replay.cu
)
target_compile_options(manhattan_union_replay PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
)
set_target_properties(manhattan_union_replay PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

add_library(manhattan_union_gpu_core STATIC
  manhattan_union_gpu.cu
)
target_compile_options(manhattan_union_gpu_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
)
set_target_properties(manhattan_union_gpu_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  POSITION_INDEPENDENT_CODE ON
)

add_executable(manhattan_union_gpu_core_smoke
  manhattan_union_gpu_core_smoke.cu
)
target_compile_options(manhattan_union_gpu_core_smoke PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
)
set_target_properties(manhattan_union_gpu_core_smoke PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)
target_link_libraries(manhattan_union_gpu_core_smoke PRIVATE
  manhattan_union_gpu_core
)

add_executable(m2_via2_certificate
  m2_via2_certificate.cu
)
target_compile_options(m2_via2_certificate PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
)
set_target_properties(m2_via2_certificate PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

add_library(klayout_cuda_spatial_backend SHARED
  m2_union_backend.cu
  poly34_backend.cu
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
target_link_libraries(klayout_cuda_spatial_backend PRIVATE
  manhattan_union_gpu_core
)

add_executable(cuda_spatial_backend_smoke backend_smoke.cc)
target_include_directories(cuda_spatial_backend_smoke PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
)
target_link_libraries(cuda_spatial_backend_smoke PRIVATE
  klayout_cuda_spatial_backend
)

add_executable(poly34_backend_smoke poly34_backend_smoke.cc)
target_include_directories(poly34_backend_smoke PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
)
target_link_libraries(poly34_backend_smoke PRIVATE
  klayout_cuda_spatial_backend
)

add_library(klayout_cuda_spatial_legacy_m1_backend SHARED
  legacy_m1_width_space_backend.cc
)
target_include_directories(klayout_cuda_spatial_legacy_m1_backend PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
)
target_compile_definitions(klayout_cuda_spatial_legacy_m1_backend PRIVATE
  KLAYOUT_CUDA_SPATIAL_BACKEND_BUILD
)
set_target_properties(klayout_cuda_spatial_legacy_m1_backend PROPERTIES
  CXX_VISIBILITY_PRESET hidden
)

foreach(M2_UNION_FAKE_VARIANT IN ITEMS full run_only release_only)
  add_library(m2_union_contract_backend_${M2_UNION_FAKE_VARIANT} SHARED
    m2_union_contract_fake_backend.cc
  )
  target_include_directories(
    m2_union_contract_backend_${M2_UNION_FAKE_VARIANT} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
  )
  target_compile_definitions(
    m2_union_contract_backend_${M2_UNION_FAKE_VARIANT} PRIVATE
    KLAYOUT_CUDA_SPATIAL_BACKEND_BUILD
  )
  set_target_properties(
    m2_union_contract_backend_${M2_UNION_FAKE_VARIANT} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
  )
endforeach()
target_compile_definitions(m2_union_contract_backend_full PRIVATE
  KLAYOUT_M2_UNION_FAKE_WITH_RUN=1
  KLAYOUT_M2_UNION_FAKE_WITH_RELEASE=1
)
target_compile_definitions(m2_union_contract_backend_run_only PRIVATE
  KLAYOUT_M2_UNION_FAKE_WITH_RUN=1
  KLAYOUT_M2_UNION_FAKE_WITH_RELEASE=0
)
target_compile_definitions(m2_union_contract_backend_release_only PRIVATE
  KLAYOUT_M2_UNION_FAKE_WITH_RUN=0
  KLAYOUT_M2_UNION_FAKE_WITH_RELEASE=1
)

add_executable(m2_backend_capability_smoke
  m2_backend_capability_smoke.cc
)
target_include_directories(m2_backend_capability_smoke PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
)
if(CMAKE_DL_LIBS)
  target_link_libraries(m2_backend_capability_smoke PRIVATE
    ${CMAKE_DL_LIBS}
  )
endif()

add_executable(via1_stack_backend_smoke via1_stack_backend_smoke.cc)
target_include_directories(via1_stack_backend_smoke PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
)
target_link_libraries(via1_stack_backend_smoke PRIVATE
  klayout_cuda_spatial_backend
)

add_executable(m2_union_production_backend_smoke
  m2_union_production_backend_smoke.cc
)
target_include_directories(m2_union_production_backend_smoke PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
)
target_compile_options(m2_union_production_backend_smoke PRIVATE
  -O3 -Wall -Wextra
)
target_link_libraries(m2_union_production_backend_smoke PRIVATE
  klayout_cuda_spatial_backend
)

add_executable(m2_union_production_backend_gate
  m2_union_production_backend_gate.cu
)
target_include_directories(m2_union_production_backend_gate PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
)
target_compile_options(m2_union_production_backend_gate PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
)
set_target_properties(m2_union_production_backend_gate PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)
target_link_libraries(m2_union_production_backend_gate PRIVATE
  klayout_cuda_spatial_backend
  m2_merged_boundary_oracle
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

add_executable(manhattan_contour_validator_test
  manhattan_contour_validator_test.cc
)
target_include_directories(manhattan_contour_validator_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
)
target_compile_options(manhattan_contour_validator_test PRIVATE
  -O3 -Wall -Wextra
)

add_executable(implant12_exact_predicate_test
  implant12_exact_predicate.cu
  implant12_exact_predicate_test.cc
  ../../src/db/db/dbEdgePairRelations.cc
)
target_include_directories(implant12_exact_predicate_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/tl/tl
)
target_compile_options(implant12_exact_predicate_test PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
)
set_target_properties(implant12_exact_predicate_test PROPERTIES
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

add_executable(m2_manhattan_production_census
  m2_manhattan_production_census.cu
)
target_compile_options(m2_manhattan_production_census PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
)
set_target_properties(m2_manhattan_production_census PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

add_library(m2_manhattan_production_loader STATIC
  m2_manhattan_production_census.cu
)
target_compile_definitions(m2_manhattan_production_loader PRIVATE
  KLAYOUT_M2_PRODUCTION_LOADER_NO_MAIN
)
target_compile_options(m2_manhattan_production_loader PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
)
set_target_properties(m2_manhattan_production_loader PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

add_executable(m2_manhattan_production_loader_smoke
  m2_manhattan_production_loader_smoke.cc
)
target_link_libraries(m2_manhattan_production_loader_smoke PRIVATE
  m2_manhattan_production_loader
)
set_target_properties(m2_manhattan_production_loader_smoke PROPERTIES
  LINKER_LANGUAGE CUDA
)

add_library(m2_merged_boundary_oracle STATIC
  m2_merged_boundary_oracle.cc
)
target_include_directories(m2_merged_boundary_oracle PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
)
target_compile_options(m2_merged_boundary_oracle PRIVATE
  -O3 -Wall -Wextra
)

# The flat bridge deliberately exercises KLayout's real stock Region
# morphology.  It is optional because this standalone CUDA benchmark project
# does not normally know where a qmake KLayout build placed its shared
# libraries.
set(KLAYOUT_QMAKE_BUILD_DIR "" CACHE PATH
  "KLayout qmake build directory containing libklayout_db/tl/gsi")
if(KLAYOUT_QMAKE_BUILD_DIR)
  find_library(KLAYOUT_DB_LIBRARY
    NAMES klayout_db
    PATHS "${KLAYOUT_QMAKE_BUILD_DIR}"
    NO_DEFAULT_PATH REQUIRED)
  find_library(KLAYOUT_TL_LIBRARY
    NAMES klayout_tl
    PATHS "${KLAYOUT_QMAKE_BUILD_DIR}"
    NO_DEFAULT_PATH REQUIRED)
  find_library(KLAYOUT_GSI_LIBRARY
    NAMES klayout_gsi
    PATHS "${KLAYOUT_QMAKE_BUILD_DIR}"
    NO_DEFAULT_PATH REQUIRED)

  add_executable(m2_flat_region_bridge
    m2_flat_region_bridge.cc
  )
  target_include_directories(m2_flat_region_bridge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/tl/tl
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gsi/gsi
  )
  target_compile_options(m2_flat_region_bridge PRIVATE
    -O3 -Wall -Wextra
  )
  target_link_libraries(m2_flat_region_bridge PRIVATE
    m2_merged_boundary_oracle
    "${KLAYOUT_DB_LIBRARY}"
    "${KLAYOUT_TL_LIBRARY}"
    "${KLAYOUT_GSI_LIBRARY}"
    Threads::Threads
  )
  set_target_properties(m2_flat_region_bridge PROPERTIES
    BUILD_RPATH "${KLAYOUT_QMAKE_BUILD_DIR}"
  )
endif()

target_link_libraries(manhattan_union_replay PRIVATE
  manhattan_union_gpu_core
  m2_manhattan_production_loader
  m2_merged_boundary_oracle
  Threads::Threads
)

add_library(m2_resident_morphology_gpu STATIC
  m2_resident_morphology_gpu.cu
)
target_compile_options(m2_resident_morphology_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
)
target_link_libraries(m2_resident_morphology_gpu PUBLIC
  manhattan_union_gpu_core
)
set_target_properties(m2_resident_morphology_gpu PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
  POSITION_INDEPENDENT_CODE ON
)

add_executable(m2_resident_morphology_replay
  m2_resident_morphology_replay.cc
)
target_compile_options(m2_resident_morphology_replay PRIVATE
  -O3 -Wall -Wextra
)
target_link_libraries(m2_resident_morphology_replay PRIVATE
  m2_resident_morphology_gpu
  manhattan_union_gpu_core
  m2_manhattan_production_loader
  m2_merged_boundary_oracle
  Threads::Threads
)
set_target_properties(m2_resident_morphology_replay PROPERTIES
  LINKER_LANGUAGE CUDA
)

add_executable(m2_merged_boundary_oracle_cli
  m2_merged_boundary_oracle_main.cc
)
target_link_libraries(m2_merged_boundary_oracle_cli PRIVATE
  m2_merged_boundary_oracle
)
target_include_directories(m2_merged_boundary_oracle_cli PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
)
target_compile_options(m2_merged_boundary_oracle_cli PRIVATE
  -O3 -Wall -Wextra
)

add_executable(m2_merged_boundary_oracle_test
  m2_merged_boundary_oracle_test.cc
)
target_link_libraries(m2_merged_boundary_oracle_test PRIVATE
  m2_merged_boundary_oracle
)
target_compile_options(m2_merged_boundary_oracle_test PRIVATE
  -O3 -Wall -Wextra
)

# This loader-level test compiles the modified host wrapper itself and links
# its existing digest/log dependencies from a qmake KLayout build.  The fake
# DSOs above need no CUDA device and deliberately exercise malformed output
# and missing-capability paths.
if(KLAYOUT_QMAKE_BUILD_DIR)
  add_executable(m2_union_backend_contract_smoke
    m2_union_backend_contract_smoke.cc
    ../../src/db/db/dbCudaSpatialBackend.cc
  )
  target_include_directories(m2_union_backend_contract_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/tl/tl
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gsi/gsi
  )
  target_compile_options(m2_union_backend_contract_smoke PRIVATE
    -O2 -Wall -Wextra
  )
  target_link_libraries(m2_union_backend_contract_smoke PRIVATE
    "${KLAYOUT_DB_LIBRARY}"
    "${KLAYOUT_TL_LIBRARY}"
    "${KLAYOUT_GSI_LIBRARY}"
    Threads::Threads
    ${CMAKE_DL_LIBS}
  )
  set_target_properties(m2_union_backend_contract_smoke PROPERTIES
    BUILD_RPATH "${KLAYOUT_QMAKE_BUILD_DIR}"
  )

  add_custom_target(m2_union_backend_contract_gate
    COMMAND m2_union_backend_contract_smoke
      $<TARGET_FILE:m2_union_contract_backend_run_only>
      capability unavailable
    COMMAND m2_union_backend_contract_smoke
      $<TARGET_FILE:m2_union_contract_backend_release_only>
      capability unavailable
    COMMAND m2_union_backend_contract_smoke
      $<TARGET_FILE:m2_union_contract_backend_full>
      ok complete
    COMMAND m2_union_backend_contract_smoke
      $<TARGET_FILE:m2_union_contract_backend_full>
      side_order complete
    COMMAND m2_union_backend_contract_smoke
      $<TARGET_FILE:m2_union_contract_backend_full>
      fallback fallback
    COMMAND m2_union_backend_contract_smoke
      $<TARGET_FILE:m2_union_contract_backend_full>
      backend_throw error
    COMMAND m2_union_backend_contract_smoke
      $<TARGET_FILE:m2_union_contract_backend_full>
      bad_echo invalid
    COMMAND m2_union_backend_contract_smoke
      $<TARGET_FILE:m2_union_contract_backend_full>
      bad_order invalid
    COMMAND m2_union_backend_contract_smoke
      $<TARGET_FILE:m2_union_contract_backend_full>
      bad_digest invalid
    COMMAND m2_union_backend_contract_smoke
      $<TARGET_FILE:m2_union_contract_backend_full>
      bad_count invalid
    COMMAND m2_union_backend_contract_smoke
      $<TARGET_FILE:m2_union_contract_backend_full>
      copy_throw error
    DEPENDS
      m2_union_backend_contract_smoke
      m2_union_contract_backend_full
      m2_union_contract_backend_run_only
      m2_union_contract_backend_release_only
    VERBATIM
  )

  add_executable(m2_union_live_transaction_smoke
    m2_union_live_transaction_smoke.cc
  )
  target_include_directories(m2_union_live_transaction_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/tl/tl
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gsi/gsi
  )
  target_compile_options(m2_union_live_transaction_smoke PRIVATE
    -O2 -Wall -Wextra
  )
  target_link_libraries(m2_union_live_transaction_smoke PRIVATE
    "${KLAYOUT_DB_LIBRARY}"
    "${KLAYOUT_TL_LIBRARY}"
    "${KLAYOUT_GSI_LIBRARY}"
    Threads::Threads
    ${CMAKE_DL_LIBS}
  )
  set_target_properties(m2_union_live_transaction_smoke PROPERTIES
    BUILD_RPATH "${KLAYOUT_QMAKE_BUILD_DIR}"
  )

  add_executable(m2_union_real_transaction_gate
    m2_union_real_transaction_gate.cu
  )
  target_include_directories(m2_union_real_transaction_gate PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/db/db
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/tl/tl
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/gsi/gsi
  )
  target_compile_options(m2_union_real_transaction_gate PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3;-Wall;-Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--expt-relaxed-constexpr;-Xcompiler=-Wall,-Wextra>
  )
  # nvcc contributes its (older) CUDA host compiler directory to the final
  # C++ link.  Keep the CMake C++ driver's runtime first: the qmake libraries
  # were built by that driver and may require its newer CXXABI symbols.
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libstdc++.so
    OUTPUT_VARIABLE KLAYOUT_CXX_STDLIB
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(EXISTS "${KLAYOUT_CXX_STDLIB}")
    get_filename_component(
      KLAYOUT_CXX_STDLIB_DIR "${KLAYOUT_CXX_STDLIB}" DIRECTORY)
    target_link_options(m2_union_real_transaction_gate BEFORE PRIVATE
      "-L${KLAYOUT_CXX_STDLIB_DIR}"
    )
  endif()
  target_link_libraries(m2_union_real_transaction_gate PRIVATE
    m2_merged_boundary_oracle
    "${KLAYOUT_DB_LIBRARY}"
    "${KLAYOUT_TL_LIBRARY}"
    "${KLAYOUT_GSI_LIBRARY}"
    Threads::Threads
    ${CMAKE_DL_LIBS}
  )
  add_dependencies(m2_union_real_transaction_gate
    klayout_cuda_spatial_backend
  )
  set_target_properties(m2_union_real_transaction_gate PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    BUILD_RPATH "${KLAYOUT_QMAKE_BUILD_DIR}"
  )

  add_custom_target(m2_union_live_transaction_gate
    COMMAND m2_union_live_transaction_smoke
      $<TARGET_FILE:m2_union_contract_backend_run_only>
      capability disabled
    COMMAND m2_union_live_transaction_smoke
      $<TARGET_FILE:m2_union_contract_backend_release_only>
      capability disabled
    COMMAND m2_union_live_transaction_smoke
      $<TARGET_FILE:m2_union_contract_backend_full>
      invalid_scene host
    COMMAND m2_union_live_transaction_smoke
      $<TARGET_FILE:m2_union_contract_backend_full>
      live_caps complete
    COMMAND m2_union_live_transaction_smoke
      $<TARGET_FILE:m2_union_contract_backend_full>
      fallback fallback
    COMMAND m2_union_live_transaction_smoke
      $<TARGET_FILE:m2_union_contract_backend_full>
      bad_order invalid
    COMMAND m2_union_live_transaction_smoke
      $<TARGET_FILE:m2_union_contract_backend_full>
      side_order topology
    COMMAND m2_union_live_transaction_smoke
      $<TARGET_FILE:m2_union_contract_backend_full>
      backend_throw error
    DEPENDS
      m2_union_live_transaction_smoke
      m2_union_contract_backend_full
      m2_union_contract_backend_run_only
      m2_union_contract_backend_release_only
    VERBATIM
  )
endif()
