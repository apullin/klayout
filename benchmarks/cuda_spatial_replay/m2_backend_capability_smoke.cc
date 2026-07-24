/*
 * Loader-level regression for the additive M2 symbol.  An old ABI-v1 DSO
 * with the M1 entry point must not advertise M2 capability, while the current
 * DSO must expose both profile-specific entry points.
 */

#include "dbCudaSpatialApi.h"

#include <cstdint>
#include <iostream>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace
{

struct Symbols
{
  klayout_cuda_spatial_abi_version_func version;
  klayout_cuda_spatial_run_m1_width_space_empty_v1_func run_m1;
  klayout_cuda_spatial_run_m2_width_space_empty_v1_func run_m2;
};

#if defined(_WIN32)
Symbols load_symbols (const char *path, HMODULE &handle)
{
  handle = LoadLibraryA (path);
  if (! handle) {
    return Symbols { 0, 0, 0 };
  }
  return Symbols {
    reinterpret_cast<klayout_cuda_spatial_abi_version_func> (
      GetProcAddress (handle, "klayout_cuda_spatial_abi_version")),
    reinterpret_cast<klayout_cuda_spatial_run_m1_width_space_empty_v1_func> (
      GetProcAddress (
        handle, "klayout_cuda_spatial_run_m1_width_space_empty_v1")),
    reinterpret_cast<klayout_cuda_spatial_run_m2_width_space_empty_v1_func> (
      GetProcAddress (
        handle, "klayout_cuda_spatial_run_m2_width_space_empty_v1"))
  };
}
#else
Symbols load_symbols (const char *path, void *&handle)
{
  handle = dlopen (path, RTLD_NOW | RTLD_LOCAL);
  if (! handle) {
    return Symbols { 0, 0, 0 };
  }
  return Symbols {
    reinterpret_cast<klayout_cuda_spatial_abi_version_func> (
      dlsym (handle, "klayout_cuda_spatial_abi_version")),
    reinterpret_cast<klayout_cuda_spatial_run_m1_width_space_empty_v1_func> (
      dlsym (handle, "klayout_cuda_spatial_run_m1_width_space_empty_v1")),
    reinterpret_cast<klayout_cuda_spatial_run_m2_width_space_empty_v1_func> (
      dlsym (handle, "klayout_cuda_spatial_run_m2_width_space_empty_v1"))
  };
}
#endif

bool abi1_with_m1 (const Symbols &symbols)
{
  return symbols.version &&
         symbols.version () == KLAYOUT_CUDA_SPATIAL_ABI_VERSION &&
         symbols.run_m1;
}

} // anonymous namespace

int main (int argc, char **argv)
{
  if (argc != 3) {
    std::cerr << "usage: m2_backend_capability_smoke LEGACY_DSO CURRENT_DSO\n";
    return 2;
  }

#if defined(_WIN32)
  HMODULE legacy_handle = 0;
  HMODULE current_handle = 0;
#else
  void *legacy_handle = 0;
  void *current_handle = 0;
#endif
  const Symbols legacy = load_symbols (argv [1], legacy_handle);
  const Symbols current = load_symbols (argv [2], current_handle);
  const bool good =
    abi1_with_m1 (legacy) && ! legacy.run_m2 &&
    abi1_with_m1 (current) && current.run_m2;

#if defined(_WIN32)
  if (legacy_handle) FreeLibrary (legacy_handle);
  if (current_handle) FreeLibrary (current_handle);
#else
  if (legacy_handle) dlclose (legacy_handle);
  if (current_handle) dlclose (current_handle);
#endif

  if (! good) {
    std::cerr
      << "M2 capability-symbol gate failed: legacy(m1="
      << bool (legacy.run_m1) << ",m2=" << bool (legacy.run_m2)
      << ") current(m1=" << bool (current.run_m1)
      << ",m2=" << bool (current.run_m2) << ")\n";
    return 1;
  }
  std::cout
    << "M2 capability-symbol gate passed: ABI-v1 legacy DSO is M1-only; "
       "current DSO exposes independent M1 and M2 entries\n";
  return 0;
}
