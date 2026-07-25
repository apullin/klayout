#include "m2_manhattan_production_loader.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv)
{
  if (argc != 3) {
    std::cerr << "usage: m2_manhattan_production_loader_smoke "
                 "SCENE.kact SCENE_SHA256\n";
    return 2;
  }
  try {
    klayout_cuda::m2_production::LoadOptions options;
    options.expected_scene_sha256 = argv[2];
    options.expected_flat_polygons = UINT64_C(22945976);
    options.expected_flat_rectangles = UINT64_C(22946444);
    const auto scene =
        klayout_cuda::m2_production::load_kact_templates(argv[1], options);
    if (scene.contexts.size() != 587201 ||
        scene.m2_contexts.size() != 568632 ||
        scene.rectangles.size() != 45966 ||
        scene.local_polygons != 45960 || scene.local_l_shapes != 6 ||
        scene.flat_l_shapes != 468) {
      throw std::runtime_error("qualified compact-scene census mismatch");
    }
    std::cout << "M2_PRODUCTION_LOADER_SMOKE"
              << " verdict=COMPLETE"
              << " scene_sha256=" << scene.scene_sha256
              << " contexts=" << scene.contexts.size()
              << " m2_contexts=" << scene.m2_contexts.size()
              << " templates=" << scene.rectangles.size()
              << " flat_polygons=" << scene.flat_polygons
              << " flat_rectangles=" << scene.flat_rectangles << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "M2_PRODUCTION_LOADER_SMOKE"
              << " verdict=UNCERTAIN error=\"" << error.what() << "\"\n";
    return 2;
  }
}
