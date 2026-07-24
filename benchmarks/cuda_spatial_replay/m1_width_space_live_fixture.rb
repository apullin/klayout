# frozen_string_literal: true

# Deterministic hierarchical fixtures for the live METAL1.1/METAL1.2 CUDA
# empty-certificate gate.

include RBA

output = File.expand_path(($output || "").to_s)
raise("missing -rd output=PATH") if output.empty?
raise("refusing to overwrite #{output}") if File.exist?(output) || File.symlink?(output)
raise("output parent does not exist") unless File.directory?(File.dirname(output))

layout = RBA::Layout.new
layout.dbu = 0.0005
metal1 = layout.layer(11, 0)

# The clean case is deliberately hierarchical.  Every rectangle is wider than
# 130 DBU and the 1000-DBU array pitch leaves a 600-DBU spacing.
clean_leaf = layout.create_cell("M1_WIDTH_SPACE_CLEAN_LEAF")
clean_leaf.shapes(metal1).insert(RBA::Box.new(0, 0, 400, 400))

clean_array = layout.create_cell("M1_WIDTH_SPACE_CLEAN_ARRAY")
clean_array.insert(
  RBA::CellInstArray.new(
    clean_leaf.cell_index,
    RBA::Trans.new,
    RBA::Vector.new(1000, 0),
    RBA::Vector.new(0, 1000),
    2,
    2
  )
)

clean = layout.create_cell("M1_WIDTH_SPACE_CLEAN")
clean.insert(
  RBA::CellInstArray.new(
    clean_array.cell_index,
    RBA::Trans.new(RBA::Trans::R90, 5_000, 5_000)
  )
)
clean.insert(
  RBA::CellInstArray.new(
    clean_array.cell_index,
    RBA::Trans.new(RBA::Trans::M90, 15_000, 5_000)
  )
)

# A 120-DBU width is strictly below the qualified 130-DBU limit.  Its long
# axis also makes a MAX_GRID_CELLS=1 decline deterministic in the capacity
# fallback lane.
width_hit = layout.create_cell("M1_WIDTH_SPACE_WIDTH_HIT")
width_hit.shapes(metal1).insert(RBA::Box.new(0, 0, 120, 3000))

# Both rectangles are individually legal, but their 120-DBU axial gap is a
# strict spacing violation.
space_hit = layout.create_cell("M1_WIDTH_SPACE_SPACE_HIT")
space_hit.shapes(metal1).insert(RBA::Box.new(0, 0, 300, 300))
space_hit.shapes(metal1).insert(RBA::Box.new(420, 0, 720, 300))

expected_tops = %w[
  M1_WIDTH_SPACE_CLEAN
  M1_WIDTH_SPACE_SPACE_HIT
  M1_WIDTH_SPACE_WIDTH_HIT
].sort
actual_tops = layout.top_cells.map(&:name).sort
raise("unexpected top cells: #{actual_tops.join(',')}") unless actual_tops == expected_tops

layout.write(output)
puts("M1_WIDTH_SPACE_LIVE_FIXTURE ok path=#{output} tops=#{expected_tops.length}")
