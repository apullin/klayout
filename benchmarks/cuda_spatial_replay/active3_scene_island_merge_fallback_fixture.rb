# frozen_string_literal: true

# Builds and checks a case where a raw ACTIVE edge is a possible ACTIVE.3 hit
# but KLayout's required ACTIVE union removes that internal edge. The GPU scene
# island must return RAW_HIT_FALLBACK, never an exact production violation.

include RBA

output = File.expand_path(($output || "").to_s)
raise("missing -rd output=PATH") if output.empty?
raise("refusing to overwrite #{output}") if File.exist?(output) || File.symlink?(output)
raise("output parent does not exist") unless File.directory?(File.dirname(output))

well_box = RBA::Box.new(0, 0, 1000, 1000)
first_box = RBA::Box.new(200, 200, 950, 800)
covering_box = RBA::Box.new(900, 200, 1100, 800)

well = RBA::Region.new(well_box)
first = RBA::Region.new(first_box)
both = RBA::Region.new
both.insert(first_box)
both.insert(covering_box)

first_result = well.enclosing_check(
  first, 110, false, RBA::Region::Euclidian, 90.0, 0, nil, true,
  RBA::Region::NoOppositeFilter, RBA::Region::NoRectFilter, false,
  RBA::Region::IgnoreProperties,
  RBA::Region::IncludeZeroDistanceWhenTouching
)
both_result = well.enclosing_check(
  both, 110, false, RBA::Region::Euclidian, 90.0, 0, nil, true,
  RBA::Region::NoOppositeFilter, RBA::Region::NoRectFilter, false,
  RBA::Region::IgnoreProperties,
  RBA::Region::IncludeZeroDistanceWhenTouching
)
raise("single raw ACTIVE polygon should violate once") unless first_result.size == 1
raise("merged ACTIVE counterexample should be clean") unless both_result.is_empty?

layout = RBA::Layout.new
layout.dbu = 0.0005
well_layer = layout.layer(101, 0)
active_layer = layout.layer(102, 0)
top = layout.create_cell("KLAYOUT_CUDA_ACTIVE3_SCENE")
top.shapes(well_layer).insert(well_box)
top.shapes(active_layer).insert(first_box)
top.shapes(active_layer).insert(covering_box)
layout.write(output)

puts(
  "ACTIVE3_MERGE_FALLBACK_FIXTURE ok path=#{output} " \
  "raw_single_markers=#{first_result.size} merged_markers=#{both_result.size}"
)
