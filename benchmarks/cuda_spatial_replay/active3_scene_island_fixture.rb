# frozen_string_literal: true

# Builds a small nonempty ACTIVE.3 scene with one violating child under each
# simple orthogonal transform.  This is a differential fixture for hierarchy
# composition, reflected edge direction, candidate generation, and exact
# device classification.

include RBA

output = File.expand_path(($output || "").to_s)
raise("missing -rd output=PATH") if output.empty?
raise("refusing to overwrite #{output}") if File.exist?(output) || File.symlink?(output)
raise("output parent does not exist") unless File.directory?(File.dirname(output))

layout = RBA::Layout.new
layout.dbu = 0.0005
well_layer = layout.layer(101, 0)
active_layer = layout.layer(102, 0)
child = layout.create_cell("ACTIVE3_VIOLATING_CHILD")
top = layout.create_cell("KLAYOUT_CUDA_ACTIVE3_SCENE")

# 50 DBU = 25nm enclosure, strictly below ACTIVE.3's 110 DBU = 55nm.
child.shapes(well_layer).insert(RBA::Box.new(0, 0, 1000, 1000))
child.shapes(active_layer).insert(RBA::Box.new(50, 50, 950, 950))

transforms = [
  RBA::Trans::R0, RBA::Trans::R90, RBA::Trans::R180, RBA::Trans::R270,
  RBA::Trans::M0, RBA::Trans::M45, RBA::Trans::M90, RBA::Trans::M135
]
transforms.each_with_index do |transform, index|
  top.insert(
    RBA::CellInstArray.new(
      child.cell_index,
      RBA::Trans.new(transform, index * 3000, 3000)
    )
  )
end

layout.write(output)
puts("ACTIVE3_SCENE_ISLAND_FIXTURE ok path=#{output} transforms=8")
