# frozen_string_literal: true

# Builds a small, empty ACTIVE.3 scene for exercising the live DeepRegion
# CUDA seam.  A 3x2 regular array is instantiated below each of KLayout's
# eight simple orthogonal transforms, so the expanded scene has:
#
#   1 top context + 8 array contexts + 48 leaf contexts = 57 contexts
#   48 WELL polygons and 48 ACTIVE polygons (192 edges per operand)
#
# ACTIVE is enclosed by 200 DBU (100nm at dbu=0.0005um), comfortably above
# the qualified ACTIVE.3 distance of 110 DBU (55nm).

include RBA

output = File.expand_path(($output || "").to_s)
raise("missing -rd output=PATH") if output.empty?
raise("refusing to overwrite #{output}") if File.exist?(output) || File.symlink?(output)
raise("output parent does not exist") unless File.directory?(File.dirname(output))

layout = RBA::Layout.new
layout.dbu = 0.0005
well_layer = layout.layer(101, 0)
active_layer = layout.layer(102, 0)
leaf = layout.create_cell("ACTIVE3_CLEAN_LEAF")
array = layout.create_cell("ACTIVE3_CLEAN_ARRAY")
top = layout.create_cell("KLAYOUT_CUDA_ACTIVE3_SCENE")

leaf.shapes(well_layer).insert(RBA::Box.new(0, 0, 1000, 1000))
leaf.shapes(active_layer).insert(RBA::Box.new(200, 200, 800, 800))

array.insert(
  RBA::CellInstArray.new(
    leaf.cell_index,
    RBA::Trans.new,
    RBA::Vector.new(1500, 0),
    RBA::Vector.new(0, 1500),
    3,
    2
  )
)

transforms = [
  RBA::Trans::R0, RBA::Trans::R90, RBA::Trans::R180, RBA::Trans::R270,
  RBA::Trans::M0, RBA::Trans::M45, RBA::Trans::M90, RBA::Trans::M135
]
transforms.each_with_index do |transform, index|
  top.insert(
    RBA::CellInstArray.new(
      array.cell_index,
      RBA::Trans.new(transform, index * 10_000, 10_000)
    )
  )
end

# Verify the exact local polygon qualifications consumed by the live seam.
{ "WELL" => well_layer, "ACTIVE" => active_layer }.each do |name, layer|
  shapes = leaf.shapes(layer)
  raise("#{name} fixture must contain exactly one shape") unless shapes.size == 1
  shapes.each do |shape|
    raise("#{name} fixture shape has properties") unless shape.prop_id == 0
    raise("#{name} fixture shape is not polygonal") unless shape.is_box? || shape.is_polygon?
    polygon = shape.polygon
    raise("#{name} fixture polygon has holes") unless polygon.holes == 0
    twice_area = 0
    edge_count = 0
    polygon.each_edge do |edge|
      raise("#{name} fixture edge is not Manhattan") unless
        edge.p1.x == edge.p2.x || edge.p1.y == edge.p2.y
      twice_area += edge.p1.x * edge.p2.y - edge.p2.x * edge.p1.y
      edge_count += 1
    end
    raise("#{name} fixture polygon is not a clockwise rectangle") unless
      edge_count == 4 && twice_area.negative?
  end
end

layout.write(output)
puts(
  "ACTIVE3_LIVE_NESTED_FIXTURE ok path=#{output} contexts=57 " \
  "well_contexts=48 active_contexts=48 well_edges=192 active_edges=192"
)
