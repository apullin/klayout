# Synthetic ACTIVE.1/.2 endpoint-orientation fixtures.
#
# Run with:
#   klayout -b -r active12_endpoint_cpu_fixture.rb -rd output=PATH

output = File.expand_path(($output || "").to_s)
raise("missing -rd output=PATH") if output.empty?
raise("refusing to overwrite #{output}") if File.exist?(output) || File.symlink?(output)
raise("output parent does not exist") unless File.directory?(File.dirname(output))

layout = RBA::Layout.new
layout.dbu = 0.0005
active_layer = layout.layer(1, 0)

def transform_boxes(boxes, swap_xy, mirror_x, mirror_y)
  boxes.map do |left, bottom, right, top|
    if swap_xy
      left, bottom, right, top = bottom, left, top, right
    end
    if mirror_x
      left, right = 1000 - right, 1000 - left
    end
    if mirror_y
      bottom, top = 1000 - top, 1000 - bottom
    end
    [left, bottom, right, top]
  end
end

def add_fixture(layout, layer, name, boxes)
  cell = layout.create_cell(name)
  raise("duplicate fixture cell #{name}") unless cell
  boxes.each do |left, bottom, right, top|
    raise("invalid fixture box in #{name}") unless left < right && bottom < top
    cell.shapes(layer).insert(RBA::Box.new(left, bottom, right, top))
  end
end

deltas = {
  "minus" => -1,
  "equal" => 0,
  "plus" => 1,
}
orientations = {
  "right_up" => [false, false, false],
  "left_up" => [false, true, false],
  "right_down" => [false, false, true],
  "left_down" => [false, true, true],
  "swap_right_up" => [true, false, false],
  "swap_left_up" => [true, true, false],
  "swap_right_down" => [true, false, true],
  "swap_left_down" => [true, true, true],
}

# Exterior-facing finite edges.  The closest corners have (dx, dy)
# (95/96/97, 128), bracketing the exact 160-DBU ACTIVE.2 threshold.
orientations.each do |orientation, (swap_xy, mirror_x, mirror_y)|
  deltas.each do |limit, delta|
    dx = 96 + delta
    boxes = [
      [200, 200, 400, 400],
      [400 + dx, 528, 600 + dx, 728],
    ]
    add_fixture(
      layout,
      active_layer,
      "space_corner_#{orientation}_#{limit}",
      transform_boxes(boxes, swap_xy, mirror_x, mirror_y),
    )
  end
end

# Convex disconnected corners can have bottom/top half-planes that appear to
# face like a width relation.  Their 159/160/161 DBU horizontal separation and
# 10 DBU overlap reproduce the live SRAM shape that distinguishes the
# ACTIVE.2 threshold from ACTIVE.1.
orientations.each do |orientation, (swap_xy, mirror_x, mirror_y)|
  deltas.each do |limit, delta|
    gap = 160 + delta
    boxes = [
      [200, 400, 400, 700],
      [400 + gap, 100, 600 + gap, 410],
    ]
    add_fixture(
      layout,
      active_layer,
      "convex_width_facing_space_#{orientation}_#{limit}",
      transform_boxes(boxes, swap_xy, mirror_x, mirror_y),
    )
  end
end

# Disconnected rectangles whose orthogonal projections meet at exactly one
# endpoint.  Their 159/160/161-DBU separation is invisible to both strip scans,
# so it guards the endpoint pass's zero-projection-gap ownership.
orientations.each do |orientation, (swap_xy, mirror_x, mirror_y)|
  deltas.each do |limit, delta|
    gap = 160 + delta
    boxes = [
      [200, 200, 400, 400],
      [400 + gap, 400, 600 + gap, 600],
    ]
    add_fixture(
      layout,
      active_layer,
      "projection_touch_space_#{orientation}_#{limit}",
      transform_boxes(boxes, swap_xy, mirror_x, mirror_y),
    )
  end
end

# Interior-facing concave finite edges.  The two notch tips have (dx, dy)
# (107/108/109, 144), bracketing the exact 180-DBU ACTIVE.1 threshold.
# Every axis-aligned slice is at least 500 DBU thick, so only the Euclidean
# endpoint relation can expose the threshold-minus case.
orientations.each do |orientation, (swap_xy, mirror_x, mirror_y)|
  deltas.each do |limit, delta|
    left_tip = 446
    right_tip = left_tip + 108 + delta
    boxes = [
      [0, 0, left_tip, 500],
      [left_tip, 0, right_tip, 1000],
      [right_tip, 356, 1000, 1000],
    ]
    add_fixture(
      layout,
      active_layer,
      "width_corner_#{orientation}_#{limit}",
      transform_boxes(boxes, swap_xy, mirror_x, mirror_y),
    )
  end
end

# Same-rectangle endpoint pairs are deliberately redundant: the strip
# certificate, not a diagonal endpoint pair, owns these width decisions.
deltas.each do |limit, delta|
  add_fixture(
    layout,
    active_layer,
    "same_rectangle_width_#{limit}",
    [[100, 100, 500, 280 + delta]],
  )
end

# Canonical-union invariants: duplicate, overlapping, and edge-touching source
# boxes all reduce to wide material with no ACTIVE.1/.2 error.  A diagonal
# point touch is deliberately non-manifold and must produce at least one CPU
# rule hit; the GPU empty-only proof declines it rather than guessing.
add_fixture(
  layout,
  active_layer,
  "duplicate_union",
  [[100, 100, 600, 600], [100, 100, 600, 600]],
)
add_fixture(
  layout,
  active_layer,
  "overlap_union",
  [[100, 100, 700, 700], [300, 300, 900, 900]],
)
add_fixture(
  layout,
  active_layer,
  "edge_touch_union",
  [[100, 100, 500, 700], [500, 100, 900, 700]],
)
add_fixture(
  layout,
  active_layer,
  "point_touch_union",
  [[100, 100, 500, 500], [500, 500, 900, 900]],
)
add_fixture(
  layout,
  active_layer,
  "mixed_topology",
  [
    [0, 0, 446, 500],
    [446, 0, 451, 1000],
    [611, 510, 900, 900],
  ],
)

# A hole exercises inner and outer boundary orientation.  Only wall thickness
# owns the decision; the 600-DBU hole itself is far beyond ACTIVE.2.
deltas.each do |limit, delta|
  wall = 180 + delta
  add_fixture(
    layout,
    active_layer,
    "hole_wall_width_#{limit}",
    [
      [0, 0, 1000, wall],
      [0, 1000 - wall, 1000, 1000],
      [0, wall, wall, 1000 - wall],
      [1000 - wall, wall, 1000, 1000 - wall],
    ],
  )
end

layout.write(output)
puts("ACTIVE12_ENDPOINT_FIXTURE wrote=#{output} cells=#{layout.cells}")
