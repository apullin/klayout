# frozen_string_literal: true

# Generates deterministic and randomized rectangle-union cases for the
# standalone POLY.3/POLY.4 terminal-empty CUDA island.  Expected geometric
# outcomes come from KLayout's real projection-enclosure implementation and
# include the deck's edge-pair-to-polygon and zero-area filtering chain.

include RBA

POLY3_DISTANCE = 110
POLY4_DISTANCE = 140
MAXIMUM_CANDIDATE_BOXES = 64
RANDOM_SEED = 0x5033_5034

UNSUPPORTED = 0
FALLBACK = 1
TERMINAL_EMPTY = 2

def fixture_error(message)
  raise("POLY34 terminal-empty fixture: #{message}")
end

def required_rd(name, value)
  fixture_error("missing -rd #{name}=VALUE") if value.nil? || value.to_s.empty?
  value.to_s
end

def positive_intersection?(first, second)
  first.left < second.right && first.right > second.left &&
    first.bottom < second.top && first.top > second.bottom
end

def union_covers?(target, boxes)
  return false if target.empty? || boxes.empty?

  y_endpoints = [target.bottom, target.top]
  boxes.each do |box|
    y_endpoints << box.bottom if box.bottom > target.bottom && box.bottom < target.top
    y_endpoints << box.top if box.top > target.bottom && box.top < target.top
  end
  y_endpoints.sort!.uniq!
  y_endpoints.each_cons(2).all? do |bottom, top|
    x = target.left
    while x < target.right
      farthest = x
      boxes.each do |box|
        next unless box.bottom <= bottom && box.top >= top
        next unless box.left <= x && box.right > farthest

        farthest = box.right
      end
      break false if farthest <= x
      x = farthest
    end
    x >= target.right
  end
end

def union_misses?(target, boxes)
  boxes.none? { |box| positive_intersection?(target, box) }
end

def independent_certificate(gate, boxes, distance, supported)
  return UNSUPPORTED unless supported
  return UNSUPPORTED if gate.empty? || boxes.empty? ||
                        boxes.length > MAXIMUM_CANDIDATE_BOXES
  return FALLBACK unless union_covers?(gate, boxes)

  bands = [
    RBA::Box.new(gate.left - distance, gate.bottom, gate.left, gate.top),
    RBA::Box.new(gate.right, gate.bottom, gate.right + distance, gate.top),
    RBA::Box.new(gate.left, gate.bottom - distance, gate.right, gate.bottom),
    RBA::Box.new(gate.left, gate.top, gate.right, gate.top + distance)
  ]
  return FALLBACK unless bands.all? do |band|
    union_misses?(band, boxes) || union_covers?(band, boxes)
  end

  TERMINAL_EMPTY
end

def klayout_terminal_empty?(gate, boxes, distance)
  primary = RBA::Region.new
  boxes.each { |box| primary.insert(box) }
  gate_region = RBA::Region.new(gate)
  edge_pairs = primary.enclosing_check(
    gate_region,
    distance,
    false,
    RBA::Region::Projection,
    nil,
    nil,
    nil
  )
  # Region#with_area(0, true) is the direct API equivalent of the deck's
  # `.without_area(0)`. EdgePairs#polygons performs normalized to_polygon(0).
  [
    edge_pairs.polygons.with_area(0, true).is_empty?,
    edge_pairs.size
  ]
end

def expanded_box(gate, left, bottom, right, top)
  RBA::Box.new(
    gate.left - left,
    gate.bottom - bottom,
    gate.right + right,
    gate.top + top
  )
end

def side_tiles(gate, distance)
  middle_y = gate.bottom + gate.height / 2
  middle_x = gate.left + gate.width / 2
  [
    gate,
    RBA::Box.new(gate.left - distance, gate.bottom, gate.left, middle_y),
    RBA::Box.new(gate.left - distance, middle_y, gate.left, gate.top),
    RBA::Box.new(gate.right, gate.bottom, gate.right + distance, middle_y),
    RBA::Box.new(gate.right, middle_y, gate.right + distance, gate.top),
    RBA::Box.new(gate.left, gate.bottom - distance, middle_x, gate.bottom),
    RBA::Box.new(middle_x, gate.bottom - distance, gate.right, gate.bottom),
    RBA::Box.new(gate.left, gate.top, middle_x, gate.top + distance),
    RBA::Box.new(middle_x, gate.top, gate.right, gate.top + distance)
  ]
end

CaseData = Struct.new(
  :name,
  :kind,
  :gate,
  :poly_boxes,
  :active_boxes,
  :poly_supported,
  :active_supported,
  keyword_init: true
)

def deterministic_cases
  gate = RBA::Box.new(0, 0, 100, 180)
  [
    CaseData.new(
      name: "all_coincident",
      kind: "deterministic",
      gate: gate,
      poly_boxes: [gate],
      active_boxes: [gate],
      poly_supported: true,
      active_supported: true
    ),
    CaseData.new(
      name: "exact_profiles",
      kind: "deterministic",
      gate: gate,
      poly_boxes: [expanded_box(gate, 110, 110, 110, 110)],
      active_boxes: [expanded_box(gate, 140, 140, 140, 140)],
      poly_supported: true,
      active_supported: true
    ),
    CaseData.new(
      name: "profile_split_110_140",
      kind: "deterministic",
      gate: gate,
      poly_boxes: [expanded_box(gate, 110, 110, 110, 110)],
      active_boxes: [expanded_box(gate, 110, 110, 110, 110)],
      poly_supported: true,
      active_supported: true
    ),
    CaseData.new(
      name: "poly_one_dbu_partial",
      kind: "deterministic",
      gate: gate,
      poly_boxes: [expanded_box(gate, 1, 110, 110, 110)],
      active_boxes: [expanded_box(gate, 140, 140, 140, 140)],
      poly_supported: true,
      active_supported: true
    ),
    CaseData.new(
      name: "active_one_dbu_partial",
      kind: "deterministic",
      gate: gate,
      poly_boxes: [expanded_box(gate, 110, 110, 110, 110)],
      active_boxes: [expanded_box(gate, 140, 1, 140, 140)],
      poly_supported: true,
      active_supported: true
    ),
    CaseData.new(
      name: "threshold_minus_one",
      kind: "deterministic",
      gate: gate,
      poly_boxes: [expanded_box(gate, 109, 110, 110, 110)],
      active_boxes: [expanded_box(gate, 140, 139, 140, 140)],
      poly_supported: true,
      active_supported: true
    ),
    CaseData.new(
      name: "mixed_zero_full_declined",
      kind: "deterministic",
      gate: gate,
      poly_boxes: [
        gate,
        RBA::Box.new(gate.left - 110, gate.bottom, gate.left, gate.bottom + 90)
      ],
      active_boxes: [
        gate,
        RBA::Box.new(gate.right, gate.bottom + 90, gate.right + 140, gate.top)
      ],
      poly_supported: true,
      active_supported: true
    ),
    CaseData.new(
      name: "disconnected_near_declined",
      kind: "deterministic",
      gate: gate,
      poly_boxes: [
        gate,
        RBA::Box.new(gate.right + 10, gate.bottom, gate.right + 40, gate.top)
      ],
      active_boxes: [
        gate,
        RBA::Box.new(gate.left - 80, gate.bottom, gate.left - 10, gate.top)
      ],
      poly_supported: true,
      active_supported: true
    ),
    CaseData.new(
      name: "disconnected_exact_boundary",
      kind: "deterministic",
      gate: gate,
      poly_boxes: [
        gate,
        RBA::Box.new(gate.right + 110, gate.bottom, gate.right + 150, gate.top)
      ],
      active_boxes: [
        gate,
        RBA::Box.new(gate.left - 180, gate.bottom, gate.left - 140, gate.top)
      ],
      poly_supported: true,
      active_supported: true
    ),
    CaseData.new(
      name: "union_tiled_full_bands",
      kind: "deterministic",
      gate: gate,
      poly_boxes: side_tiles(gate, 110),
      active_boxes: side_tiles(gate, 140),
      poly_supported: true,
      active_supported: true
    ),
    CaseData.new(
      name: "one_dbu_band_gap_declined",
      kind: "deterministic",
      gate: gate,
      poly_boxes: [
        gate,
        RBA::Box.new(gate.left - 110, gate.bottom, gate.left, gate.bottom + 89),
        RBA::Box.new(gate.left - 110, gate.bottom + 90, gate.left, gate.top)
      ],
      active_boxes: [
        gate,
        RBA::Box.new(gate.right, gate.bottom, gate.right + 140, gate.bottom + 89),
        RBA::Box.new(gate.right, gate.bottom + 90, gate.right + 140, gate.top)
      ],
      poly_supported: true,
      active_supported: true
    ),
    CaseData.new(
      name: "gate_not_covered",
      kind: "deterministic",
      gate: gate,
      poly_boxes: [RBA::Box.new(0, 0, 50, 180)],
      active_boxes: [RBA::Box.new(50, 0, 100, 180)],
      poly_supported: true,
      active_supported: true
    ),
    CaseData.new(
      name: "unsupported_poly_shape",
      kind: "deterministic",
      gate: gate,
      poly_boxes: [gate],
      active_boxes: [gate],
      poly_supported: false,
      active_supported: true
    ),
    CaseData.new(
      name: "unsupported_active_shape",
      kind: "deterministic",
      gate: gate,
      poly_boxes: [gate],
      active_boxes: [gate],
      poly_supported: true,
      active_supported: false
    ),
    CaseData.new(
      name: "candidate_capacity",
      kind: "deterministic",
      gate: gate,
      poly_boxes: Array.new(MAXIMUM_CANDIDATE_BOXES + 1) { gate },
      active_boxes: Array.new(MAXIMUM_CANDIDATE_BOXES + 1) { gate },
      poly_supported: true,
      active_supported: true
    )
  ]
end

def random_primary(rng, gate, distance, mode)
  boundary_values = [0, 1, distance - 1, distance, distance + 1]
  case mode
  when 0
    [expanded_box(
      gate,
      boundary_values.sample(random: rng),
      boundary_values.sample(random: rng),
      boundary_values.sample(random: rng),
      boundary_values.sample(random: rng)
    )]
  when 1
    boxes = [gate]
    rng.rand(1..7).times do
      x1 = rng.rand((gate.left - distance - 40)..(gate.right + distance + 40))
      x2 = rng.rand((gate.left - distance - 40)..(gate.right + distance + 40))
      y1 = rng.rand((gate.bottom - distance - 40)..(gate.top + distance + 40))
      y2 = rng.rand((gate.bottom - distance - 40)..(gate.top + distance + 40))
      next if x1 == x2 || y1 == y2
      boxes << RBA::Box.new(
        [x1, x2].min, [y1, y2].min, [x1, x2].max, [y1, y2].max
      )
    end
    boxes
  when 2
    side_tiles(gate, distance)
  when 3
    side = rng.rand(4)
    gap = rng.rand(1...distance)
    nearby = case side
             when 0
               RBA::Box.new(
                 gate.left - gap - 20, gate.bottom,
                 gate.left - gap, gate.top
               )
             when 1
               RBA::Box.new(
                 gate.right + gap, gate.bottom,
                 gate.right + gap + 20, gate.top
               )
             when 2
               RBA::Box.new(
                 gate.left, gate.bottom - gap - 20,
                 gate.right, gate.bottom - gap
               )
             else
               RBA::Box.new(
                 gate.left, gate.top + gap,
                 gate.right, gate.top + gap + 20
               )
             end
    [gate, nearby]
  else
    split = rng.rand(1...gate.height)
    [
      gate,
      RBA::Box.new(
        gate.left - distance, gate.bottom,
        gate.left, gate.bottom + split
      )
    ]
  end
end

def randomized_cases(count)
  rng = Random.new(RANDOM_SEED)
  heights = [180, 270, 410, 735]
  Array.new(count) do |index|
    left = rng.rand(-2_000..2_000)
    bottom = rng.rand(-2_000..2_000)
    gate = RBA::Box.new(left, bottom, left + 100, bottom + heights.sample(random: rng))
    CaseData.new(
      name: format("random_%05d", index),
      kind: "random",
      gate: gate,
      poly_boxes: random_primary(rng, gate, POLY3_DISTANCE, rng.rand(5)),
      active_boxes: random_primary(rng, gate, POLY4_DISTANCE, rng.rand(5)),
      poly_supported: true,
      active_supported: true
    )
  end
end

output_expanded = File.expand_path(required_rd("output", $output))
output_parent = File.realpath(File.dirname(output_expanded))
output_path = File.join(output_parent, File.basename(output_expanded))
fixture_error("refusing to overwrite output: #{output_path}") if
  File.exist?(output_path) || File.symlink?(output_path)
random_count = required_rd("random_count", $random_count).to_i
fixture_error("random_count must be in 1..100000") unless
  random_count.between?(1, 100_000)

cases = deterministic_cases + randomized_cases(random_count)
false_clean = 0
terminal_polygons_nonempty = 0
raw_edge_pairs = 0
zero_area_only_profiles = 0
File.open(output_path, "wb") do |output|
  output.puts("POLY34CASE1 #{cases.length}")
  cases.each do |test|
    actual_poly, poly_edge_pairs = klayout_terminal_empty?(
      test.gate, test.poly_boxes, POLY3_DISTANCE
    )
    actual_active, active_edge_pairs = klayout_terminal_empty?(
      test.gate, test.active_boxes, POLY4_DISTANCE
    )
    expected_poly = independent_certificate(
      test.gate, test.poly_boxes, POLY3_DISTANCE, test.poly_supported
    )
    expected_active = independent_certificate(
      test.gate, test.active_boxes, POLY4_DISTANCE, test.active_supported
    )
    false_clean += 1 if expected_poly == TERMINAL_EMPTY && !actual_poly
    false_clean += 1 if expected_active == TERMINAL_EMPTY && !actual_active
    terminal_polygons_nonempty += 1 unless actual_poly
    terminal_polygons_nonempty += 1 unless actual_active
    raw_edge_pairs += poly_edge_pairs + active_edge_pairs
    zero_area_only_profiles += 1 if actual_poly && poly_edge_pairs.positive?
    zero_area_only_profiles += 1 if actual_active && active_edge_pairs.positive?

    fields = [
      "CASE",
      test.name,
      test.kind,
      test.gate.left,
      test.gate.bottom,
      test.gate.right,
      test.gate.top,
      actual_poly ? 1 : 0,
      actual_active ? 1 : 0,
      expected_poly,
      expected_active,
      test.poly_supported ? 1 : 0,
      test.active_supported ? 1 : 0,
      test.poly_boxes.length
    ]
    test.poly_boxes.each do |box|
      fields.concat([box.left, box.bottom, box.right, box.top])
    end
    fields << test.active_boxes.length
    test.active_boxes.each do |box|
      fields.concat([box.left, box.bottom, box.right, box.top])
    end
    output.puts(fields.join(" "))
  end
end

fixture_error("independent classifier produced #{false_clean} false-clean outcomes") unless
  false_clean.zero?
puts(
  "POLY34_TERMINAL_EMPTY_FIXTURE ok output=#{output_path} " \
  "cases=#{cases.length} deterministic=#{deterministic_cases.length} " \
  "randomized=#{random_count} random_seed=#{RANDOM_SEED} " \
  "raw_edge_pairs=#{raw_edge_pairs} " \
  "zero_area_only_profiles=#{zero_area_only_profiles} " \
  "actual_nonempty_profiles=#{terminal_polygons_nonempty} false_clean=0"
)
