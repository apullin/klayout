# frozen_string_literal: true

# Representative differential gate against KLayout's production
# Edges#enclosing_check path.  The exhaustive/random CUDA gate lives in
# active3_exact_predicate_test.cc; this script anchors its semantic derivation
# directly to EdgeRelationFilter and the real edge scanner.

Case = Struct.new(:name, :well, :active, :distance, :violation)

RULE_DISTANCE_UM = 0.055
QUALIFIED_SCENE_DBU_UM = 0.0005
DISTANCE_DBU = 110
unless RULE_DISTANCE_UM / QUALIFIED_SCENE_DBU_UM == DISTANCE_DBU
  raise "ACTIVE.3 physical-to-DBU conversion is not exactly 110"
end

def e(x1, y1, x2, y2)
  RBA::Edge.new(x1, y1, x2, y2)
end

cases = [
  Case.new("east/right/d-1 endpoint", e(0, 0, 100, 0),
           e(100, -109, 200, -109), DISTANCE_DBU, true),
  Case.new("east/right/d endpoint", e(0, 0, 100, 0),
           e(100, -110, 200, -110), DISTANCE_DBU, false),
  Case.new("east/right/d+1 endpoint", e(0, 0, 100, 0),
           e(100, -111, 200, -111), DISTANCE_DBU, false),
  Case.new("wrong half-plane", e(0, 0, 100, 0),
           e(0, 109, 100, 109), DISTANCE_DBU, false),
  Case.new("opposite direction", e(0, 0, 100, 0),
           e(100, -1, 0, -1), DISTANCE_DBU, false),
  Case.new("perpendicular", e(0, 0, 100, 0),
           e(50, -10, 50, 10), DISTANCE_DBU, false),
  Case.new("west/right", e(100, 0, 0, 0),
           e(100, 109, 0, 109), DISTANCE_DBU, true),
  Case.new("north/right", e(0, 0, 0, 100),
           e(109, 0, 109, 100), DISTANCE_DBU, true),
  Case.new("south/right", e(0, 100, 0, 0),
           e(-109, 100, -109, 0), DISTANCE_DBU, true),
  Case.new("projection overlap", e(0, 0, 100, 0),
           e(50, -109, 150, -109), DISTANCE_DBU, true),
  Case.new("projection gap inside circle", e(0, 0, 100, 0),
           e(101, -109, 200, -109), DISTANCE_DBU, true),
  Case.new("3-4-5 boundary minus one", e(0, 0, 100, 0),
           e(187, -66, 250, -66), DISTANCE_DBU, true),
  Case.new("3-4-5 exact boundary", e(0, 0, 100, 0),
           e(188, -66, 250, -66), DISTANCE_DBU, false),
  Case.new("3-4-5 boundary plus one", e(0, 0, 100, 0),
           e(189, -66, 250, -66), DISTANCE_DBU, false),
  Case.new("collinear endpoint touch", e(0, 0, 100, 0),
           e(100, 0, 200, 0), DISTANCE_DBU, true),
  Case.new("collinear overlap", e(0, 0, 100, 0),
           e(50, 0, 150, 0), DISTANCE_DBU, true),
  Case.new("identical directed edges", e(0, 0, 100, 0),
           e(0, 0, 100, 0), DISTANCE_DBU, true),
  Case.new("collinear opposite direction", e(0, 0, 100, 0),
           e(100, 0, 0, 0), DISTANCE_DBU, false),
  Case.new("collinear gap excluded", e(0, 0, 100, 0),
           e(101, 0, 200, 0), DISTANCE_DBU, false),
  Case.new("negative coordinates", e(-200, -100, -100, -100),
           e(-150, -209, -50, -209), DISTANCE_DBU, true)
]

failures = []
cases.each do |test|
  well = RBA::Edges.new
  active = RBA::Edges.new
  well.insert(test.well)
  active.insert(test.active)
  result = well.enclosing_check(
    active, test.distance, false, RBA::Edges::Euclidian, 90.0, 0, nil,
    RBA::Edges::IncludeZeroDistanceWhenTouching
  )
  actual = !result.is_empty?
  next if actual == test.violation

  failures << "#{test.name}: expected=#{test.violation} actual=#{actual} " \
              "result=#{result}"
end

unless failures.empty?
  warn "ACTIVE.3 KLayout differential: FAIL"
  failures.each { |failure| warn failure }
  exit 1
end

puts "ACTIVE.3 KLayout differential: PASS (#{cases.size} production checks)"
