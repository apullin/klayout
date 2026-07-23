# frozen_string_literal: true

# Differential gate for KLayout's directed-edge normalization under all eight
# simple orthogonal transforms.  The CUDA scene island applies the affine
# transform to each stored clockwise edge, then reverses reflected edges.

include RBA

TRANSFORMS = [
  RBA::Trans::R0, RBA::Trans::R90, RBA::Trans::R180, RBA::Trans::R270,
  RBA::Trans::M0, RBA::Trans::M45, RBA::Trans::M90, RBA::Trans::M135
].freeze

MATRICES = [
  [1, 0, 0, 1], [0, -1, 1, 0], [-1, 0, 0, -1], [0, 1, -1, 0],
  [1, 0, 0, -1], [0, 1, 1, 0], [-1, 0, 0, 1], [0, -1, -1, 0]
].freeze

def edges_of(polygon)
  result = []
  polygon.each_edge do |edge|
    result << [edge.p1.x, edge.p1.y, edge.p2.x, edge.p2.y]
  end
  result
end

def transform_point(matrix, x, y)
  [
    matrix[0] * x + matrix[1] * y,
    matrix[2] * x + matrix[3] * y
  ]
end

def signed_area2(edges)
  edges.sum { |x1, y1, x2, y2| x1 * y2 - x2 * y1 }
end

source = RBA::Polygon.new(RBA::Box.new(0, 0, 10, 20))
source_edges = edges_of(source)
raise("source polygon is not clockwise") unless signed_area2(source_edges).negative?

TRANSFORMS.each_with_index do |transform_constant, code|
  actual_polygon = source.transformed(RBA::Trans.new(transform_constant, 13, -7))
  actual = edges_of(actual_polygon).sort
  expected = source_edges.map do |x1, y1, x2, y2|
    p1 = transform_point(MATRICES.fetch(code), x1, y1)
    p2 = transform_point(MATRICES.fetch(code), x2, y2)
    p1 = [p1[0] + 13, p1[1] - 7]
    p2 = [p2[0] + 13, p2[1] - 7]
    p1, p2 = p2, p1 if code >= 4
    [p1[0], p1[1], p2[0], p2[1]]
  end.sort
  raise("transform #{code}: directed-edge mismatch") unless actual == expected
  raise("transform #{code}: KLayout hull is not clockwise") unless signed_area2(actual).negative?
end

puts("ACTIVE3_MIRROR_TRANSFORM_DIFFERENTIAL ok transforms=8")
