# frozen_string_literal: true

# Three tiny deterministic layouts for the pre-WELL ACTIVE.3 integrity gate.

include RBA

output = File.expand_path(($output || "").to_s)
raise("missing -rd output=PATH") if output.empty?
raise("refusing to overwrite #{output}") if File.exist?(output) || File.symlink?(output)
raise("output parent does not exist") unless File.directory?(File.dirname(output))

layout = RBA::Layout.new
layout.dbu = 0.0005
active = layout.layer(1, 0)
pwell = layout.layer(2, 0)
nwell = layout.layer(3, 0)

def active3_cell(layout, name, active, pwell, nwell, active_box, pwell_box, nwell_box)
  cell = layout.create_cell(name)
  cell.shapes(active).insert(active_box)
  cell.shapes(pwell).insert(pwell_box)
  cell.shapes(nwell).insert(nwell_box)
  cell
end

# Both raw well streams participate, while ACTIVE is at least 200 DBU from
# either merged WELL boundary.  The 110-DBU raw certificate must be clean.
active3_cell(
  layout,
  "ACTIVE3_RAW_WELLS_CLEAN",
  active,
  pwell,
  nwell,
  RBA::Box.new(200, 200, 800, 800),
  RBA::Box.new(2000, 0, 3000, 1000),
  RBA::Box.new(0, 0, 1000, 1000)
)

# The merged WELL is the clean outer box (0,0)-(1000,1000).  Raw NWELL's
# internal x=150 boundary is only 50 DBU from ACTIVE's x=200 boundary.  Both
# clockwise left edges point north, with ACTIVE on NWELL's material side, so
# this is deliberately inside the exact enclosing-predicate hit domain.  The
# conservative raw certificate must decline, then exact CPU union/enclosing
# must still report an empty category.
active3_cell(
  layout,
  "ACTIVE3_RAW_WELLS_FALSE_POSITIVE",
  active,
  pwell,
  nwell,
  RBA::Box.new(200, 200, 800, 800),
  RBA::Box.new(0, 0, 250, 1000),
  RBA::Box.new(150, 0, 1000, 1000)
)

# A real 50-DBU outer-boundary violation.  Both the raw certificate and the
# exact CPU fallback must decline/catch it.
active3_cell(
  layout,
  "ACTIVE3_RAW_WELLS_TRUE_HIT",
  active,
  pwell,
  nwell,
  RBA::Box.new(50, 200, 800, 800),
  RBA::Box.new(2000, 0, 3000, 1000),
  RBA::Box.new(0, 0, 1000, 1000)
)

expected_tops = %w[
  ACTIVE3_RAW_WELLS_CLEAN
  ACTIVE3_RAW_WELLS_FALSE_POSITIVE
  ACTIVE3_RAW_WELLS_TRUE_HIT
].sort
actual_tops = layout.top_cells.map(&:name).sort
raise("unexpected top cells: #{actual_tops.join(',')}") unless actual_tops == expected_tops

layout.write(output)
puts("ACTIVE3_RAW_WELLS_LIVE_FIXTURE ok path=#{output} tops=3 dbu=#{layout.dbu}")
