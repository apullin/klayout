# frozen_string_literal: true

# Small, deterministic production-dry-run fixture.  The two gate rectangles
# exercise both reasons a projected side can safely normalize away:
# coincident primary/gate boundaries and complete primary coverage out to the
# exact rule distance.

include RBA

output = $output.to_s
raise("missing -rd output=PATH") if output.empty?

layout = RBA::Layout.new
layout.dbu = ($dbu || 0.0005).to_f
top = layout.create_cell("POLY34_PRODUCTION_DRY_RUN")
poly = layout.layer(9, 0)
active = layout.layer(1, 0)

if $coordinate_limit
  # Valid GDS coordinates, but too close to INT32_MAX for the box scanner to
  # add the 140-DBU search distance safely.
  near_limit = RBA::Box.new(2_147_483_447, 0, 2_147_483_637, 200)
  top.shapes(poly).insert(near_limit)
  top.shapes(active).insert(near_limit)
else
  # All four sides coincide for both profiles.
  top.shapes(poly).insert(RBA::Box.new(1000, 1000, 1100, 1200))
  top.shapes(active).insert(RBA::Box.new(1000, 1000, 1100, 1200))

  # POLY fully covers the 110-DBU left/right bands and coincides at top/bottom.
  # ACTIVE fully covers the 140-DBU top/bottom bands and coincides at left/right.
  # Their intersection remains exactly the 100x200 gate.
  top.shapes(poly).insert(RBA::Box.new(-110, 0, 210, 200))
  top.shapes(active).insert(RBA::Box.new(0, -140, 100, 340))
end

layout.write(output)
puts("POLY34_PRODUCTION_DRY_RUN_FIXTURE ok output=#{output}")
