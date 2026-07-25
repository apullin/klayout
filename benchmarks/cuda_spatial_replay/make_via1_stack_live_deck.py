#!/usr/bin/env python3
"""Create qualified live-CUDA variants of the FreePDK45 shard deck."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one source block, found {count}")
    return text.replace(old, new, 1)


def transform(text: str) -> str:
    text = replace_once(
        text,
        """metal10    = polygons(29, 0)

# Computed layers""",
        """metal10    = polygons(29, 0)

# CUDA off preserves the source deck's original shard owners and CPU chains.
# CUDA on moves all six decisions into via1_upper_active12.  A missing method,
# disabled/missing backend, unsupported hierarchy, proof miss, or validation
# failure then selects all six local historical CPU chains.
via1_stack_request = ENV["KLAYOUT_CUDA_VIA1_STACK"].to_s
via1_stack_requested = !via1_stack_request.empty? &amp;&amp; via1_stack_request != "0" &amp;&amp; via1_stack_request != "false" &amp;&amp; via1_stack_request != "off"
via1_stack_owner = via1_stack_requested &amp;&amp; run_via1_upper_active12
via1_stack_clean = via1_stack_owner &amp;&amp; via1.respond_to?(:cuda_via1_stack_clean?) &amp;&amp; via1.cuda_via1_stack_clean?(metal1, metal2)
via1_stack_empty = polygon_layer if via1_stack_clean
info("CUDA VIA1 stack transaction: #{via1_stack_clean ? 'certified-empty' : 'full-cpu-fallback'}") if via1_stack_owner

# Computed layers""",
        "transaction initialization",
    )

    text = replace_once(
        text,
        """if run_m1_via_class

via1_edges_with_less_enclosure = metal1.enclosing(via1, 35.nm, projection).second_edges
error_corners = via1_edges_with_less_enclosure.width(angle_limit(100.0), 1.dbu)
via1_edges_with_less_enclosure.forget
via1.interacting(error_corners.polygons(1.dbu)).output("METAL1.4", "METAL1.4 : Minimum enclosure around via1 on two opposite sides : 35nm")
error_corners.forget
metal1_gt90""",
        """if run_m1_via_class

unless via1_stack_requested
  via1_edges_with_less_enclosure = metal1.enclosing(via1, 35.nm, projection).second_edges
  error_corners = via1_edges_with_less_enclosure.width(angle_limit(100.0), 1.dbu)
  via1_edges_with_less_enclosure.forget
  via1.interacting(error_corners.polygons(1.dbu)).output("METAL1.4", "METAL1.4 : Minimum enclosure around via1 on two opposite sides : 35nm")
  error_corners.forget
end

metal1_gt90""",
        "METAL1.4 original ownership",
    )

    text = replace_once(
        text,
        """#   Via1
via1.edges.without_length(65.nm).output("VIA1.1", "VIA1.1 : Minimum/Maximum width of via1 : 65nm")
via1.space(75.nm, euclidian).output("VIA1.2", "VIA1.2 : Minimum spacing of via1 : 75nm")
via1.not(metal1).output("VIA1.3", "VIA1.3 : via1 must be inside metal1")""",
        """if via1_stack_owner
  if via1_stack_clean
    via1_stack_empty.output("METAL1.4", "METAL1.4 : Minimum enclosure around via1 on two opposite sides : 35nm")
  else
    via1_edges_with_less_enclosure = metal1.enclosing(via1, 35.nm, projection).second_edges
    error_corners = via1_edges_with_less_enclosure.width(angle_limit(100.0), 1.dbu)
    via1_edges_with_less_enclosure.forget
    via1.interacting(error_corners.polygons(1.dbu)).output("METAL1.4", "METAL1.4 : Minimum enclosure around via1 on two opposite sides : 35nm")
    error_corners.forget
  end
end

#   Via1
if via1_stack_clean
  via1_stack_empty.output("VIA1.1", "VIA1.1 : Minimum/Maximum width of via1 : 65nm")
  via1_stack_empty.output("VIA1.2", "VIA1.2 : Minimum spacing of via1 : 75nm")
  via1_stack_empty.output("VIA1.3", "VIA1.3 : via1 must be inside metal1")
else
  via1.edges.without_length(65.nm).output("VIA1.1", "VIA1.1 : Minimum/Maximum width of via1 : 65nm")
  via1.space(75.nm, euclidian).output("VIA1.2", "VIA1.2 : Minimum spacing of via1 : 75nm")
  via1.not(metal1).output("VIA1.3", "VIA1.3 : via1 must be inside metal1")
end""",
        "VIA1.1-.3 transaction",
    )

    text = replace_once(
        text,
        """via1.not(metal2).output("VIA1.4", "VIA1.4 : via1 must be inside metal2")""",
        """if via1_stack_clean
  via1_stack_empty.output("VIA1.4", "VIA1.4 : via1 must be inside metal2")
else
  via1.not(metal2).output("VIA1.4", "VIA1.4 : via1 must be inside metal2")
end

if via1_stack_owner
  if via1_stack_clean
    via1_stack_empty.output("METAL2.3", "METAL2.3 : Minimum enclosure around via1 on two opposite sides : 35nm")
  else
    via1_edges_with_less_enclosure = metal2.enclosing(via1, 35.nm, projection).second_edges
    error_corners = via1_edges_with_less_enclosure.width(angle_limit(100.0), 1.dbu)
    via1_edges_with_less_enclosure.forget
    via1.interacting(error_corners.polygons(1.dbu)).output("METAL2.3", "METAL2.3 : Minimum enclosure around via1 on two opposite sides : 35nm")
    error_corners.forget
  end
end""",
        "VIA1.4 and METAL2.3 transaction",
    )

    text = replace_once(
        text,
        """metal2_space.output("METAL2.2", "METAL2.2 : Minimum spacing of  intermediate metal2 : 70nm")
via1_edges_with_less_enclosure = metal2.enclosing(via1, 35.nm, projection).second_edges
error_corners = via1_edges_with_less_enclosure.width(angle_limit(100.0), 1.dbu)
via1_edges_with_less_enclosure.forget
via1.interacting(error_corners.polygons(1.dbu)).output("METAL2.3", "METAL2.3 : Minimum enclosure around via1 on two opposite sides : 35nm")
error_corners.forget
via2_edges_with_less_enclosure""",
        """metal2_space.output("METAL2.2", "METAL2.2 : Minimum spacing of  intermediate metal2 : 70nm")
unless via1_stack_requested
  via1_edges_with_less_enclosure = metal2.enclosing(via1, 35.nm, projection).second_edges
  error_corners = via1_edges_with_less_enclosure.width(angle_limit(100.0), 1.dbu)
  via1_edges_with_less_enclosure.forget
  via1.interacting(error_corners.polygons(1.dbu)).output("METAL2.3", "METAL2.3 : Minimum enclosure around via1 on two opposite sides : 35nm")
  error_corners.forget
end

via2_edges_with_less_enclosure""",
        "METAL2.3 original ownership",
    )

    # The source owners remain intact and are only suppressed when the atomic
    # owner is explicitly requested. Exact guard balance is parsed by KLayout
    # in the live gate.
    return text


def add_m2_rules(text: str) -> str:
    text = replace_once(
        text,
        """if run_m2_rules

#   metal2
metal2_width, metal2_space = metal2.drc_batch([
  width(euclidian) &lt; 70.nm,
  space(euclidian) &lt; 70.nm
])
metal2_width.output("METAL2.1", "METAL2.1 : Minimum width of  intermediate metal2 : 70nm")
metal2_space.output("METAL2.2", "METAL2.2 : Minimum spacing of  intermediate metal2 : 70nm")""",
        """if run_m2_rules

# The optional exact M2 transaction returns owned flat operands only when its
# backend also certifies the fixed M2.5-.9 suffix empty.  Run M2.1/.2/.4
# speculatively on those flat operands and publish empty categories only if
# all three CPU results are empty.  Any missing method, backend decline,
# prefix hit, exception, or cleanup failure selects the pristine complete CPU
# block below.  M2.3 remains under its existing VIA1 transaction owner and
# VIA2.1-.4 remain under run_via1_upper_active12.
m2_rules_request = ENV["KLAYOUT_CUDA_M2_RULES"].to_s
m2_rules_requested = !m2_rules_request.empty? &amp;&amp; m2_rules_request != "0" &amp;&amp; m2_rules_request != "false" &amp;&amp; m2_rules_request != "off"
m2_rules_owner = m2_rules_requested &amp;&amp; run_m2_rules
m2_rules_clean = false
m2_rules_reason = "not-requested"
m2_rules_flat_results = []
m2_rules_flat_temps = []

if m2_rules_owner
  m2_rules_reason = "method-unavailable"
  begin
    if metal2.respond_to?(:cuda_m2_flat_union)
      m2_rules_flat_operands = metal2.cuda_m2_flat_union(via2)
      if m2_rules_flat_operands &amp;&amp; m2_rules_flat_operands.length == 2
        m2_rules_flat_metal2 = m2_rules_flat_operands[0]
        m2_rules_flat_via2 = m2_rules_flat_operands[1]
        m2_rules_flat_temps &lt;&lt; m2_rules_flat_metal2
        m2_rules_flat_temps &lt;&lt; m2_rules_flat_via2

        m2_rules_flat_width, m2_rules_flat_space = m2_rules_flat_metal2.drc_batch([
          width(euclidian) &lt; 70.nm,
          space(euclidian) &lt; 70.nm
        ])
        m2_rules_flat_temps &lt;&lt; m2_rules_flat_width
        m2_rules_flat_results &lt;&lt; m2_rules_flat_width
        m2_rules_flat_temps &lt;&lt; m2_rules_flat_space
        m2_rules_flat_results &lt;&lt; m2_rules_flat_space

        m2_rules_flat_via2_enclosure_pairs = m2_rules_flat_metal2.enclosing(m2_rules_flat_via2, 35.nm, projection)
        m2_rules_flat_temps &lt;&lt; m2_rules_flat_via2_enclosure_pairs
        m2_rules_flat_via2_enclosure_edges = m2_rules_flat_via2_enclosure_pairs.second_edges
        m2_rules_flat_temps &lt;&lt; m2_rules_flat_via2_enclosure_edges
        m2_rules_flat_error_corners = m2_rules_flat_via2_enclosure_edges.width(angle_limit(100.0), 1.dbu)
        m2_rules_flat_temps &lt;&lt; m2_rules_flat_error_corners
        m2_rules_flat_corner_polygons = m2_rules_flat_error_corners.polygons(1.dbu)
        m2_rules_flat_temps &lt;&lt; m2_rules_flat_corner_polygons
        m2_rules_flat_metal2_4 = m2_rules_flat_via2.interacting(m2_rules_flat_corner_polygons)
        m2_rules_flat_temps &lt;&lt; m2_rules_flat_metal2_4
        m2_rules_flat_results &lt;&lt; m2_rules_flat_metal2_4

        # A successful operand transaction is itself the exact all-five-bit
        # M2.5-.9 certificate.  Do not repeat those rules on the host.
        m2_rules_clean = m2_rules_flat_results.length == 3 &amp;&amp; m2_rules_flat_results.all? { |result| result.is_empty? }
        m2_rules_reason = m2_rules_clean ? "prefix-clean+suffix-certified" : "prefix-rule-hit"
      else
        m2_rules_reason = "operand-decline"
      end
    end
  rescue StandardError =&gt; m2_rules_error
    m2_rules_clean = false
    m2_rules_reason = "exception:#{m2_rules_error.class}"
  ensure
    m2_rules_flat_temps.reverse_each do |layer|
      begin
        layer.forget if layer
      rescue StandardError =&gt; m2_rules_cleanup_error
        m2_rules_clean = false
        m2_rules_reason = "cleanup-exception:#{m2_rules_cleanup_error.class}"
      end
    end
  end
end

m2_rules_empty = polygon_layer if m2_rules_clean
info("CUDA M2 rules transaction: #{m2_rules_clean ? 'certified-empty' : 'full-cpu-fallback'} reason=#{m2_rules_reason}") if m2_rules_owner

#   metal2
if m2_rules_clean
  m2_rules_empty.output("METAL2.1", "METAL2.1 : Minimum width of  intermediate metal2 : 70nm")
  m2_rules_empty.output("METAL2.2", "METAL2.2 : Minimum spacing of  intermediate metal2 : 70nm")
else
  metal2_width, metal2_space = metal2.drc_batch([
    width(euclidian) &lt; 70.nm,
    space(euclidian) &lt; 70.nm
  ])
  metal2_width.output("METAL2.1", "METAL2.1 : Minimum width of  intermediate metal2 : 70nm")
  metal2_space.output("METAL2.2", "METAL2.2 : Minimum spacing of  intermediate metal2 : 70nm")
end""",
        "M2.1/.2 speculative transaction",
    )

    text = replace_once(
        text,
        """via2_edges_with_less_enclosure = metal2.enclosing(via2, 35.nm, projection).second_edges
error_corners = via2_edges_with_less_enclosure.width(angle_limit(100.0), 1.dbu)
via2_edges_with_less_enclosure.forget
via2.interacting(error_corners.polygons(1.dbu)).output("METAL2.4", "METAL2.4 : Minimum enclosure around via2 on two opposite sides : 35nm")
error_corners.forget
metal2_gt90, metal2_gt270, metal2_gt500, metal2_gt900, metal2_gt1500 = classify_by_width(metal2, 90.nm, 270.nm, 500.nm, 900.nm, 1500.nm)
metal2_gt90.edges.with_length(300.nm,nil).space(90.nm,euclidian).output("METAL2.5", "METAL2.5 : Minimum spacing of  intermediate metal2 wider than 90 nm and longer than 300 nm : 90nm")
metal2_gt270.edges.with_length(900.nm,nil).space(270.nm,euclidian).output("METAL2.6", "METAL2.6 : Minimum spacing of  intermediate metal2 wider than 270 nm and longer than 900 nm : 270nm")
metal2_gt500.edges.with_length(1.8.um,nil).space(500.nm,euclidian).output("METAL2.7", "METAL2.7 : Minimum spacing of  intermediate metal2 wider than 500 nm and longer than 1.8 um : 500nm")
metal2_gt900.edges.with_length(2.7.um,nil).space(900.nm,euclidian).output("METAL2.8", "METAL2.8 : Minimum spacing of  intermediate metal2 wider than 900 nm and longer than 2.7 um : 900nm")
metal2_gt1500.edges.with_length(4.um,nil).space(1500.nm,euclidian).output("METAL2.9", "METAL2.9 : Minimum spacing of  intermediate metal2 wider than 1500 nm and longer than 4.0 um : 1500nm")
[ metal2_gt90, metal2_gt270, metal2_gt500, metal2_gt900, metal2_gt1500 ].each { |l| l.forget }""",
        """if m2_rules_clean
  m2_rules_empty.output("METAL2.4", "METAL2.4 : Minimum enclosure around via2 on two opposite sides : 35nm")
  m2_rules_empty.output("METAL2.5", "METAL2.5 : Minimum spacing of  intermediate metal2 wider than 90 nm and longer than 300 nm : 90nm")
  m2_rules_empty.output("METAL2.6", "METAL2.6 : Minimum spacing of  intermediate metal2 wider than 270 nm and longer than 900 nm : 270nm")
  m2_rules_empty.output("METAL2.7", "METAL2.7 : Minimum spacing of  intermediate metal2 wider than 500 nm and longer than 1.8 um : 500nm")
  m2_rules_empty.output("METAL2.8", "METAL2.8 : Minimum spacing of  intermediate metal2 wider than 900 nm and longer than 2.7 um : 900nm")
  m2_rules_empty.output("METAL2.9", "METAL2.9 : Minimum spacing of  intermediate metal2 wider than 1500 nm and longer than 4.0 um : 1500nm")
else
  via2_edges_with_less_enclosure = metal2.enclosing(via2, 35.nm, projection).second_edges
  error_corners = via2_edges_with_less_enclosure.width(angle_limit(100.0), 1.dbu)
  via2_edges_with_less_enclosure.forget
  via2.interacting(error_corners.polygons(1.dbu)).output("METAL2.4", "METAL2.4 : Minimum enclosure around via2 on two opposite sides : 35nm")
  error_corners.forget
  metal2_gt90, metal2_gt270, metal2_gt500, metal2_gt900, metal2_gt1500 = classify_by_width(metal2, 90.nm, 270.nm, 500.nm, 900.nm, 1500.nm)
  metal2_gt90.edges.with_length(300.nm,nil).space(90.nm,euclidian).output("METAL2.5", "METAL2.5 : Minimum spacing of  intermediate metal2 wider than 90 nm and longer than 300 nm : 90nm")
  metal2_gt270.edges.with_length(900.nm,nil).space(270.nm,euclidian).output("METAL2.6", "METAL2.6 : Minimum spacing of  intermediate metal2 wider than 270 nm and longer than 900 nm : 270nm")
  metal2_gt500.edges.with_length(1.8.um,nil).space(500.nm,euclidian).output("METAL2.7", "METAL2.7 : Minimum spacing of  intermediate metal2 wider than 500 nm and longer than 1.8 um : 500nm")
  metal2_gt900.edges.with_length(2.7.um,nil).space(900.nm,euclidian).output("METAL2.8", "METAL2.8 : Minimum spacing of  intermediate metal2 wider than 900 nm and longer than 2.7 um : 900nm")
  metal2_gt1500.edges.with_length(4.um,nil).space(1500.nm,euclidian).output("METAL2.9", "METAL2.9 : Minimum spacing of  intermediate metal2 wider than 1500 nm and longer than 4.0 um : 1500nm")
  [ metal2_gt90, metal2_gt270, metal2_gt500, metal2_gt900, metal2_gt1500 ].each { |l| l.forget }
end""",
        "M2.4-.9 speculative transaction",
    )
    return text


def add_m1_contact(text: str) -> str:
    text = replace_once(
        text,
        """if run_implant_contact

#   Implant""",
        """# The fixed M1-contact certificate is stronger than both consumers:
# exact 65nm contact boxes, 75nm contact spacing, full M1 containment, and
# the METAL1.3 two-opposite-side enclosure relation.  Keep one transaction
# when the unsplit deck owns both consumers; each split owner otherwise gets
# its own fail-closed transaction.
m1_contact_request = ENV["KLAYOUT_CUDA_M1_CONTACT"].to_s
m1_contact_requested = !m1_contact_request.empty? &amp;&amp; m1_contact_request != "0" &amp;&amp; m1_contact_request != "false" &amp;&amp; m1_contact_request != "off"
m1_contact_owner = m1_contact_requested &amp;&amp; (run_implant_contact || run_m1_enclosure)
m1_contact_clean = m1_contact_owner &amp;&amp; cont.respond_to?(:cuda_m1_contact_clean?) &amp;&amp; cont.cuda_m1_contact_clean?(metal1)
m1_contact_empty = polygon_layer if m1_contact_clean
info("CUDA M1 contact transaction: #{m1_contact_clean ? 'certified-empty' : 'full-cpu-fallback'}") if m1_contact_owner

if run_implant_contact

#   Implant""",
        "M1-contact owner transaction",
    )
    text = replace_once(
        text,
        """#   Contact
cont.edges.without_length(65.nm).output("CONTACT.1", "CONTACT.1 : Minimum/Maximum width of contact : 65nm")
cont.space(75.nm, euclidian).output("CONTACT.2", "CONTACT.2 : Minimum spacing of contact : 75nm")
cont.not(active).not(poly).not(metal1).output("CONTACT.3", "CONTACT.3 : contact must be inside active or poly or metal1")
active.enclosing(cont, 5.nm, euclidian).output("CONTACT.4", "CONTACT.4 : Minimum enclosure of active around contact : 5nm")""",
        """#   Contact
# Match the three original expressions as one exact transaction.  The CUDA
# certificate's M1-containment proof is stronger than CONTACT.3's
# active-or-poly-or-M1 union; no CONTACT.1-.3 result can survive on success.
# Every decline executes each untouched CPU expression exactly once.
if m1_contact_clean
  m1_contact_empty.output("CONTACT.1", "CONTACT.1 : Minimum/Maximum width of contact : 65nm")
  m1_contact_empty.output("CONTACT.2", "CONTACT.2 : Minimum spacing of contact : 75nm")
  m1_contact_empty.output("CONTACT.3", "CONTACT.3 : contact must be inside active or poly or metal1")
else
  cont.edges.without_length(65.nm).output("CONTACT.1", "CONTACT.1 : Minimum/Maximum width of contact : 65nm")
  cont.space(75.nm, euclidian).output("CONTACT.2", "CONTACT.2 : Minimum spacing of contact : 75nm")
  cont.not(active).not(poly).not(metal1).output("CONTACT.3", "CONTACT.3 : contact must be inside active or poly or metal1")
end
active.enclosing(cont, 5.nm, euclidian).output("CONTACT.4", "CONTACT.4 : Minimum enclosure of active around contact : 5nm")""",
        "CONTACT.1-.3 transaction",
    )
    text = replace_once(
        text,
        """if run_m1_enclosure

# The inside/enclosing relation""",
        """if run_m1_enclosure

if m1_contact_clean
  m1_contact_empty.output("METAL1.3", "METAL1.3 : Minimum enclosure around contact on two opposite sides : 35nm")
else

# The inside/enclosing relation""",
        "METAL1.3 transaction entry",
    )
    text = replace_once(
        text,
        """end


if run_m1_via_class""",
        """end

end


if run_m1_via_class""",
        "METAL1.3 transaction exit",
    )
    return text


def add_m1_5_9(text: str) -> str:
    """Add the fail-closed exact raw-M1 M1.5-.9 certificate transaction."""

    return replace_once(
        text,
        """metal1_gt90, metal1_gt270, metal1_gt500, metal1_gt900, metal1_gt1500 = classify_by_width(metal1, 90.nm, 270.nm, 500.nm, 900.nm, 1500.nm)
metal1_gt90.edges.with_length(300.nm,nil).space(90.nm,euclidian).output("METAL1.5", "METAL1.5 : Minimum spacing of metal1 wider than 90 nm and longer than 300 nm : 90nm")
metal1_gt270.edges.with_length(900.nm,nil).space(270.nm,euclidian).output("METAL1.6", "METAL1.6 : Minimum spacing of metal1 wider than 270 nm and longer than 900 nm : 270nm")
metal1_gt500.edges.with_length(1.8.um,nil).space(500.nm,euclidian).output("METAL1.7", "METAL1.7 : Minimum spacing of metal1 wider than 500 nm and longer than 1.8 um : 500nm")
metal1_gt900.edges.with_length(2.7.um,nil).space(900.nm,euclidian).output("METAL1.8", "METAL1.8 : Minimum spacing of metal1 wider than 900 nm and longer than 2.7 um : 900nm")
metal1_gt1500.edges.with_length(4.um,nil).space(1500.nm,euclidian).output("METAL1.9", "METAL1.9 : Minimum spacing of metal1 wider than 1500 nm and longer than 4.0 um : 1500nm")
[ metal1_gt90, metal1_gt270, metal1_gt500, metal1_gt900, metal1_gt1500 ].each { |l| l.forget }""",
        """# The optional certificate serializes pristine physical M1, expands and
# unions it exactly on the selected device, and evaluates the unchanged
# M1.5-.9 morphology sequence without returning geometry.  Any unavailable
# method, exception, capacity miss, rule hit, or proof-validation failure
# executes the five historical CPU expressions below byte-for-byte.
m1_5_9_request = ENV["KLAYOUT_CUDA_M1_5_9"].to_s
m1_5_9_requested = !m1_5_9_request.empty? &amp;&amp; m1_5_9_request != "0" &amp;&amp; m1_5_9_request != "false" &amp;&amp; m1_5_9_request != "off"
m1_5_9_clean = false
if m1_5_9_requested
  begin
    m1_5_9_clean = metal1.respond_to?(:cuda_m1_5_9_clean?) &amp;&amp; metal1.cuda_m1_5_9_clean?
  rescue StandardError =&gt; m1_5_9_error
    m1_5_9_clean = false
    info("CUDA M1.5-.9 Ruby fallback: #{m1_5_9_error}")
  end
  info("CUDA M1.5-.9 transaction: #{m1_5_9_clean ? 'certified-empty' : 'full-cpu-fallback'}")
end
m1_5_9_empty = polygon_layer if m1_5_9_clean

if m1_5_9_clean
  m1_5_9_empty.output("METAL1.5", "METAL1.5 : Minimum spacing of metal1 wider than 90 nm and longer than 300 nm : 90nm")
  m1_5_9_empty.output("METAL1.6", "METAL1.6 : Minimum spacing of metal1 wider than 270 nm and longer than 900 nm : 270nm")
  m1_5_9_empty.output("METAL1.7", "METAL1.7 : Minimum spacing of metal1 wider than 500 nm and longer than 1.8 um : 500nm")
  m1_5_9_empty.output("METAL1.8", "METAL1.8 : Minimum spacing of metal1 wider than 900 nm and longer than 2.7 um : 900nm")
  m1_5_9_empty.output("METAL1.9", "METAL1.9 : Minimum spacing of metal1 wider than 1500 nm and longer than 4.0 um : 1500nm")
else
  metal1_gt90, metal1_gt270, metal1_gt500, metal1_gt900, metal1_gt1500 = classify_by_width(metal1, 90.nm, 270.nm, 500.nm, 900.nm, 1500.nm)
  metal1_gt90.edges.with_length(300.nm,nil).space(90.nm,euclidian).output("METAL1.5", "METAL1.5 : Minimum spacing of metal1 wider than 90 nm and longer than 300 nm : 90nm")
  metal1_gt270.edges.with_length(900.nm,nil).space(270.nm,euclidian).output("METAL1.6", "METAL1.6 : Minimum spacing of metal1 wider than 270 nm and longer than 900 nm : 270nm")
  metal1_gt500.edges.with_length(1.8.um,nil).space(500.nm,euclidian).output("METAL1.7", "METAL1.7 : Minimum spacing of metal1 wider than 500 nm and longer than 1.8 um : 500nm")
  metal1_gt900.edges.with_length(2.7.um,nil).space(900.nm,euclidian).output("METAL1.8", "METAL1.8 : Minimum spacing of metal1 wider than 900 nm and longer than 2.7 um : 900nm")
  metal1_gt1500.edges.with_length(4.um,nil).space(1500.nm,euclidian).output("METAL1.9", "METAL1.9 : Minimum spacing of metal1 wider than 1500 nm and longer than 4.0 um : 1500nm")
  [ metal1_gt90, metal1_gt270, metal1_gt500, metal1_gt900, metal1_gt1500 ].each { |l| l.forget }
end""",
        "M1.5-.9 exact resident morphology transaction",
    )


def add_implant12(text: str) -> str:
    return replace_once(
        text,
        """implant.separation(gate, 70.nm, projection).polygons.without_area(0).output("IMPLANT.1", "IMPLANT.1 : Minimum spacing of nimplant/ pimplant to channel : 70nm")
implant.separation(cont, 25.nm, projection).polygons.without_area(0).output("IMPLANT.2", "IMPLANT.1 : Minimum spacing of nimplant/ pimplant to contact : 25nm")""",
        """# One qualified transaction may prove both fixed projection-separation
# categories empty.  The generated opt-in is fail-closed: a missing method or
# any host/backend decline executes each historical CPU expression unchanged.
implant12_request = ENV["KLAYOUT_CUDA_IMPLANT12"].to_s
implant12_requested = !implant12_request.empty? &amp;&amp; implant12_request != "0" &amp;&amp; implant12_request != "false" &amp;&amp; implant12_request != "off"
implant12_clean = implant12_requested &amp;&amp; implant.respond_to?(:cuda_implant12_clean?) &amp;&amp; implant.cuda_implant12_clean?(gate, cont)
implant12_empty = polygon_layer if implant12_clean
info("CUDA IMPLANT.1/.2 transaction: #{implant12_clean ? 'certified-empty' : 'full-cpu-fallback'}") if implant12_requested

if implant12_clean
  implant12_empty.output("IMPLANT.1", "IMPLANT.1 : Minimum spacing of nimplant/ pimplant to channel : 70nm")
  implant12_empty.output("IMPLANT.2", "IMPLANT.1 : Minimum spacing of nimplant/ pimplant to contact : 25nm")
else
  implant.separation(gate, 70.nm, projection).polygons.without_area(0).output("IMPLANT.1", "IMPLANT.1 : Minimum spacing of nimplant/ pimplant to channel : 70nm")
  implant.separation(cont, 25.nm, projection).polygons.without_area(0).output("IMPLANT.2", "IMPLANT.1 : Minimum spacing of nimplant/ pimplant to contact : 25nm")
end""",
        "IMPLANT.1/.2 transaction",
    )


def add_poly34(text: str) -> str:
    return replace_once(
        text,
        """poly.enclosing(gate, 55.nm, projection).polygons.without_area(0).output("POLY.3", "POLY.3 : Minimum poly extension beyond active : 55nm")
active.enclosing(gate, 70.nm, projection).polygons.without_area(0).output("POLY.4", "POLY.4 : Minimum enclosure of active around gate : 70nm")""",
        """# BEGIN KLAYOUT CUDA POLY34 TRANSACTION
# One qualified transaction may prove both fixed projection-enclosure
# categories empty.  The generated opt-in is fail-closed: a missing method or
# any Ruby/host/backend decline executes both historical CPU expressions
# unchanged and in their original output order.
poly34_request = ENV["KLAYOUT_CUDA_POLY34"].to_s
poly34_requested = !poly34_request.empty? &amp;&amp; poly34_request != "0" &amp;&amp; poly34_request != "false" &amp;&amp; poly34_request != "off"
poly34_clean = false
poly34_error = nil
if poly34_requested
  begin
    poly34_clean = poly.respond_to?(:cuda_poly34_clean?) &amp;&amp; poly.cuda_poly34_clean?(active, gate)
  rescue StandardError =&gt; error
    poly34_clean = false
    poly34_error = "#{error.class}: #{error.message}"
  end
end
poly34_empty = polygon_layer if poly34_clean
info("CUDA POLY.3/.4 transaction: #{poly34_clean ? 'certified-empty' : 'full-cpu-fallback'}") if poly34_requested
info("CUDA POLY.3/.4 Ruby fallback: #{poly34_error}") if poly34_error

if poly34_clean
  poly34_empty.output("POLY.3", "POLY.3 : Minimum poly extension beyond active : 55nm")
  poly34_empty.output("POLY.4", "POLY.4 : Minimum enclosure of active around gate : 70nm")
else
  poly.enclosing(gate, 55.nm, projection).polygons.without_area(0).output("POLY.3", "POLY.3 : Minimum poly extension beyond active : 55nm")
  active.enclosing(gate, 70.nm, projection).polygons.without_area(0).output("POLY.4", "POLY.4 : Minimum enclosure of active around gate : 70nm")
end
# END KLAYOUT CUDA POLY34 TRANSACTION""",
        "POLY.3/.4 transaction",
    )


def add_active3_raw_wells(text: str) -> str:
    text = replace_once(
        text,
        """well = nwell.or(pwell) if need_well""",
        """# BEGIN KLAYOUT CUDA ACTIVE3 RAW WELLS TRANSACTION
# This transaction may bypass WELL construction only when ACTIVE.3 is its sole
# consumer.  Every missing method, exception, host/backend decline, raw hit,
# or uncertainty leaves this flag false and executes the literal source union.
active3_raw_wells_request = ENV["KLAYOUT_CUDA_ACTIVE3_RAW_WELLS"].to_s
active3_raw_wells_requested = !active3_raw_wells_request.empty? &amp;&amp; active3_raw_wells_request != "0" &amp;&amp; active3_raw_wells_request != "false" &amp;&amp; active3_raw_wells_request != "off"
active3_raw_wells_owner = active3_raw_wells_requested &amp;&amp; DRC &amp;&amp; run_active3 &amp;&amp; !run_well &amp;&amp; !run_active4 &amp;&amp; !(OFFGRID &amp;&amp; run_grid)
active3_raw_wells_clean = false
active3_raw_wells_reason = "not-owner"
if active3_raw_wells_owner
  active3_raw_wells_reason = "method-unavailable"
  begin
    if nwell.respond_to?(:cuda_active3_raw_wells_clean?)
      active3_raw_wells_clean = nwell.cuda_active3_raw_wells_clean?(pwell, active)
      active3_raw_wells_reason = active3_raw_wells_clean ? "certified-empty" : "certificate-declined"
    end
  rescue StandardError =&gt; active3_raw_wells_error
    active3_raw_wells_clean = false
    active3_raw_wells_reason = "exception:#{active3_raw_wells_error.class}"
  end
end
active3_raw_wells_empty = polygon_layer if active3_raw_wells_clean
info("CUDA ACTIVE.3 raw-WELL transaction: #{active3_raw_wells_clean ? 'certified-empty' : 'full-cpu-fallback'} reason=#{active3_raw_wells_reason}") if active3_raw_wells_owner

unless active3_raw_wells_clean
  well = nwell.or(pwell) if need_well
end
# END KLAYOUT CUDA ACTIVE3 RAW WELLS TRANSACTION""",
        "ACTIVE.3 raw-WELL union transaction",
    )
    return replace_once(
        text,
        """well.enclosing(active, 55.nm, euclidian).output("ACTIVE.3", "ACTIVE.3 : Minimum enclosure/spacing of nwell/pwell to active: 55nm")""",
        """if active3_raw_wells_clean
  active3_raw_wells_empty.output("ACTIVE.3", "ACTIVE.3 : Minimum enclosure/spacing of nwell/pwell to active: 55nm")
else
  well.enclosing(active, 55.nm, euclidian).output("ACTIVE.3", "ACTIVE.3 : Minimum enclosure/spacing of nwell/pwell to active: 55nm")
end""",
        "ACTIVE.3 raw-WELL output transaction",
    )


def add_active3_well_union(text: str) -> str:
    text = replace_once(
        text,
        """well = nwell.or(pwell) if need_well""",
        """# BEGIN KLAYOUT CUDA ACTIVE3 EXACT WELL UNION TRANSACTION
# This transaction bypasses WELL construction only after an exact resident
# NWELL/PWELL union and complete ACTIVE.3 zero-hit certificate.  Every missing
# method, exception, bounded decline, hit, or uncertainty executes the literal
# source union and rule.
active3_well_union_request = ENV["KLAYOUT_CUDA_ACTIVE3_WELL_UNION"].to_s
active3_well_union_requested = !active3_well_union_request.empty? &amp;&amp; active3_well_union_request != "0" &amp;&amp; active3_well_union_request != "false" &amp;&amp; active3_well_union_request != "off"
active3_well_union_owner = active3_well_union_requested &amp;&amp; DRC &amp;&amp; run_active3 &amp;&amp; !run_well &amp;&amp; !run_active4 &amp;&amp; !(OFFGRID &amp;&amp; run_grid)
active3_well_union_clean = false
active3_well_union_reason = "not-owner"
if active3_well_union_owner
  active3_well_union_reason = "method-unavailable"
  begin
    if nwell.respond_to?(:cuda_active3_well_union_clean?)
      active3_well_union_clean = nwell.cuda_active3_well_union_clean?(pwell, active)
      active3_well_union_reason = active3_well_union_clean ? "certified-empty" : "certificate-declined"
    end
  rescue StandardError =&gt; active3_well_union_error
    active3_well_union_clean = false
    active3_well_union_reason = "exception:#{active3_well_union_error.class}"
  end
end
active3_well_union_empty = polygon_layer if active3_well_union_clean
info("CUDA ACTIVE.3 exact WELL-union transaction: #{active3_well_union_clean ? 'certified-empty' : 'full-cpu-fallback'} reason=#{active3_well_union_reason}") if active3_well_union_owner

unless active3_well_union_clean
  well = nwell.or(pwell) if need_well
end
# END KLAYOUT CUDA ACTIVE3 EXACT WELL UNION TRANSACTION""",
        "ACTIVE.3 exact WELL-union transaction",
    )
    return replace_once(
        text,
        """well.enclosing(active, 55.nm, euclidian).output("ACTIVE.3", "ACTIVE.3 : Minimum enclosure/spacing of nwell/pwell to active: 55nm")""",
        """if active3_well_union_clean
  active3_well_union_empty.output("ACTIVE.3", "ACTIVE.3 : Minimum enclosure/spacing of nwell/pwell to active: 55nm")
else
  well.enclosing(active, 55.nm, euclidian).output("ACTIVE.3", "ACTIVE.3 : Minimum enclosure/spacing of nwell/pwell to active: 55nm")
end""",
        "ACTIVE.3 exact WELL-union output transaction",
    )


def inject_poly34_ruby_exception(text: str) -> str:
    return replace_once(
        text,
        """poly34_clean = poly.respond_to?(:cuda_poly34_clean?) &amp;&amp; poly.cuda_poly34_clean?(active, gate)""",
        """raise("injected POLY34 Ruby exception")""",
        "POLY.3/.4 injected Ruby exception",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--m1-contact",
        action="store_true",
        help="also add the fail-closed CONTACT/METAL1.3 certificate",
    )
    parser.add_argument(
        "--implant12",
        action="store_true",
        help="also add the fail-closed IMPLANT.1/.2 transaction",
    )
    parser.add_argument(
        "--m2-rules",
        action="store_true",
        help="also add the fail-closed live M2.1/.2/.4-.9 transaction",
    )
    parser.add_argument(
        "--m1-5-9",
        action="store_true",
        help="also add the fail-closed exact raw-M1 M1.5-.9 transaction",
    )
    parser.add_argument(
        "--poly34",
        action="store_true",
        help="also add the fail-closed POLY.3/.4 transaction",
    )
    parser.add_argument(
        "--active3-raw-wells",
        action="store_true",
        help="also try ACTIVE.3 before constructing the WELL union",
    )
    parser.add_argument(
        "--active3-well-union",
        action="store_true",
        help="also try exact resident WELL union followed by ACTIVE.3",
    )
    parser.add_argument(
        "--inject-poly34-ruby-exception",
        action="store_true",
        help="gate-only: replace the qualified POLY.3/.4 hook with an exception",
    )
    args = parser.parse_args()
    if args.inject_poly34_ruby_exception and not args.poly34:
        parser.error("--inject-poly34-ruby-exception requires --poly34")
    if args.active3_raw_wells and args.active3_well_union:
        parser.error(
            "--active3-raw-wells and --active3-well-union are alternatives"
        )

    source = args.input.read_text(encoding="utf-8")
    output = transform(source)
    if args.active3_raw_wells:
        output = add_active3_raw_wells(output)
    if args.active3_well_union:
        output = add_active3_well_union(output)
    if args.m2_rules:
        output = add_m2_rules(output)
    if args.m1_5_9:
        output = add_m1_5_9(output)
    if args.implant12:
        output = add_implant12(output)
    if args.poly34:
        output = add_poly34(output)
    if args.inject_poly34_ruby_exception:
        output = inject_poly34_ruby_exception(output)
    if args.m1_contact:
        output = add_m1_contact(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"VIA1_STACK_LIVE_DECK input={args.input} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
