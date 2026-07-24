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
    args = parser.parse_args()

    source = args.input.read_text(encoding="utf-8")
    output = transform(source)
    if args.implant12:
        output = add_implant12(output)
    if args.m1_contact:
        output = add_m1_contact(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(f"VIA1_STACK_LIVE_DECK input={args.input} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
