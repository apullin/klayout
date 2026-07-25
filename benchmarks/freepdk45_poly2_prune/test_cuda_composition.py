#!/usr/bin/env python3
"""Composition tests for the POLY.2 prune and live-CUDA deck rewrite."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from prune_deck import PRUNED_BLOCK, SOURCE_BLOCK, prune_deck


HERE = Path(__file__).resolve().parent
CUDA_GENERATOR = (
    HERE.parent / "cuda_spatial_replay" / "make_via1_stack_live_deck.py"
)


def load_cuda_generator():
    spec = importlib.util.spec_from_file_location(
        "poly2_cuda_generator", CUDA_GENERATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the live-CUDA deck generator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


POLY3 = (
    "poly.enclosing(gate, 55.nm, projection).polygons.without_area(0)"
    '.output("POLY.3", "POLY.3 : Minimum poly extension beyond active : '
    '55nm")'
)
POLY4 = (
    "active.enclosing(gate, 70.nm, projection).polygons.without_area(0)"
    '.output("POLY.4", "POLY.4 : Minimum enclosure of active around gate : '
    '70nm")'
)
POLY34_SOURCE_BLOCK = f"{POLY3}\n{POLY4}"


class CudaCompositionTest(unittest.TestCase):
    def test_prune_composes_after_poly34_transaction(self) -> None:
        generator = load_cuda_generator()
        source = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            "<klayout-macro>\n"
            "<text>\n"
            f"{SOURCE_BLOCK}"
            f"{POLY34_SOURCE_BLOCK}\n"
            "</text>\n"
            "</klayout-macro>\n"
        )

        cuda = generator.add_poly34(source)
        composed = prune_deck(cuda)

        self.assertIn(PRUNED_BLOCK, composed)
        self.assertNotIn("poly_sep_active", composed)
        self.assertNotIn(
            "poly.separation(active, 140.nm, projection)", composed
        )
        self.assertNotIn('.output("POLY.2"', composed)
        self.assertEqual(
            composed.count(
                'poly34_request = ENV["KLAYOUT_CUDA_POLY34"].to_s'
            ),
            1,
        )
        self.assertEqual(composed.count("poly34_empty.output"), 2)
        self.assertEqual(composed.count(POLY3), 1)
        self.assertEqual(composed.count(POLY4), 1)
        self.assertEqual(composed, prune_deck(cuda))

    def test_default_cuda_rewrite_preserves_poly2_source(self) -> None:
        generator = load_cuda_generator()
        source = f"before\n{SOURCE_BLOCK}{POLY34_SOURCE_BLOCK}\nafter\n"

        cuda = generator.add_poly34(source)

        self.assertIn(SOURCE_BLOCK, cuda)
        self.assertIn(
            "poly.separation(active, 140.nm, projection)", cuda
        )
        self.assertEqual(cuda.count('.output("POLY.2"'), 1)


if __name__ == "__main__":
    unittest.main()
