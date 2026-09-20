"""plot_resultant_diagram.py: writes figures/report/resultant_vector_diagram.png (issue #10).

Draws one HYCOM grid cell to scale with four synthetic corner vectors, the particle and the
resultant, beside a comparison of the correct and the wrong average. It needs no data.

The same figure comes out of `python -m sar.model.interpolate --diagram`; this wrapper
exists so the figure sits with the project's other figure scripts.

Run:
    python scripts/plot_resultant_diagram.py [--out path.png]
"""

import argparse

from sar.model.interpolate import plot_resultant_cell, synthetic_sample


def main() -> None:
    p = argparse.ArgumentParser(description="Draw the resultant vector diagram")
    p.add_argument("--out", default="figures/report/resultant_vector_diagram.png")
    args = p.parse_args()
    plot_resultant_cell(synthetic_sample(), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
