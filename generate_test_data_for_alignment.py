"""Generate simple CSV datasets for testing Tucker alignment.

The generated CSV columns are:
    x,z,t,fracture

Edit target_function(x, z, t) below to choose the multivariate function.

Example:
    python generate_test_data_for_alignment.py --name align_test_ref
"""
import argparse
import csv
import math
from pathlib import Path


def target_function(x, z, t):
    """Edit this function to generate any scalar field f(x, z, t)."""
    return (
        math.sin(math.pi * x)
        * math.cos(math.pi * z)
        * math.exp(-t)
        + 0.25 * x * z
        + 0.1 * t
    )


def linspace(n, start=1.0, stop=2.0):
    if n <= 1:
        return [start]
    return [start + (stop - start) * i / (n - 1) for i in range(n)]


def write_dataset(path, nx, nz, nt, x_range, z_range, t_range):
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = linspace(nx,1,2)
    zs = linspace(nz,1,2)
    ts = linspace(nt,1,2)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "z", "t", "target"])

        for t in ts:
            for x in xs:
                for z in zs:
                    writer.writerow([x, z, t, target_function(x, z, t)])


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic alignment-test CSV data.")
    parser.add_argument("--name", default="align_test_ref", help="Output dataset name without .csv.")
    parser.add_argument("--nx", type=int, default=32, help="Number of x grid points.")
    parser.add_argument("--nz", type=int, default=32, help="Number of z grid points.")
    parser.add_argument("--nt", type=int, default=16, help="Number of time points.")
    parser.add_argument("--x-range", type=float, nargs=2, default=[0.0, 1.0], metavar=("MIN", "MAX"))
    parser.add_argument("--z-range", type=float, nargs=2, default=[0.0, 1.0], metavar=("MIN", "MAX"))
    parser.add_argument("--t-range", type=float, nargs=2, default=[0.0, 1.0], metavar=("MIN", "MAX"))
    parser.add_argument("--output-dir", default="data", help="Directory for generated CSV.")
    args = parser.parse_args()

    output_path = Path(args.output_dir) / f"{args.name}.csv"
    write_dataset(output_path, args.nx, args.nz, args.nt, args.x_range, args.z_range, args.t_range)

    nrows = args.nx * args.nz * args.nt
    print(f"Wrote {nrows} rows to {output_path}")


if __name__ == "__main__":
    main()
