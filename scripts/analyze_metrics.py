#!/usr/bin/env python3
import argparse
import csv
import os
import sys


LOWER_IS_BETTER = ["L1", "rmse_linear", "abs_relative_difference"]
HIGHER_IS_BETTER = ["delta4_acc_105", "delta5_acc110", "delta1_acc"]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze evaluation metrics CSV")
    parser.add_argument("--csv", type=str, required=True, help="Path to all_metrics_*.csv")
    return parser.parse_args()


def read_rows(csv_path):
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        for key, value in list(row.items()):
            if key != "name" and value != "":
                row[key] = float(value)
    return rows


def print_worst(rows, metric, reverse):
    sorted_rows = sorted(rows, key=lambda row: row[metric], reverse=reverse)
    worst_samples = sorted_rows[-10:]
    if not worst_samples:
        return

    print(f"\n[{metric}] worst 10 samples:")
    print(f"Worst value: {worst_samples[-1][metric]:.6f}")
    for row in worst_samples:
        print(f"  {row['name']} : {row[metric]:.6f}")


def print_stats(rows, metric):
    values = [row[metric] for row in rows]
    values_sorted = sorted(values)
    count = len(values)
    mean = sum(values) / count
    median = (
        values_sorted[count // 2]
        if count % 2
        else (values_sorted[count // 2 - 1] + values_sorted[count // 2]) / 2
    )
    print(
        f"{metric}: count={count}, mean={mean:.6f}, median={median:.6f}, "
        f"min={min(values):.6f}, max={max(values):.6f}"
    )


def main():
    args = parse_args()
    if not os.path.exists(args.csv):
        print(f"File not found: {args.csv}")
        sys.exit(1)

    rows = read_rows(args.csv)
    if not rows:
        print("No rows found.")
        return

    print("=" * 80)
    print("CSV metrics analysis")
    print("=" * 80)
    print(f"Total records: {len(rows)}")

    print("\n" + "=" * 80)
    print("Lower-is-better metrics")
    print("=" * 80)
    for metric in LOWER_IS_BETTER:
        if metric in rows[0]:
            print_worst(rows, metric, reverse=False)

    print("\n" + "=" * 80)
    print("Higher-is-better metrics")
    print("=" * 80)
    for metric in HIGHER_IS_BETTER:
        if metric in rows[0]:
            print_worst(rows, metric, reverse=True)

    print("\n" + "=" * 80)
    print("Summary statistics")
    print("=" * 80)
    for metric in LOWER_IS_BETTER + HIGHER_IS_BETTER:
        if metric in rows[0]:
            print_stats(rows, metric)


if __name__ == "__main__":
    main()
