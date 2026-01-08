#!/usr/bin/env python3
import pandas as pd
import glob
import os

results_dir = "/Users/finnrobl/Documents/k-NN-2/k-NN/benchmark_results_20260106_165958/osb-results"

metrics = ["50th percentile latency", "90th percentile latency", "99th percentile latency", "Mean Throughput"]

def extract_metrics(csv_path):
    df = pd.read_csv(csv_path)
    results = {}
    for metric in metrics:
        row = df[(df["Metric"] == metric) & (df["Task"] == "prod-queries")]
        if not row.empty:
            results[metric] = row["Value"].values[0]
    return results

print("=" * 70)
print("BASELINE RESULTS")
print("=" * 70)
baseline_data = []
for f in sorted(glob.glob(f"{results_dir}/baseline_run*.csv")):
    m = extract_metrics(f)
    baseline_data.append(m)
    print(f"{os.path.basename(f)}: p50={m.get('50th percentile latency', 'N/A'):.2f}ms, p90={m.get('90th percentile latency', 'N/A'):.2f}ms, p99={m.get('99th percentile latency', 'N/A'):.2f}ms, throughput={m.get('Mean Throughput', 'N/A'):.2f} ops/s")

print("\n" + "=" * 70)
print("OPTIMIZED RESULTS")
print("=" * 70)
optimized_data = []
for f in sorted(glob.glob(f"{results_dir}/optimized_run*.csv")):
    m = extract_metrics(f)
    optimized_data.append(m)
    print(f"{os.path.basename(f)}: p50={m.get('50th percentile latency', 'N/A'):.2f}ms, p90={m.get('90th percentile latency', 'N/A'):.2f}ms, p99={m.get('99th percentile latency', 'N/A'):.2f}ms, throughput={m.get('Mean Throughput', 'N/A'):.2f} ops/s")

print("\n" + "=" * 70)
print("AVERAGES")
print("=" * 70)
for metric in metrics:
    baseline_avg = sum(d.get(metric, 0) for d in baseline_data) / len(baseline_data)
    optimized_avg = sum(d.get(metric, 0) for d in optimized_data) / len(optimized_data)
    diff = ((optimized_avg - baseline_avg) / baseline_avg) * 100 if baseline_avg else 0
    print(f"{metric}: baseline={baseline_avg:.2f}, optimized={optimized_avg:.2f}, change={diff:+.1f}%")
