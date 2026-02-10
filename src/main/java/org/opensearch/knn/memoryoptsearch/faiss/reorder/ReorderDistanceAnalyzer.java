/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.memoryoptsearch.faiss.reorder;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.DoubleSummaryStatistics;

/**
 * Utility to analyze the distance between internal vector ordinals before and after reordering.
 * 
 * Given a permutation file (newOrd2Old mapping), this computes statistics on how far
 * vectors moved from their original positions.
 */
public class ReorderDistanceAnalyzer {

    public record Stats(double mean, double stdDev, long count, double min, double max) {
        @Override
        public String toString() {
            return String.format("count=%d, mean=%.2f, stdDev=%.2f, min=%.0f, max=%.0f", count, mean, stdDev, min, max);
        }
    }

    /**
     * Saves the permutation (newOrd2Old) to a file.
     */
    public static void savePermutation(int[] newOrd2Old, Path outputPath) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(outputPath)) {
            for (int i = 0; i < newOrd2Old.length; i++) {
                writer.write(Integer.toString(newOrd2Old[i]));
                writer.newLine();
            }
        }
    }

    /**
     * Loads a permutation from file.
     */
    public static int[] loadPermutation(Path inputPath) throws IOException {
        try (BufferedReader reader = Files.newBufferedReader(inputPath)) {
            return reader.lines().mapToInt(Integer::parseInt).toArray();
        }
    }

    /**
     * Computes the displacement statistics for a reordering permutation.
     * Displacement = |newOrd - oldOrd| for each vector.
     */
    public static Stats computeDisplacementStats(int[] newOrd2Old) {
        if (newOrd2Old == null || newOrd2Old.length == 0) {
            return new Stats(0, 0, 0, 0, 0);
        }

        double[] displacements = new double[newOrd2Old.length];
        for (int newOrd = 0; newOrd < newOrd2Old.length; newOrd++) {
            int oldOrd = newOrd2Old[newOrd];
            displacements[newOrd] = Math.abs(newOrd - oldOrd);
        }

        DoubleSummaryStatistics summary = java.util.Arrays.stream(displacements).summaryStatistics();
        double mean = summary.getAverage();
        double variance = java.util.Arrays.stream(displacements)
            .map(d -> (d - mean) * (d - mean))
            .average()
            .orElse(0);

        return new Stats(mean, Math.sqrt(variance), summary.getCount(), summary.getMin(), summary.getMax());
    }

    /**
     * Computes baseline stats (1:1 mapping where newOrd == oldOrd).
     */
    public static Stats computeBaselineStats(int numVectors) {
        return new Stats(0, 0, numVectors, 0, 0);
    }

    /**
     * Compares reordered vs baseline and prints analysis.
     */
    public static String compareStats(Stats reordered, Stats baseline) {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Reorder Distance Analysis ===\n");
        sb.append(String.format("Baseline (1:1):  %s%n", baseline));
        sb.append(String.format("Reordered:       %s%n", reordered));
        sb.append(String.format("Mean increase:   %.2f%n", reordered.mean - baseline.mean));
        sb.append(String.format("StdDev increase: %.2f%n", reordered.stdDev - baseline.stdDev));
        return sb.toString();
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.out.println("Usage: ReorderDistanceAnalyzer <permutation-file>");
            System.out.println("  Analyzes displacement statistics from a permutation file.");
            return;
        }

        Path permPath = Path.of(args[0]);
        int[] permutation = loadPermutation(permPath);

        Stats reordered = computeDisplacementStats(permutation);
        Stats baseline = computeBaselineStats(permutation.length);

        System.out.println(compareStats(reordered, baseline));
    }
}
