use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use renegade::{DataPoint, Renegade};
use std::time::Instant;

#[derive(Clone, Debug)]
struct ProfilePoint {
    features: Vec<f64>,
    ranges: Vec<(f64, f64)>,
}

impl DataPoint for ProfilePoint {
    fn feature_distances(&self, other: &Self) -> Vec<f64> {
        self.features
            .iter()
            .zip(other.features.iter())
            .zip(self.ranges.iter())
            .map(|((a, b), (lo, hi))| {
                let range = hi - lo;
                if range == 0.0 {
                    0.0
                } else {
                    (a - b).abs() / range
                }
            })
            .collect()
    }

    fn feature_values(&self) -> Vec<f64> {
        self.features.clone()
    }
}

fn make_dataset(n: usize, d: usize, seed: u64) -> Vec<(ProfilePoint, f64)> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let ranges = vec![(0.0, 1.0); d];
    (0..n)
        .map(|_| {
            let features: Vec<f64> = (0..d).map(|_| rng.gen()).collect();
            let output: f64 = features.iter().take(3.min(d)).sum();
            (
                ProfilePoint {
                    features,
                    ranges: ranges.clone(),
                },
                output,
            )
        })
        .collect()
}

#[test]
fn profile_training_and_inference() {
    eprintln!();
    eprintln!("=== Performance Profile ===");
    eprintln!(
        "{:<8} {:<8} {:<15} {:<15} {:<15} {:<15}",
        "n", "d", "train (ms)", "query 1 (µs)", "query 100 (ms)", "predict (µs)"
    );
    eprintln!("{}", "-".repeat(80));

    for &(n, d) in &[
        (100, 5),
        (500, 5),
        (1000, 5),
        (1000, 20),
        (5000, 5),
        (5000, 20),
        (10000, 5),
    ] {
        let data = make_dataset(n, d, 42);
        let query_point = data[0].0.clone();

        // Measure training (ensure_trained via get_optimal_k)
        let mut model = Renegade::new();
        for (p, o) in &data {
            model.add(p.clone(), *o);
        }

        let t0 = Instant::now();
        model.get_optimal_k();
        let train_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Measure single query (inference)
        let t0 = Instant::now();
        let _ = model.query_k(&query_point, 5);
        let query_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

        // Measure 100 queries
        let t0 = Instant::now();
        for i in 0..100 {
            let _ = model.query_k(&data[i % data.len()].0, 5);
        }
        let query_100_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Measure predict (includes weighted mean computation)
        let t0 = Instant::now();
        let _ = model.predict_k(&query_point, 5);
        let predict_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

        eprintln!(
            "{:<8} {:<8} {:<15.1} {:<15.0} {:<15.1} {:<15.0}",
            n, d, train_ms, query_us, query_100_ms, predict_us
        );
    }
    eprintln!();

    // Profile the hot path: what takes time in a single query?
    eprintln!("=== Query Hot Path Breakdown (n=5000, d=5) ===");
    let data = make_dataset(5000, 5, 42);
    let mut model = Renegade::new();
    for (p, o) in &data {
        model.add(p.clone(), *o);
    }
    model.get_optimal_k();

    let query = &data[0].0;
    let query_values = query.feature_values();

    let entries_count = data.len();
    let t0 = Instant::now();
    let mut dists = Vec::with_capacity(entries_count);
    for (p, _) in &data {
        let d = query.feature_distances(p);
        let mean = d.iter().sum::<f64>() / d.len() as f64;
        dists.push(mean);
    }
    let dist_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Sort
    let t0 = Instant::now();
    let mut indexed: Vec<(usize, f64)> = dists.iter().enumerate().map(|(i, &d)| (i, d)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let sort_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Vec allocation for feature_distances
    let t0 = Instant::now();
    for (p, _) in &data {
        let _ = query.feature_distances(p);
    }
    let alloc_ms = t0.elapsed().as_secs_f64() * 1000.0;

    eprintln!("  Distance computation (n=5000): {:.2} ms", dist_ms);
    eprintln!("  Vec allocation overhead (n=5000): {:.2} ms", alloc_ms);
    eprintln!("  Sort (n=5000): {:.2} ms", sort_ms);
    eprintln!();
}
