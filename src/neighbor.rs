use crate::predict::ExtrapolatedPrediction;

/// A single nearest neighbor result.
#[derive(Debug, Clone)]
pub struct Neighbor {
    /// Distance from query point (0 = identical).
    pub distance: f64,
    /// Output value of this training point.
    pub output: f64,
    /// Instance weight (default 1.0). Higher weight means this point
    /// has more influence on predictions.
    pub weight: f64,
}

/// A set of nearest neighbors, sorted by distance.
#[derive(Debug, Clone)]
pub struct Neighbors {
    pub neighbors: Vec<Neighbor>,
}

impl Neighbors {
    /// Weighted average of neighbor outputs.
    /// Combines inverse-distance weighting with instance weights:
    /// effective_weight = instance_weight / distance.
    pub fn weighted_mean(&self) -> f64 {
        if self.neighbors.is_empty() {
            return f64::NAN;
        }

        // If any neighbor has distance 0, return weighted average of exact matches.
        let exact: Vec<&Neighbor> = self
            .neighbors
            .iter()
            .filter(|n| n.distance == 0.0)
            .collect();
        if !exact.is_empty() {
            let total_w: f64 = exact.iter().map(|n| n.weight).sum();
            if total_w > 0.0 {
                return exact.iter().map(|n| n.weight * n.output).sum::<f64>() / total_w;
            }
            return exact[0].output;
        }

        let mut weight_sum = 0.0;
        let mut value_sum = 0.0;
        for n in &self.neighbors {
            let w = n.weight / n.distance;
            weight_sum += w;
            value_sum += w * n.output;
        }
        value_sum / weight_sum
    }

    /// Extrapolate output to distance=0 by fitting a linear trend.
    pub fn extrapolate(&self) -> ExtrapolatedPrediction {
        ExtrapolatedPrediction::from_neighbors(&self.neighbors)
    }

    /// Class probabilities: weighted fraction of neighbors with each distinct output value.
    /// Combines inverse-distance weighting with instance weights.
    pub fn class_votes(&self) -> Vec<(f64, f64)> {
        if self.neighbors.is_empty() {
            return Vec::new();
        }

        let mut counts: Vec<(f64, f64)> = Vec::new(); // (class, total_weight)
        for n in &self.neighbors {
            let w = if n.distance == 0.0 {
                n.weight * 1e6 // very large but finite weight for exact matches
            } else {
                n.weight / n.distance
            };
            if let Some(entry) = counts
                .iter_mut()
                .find(|(v, _)| (*v - n.output).abs() < 1e-10)
            {
                entry.1 += w;
            } else {
                counts.push((n.output, w));
            }
        }

        let total: f64 = counts.iter().map(|(_, w)| w).sum();
        let n_classes = counts.len() as f64;
        if total > 0.0 {
            counts
                .into_iter()
                .map(|(class, w)| (class, w / total))
                .collect()
        } else {
            counts
                .into_iter()
                .map(|(class, _)| (class, 1.0 / n_classes))
                .collect()
        }
    }

    /// Random sample from neighbors (uniform).
    pub fn sample(&self, rng_value: f64) -> Option<f64> {
        if self.neighbors.is_empty() {
            return None;
        }
        let idx = (rng_value * self.neighbors.len() as f64) as usize;
        let idx = idx.min(self.neighbors.len() - 1);
        Some(self.neighbors[idx].output)
    }
}
