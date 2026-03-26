use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A vantage-point tree for efficient nearest neighbor search in metric spaces.
///
/// Works with any distance function. Provides exact KNN results (not approximate).
/// Build is O(n log n), query is O(log n) average case for low intrinsic dimensionality,
/// degrading toward O(n) for high-dimensional data (never worse than brute force).
pub struct VpTree {
    nodes: Vec<VpNode>,
}

struct VpNode {
    /// Index of the data point this node represents.
    index: usize,
    /// Median distance from this vantage point to all points in its subtree.
    threshold: f64,
    /// Left child: points closer than threshold. None if leaf.
    left: Option<usize>,
    /// Right child: points farther than threshold. None if leaf.
    right: Option<usize>,
}

/// A candidate neighbor during KNN search.
#[derive(PartialEq)]
struct Candidate {
    distance: f64,
    index: usize,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: largest distance at top (so we can pop the farthest neighbor)
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl VpTree {
    /// Build a VP-tree from a set of point indices and a distance function.
    ///
    /// `distances` is a function that returns the distance between two points
    /// given their indices. Must satisfy metric properties (symmetry, triangle inequality).
    pub fn build<F>(n: usize, distances: &F) -> Self
    where
        F: Fn(usize, usize) -> f64,
    {
        if n == 0 {
            return VpTree { nodes: Vec::new() };
        }

        let mut indices: Vec<usize> = (0..n).collect();
        let mut nodes = Vec::with_capacity(n);

        Self::build_recursive(&mut indices, 0, n, distances, &mut nodes);

        VpTree { nodes }
    }

    fn build_recursive<F>(
        indices: &mut [usize],
        start: usize,
        end: usize,
        distances: &F,
        nodes: &mut Vec<VpNode>,
    ) -> usize
    where
        F: Fn(usize, usize) -> f64,
    {
        if start >= end {
            return usize::MAX; // sentinel for "no node"
        }

        let node_idx = nodes.len();

        if end - start == 1 {
            // Leaf node
            nodes.push(VpNode {
                index: indices[start],
                threshold: 0.0,
                left: None,
                right: None,
            });
            return node_idx;
        }

        // Use the first point as vantage point
        let vp = indices[start];

        // Compute distances from vantage point to all other points in this subset
        let subset = &mut indices[start + 1..end];
        let count = subset.len();

        // Sort subset by distance to vantage point
        subset.sort_by(|&a, &b| {
            let da = distances(vp, a);
            let db = distances(vp, b);
            da.partial_cmp(&db).unwrap_or(Ordering::Equal)
        });

        // Median distance
        let median_idx = count / 2;
        let threshold = distances(vp, subset[median_idx]);

        // Placeholder node — we'll fill in children after recursing
        nodes.push(VpNode {
            index: vp,
            threshold,
            left: None,
            right: None,
        });

        // Left subtree: points closer than or equal to threshold
        // These are indices[start+1..start+1+median_idx+1]
        let left_start = start + 1;
        let left_end = start + 1 + median_idx;
        let left = if left_start < left_end {
            Some(Self::build_recursive(
                indices, left_start, left_end, distances, nodes,
            ))
        } else {
            None
        };

        // Right subtree: points farther than threshold
        let right_start = left_end;
        let right_end = end;
        let right = if right_start < right_end {
            Some(Self::build_recursive(
                indices,
                right_start,
                right_end,
                distances,
                nodes,
            ))
        } else {
            None
        };

        nodes[node_idx].left = left;
        nodes[node_idx].right = right;

        node_idx
    }

    /// Find the k nearest neighbors to a query point.
    ///
    /// `distance_to_query` returns the distance from the query point to the
    /// point at the given index.
    ///
    /// Returns Vec of (index, distance) sorted by distance (closest first).
    pub fn query_nearest<F>(&self, k: usize, distance_to_query: &F) -> Vec<(usize, f64)>
    where
        F: Fn(usize) -> f64,
    {
        if self.nodes.is_empty() || k == 0 {
            return Vec::new();
        }

        // Max-heap of size k: tracks the k best candidates so far.
        // The top of the heap is the farthest of the k best.
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);

        self.search_recursive(0, k, distance_to_query, &mut heap);

        // Extract results sorted by distance (closest first)
        let mut results: Vec<(usize, f64)> =
            heap.into_iter().map(|c| (c.index, c.distance)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    fn search_recursive<F>(
        &self,
        node_idx: usize,
        k: usize,
        distance_to_query: &F,
        heap: &mut BinaryHeap<Candidate>,
    ) where
        F: Fn(usize) -> f64,
    {
        let node = &self.nodes[node_idx];
        let dist = distance_to_query(node.index);

        // Consider this point as a candidate
        if heap.len() < k {
            heap.push(Candidate {
                distance: dist,
                index: node.index,
            });
        } else if let Some(worst) = heap.peek() {
            if dist < worst.distance {
                heap.pop();
                heap.push(Candidate {
                    distance: dist,
                    index: node.index,
                });
            }
        }

        // Decide which subtrees to search
        if dist < node.threshold {
            // Query is closer to vantage point than threshold — search left first
            if let Some(left) = node.left {
                // Left contains points within threshold of vantage point.
                // Any point in left subtree has distance from vp <= threshold.
                // By triangle inequality, closest possible distance to query is
                // >= |dist - threshold| ... but we should search if dist - tau < threshold
                self.search_recursive(left, k, distance_to_query, heap);
            }
            // Update tau after searching left
            let tau = if heap.len() < k {
                f64::MAX
            } else {
                heap.peek().unwrap().distance
            };
            if let Some(right) = node.right {
                // Right contains points with distance from vp > threshold.
                // Closest possible point in right subtree to query:
                // By triangle inequality, at least (threshold - dist).
                // Search right only if that minimum could beat tau.
                if dist + tau >= node.threshold {
                    self.search_recursive(right, k, distance_to_query, heap);
                }
            }
        } else {
            // Query is farther from vantage point — search right first
            if let Some(right) = node.right {
                self.search_recursive(right, k, distance_to_query, heap);
            }
            let tau = if heap.len() < k {
                f64::MAX
            } else {
                heap.peek().unwrap().distance
            };
            if let Some(left) = node.left {
                // Closest possible point in left subtree to query:
                // at least (dist - threshold)
                if dist - tau <= node.threshold {
                    self.search_recursive(left, k, distance_to_query, heap);
                }
            }
        }
    }

    /// Number of points in the tree.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1D points for easy verification.
    fn distance_1d(points: &[f64]) -> impl Fn(usize, usize) -> f64 + '_ {
        move |a, b| (points[a] - points[b]).abs()
    }

    fn query_1d<'a>(points: &'a [f64], query: f64) -> impl Fn(usize) -> f64 + 'a {
        move |i| (points[i] - query).abs()
    }

    /// Brute-force KNN for verification.
    fn brute_force_knn(
        n: usize,
        k: usize,
        distance_to_query: &dyn Fn(usize) -> f64,
    ) -> Vec<(usize, f64)> {
        let mut dists: Vec<(usize, f64)> = (0..n).map(|i| (i, distance_to_query(i))).collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        dists.truncate(k);
        dists
    }

    #[test]
    fn single_point() {
        let points = vec![5.0];
        let tree = VpTree::build(1, &distance_1d(&points));
        let results = tree.query_nearest(1, &query_1d(&points, 3.0));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!((results[0].1 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn two_points() {
        let points = vec![0.0, 10.0];
        let tree = VpTree::build(2, &distance_1d(&points));

        let results = tree.query_nearest(1, &query_1d(&points, 3.0));
        assert_eq!(results[0].0, 0); // 0.0 is closer to 3.0

        let results = tree.query_nearest(1, &query_1d(&points, 8.0));
        assert_eq!(results[0].0, 1); // 10.0 is closer to 8.0

        let results = tree.query_nearest(2, &query_1d(&points, 5.0));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn knn_matches_brute_force_1d() {
        let points: Vec<f64> = (0..100).map(|i| i as f64 * 0.7 + 3.0).collect();
        let tree = VpTree::build(100, &distance_1d(&points));

        for &query in &[0.0, 5.0, 25.0, 50.0, 75.0, 100.0] {
            for k in [1, 3, 5, 10, 20] {
                let vp_results = tree.query_nearest(k, &query_1d(&points, query));
                let bf_results = brute_force_knn(100, k, &|i| (points[i] - query).abs());

                assert_eq!(
                    vp_results.len(),
                    bf_results.len(),
                    "Length mismatch at query={}, k={}",
                    query,
                    k
                );

                // Same set of indices (order may differ for equidistant points)
                let mut vp_indices: Vec<usize> = vp_results.iter().map(|r| r.0).collect();
                let mut bf_indices: Vec<usize> = bf_results.iter().map(|r| r.0).collect();
                vp_indices.sort();
                bf_indices.sort();
                assert_eq!(
                    vp_indices, bf_indices,
                    "Index mismatch at query={}, k={}",
                    query, k
                );

                // Same distances
                for (vp, bf) in vp_results.iter().zip(bf_results.iter()) {
                    assert!(
                        (vp.1 - bf.1).abs() < 1e-10,
                        "Distance mismatch at query={}, k={}: vp={}, bf={}",
                        query,
                        k,
                        vp.1,
                        bf.1
                    );
                }
            }
        }
    }

    #[test]
    fn knn_matches_brute_force_2d() {
        // 2D points with Euclidean distance
        let points: Vec<(f64, f64)> = (0..200)
            .map(|i| {
                let x = (i as f64 * 1.3) % 10.0;
                let y = (i as f64 * 0.7) % 10.0;
                (x, y)
            })
            .collect();

        let dist_fn = |a: usize, b: usize| -> f64 {
            let dx = points[a].0 - points[b].0;
            let dy = points[a].1 - points[b].1;
            (dx * dx + dy * dy).sqrt()
        };

        let tree = VpTree::build(200, &dist_fn);

        let queries = [(0.0, 0.0), (5.0, 5.0), (10.0, 0.0), (3.3, 7.7)];
        for (qx, qy) in queries {
            let query_dist = |i: usize| -> f64 {
                let dx = points[i].0 - qx;
                let dy = points[i].1 - qy;
                (dx * dx + dy * dy).sqrt()
            };

            for k in [1, 5, 10, 20] {
                let vp_results = tree.query_nearest(k, &query_dist);
                let bf_results = brute_force_knn(200, k, &query_dist);

                // Compare distances (not indices — ties in distance could reorder)
                let vp_dists: Vec<f64> = vp_results.iter().map(|r| (r.1 * 1e6).round()).collect();
                let bf_dists: Vec<f64> = bf_results.iter().map(|r| (r.1 * 1e6).round()).collect();
                assert_eq!(
                    vp_dists, bf_dists,
                    "Distance mismatch at query=({},{}), k={}",
                    qx, qy, k
                );
            }
        }
    }

    #[test]
    fn knn_matches_brute_force_high_dimensional() {
        // 20D random-ish points
        let d = 20;
        let n = 500;
        let points: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..d)
                    .map(|j| ((i * 37 + j * 53) % 100) as f64 / 100.0)
                    .collect()
            })
            .collect();

        let dist_fn = |a: usize, b: usize| -> f64 {
            points[a]
                .iter()
                .zip(points[b].iter())
                .map(|(x, y)| (x - y).abs())
                .sum::<f64>()
                / d as f64
        };

        let tree = VpTree::build(n, &dist_fn);

        // Test with several query points
        for qi in [0, 50, 100, 250, 499] {
            let query_dist = |i: usize| -> f64 { dist_fn(qi, i) };

            for k in [1, 5, 10] {
                let vp_results = tree.query_nearest(k, &query_dist);
                let bf_results = brute_force_knn(n, k, &query_dist);

                let vp_dists: Vec<f64> = vp_results.iter().map(|r| (r.1 * 1e8).round()).collect();
                let bf_dists: Vec<f64> = bf_results.iter().map(|r| (r.1 * 1e8).round()).collect();
                assert_eq!(
                    vp_dists, bf_dists,
                    "Distance mismatch at qi={}, k={}",
                    qi, k
                );
            }
        }
    }

    #[test]
    fn duplicate_points() {
        let points = vec![1.0, 1.0, 1.0, 5.0, 5.0];
        let tree = VpTree::build(5, &distance_1d(&points));

        let results = tree.query_nearest(3, &query_1d(&points, 1.0));
        assert_eq!(results.len(), 3);
        // All three nearest should be the duplicates at 1.0
        for r in &results {
            assert_eq!(r.1, 0.0);
        }
    }

    #[test]
    fn k_larger_than_n() {
        let points = vec![1.0, 2.0, 3.0];
        let tree = VpTree::build(3, &distance_1d(&points));

        let results = tree.query_nearest(10, &query_1d(&points, 0.0));
        assert_eq!(results.len(), 3); // can't return more than n
    }

    #[test]
    fn empty_tree() {
        let tree = VpTree::build(0, &|_a: usize, _b: usize| 0.0);
        let results = tree.query_nearest(5, &|_i: usize| 0.0);
        assert!(results.is_empty());
    }

    #[test]
    fn query_point_in_dataset() {
        // Query for a point that exists in the dataset — distance 0
        let points: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let tree = VpTree::build(50, &distance_1d(&points));

        let results = tree.query_nearest(1, &query_1d(&points, 25.0));
        assert_eq!(results[0].0, 25);
        assert_eq!(results[0].1, 0.0);
    }

    #[test]
    fn stress_test_random() {
        // Larger random test to shake out edge cases
        let n = 1000;
        let d = 10;
        let points: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..d)
                    .map(|j| ((i * 97 + j * 31 + 17) % 1000) as f64 / 1000.0)
                    .collect()
            })
            .collect();

        let dist_fn = |a: usize, b: usize| -> f64 {
            points[a]
                .iter()
                .zip(points[b].iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt()
        };

        let tree = VpTree::build(n, &dist_fn);

        // Test 50 random queries
        for qi in (0..n).step_by(20) {
            let query_dist = |i: usize| -> f64 { dist_fn(qi, i) };

            let vp_results = tree.query_nearest(7, &query_dist);
            let bf_results = brute_force_knn(n, 7, &query_dist);

            let vp_dists: Vec<f64> = vp_results.iter().map(|r| (r.1 * 1e8).round()).collect();
            let bf_dists: Vec<f64> = bf_results.iter().map(|r| (r.1 * 1e8).round()).collect();
            assert_eq!(vp_dists, bf_dists, "Mismatch at qi={}", qi);
        }
    }
}
