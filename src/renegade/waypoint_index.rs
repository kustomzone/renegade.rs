use std::ops::Add;

use bit_vec::BitVec;
use rand::prelude::{Rng, ThreadRng};
use std::collections::HashSet;

#[derive(Clone)]
pub struct WaypointIndex<I: Clone + PartialEq> {
    waypoints: Vec<(I, I)>,
    dist: Box<fn(&I, &I) -> f64>,
}

impl<I: Clone + PartialEq> WaypointIndex<I> {
    pub fn new(
        data: &Vec<I>,
        dist: fn(&I, &I) -> f64,
        waypoint_count: u8,
        sample_count: usize,
        rng: &mut ThreadRng,
    ) -> WaypointIndex<I> {
        let samples = Self::sample(rng, data, sample_count);

        let mut waypoints: Vec<(I, I)> = vec![];
        'w: for _ in 0..waypoint_count {
            let mut first_best: Option<(I, I, f64)> = Option::None;
            for _ in 0..sample_count {
                let (a, b) = Self::select_distinct_pair(rng, data);
                let dst = if waypoints.is_empty() {
                    dist(&a, &b)
                } else {
                    1.0
                };
                match first_best {
                    None => first_best = Option::Some((a, b, dst)),
                    Some((_, _, d)) => {
                        if dst > d {
                            first_best = Option::Some((a, b, dst))
                        }
                    }
                }
            }
            match first_best {
                Some((a, b, _)) => {
                    waypoints.push((a, b));
                }
                None => {
                    panic!("Unable to find waypoint")
                }
            }
        }
        WaypointIndex {
            waypoints,
            dist: Box::new(dist),
        }
    }

    pub fn bit_vec(&self, item: &I) -> BitVec {
        let mut bv = BitVec::from_elem(self.waypoints.len(), false);
        for (ix, wp_pair) in self.waypoints.iter().enumerate() {
            let d1 = (self.dist)(&wp_pair.0, item);
            let d2 = (self.dist)(&wp_pair.1, item);
            bv.set(ix, d1 < d2);
        }
        bv
    }

    fn select_distinct_pair(rng: &mut ThreadRng, vec: &Vec<I>) -> (I, I) {
        loop {
            let a = vec[rng.gen_range(0..vec.len())].clone();
            let b = vec[rng.gen_range(0..vec.len())].clone();
            if a != b {
                return (a, b);
            }
        }
    }

    fn select_random(rng: &mut ThreadRng, vec: &Vec<I>) -> I {
        vec[rng.gen_range(0..vec.len())].clone()
    }

    fn sample(rng: &mut ThreadRng, data: &Vec<I>, sample_count: usize) -> Vec<I> {
        assert!(data.len() > sample_count*2);
        let mut samples: HashSet<I> = HashSet::new();
        while samples.len() < sample_count {
            samples.insert(Self::select_random(rng, data));
        }
        samples.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn simple_test() {}
}
