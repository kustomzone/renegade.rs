use bit_vec::BitVec;
use rand::prelude::{Rng, ThreadRng};

#[derive(Debug, Clone)]
struct WaypointIndex<I: Clone + PartialEq> {
    waypoints: Vec<(I, I)>,
}

impl<I: Clone + PartialEq> WaypointIndex<I> {
    pub fn new(
        data: &Vec<I>,
        dist: &fn(&I, &I) -> f64,
        waypoint_count: u8,
        sample_count: u32,
        rng: &mut ThreadRng,
    ) -> WaypointIndex<I> {
        let mut waypoints: Vec<(I, I)> = vec![];
        'w: for _ in 0..waypoint_count {
            let mut first_best: Option<(I, I, f64)> = Option::None;
            for _ in 0..sample_count {
                let a = Self::select_random(rng, data);
                let b = Self::select_random(rng, data);
                for w in &waypoints {
                    if a == w.0 || a == w.1 || b == w.0 || b == w.1 {
                        continue 'w;
                    }
                }
                let dst = if waypoints.is_empty() { dist(&a, &b) } else {
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
                None => { panic!("Unable to find waypoint") }
            }
        }
        WaypointIndex { waypoints }
    }

    pub fn bit_vec(&self, item: &I, dist: fn(&I, &I) -> f64) -> BitVec {
        let mut bv = BitVec::from_elem(self.waypoints.len(), false);
        for (ix, wp_pair) in self.waypoints.iter().enumerate() {
            let d1 = dist(&wp_pair.0, item);
            let d2 = dist(&wp_pair.1, item);
            bv.set(ix, d1 < d2);
        }
        bv
    }

    fn select_random(rng: &mut ThreadRng, vec: &Vec<I>) -> I {
        vec[rng.gen_range(0..vec.len())].clone()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn simple_test() {
        
    }
}