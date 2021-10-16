extern crate renegade;

use rand::prelude::*;

#[test]
fn simple_model_build() {
    let mut rng = thread_rng();
    let mut data: Vec<(f64, f64)> = vec![];
    for _ in 0..1000 {
        data.push((rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)));
    }

    let wi = renegade::learn_metrics::learn_metrics();
}
