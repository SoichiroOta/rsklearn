use std::error::Error;

use ndarray::{Array, Ix2};
use ndarray::prelude::*;

use rsklearn_zeror::ZeroRule;

fn main() -> Result<(), Box<dyn Error>> {
    // load Diabetes dataset
    let dataset = rsklearn_datasets::iris();
    let mut targets = Array::<f64, Ix2>::zeros((dataset.targets.shape()[0], 3));
    for i in 0..dataset.targets.shape()[0] {
        if dataset.targets[[i]] == 0 as usize {
            targets.slice_mut(s![i, 0]).fill(1.0 as f64);
        } else if dataset.targets[[i]] == 1 as usize {
            targets.slice_mut(s![i, 1]).fill(1.0 as f64);
        } else if dataset.targets[[i]] == 2 as usize {
            targets.slice_mut(s![i, 2]).fill(1.0 as f64);
        }
    }

    let mut plf = ZeroRule::new();
    let model = plf.fit(dataset.records, targets);

    println!("Model: {}", model.str());

    Ok(())
}
