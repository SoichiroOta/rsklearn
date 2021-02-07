use std::error::Error;

use rsklearn_linear::Linear;

fn main() -> Result<(), Box<dyn Error>> {
    // load Diabetes dataset
    let dataset = rsklearn_datasets::diabetes();

    let mut lin_reg = Linear::new(20, 0.01, None);
    let model = lin_reg.fit(&dataset.records, &dataset.targets);

    println!("Model: {}", model.str());

    Ok(())
}
