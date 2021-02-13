use std::error::Error;

use rsklearn_dstump::DecisionStump;
use rsklearn_linear::Linear;

fn main() -> Result<(), Box<dyn Error>> {
    // load Diabetes dataset
    let dataset = rsklearn_datasets::diabetes();

    let dstump = DecisionStump::<Linear>::default();
    let model = dstump.fit(dataset.records, dataset.targets);

    println!("Model: {}", model.str());

    Ok(())
}
