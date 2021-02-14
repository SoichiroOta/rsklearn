use std::error::Error;

use rsklearn::entropy::gini;
use rsklearn_dtree::DecisionTree;
use rsklearn_linear::Linear;

fn main() -> Result<(), Box<dyn Error>> {
    // load Diabetes dataset
    let dataset = rsklearn_datasets::diabetes();

    let dtree = DecisionTree::<Linear>::new(gini, Some(20), Some(0.01), None, 1, 1);
    let model = dtree.fit(dataset.records, dataset.targets);

    println!("Model: \n{}", model.str());

    Ok(())
}
