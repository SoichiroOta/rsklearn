//! `rsklearn` aims to provide a comprehensive toolkit to build Machine Learning applications
//! with Rust.
//!
//! Kin in spirit to Python's `scikit-learn`, it focuses on common preprocessing tasks
//! and classical ML algorithms for your everyday ML tasks.
//!

pub mod dataset;
pub mod error;
mod metrics_classification;
mod metrics_regression;
pub mod prelude;
pub mod traits;
pub mod entropy;

pub use dataset::{Dataset, DatasetBase, DatasetPr, DatasetView, Float, Label};

#[cfg(feature = "ndarray-linalg")]
pub use ndarray_linalg as linalg;

#[cfg(any(feature = "intel-mkl-system", feature = "intel-mkl-static"))]
extern crate intel_mkl_src;

#[cfg(any(feature = "openblas-system", feature = "openblas-static"))]
extern crate openblas_src;

#[cfg(any(feature = "netblas-system", feature = "netblas-static"))]
extern crate netblas_src;

/// Common metrics functions for classification and regression
pub mod metrics {
    pub use crate::metrics_classification::{
        BinaryClassification, ConfusionMatrix, ReceiverOperatingCharacteristic, ToConfusionMatrix,
    };
    pub use crate::metrics_regression::Regression;
}
