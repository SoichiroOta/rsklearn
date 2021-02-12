extern crate ndarray;
use ndarray::{Array, Ix1, Ix2};
use ndarray::Zip;

pub struct ZeroRule {
    r:  Option<Array::<f64, Ix1>>,
}

impl Default for ZeroRule {
    fn default() -> Self {
        ZeroRule {
            r: None,
        }
    }
}

impl ZeroRule {
    pub fn new() -> ZeroRule {
        ZeroRule {
            r: None,
        }
    }

    pub fn fit(&mut self, _x: &Array::<f64, Ix2>, y: &Array::<f64, Ix2>) -> &Self {
        let mut totals = Array::<f64, Ix1>::zeros(y.shape()[1]);
        Zip::from(&mut totals)
            .and(y.gencolumns())
            .apply(|totals, col| *totals = col.sum());
        self.r = Some(totals.mapv(|v: f64| v / y.shape()[0] as f64));
        self
    }

    pub fn predict(&self, x: Array::<f64, Ix2>) -> Array::<f64, Ix2> {
        let z = Array::<f64, _>::zeros((x.shape()[0], self.r.as_ref().unwrap().clone().shape()[0]));
        z.clone() + self.r.as_ref().unwrap().clone()
    }

    pub fn str(&self) -> String {
        match self.r {
            None => String::from("None"),
            _ => {
                let mut s = Vec::new();
                let r = self.r.as_ref().unwrap();
                s.push(format!("[{},", r[[0]].to_string()));
                for i in 1..self.r.as_ref().unwrap().shape()[0] {
                    if i < self.r.as_ref().unwrap().shape()[0] - 1 {
                        s.push(format!(" {},", r[[i]].to_string()));
                    } else {
                        s.push(format!(" {}]", r[[i]].to_string()));
                    }                    
                }
                String::from(s.iter().cloned().collect::<String>())
            }
        }      
    }
}
