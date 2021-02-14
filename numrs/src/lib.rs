use ndarray::{Array, Ix1};

pub fn argsort<T: Ord>(a: &Array<T, Ix1>, reverse: bool) -> Array<usize, Ix1> {
    let mut idx = (0..a.len()).collect::<Vec<usize>>();
    if reverse == true {
        idx.sort_unstable_by(|&i, &j| a[j].cmp(&a[i]));
    } else {
        idx.sort_unstable_by(|&i, &j| a[i].cmp(&a[j]));
    }
    Array::from(idx)
}

pub fn partial_argsort<T: PartialOrd>(a: &Array<T, Ix1>, reverse: bool) -> Array<usize, Ix1> {
    let mut idx = (0..a.len()).collect::<Vec<usize>>();
    if reverse == true {
        idx.sort_unstable_by(|&i, &j| a[j].partial_cmp(&a[i]).unwrap());
    } else {
        idx.sort_unstable_by(|&i, &j| a[i].partial_cmp(&a[j]).unwrap());
    }
    Array::from(idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argsort_works() {
        let a = Array::from(vec![-5, 4, 1, -3, 2]);
        let idx = argsort(&a, false);

        assert_eq!(idx, Array::from(vec![0, 3, 2, 4, 1]));
    }

    #[test]
    fn argsort_works_in_case_of_reverse() {
        let a = Array::from(vec![-5, 4, 1, -3, 2]);
        let idx = argsort(&a, true);

        assert_eq!(idx, Array::from(vec![1, 4, 2, 3, 0]));
    }

    #[test]
    fn partial_argsort_works() {
        let a = Array::from(vec![-5.0_f64, 4.0_f64, 1.0_f64, -3.0_f64, 2.0_f64]);
        let idx = partial_argsort(&a, false);

        assert_eq!(idx, Array::from(vec![0, 3, 2, 4, 1]));
    }

    #[test]
    fn partial_argsort_works_in_case_of_reverse() {
        let a = Array::from(vec![-5.0_f64, 4.0_f64, 1.0_f64, -3.0_f64, 2.0_f64]);
        let idx = partial_argsort(&a, true);

        assert_eq!(idx, Array::from(vec![1, 4, 2, 3, 0]));
    }
}
