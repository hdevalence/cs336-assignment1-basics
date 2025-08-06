use pyo3::prelude::*;

pub mod tokenizer;
pub use tokenizer::rust_run_train_bpe;
pub use tokenizer::{read_merges, read_vocab, write_merges, write_vocab};

#[pyfunction]
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[pymodule]
fn _cs336_a1_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(rust_run_train_bpe, m)?)?;
    m.add_function(wrap_pyfunction!(read_vocab, m)?)?;
    m.add_function(wrap_pyfunction!(read_merges, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
