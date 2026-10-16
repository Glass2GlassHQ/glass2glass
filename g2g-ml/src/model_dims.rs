use g2g_core::{G2gError, TensorShape};

/// The `(width, height)` of a model input that must be `[N, 3, H, W]` with static
/// `H` / `W`; `N` may be dynamic (the exported-with-dynamic-batch convention).
pub(crate) fn input_geometry(dims: &[i64]) -> Result<(u32, u32), G2gError> {
    let [n, c, h, w] = dims else {
        return Err(G2gError::CapsMismatch);
    };
    if !(*n == 1 || *n == -1) || *c != 3 || *h <= 0 || *w <= 0 {
        return Err(G2gError::CapsMismatch);
    }
    Ok((*w as u32, *h as u32))
}

/// Resolve an output shape to static dims: a dynamic leading batch dim is
/// the exported-with-dynamic-batch convention and becomes 1; any other
/// dynamic dim is rejected (v1 needs static output caps for negotiation).
pub(crate) fn static_output_dims(dims: &[i64]) -> Result<TensorShape, G2gError> {
    let mut out = Vec::with_capacity(dims.len());
    for (i, d) in dims.iter().enumerate() {
        match d {
            -1 if i == 0 => out.push(1),
            d if *d > 0 => out.push(*d as u32),
            _ => return Err(G2gError::CapsMismatch),
        }
    }
    // A fixed-rank TensorShape carries at most MAX_TENSOR_RANK dims; a model
    // with a deeper output cannot be described by tensor caps.
    TensorShape::from_slice(&out).ok_or(G2gError::CapsMismatch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_output_dims_coerces_dynamic_batch_only() {
        assert_eq!(static_output_dims(&[1, 10]), Ok(TensorShape::new([1, 10])));
        assert_eq!(static_output_dims(&[-1, 10]), Ok(TensorShape::new([1, 10])));
        assert_eq!(
            static_output_dims(&[1, -1]),
            Err(G2gError::CapsMismatch),
            "dynamic non-batch dims are rejected"
        );
        assert_eq!(static_output_dims(&[1, 0]), Err(G2gError::CapsMismatch));
    }
}
