use crate::plan::FftPlannerScalar;
use crate::{common::FftNum, Direction, Fft, FftDirection, Length};
use num_complex::Complex;
use std::any::TypeId;
use std::sync::Arc;

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use crate::avx::{Butterfly512Avx, Butterfly512Avx64};
#[cfg(feature = "sse")]
use crate::FftPlannerSse;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
use crate::FftPlannerNeon;

use crate::array_utils::{workaround_transmute, workaround_transmute_mut};

enum Fft512Inner<T: FftNum> {
    F32(Arc<dyn Fft<f32>>),
    F64(Arc<dyn Fft<f64>>),
    Generic(Arc<dyn Fft<T>>),
}

pub struct Fft512<T: FftNum> {
    inner: Fft512Inner<T>,
    direction: FftDirection,
}

impl<T: FftNum> Fft512<T> {
    pub fn new(direction: FftDirection) -> Self {
        let inner = Self::select_impl(direction);
        Self { inner, direction }
    }

    fn select_impl(direction: FftDirection) -> Fft512Inner<T> {
        Self::try_specialized(direction).unwrap_or_else(|| {
            Fft512Inner::Generic(Self::plan_scalar(direction))
        })
    }

    fn try_specialized(direction: FftDirection) -> Option<Fft512Inner<T>> {
        let id_t = TypeId::of::<T>();
        if id_t == TypeId::of::<f32>() {
            return Self::build_f32(direction);
        }
        if id_t == TypeId::of::<f64>() {
            return Self::build_f64(direction);
        }
        None
    }

    fn build_f32(direction: FftDirection) -> Option<Fft512Inner<T>> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if let Ok(fft) = Butterfly512Avx::<f32>::new(direction) {
                return Some(Fft512Inner::F32(Arc::new(fft)));
            }
        }

        #[cfg(feature = "sse")]
        {
            if let Ok(mut planner) = FftPlannerSse::<f32>::new() {
                return Some(Fft512Inner::F32(planner.plan_fft(512, direction)));
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if let Ok(mut planner) = FftPlannerNeon::<f32>::new() {
                return Some(Fft512Inner::F32(planner.plan_fft(512, direction)));
            }
        }

        Some(Fft512Inner::F32(Self::plan_scalar_f32(direction)))
    }

    fn build_f64(direction: FftDirection) -> Option<Fft512Inner<T>> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if let Ok(fft) = Butterfly512Avx64::<f64>::new(direction) {
                return Some(Fft512Inner::F64(Arc::new(fft)));
            }
        }

        #[cfg(feature = "sse")]
        {
            if let Ok(mut planner) = FftPlannerSse::<f64>::new() {
                return Some(Fft512Inner::F64(planner.plan_fft(512, direction)));
            }
        }

        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if let Ok(mut planner) = FftPlannerNeon::<f64>::new() {
                return Some(Fft512Inner::F64(planner.plan_fft(512, direction)));
            }
        }

        Some(Fft512Inner::F64(Self::plan_scalar_f64(direction)))
    }

    fn plan_scalar(direction: FftDirection) -> Arc<dyn Fft<T>> {
        let mut planner = FftPlannerScalar::<T>::new();
        planner.plan_fft(512, direction)
    }

    fn plan_scalar_f32(direction: FftDirection) -> Arc<dyn Fft<f32>> {
        let mut planner = FftPlannerScalar::<f32>::new();
        planner.plan_fft(512, direction)
    }

    fn plan_scalar_f64(direction: FftDirection) -> Arc<dyn Fft<f64>> {
        let mut planner = FftPlannerScalar::<f64>::new();
        planner.plan_fft(512, direction)
    }
}

impl<T: FftNum> Length for Fft512<T> {
    fn len(&self) -> usize {
        512
    }
}

impl<T: FftNum> Direction for Fft512<T> {
    fn fft_direction(&self) -> FftDirection {
        self.direction
    }
}

impl<T: FftNum> Fft<T> for Fft512<T> {
    fn process(&self, buffer: &mut [Complex<T>]) {
        match &self.inner {
            Fft512Inner::Generic(inner) => inner.process(buffer),
            Fft512Inner::F32(inner) => unsafe {
                let buf: &mut [Complex<f32>] = workaround_transmute_mut(buffer);
                inner.process(buf);
            },
            Fft512Inner::F64(inner) => unsafe {
                let buf: &mut [Complex<f64>] = workaround_transmute_mut(buffer);
                inner.process(buf);
            },
        }
    }

    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        match &self.inner {
            Fft512Inner::Generic(inner) => inner.process_with_scratch(buffer, scratch),
            Fft512Inner::F32(inner) => unsafe {
                let buf: &mut [Complex<f32>] = workaround_transmute_mut(buffer);
                let scr: &mut [Complex<f32>] = workaround_transmute_mut(scratch);
                inner.process_with_scratch(buf, scr);
            },
            Fft512Inner::F64(inner) => unsafe {
                let buf: &mut [Complex<f64>] = workaround_transmute_mut(buffer);
                let scr: &mut [Complex<f64>] = workaround_transmute_mut(scratch);
                inner.process_with_scratch(buf, scr);
            },
        }
    }

    fn process_outofplace_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        match &self.inner {
            Fft512Inner::Generic(inner) => {
                inner.process_outofplace_with_scratch(input, output, scratch)
            }
            Fft512Inner::F32(inner) => unsafe {
                let inp: &mut [Complex<f32>] = workaround_transmute_mut(input);
                let out: &mut [Complex<f32>] = workaround_transmute_mut(output);
                let scr: &mut [Complex<f32>] = workaround_transmute_mut(scratch);
                inner.process_outofplace_with_scratch(inp, out, scr);
            },
            Fft512Inner::F64(inner) => unsafe {
                let inp: &mut [Complex<f64>] = workaround_transmute_mut(input);
                let out: &mut [Complex<f64>] = workaround_transmute_mut(output);
                let scr: &mut [Complex<f64>] = workaround_transmute_mut(scratch);
                inner.process_outofplace_with_scratch(inp, out, scr);
            },
        }
    }

    fn process_immutable_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        match &self.inner {
            Fft512Inner::Generic(inner) => {
                inner.process_immutable_with_scratch(input, output, scratch)
            }
            Fft512Inner::F32(inner) => unsafe {
                let inp: &[Complex<f32>] = workaround_transmute(input);
                let out: &mut [Complex<f32>] = workaround_transmute_mut(output);
                let scr: &mut [Complex<f32>] = workaround_transmute_mut(scratch);
                inner.process_immutable_with_scratch(inp, out, scr);
            },
            Fft512Inner::F64(inner) => unsafe {
                let inp: &[Complex<f64>] = workaround_transmute(input);
                let out: &mut [Complex<f64>] = workaround_transmute_mut(output);
                let scr: &mut [Complex<f64>] = workaround_transmute_mut(scratch);
                inner.process_immutable_with_scratch(inp, out, scr);
            },
        }
    }

    fn get_inplace_scratch_len(&self) -> usize {
        match &self.inner {
            Fft512Inner::Generic(inner) => inner.get_inplace_scratch_len(),
            Fft512Inner::F32(inner) => inner.get_inplace_scratch_len(),
            Fft512Inner::F64(inner) => inner.get_inplace_scratch_len(),
        }
    }

    fn get_outofplace_scratch_len(&self) -> usize {
        match &self.inner {
            Fft512Inner::Generic(inner) => inner.get_outofplace_scratch_len(),
            Fft512Inner::F32(inner) => inner.get_outofplace_scratch_len(),
            Fft512Inner::F64(inner) => inner.get_outofplace_scratch_len(),
        }
    }

    fn get_immutable_scratch_len(&self) -> usize {
        match &self.inner {
            Fft512Inner::Generic(inner) => inner.get_immutable_scratch_len(),
            Fft512Inner::F32(inner) => inner.get_immutable_scratch_len(),
            Fft512Inner::F64(inner) => inner.get_immutable_scratch_len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::check_fft_algorithm;

    #[test]
    fn fft512_forward_f32_matches_reference() {
        let fft = Fft512::<f32>::new(FftDirection::Forward);
        check_fft_algorithm(&fft, 512, FftDirection::Forward);
    }

    #[test]
    fn fft512_inverse_f32_matches_reference() {
        let fft = Fft512::<f32>::new(FftDirection::Inverse);
        check_fft_algorithm(&fft, 512, FftDirection::Inverse);
    }

    #[test]
    fn fft512_forward_f64_matches_reference() {
        let fft = Fft512::<f64>::new(FftDirection::Forward);
        check_fft_algorithm(&fft, 512, FftDirection::Forward);
    }

    #[test]
    fn fft512_inverse_f64_matches_reference() {
        let fft = Fft512::<f64>::new(FftDirection::Inverse);
        check_fft_algorithm(&fft, 512, FftDirection::Inverse);
    }
}
