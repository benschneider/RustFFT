use crate::plan::FftPlannerScalar;
use crate::{common::FftNum, Direction, Fft, FftDirection, Length};
use num_complex::Complex;
use std::sync::Arc;

pub struct Fft1536<T: FftNum> {
    inner: Arc<dyn Fft<T>>,
    direction: FftDirection,
}

impl<T: FftNum> Fft1536<T> {
    pub fn new(direction: FftDirection) -> Self {
        let mut planner = FftPlannerScalar::<T>::new();
        let inner = planner.plan_fft(1536, direction);
        Self { inner, direction }
    }
}

impl<T: FftNum> Length for Fft1536<T> {
    fn len(&self) -> usize {
        1536
    }
}

impl<T: FftNum> Direction for Fft1536<T> {
    fn fft_direction(&self) -> FftDirection {
        self.direction
    }
}

impl<T: FftNum> Fft<T> for Fft1536<T> {
    fn process(&self, buffer: &mut [Complex<T>]) {
        self.inner.process(buffer);
    }

    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        self.inner.process_with_scratch(buffer, scratch);
    }

    fn process_outofplace_with_scratch(
        &self,
        input: &mut [Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.inner
            .process_outofplace_with_scratch(input, output, scratch);
    }

    fn process_immutable_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) {
        self.inner
            .process_immutable_with_scratch(input, output, scratch);
    }

    fn get_inplace_scratch_len(&self) -> usize {
        self.inner.get_inplace_scratch_len()
    }

    fn get_outofplace_scratch_len(&self) -> usize {
        self.inner.get_outofplace_scratch_len()
    }

    fn get_immutable_scratch_len(&self) -> usize {
        self.inner.get_immutable_scratch_len()
    }
}
