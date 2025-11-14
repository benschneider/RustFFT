use crate::algorithm::dft::Dft;
use crate::{common::FftNum, Direction, Fft, FftDirection, Length};
use num_complex::Complex;

pub struct Fft256<T: FftNum> {
    inner: Dft<T>,
    direction: FftDirection,
}

impl<T: FftNum> Fft256<T> {
    pub fn new(direction: FftDirection) -> Self {
        Self {
            inner: Dft::new(256, direction),
            direction,
        }
    }
}

impl<T: FftNum> Length for Fft256<T> {
    fn len(&self) -> usize {
        256
    }
}

impl<T: FftNum> Direction for Fft256<T> {
    fn fft_direction(&self) -> FftDirection {
        self.direction
    }
}

impl<T: FftNum> Fft<T> for Fft256<T> {
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