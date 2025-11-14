pub mod fixed_1000;
pub mod fixed_1536;
pub mod fixed_256;
pub mod fixed_512;

use std::sync::OnceLock;

pub use fixed_1000::Fft1000;
pub use fixed_1536::Fft1536;
pub use fixed_256::Fft256;
pub use fixed_512::Fft512;

static SPECIAL_ENABLED: OnceLock<bool> = OnceLock::new();

pub fn special_fft_enabled() -> bool {
    *SPECIAL_ENABLED.get_or_init(|| {
        let disable = std::env::var("RUSTFFT_DISABLE_SPECIAL")
            .map(|val| {
                matches!(
                    val.to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);
        !disable
    })
}
