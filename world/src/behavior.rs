use serde::{Deserialize, Serialize};

use simgame_types::impl_behavior;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bounce {
    pub progress: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fall;

impl_behavior! {
    impl Behavior for Bounce;
    impl Behavior for Fall;
}
