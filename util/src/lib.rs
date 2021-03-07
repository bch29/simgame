pub mod bounds;
pub mod bsp;
pub mod ray;
pub mod stable_map;
pub mod octree;

use cgmath::{Point3, Vector3};

pub use bounds::Bounds;

pub trait DivDown {
    /// Divides and rounds the result towards negative infinity.
    fn div_down(self, divisor: Self) -> Self;
}

pub trait DivUp {
    /// Divides and rounds the result towards positive infinity.
    fn div_up(self, divisor: Self) -> Self;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrdFloat<T>(pub T);

impl<T> Eq for OrdFloat<T> where T: PartialEq {}

impl<T> PartialOrd for OrdFloat<T>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(
            self.0
                .partial_cmp(&other.0)
                .unwrap_or(std::cmp::Ordering::Less),
        )
    }
}

impl<T> Ord for OrdFloat<T>
where
    T: PartialOrd,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Less)
    }
}

#[macro_export]
macro_rules! convert_point {
    ($val:expr, $type:ty) => {
        cgmath::Point3 {
            x: $val.x as $type,
            y: $val.y as $type,
            z: $val.z as $type,
        }
    };
}

#[macro_export]
macro_rules! convert_vec {
    ($val:expr, $type:ty) => {
        cgmath::Vector3 {
            x: $val.x as $type,
            y: $val.y as $type,
            z: $val.z as $type,
        }
    };
}

#[macro_export]
macro_rules! convert_bounds {
    ($val:expr, $type:ty) => {
        $crate::Bounds::new(
            $crate::convert_point!($val.origin(), $type),
            $crate::convert_vec!($val.size(), $type),
        )
    };
}

macro_rules! impl_div_traits_int {
    ($type:ty, $div_up_positive:ident) => {
        // `0 - x` used for negation so that unsigned types can share the implementation. We will
        // never actually compute `0 - x` for an unsigned x because there is always an `x <= 0`
        // check before a negation. These checks and the corresponding negation code should be
        // optimized away for unsigned types.

        #[allow(unused_comparisons)]
        fn $div_up_positive(dividend: $type, divisor: $type) -> $type {
            assert!(dividend >= 0);
            assert!(divisor > 0);

            let d = dividend / divisor;
            let r = dividend % divisor;
            if r > 0 {
                1 + d
            } else {
                d
            }
        }

        impl DivDown for $type {
            #[inline]
            #[allow(unused_comparisons)]
            fn div_down(mut self, mut divisor: $type) -> $type {
                assert!(divisor != 0);
                if divisor < 0 {
                    self = 0 - self;
                    divisor = 0 - divisor;
                }

                if self >= 0 {
                    self / divisor
                } else {
                    0 - $div_up_positive(0 - self, divisor)
                }
            }
        }

        impl DivUp for $type {
            #[inline]
            #[allow(unused_comparisons)]
            fn div_up(mut self, mut divisor: $type) -> $type {
                assert!(divisor != 0);
                if divisor < 0 {
                    self = 0 - self;
                    divisor = 0 - divisor;
                }

                if self >= 0 {
                    $div_up_positive(self, divisor)
                } else {
                    0 - ((0 - self) / divisor)
                }
            }
        }
    };
}

macro_rules! impl_div_trait_pv {
    ($pv:tt, $trait:path, $method:tt) => {
        impl<T> $trait for $pv<T>
        where
            T: $trait,
        {
            #[inline]
            fn $method(self, divisor: $pv<T>) -> Self {
                $pv {
                    x: self.x.$method(divisor.x),
                    y: self.y.$method(divisor.y),
                    z: self.z.$method(divisor.z),
                }
            }
        }
    };
}

impl_div_traits_int!(u8, div_up_positive_u8);
impl_div_traits_int!(u16, div_up_positive_u16);
impl_div_traits_int!(u32, div_up_positive_u32);
impl_div_traits_int!(u64, div_up_positive_u64);
impl_div_traits_int!(i8, div_up_positive_i8);
impl_div_traits_int!(i16, div_up_positive_i16);
impl_div_traits_int!(i32, div_up_positive_i32);
impl_div_traits_int!(i64, div_up_positive_i64);
impl_div_traits_int!(usize, div_up_positive_usize);
impl_div_traits_int!(isize, div_up_positive_isize);

impl_div_trait_pv!(Vector3, DivDown, div_down);
impl_div_trait_pv!(Vector3, DivUp, div_up);
impl_div_trait_pv!(Point3, DivDown, div_down);
impl_div_trait_pv!(Point3, DivUp, div_up);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_div_traits() {
        assert_eq!(38u32.div_up(16), 3);
        assert_eq!(38i32.div_up(4), 10);
        assert_eq!(38u32.div_down(4), 9);

        assert_eq!((-38).div_up(-4), 10);

        assert_eq!((-38).div_up(4), -9);
        assert_eq!((-38).div_down(4), -10);
    }
}
