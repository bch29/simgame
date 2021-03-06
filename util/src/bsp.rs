//! Binary space partitioning tree.

use cgmath::Point3;

use crate::bounds::Bounds;

#[derive(Clone, Debug)]
pub struct Bsp<T> {
    axis: Axis,
    branch: Option<Box<Branch<T>>>,
}

#[derive(Clone, Debug)]
struct Branch<T> {
    point: Point3<f64>,
    value: T,
    lower: Bsp<T>,
    upper: Bsp<T>,
}

#[derive(Clone, Copy, Debug)]
enum Axis {
    X,
    Y,
    Z,
}

#[derive(Clone, Copy, Debug)]
enum DividePointResult {
    Lower,
    Upper,
}

#[derive(Clone, Copy, Debug)]
enum DivideBoundsResult {
    Lower,
    Upper,
    Both,
}

impl<T> Bsp<T> {
    pub fn new() -> Self {
        Self {
            axis: Axis::X,
            branch: None,
        }
    }

    pub fn insert(&mut self, point: Point3<f64>, value: T) {
        match &mut self.branch {
            None => {
                self.branch = Some(Box::new(Branch {
                    point,
                    value,
                    lower: Bsp {
                        axis: self.axis.next(),
                        branch: None,
                    },
                    upper: Bsp {
                        axis: self.axis.next(),
                        branch: None,
                    },
                }))
            }
            Some(branch) => {
                let Branch {
                    point: branch_point,
                    lower,
                    upper,
                    ..
                } = &mut **branch;

                match self.axis.divide_point(*branch_point, point) {
                    DividePointResult::Lower => lower.insert(point, value),
                    DividePointResult::Upper => upper.insert(point, value),
                }
            }
        }
    }

    pub fn find<'a>(&'a self, bounds: Bounds<f64>, result: &mut Vec<(Point3<f64>, &'a T)>) {
        let Branch {
            point,
            lower,
            upper,
            value,
        } = match &self.branch {
            None => return,
            Some(branch) => &**branch,
        };

        if bounds.contains_point(*point) {
            result.push((*point, value));
        }

        let div_result = self.axis.divide_bounds(*point, bounds);

        if div_result.contains_upper() {
            upper.find(bounds, result);
        }

        if div_result.contains_lower() {
            lower.find(bounds, result);
        }
    }

    pub fn count(&self) -> usize {
        let Branch { lower, upper, .. } = match &self.branch {
            None => return 0,
            Some(branch) => &**branch,
        };

        1 + lower.count() + upper.count()
    }

    pub fn depth(&self) -> usize {
        let Branch { lower, upper, .. } = match &self.branch {
            None => return 0,
            Some(branch) => &**branch,
        };

        1 + lower.depth().max(upper.depth())
    }
}

impl<T> Default for Bsp<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl Axis {
    fn next(self) -> Self {
        match self {
            Axis::X => Axis::Y,
            Axis::Y => Axis::Z,
            Axis::Z => Axis::X,
        }
    }

    fn divide_point(self, center: Point3<f64>, dividend: Point3<f64>) -> DividePointResult {
        let offset = dividend - center;
        match self {
            Axis::X => {
                if offset.x < 0. {
                    DividePointResult::Lower
                } else {
                    DividePointResult::Upper
                }
            }
            Axis::Y => {
                if offset.y < 0. {
                    DividePointResult::Lower
                } else {
                    DividePointResult::Upper
                }
            }
            Axis::Z => {
                if offset.z < 0. {
                    DividePointResult::Lower
                } else {
                    DividePointResult::Upper
                }
            }
        }
    }

    fn divide_bounds(self, center: Point3<f64>, bounds: Bounds<f64>) -> DivideBoundsResult {
        match (
            self.divide_point(center, bounds.origin()),
            self.divide_point(center, bounds.limit()),
        ) {
            (DividePointResult::Upper, _) => DivideBoundsResult::Upper,
            (DividePointResult::Lower, DividePointResult::Upper) => DivideBoundsResult::Both,
            (_, DividePointResult::Lower) => DivideBoundsResult::Lower,
        }
    }
}

impl DivideBoundsResult {
    fn contains_upper(self) -> bool {
        !matches!(self, DivideBoundsResult::Lower)
    }

    fn contains_lower(self) -> bool {
        !matches!(self, DivideBoundsResult::Upper)
    }
}

#[cfg(test)]
mod tests {
    use super::Bsp;

    use cgmath::{Point3, Vector3};

    use crate::bounds::Bounds;

    #[test]
    fn test_bsp() {
        let mut tree: Bsp<i32> = Bsp::new();

        tree.insert(Point3::new(0., 0., 0.), 0);
        tree.insert(Point3::new(-10., 5., 4.), 1);
        tree.insert(Point3::new(10., 5., 4.), 2);
        tree.insert(Point3::new(5., 2., -8.), 3);
        tree.insert(Point3::new(0.1, 0.1, 0.1), 4);
        tree.insert(Point3::new(0.2, 0.2, 0.2), 5);

        let mut found = Vec::new();
        tree.find(
            Bounds::new(Point3::new(-0.1, -0.1, -0.1), Vector3::new(2., 2., 2.)),
            &mut found,
        );

        assert_eq!(
            found.into_iter().map(|(_, x)| *x).collect::<Vec<_>>(),
            vec![0, 4, 5]
        );
    }
}
