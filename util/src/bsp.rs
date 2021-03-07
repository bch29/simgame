//! Binary space partitioning tree.

use cgmath::Point3;

use crate::bounds::Bounds;

#[derive(Clone, Debug)]
pub struct Bsp<T> {
    root_node: BspNode,
    pool: BranchPool<T>,
}

impl<T> Bsp<T> {
    pub fn new() -> Self {
        Self {
            root_node: BspNode::new(),
            pool: Pool::new(),
        }
    }

    pub fn clear(&mut self) {
        self.root_node = BspNode::new();
        self.pool.clear();
    }

    #[inline]
    pub fn insert(&mut self, point: Point3<f64>, value: T) -> usize {
        self.root_node.insert(&mut self.pool, point, value)
    }

    #[inline]
    pub fn iter<F>(&self, bounds: Option<Bounds<f64>>, mut func: F)
    where
        F: FnMut(usize, Point3<f64>, &T),
    {
        self.root_node.iter(&self.pool, bounds, &mut func)
    }

    #[inline]
    pub fn iter_mut<F>(&mut self, bounds: Option<Bounds<f64>>, mut func: F)
    where
        F: FnMut(usize, Point3<f64>, &mut T),
    {
        self.root_node.iter_mut(&mut self.pool, bounds, &mut func)
    }

    #[inline]
    pub fn count(&self) -> usize {
        self.root_node.count(&self.pool)
    }

    #[inline]
    pub fn depth(&self) -> usize {
        self.root_node.depth(&self.pool)
    }
}

impl<T> Default for Bsp<T> {
    fn default() -> Self {
        Self::new()
    }
}

type BranchPool<T> = Pool<Branch<T>>;

#[derive(Clone, Copy, Debug)]
struct BspNode {
    axis: Axis,
    branch_ix: Option<usize>,
}

#[derive(Clone, Debug)]
struct Branch<T> {
    point: Point3<f64>,
    value: T,
    lower: BspNode,
    upper: BspNode,
}

#[derive(Clone, Debug)]
struct Pool<T> {
    pool: Vec<T>,
}

impl<T> Pool<T> {
    fn new() -> Self {
        Self { pool: Vec::new() }
    }

    fn push(&mut self, item: T) -> usize {
        let ix = self.pool.len();
        self.pool.push(item);
        ix
    }

    fn clear(&mut self) {
        self.pool.clear()
    }
}

impl<T> std::ops::Index<usize> for Pool<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.pool[index]
    }
}

impl<T> std::ops::IndexMut<usize> for Pool<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.pool[index]
    }
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

impl BspNode {
    fn new() -> Self {
        Self {
            axis: Axis::X,
            branch_ix: None,
        }
    }

    fn insert<T>(&mut self, pool: &mut BranchPool<T>, point: Point3<f64>, value: T) -> usize {
        match self.branch_ix {
            None => {
                let res = pool.push(Branch {
                    point,
                    value,
                    lower: BspNode {
                        axis: self.axis.next(),
                        branch_ix: None,
                    },
                    upper: BspNode {
                        axis: self.axis.next(),
                        branch_ix: None,
                    },
                });
                self.branch_ix = Some(res);
                res
            }
            Some(branch_ix) => {
                // look up branch and take temporary copies of nodes to avoid holding on to pool
                // reference, which we later need as mutable
                let Branch {
                    point: branch_point,
                    lower,
                    upper,
                    ..
                } = &pool[branch_ix];
                let mut tmp_lower = *lower;
                let mut tmp_upper = *upper;

                // update temporary copies
                let res = match self.axis.divide_point(*branch_point, point) {
                    DividePointResult::Lower => tmp_lower.insert(pool, point, value),
                    DividePointResult::Upper => tmp_upper.insert(pool, point, value),
                };

                // insert updated nodes back into tree
                let Branch { lower, upper, .. } = &mut pool[branch_ix];
                *lower = tmp_lower;
                *upper = tmp_upper;

                res
            }
        }
    }

    fn iter<T, F>(&self, pool: &BranchPool<T>, bounds: Option<Bounds<f64>>, func: &mut F)
    where
        F: FnMut(usize, Point3<f64>, &T),
    {
        let branch_ix = match self.branch_ix {
            None => return,
            Some(branch_ix) => branch_ix,
        };
        let Branch {
            point,
            lower,
            upper,
            value,
        } = &pool[branch_ix];

        let div_result = match bounds {
            None => DivideBoundsResult::Both,
            Some(bounds) => {
                if bounds.contains_point(*point) {
                    func(branch_ix, *point, value);
                }

                self.axis.divide_bounds(*point, bounds)
            }
        };

        if div_result.contains_upper() {
            upper.iter(pool, bounds, func);
        }

        if div_result.contains_lower() {
            lower.iter(pool, bounds, func);
        }
    }

    fn iter_mut<T, F>(&self, pool: &mut BranchPool<T>, bounds: Option<Bounds<f64>>, func: &mut F)
    where
        F: FnMut(usize, Point3<f64>, &mut T),
    {
        let branch_ix = match self.branch_ix {
            None => return,
            Some(branch_ix) => branch_ix,
        };
        let Branch {
            point,
            lower,
            upper,
            value,
        } = &mut pool[branch_ix];

        // take copies of nodes in order to drop mutable reference to pool
        let lower = *lower;
        let upper = *upper;

        let div_result = match bounds {
            None => DivideBoundsResult::Both,
            Some(bounds) => {
                if bounds.contains_point(*point) {
                    func(branch_ix, *point, value);
                }

                self.axis.divide_bounds(*point, bounds)
            }
        };

        if div_result.contains_upper() {
            upper.iter_mut(pool, bounds, func);
        }

        if div_result.contains_lower() {
            lower.iter_mut(pool, bounds, func);
        }
    }

    fn count<T>(&self, pool: &BranchPool<T>) -> usize {
        match self.branch_ix {
            None => 0,
            Some(branch_ix) => {
                let Branch { lower, upper, .. } = &pool[branch_ix];
                1 + lower.count(pool) + upper.count(pool)
            }
        }
    }

    fn depth<T>(&self, pool: &BranchPool<T>) -> usize {
        match self.branch_ix {
            None => 0,
            Some(branch_ix) => {
                let Branch { lower, upper, .. } = &pool[branch_ix];
                1 + lower.depth(pool).max(upper.depth(pool))
            }
        }
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
        tree.iter(
            Some(Bounds::new(Point3::new(-0.1, -0.1, -0.1), Vector3::new(2., 2., 2.))),
            |ix, p, x| found.push((ix, p, *x)),
        );

        assert_eq!(
            found.into_iter().map(|(_, _, x)| x).collect::<Vec<_>>(),
            vec![0, 4, 5]
        );
    }
}
