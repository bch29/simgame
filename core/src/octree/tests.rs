use super::*;

use std::collections::HashSet;
use cgmath::Point3;

fn values_vec<T: Copy>(tree: &Octree<T>) -> Vec<T> {
    tree.iter().map(|(_, &v)| v).collect()
}

fn values_set<T: Copy + Eq + std::hash::Hash>(tree: &Octree<T>) -> HashSet<T> {
    tree.iter().map(|(_, &v)| v).collect()
}

#[test]
fn test_insert_remove() {
    let mut tree: Octree<i64> = Octree::new(8);
    tree.insert(Point3::new(34, 2, 17), 6);
    assert_eq!(tree.get(Point3::new(34, 2, 17)), Some(&6));
    assert_eq!(tree.get(Point3::new(0, 2, 17)), None);
    tree.assert_height_invariant();

    {
        let inner_ref = tree.get_or_insert(Point3::new(2, 3, 4), || 7);
        assert_eq!(*inner_ref, 7);
        *inner_ref = 5;
    }

    assert_eq!(tree.get_or_insert(Point3::new(2, 3, 4), || 4), &mut 5);
    tree.assert_height_invariant();

    tree.insert(Point3::new(34, 3, 16), 2);
    tree.assert_height_invariant();

    // Remove
    assert_eq!(tree.remove(Point3::new(34, 3, 16)), Some(2));
    assert_eq!(tree.remove(Point3::new(34, 3, 16)), None);
    tree.assert_height_invariant();
}

#[test]
fn test_iter_basic() {
    let tree: Octree<i64> = Octree::new(0);
    assert_eq!(values_vec(&tree), Vec::<i64>::new());
    tree.assert_height_invariant();

    let tree: Octree<i64> = Octree::new(100);
    assert_eq!(values_vec(&tree), Vec::<i64>::new());
    tree.assert_height_invariant();

    let mut tree: Octree<i64> = Octree::new(8);
    tree.insert(Point3::new(34, 2, 17), 6);
    tree.insert(Point3::new(2, 3, 4), 5);
    tree.insert(Point3::new(34, 3, 16), 2);
    tree.assert_height_invariant();

    let mut iter = tree.iter();
    assert_eq!(iter.next(), Some((Point3::new(2, 3, 4), &5)));
    assert_eq!(iter.next(), Some((Point3::new(34, 3, 16), &2)));
    assert_eq!(iter.next(), Some((Point3::new(34, 2, 17), &6)));
    assert_eq!(iter.next(), None);
    tree.assert_height_invariant();

    tree.remove(Point3::new(34, 3, 16));
    tree.assert_height_invariant();

    // Iteration after removal still yields the values that were not removed
    assert_eq!(values_vec(&tree), vec![5, 6]);
    tree.assert_height_invariant();
}

#[test]
fn test_iter_mut_basic() {
    let mut tree: Octree<i64> = Octree::new(8);
    tree.insert(Point3::new(34, 2, 17), 6);
    tree.insert(Point3::new(2, 3, 4), 5);
    tree.insert(Point3::new(34, 3, 16), 2);
    tree.insert(Point3::new(1, 1, 1), 7);
    tree.insert(Point3::new(1, 2, 1), 8);
    tree.insert(Point3::new(1, 1, 2), 9);
    tree.assert_height_invariant();

    let mut iter_count = 0;
    for (_, x) in tree.iter_mut() {
        *x *= 2;
        iter_count += 1;
    }

    let all_vals = values_vec(&tree);
    assert_eq!(all_vals.len(), iter_count);
    assert_eq!(all_vals, vec![14, 16, 18, 10, 4, 12]);
    tree.assert_height_invariant();
}

#[test]
fn test_dense_iter_mut() {
    let mut tree: Octree<i64> = Octree::new(5);

    let bounds = tree.bounds();

    let mut expected_vals = HashSet::new();

    let size = bounds.size();
    for k in bounds.iter_points() {
        let v = (k.x + k.y * size.x + k.z * size.x * size.y) as i64;
        expected_vals.insert(v);
        tree.insert(k, v);
    }

    tree.assert_height_invariant();
    assert_eq!(values_set(&tree), expected_vals);

    for (_, v) in tree.iter_mut() {
        *v *= 3;
    }

    let expected_vals: HashSet<i64> = expected_vals.iter().map(|x| *x * 3).collect();
    assert_eq!(values_set(&tree), expected_vals);
}

#[test]
fn test_grow() {
    let mut tree: Octree<i64> = Octree::new(6);
    tree.insert(Point3::new(34, 2, 17), 6);
    tree.insert(Point3::new(2, 3, 4), 5);
    tree.insert(Point3::new(34, 3, 16), 2);
    tree.insert(Point3::new(1, 1, 1), 7);
    tree.insert(Point3::new(1, 2, 1), 8);
    tree.insert(Point3::new(1, 1, 2), 9);
    tree.assert_height_invariant();

    tree.grow(0);
    assert_eq!(tree.height(), 7);
    tree.assert_height_invariant();

    tree.grow(2);
    assert_eq!(tree.height(), 8);
    tree.assert_height_invariant();

    tree.grow(4);
    assert_eq!(tree.height(), 9);
    tree.assert_height_invariant();

    tree.grow(5);
    assert_eq!(tree.height(), 10);
    tree.assert_height_invariant();

    tree.grow(6);
    assert_eq!(tree.height(), 11);
    tree.assert_height_invariant();
}
