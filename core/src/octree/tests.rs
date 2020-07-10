use super::*;

use cgmath::Point3;
use std::collections::HashSet;

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

    let tree: Octree<i64> = Octree::new(50);
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

    // Test iteration within bounds
    tree.insert(Point3::new(34, 3, 16), 2);
    tree.insert(Point3::new(10, 11, 12), 4);
    tree.insert(Point3::new(16, 17, 18), 6);
    tree.insert(Point3::new(16, 10, 18), 8);

    let items: Vec<_> = tree
        .iter_in_bounds(Bounds::new(Point3::new(9, 9, 9), Vector3::new(10, 10, 10)))
        .collect();
    assert_eq!(
        items,
        vec![
            (Point3::new(10, 11, 12), &4),
            (Point3::new(16, 10, 18), &8),
            (Point3::new(16, 17, 18), &6),
        ]
    );
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
    let mut tree: Octree<i64> = Octree::new(7);
    tree.insert(Point3::new(34, 2, 17), 6);
    tree.insert(Point3::new(2, 3, 4), 5);
    tree.insert(Point3::new(34, 3, 16), 2);
    tree.insert(Point3::new(1, 1, 1), 7);
    tree.insert(Point3::new(1, 2, 1), 8);
    tree.insert(Point3::new(1, 1, 2), 9);
    tree.assert_height_invariant();

    let check_points = |tree: &Octree<i64>| {
        assert_eq!(tree.get(Point3::new(1, 1, 2)), Some(&9));
        assert_eq!(tree.get(Point3::new(2, 3, 4)), Some(&5));
        assert_eq!(tree.get(Point3::new(34, 3, 16)), Some(&2));
        assert_eq!(tree.get(Point3::new(1, 1, 1)), Some(&7));
        assert_eq!(tree.get(Point3::new(1, 2, 1)), Some(&8));
        assert_eq!(tree.get(Point3::new(1, 1, 2)), Some(&9));
    };

    tree.grow();
    assert_eq!(tree.height(), 8);
    check_points(&tree);
    tree.assert_height_invariant();
    println!("{:?}", tree.iter().collect::<Vec<_>>());

    tree.grow();
    assert_eq!(tree.height(), 9);
    check_points(&tree);
    tree.assert_height_invariant();

    tree.grow();
    assert_eq!(tree.height(), 10);
    check_points(&tree);
    tree.assert_height_invariant();

    tree.grow();
    assert_eq!(tree.height(), 11);
    check_points(&tree);
    tree.assert_height_invariant();

    tree.grow();
    assert_eq!(tree.height(), 12);
    check_points(&tree);
    tree.assert_height_invariant();
}
