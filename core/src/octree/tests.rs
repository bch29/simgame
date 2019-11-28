use super::*;

use cgmath::Point3;

#[test]
fn test_insert_remove() {
    let mut tree: Octree<i64> = Octree::new(8);
    tree.insert(Point3::new(34, 2, 17), 6);
    assert_eq!(tree.get(Point3::new(34, 2, 17)), Some(&6));
    assert_eq!(tree.get(Point3::new(0, 2, 17)), None);

    {
        let inner_ref = tree.get_or_insert(Point3::new(2, 3, 4), || 7);
        assert_eq!(*inner_ref, 7);
        *inner_ref = 5;
    }

    assert_eq!(tree.get_or_insert(Point3::new(2, 3, 4), || 4), &mut 5);

    tree.insert(Point3::new(34, 3, 16), 2);

    // Remove
    assert_eq!(tree.remove(Point3::new(34, 3, 16)), Some(2));
    assert_eq!(tree.remove(Point3::new(34, 3, 16)), None);
}

#[test]
fn test_iter_basic() {
    let mut tree: Octree<i64> = Octree::new(8);
    tree.insert(Point3::new(34, 2, 17), 6);
    tree.insert(Point3::new(2, 3, 4), 5);
    tree.insert(Point3::new(34, 3, 16), 2);

    let mut iter = tree.iter();
    assert_eq!(iter.next(), Some((Point3::new(2, 3, 4), &5)));
    assert_eq!(iter.next(), Some((Point3::new(34, 3, 16), &2)));
    assert_eq!(iter.next(), Some((Point3::new(34, 2, 17), &6)));
    assert_eq!(iter.next(), None);

    tree.remove(Point3::new(34, 3, 16));

    // Iteration after removal still yields the values that were not removed
    let all_vals: Vec<i64> = tree.iter().map(|(_, &v)| v).collect();
    assert_eq!(all_vals, vec![5, 6]);
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

    let mut iter_count = 0;
    for x in tree.iter_mut() {
        *x *= 2;
        iter_count += 1;
    }

    let all_vals: Vec<i64> = tree.iter().map(|(_, &v)| v).collect();
    assert_eq!(all_vals.len(), iter_count);
    assert_eq!(all_vals, vec![14, 16, 18, 10, 4, 12]);
}

#[test]
fn test_dense_iter_mut() {
    let mut tree: Octree<i64> = Octree::new(3);

    let bounds = tree.bounds();

    use std::collections::HashSet;

    let mut expected_vals = HashSet::new();

    for x in 0..bounds.x {
        for y in 0..bounds.y {
            for z in 0..bounds.z {
                let k = Point3::new(x, y, z);
                let v = (x + y * bounds.x + z * bounds.x * bounds.y) as i64;
                expected_vals.insert(v);
                tree.insert(k, v);
            }
        }
    }

    let actual_vals: HashSet<i64> = tree.iter().map(|(_, &v)| v).collect();
    assert_eq!(actual_vals, expected_vals);

    for v in tree.iter_mut() {
        *v *= 3;
    }

    let expected_vals: HashSet<i64> = expected_vals.iter().map(|x| *x * 3).collect();
    let actual_vals: HashSet<i64> = tree.iter().map(|(_, &v)| v).collect();
    assert_eq!(actual_vals, expected_vals);
}
