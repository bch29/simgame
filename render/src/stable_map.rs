use std::collections::{hash_map, HashMap, VecDeque};
use std::hash::Hash;

/// A fixed size key-value map that implements stable indices and supports a diff operation which
/// iterates over the keys that have changed since the last diff. Guarantees no allocation for any
/// operation except construction of new maps (as long as K::clone, K::eq and K::hash do not
/// allocate).
#[derive(Debug, Clone)]
pub struct StableMap<K, V>
where
    K: Eq + Hash,
{
    capacity: usize,
    index_map: HashMap<K, usize>,
    free_list: VecDeque<usize>,
    entries: Vec<Option<EntryInfo<K, V>>>,
    current_diff: Diff,
}

pub struct DiffHandle<'a, K, V>
where
    K: Eq + Hash,
{
    parent: &'a mut StableMap<K, V>,
}

#[derive(Debug, Clone)]
struct EntryInfo<K, V> {
    index: usize,
    key: K,
    value: V,
}

#[derive(Debug, Clone)]
struct Diff {
    changed_indices: Vec<bool>,
}

impl<K, V> StableMap<K, V>
where
    K: Eq + Hash,
{
    #[inline]
    pub fn new(capacity: usize) -> Self {
        let mut changed_indices = Vec::with_capacity(capacity);
        changed_indices.extend(std::iter::repeat(false).take(capacity));

        let mut entries = Vec::with_capacity(capacity);
        entries.extend(std::iter::repeat_with(|| None).take(capacity));

        let mut free_list = VecDeque::with_capacity(capacity);
        free_list.extend(0..capacity);

        StableMap {
            capacity,
            index_map: HashMap::with_capacity(capacity),
            entries,
            free_list,
            current_diff: Diff { changed_indices },
        }
    }

    pub fn set_capacity(&mut self, capacity: usize)
    where
        K: Clone,
    {
        if capacity > self.capacity {
            let additional = capacity - self.capacity;

            self.index_map.reserve(additional);

            self.free_list.reserve_exact(additional);
            self.free_list.extend(self.capacity..capacity);

            self.entries.reserve_exact(additional);
            self.entries
                .extend(std::iter::repeat_with(|| None).take(additional));

            self.current_diff.changed_indices.reserve_exact(additional);
            self.current_diff
                .changed_indices
                .extend(std::iter::repeat(false).take(additional));
        } else {
            let old = std::mem::replace(self, Self::new(capacity));

            for (k, i, v) in old.into_iter() {
                if i < capacity {
                    self.update(k, v);
                }
            }
        }

        self.capacity = capacity;
    }

    /// Take a handle that allows iteration over entries that have changed since last time
    /// take_diff() was called.
    #[inline]
    pub fn take_diff(&mut self) -> DiffHandle<K, V> {
        DiffHandle { parent: self }
    }

    /// Inserts or updates a value in the map. Returns the index assigned along with the old value,
    /// if one existed.
    #[inline]
    pub fn update(&mut self, key: K, value: V) -> (usize, Option<V>)
    where
        K: Clone,
    {
        let (index, old) = match self.index_map.entry(key.clone()) {
            hash_map::Entry::Occupied(occupied) => {
                let index = *occupied.get();
                let old =
                    std::mem::replace(&mut self.entries[index].as_mut().unwrap().value, value);
                (index, Some(old))
            }
            hash_map::Entry::Vacant(vacant) => {
                if let Some(index) = self.free_list.pop_front() {
                    self.entries[index] = Some(EntryInfo { index, key, value });
                    vacant.insert(index);
                    (index, None)
                } else {
                    panic!(
                        "Too many keys in stable map at one time: max is {}",
                        self.capacity()
                    );
                }
            }
        };

        self.current_diff.changed_indices[index] = true;
        (index, old)
    }

    /// Removes a value in the map. Returns it, if it existed.
    #[inline]
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.index_map.remove(key).map(|index| {
            self.free_list.push_back(index);
            self.current_diff.changed_indices[index] = true;
            if let Some(old) = self.entries[index].take() {
                old.value
            } else {
                panic!("entries value at index {} was None", index);
            }
        })
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Accesses the item at the given index. Panics if the index is larger than `self.capacity()`.
    #[inline]
    pub fn index(&self, index: usize) -> Option<(&K, &V)> {
        self.entries[index]
            .as_ref()
            .map(|info| (&info.key, &info.value))
    }

    /// Returns the index and value of the given key, if it exists.
    #[inline]
    pub fn get(&self, key: &K) -> Option<(usize, &V)> {
        self.index_map.get(key).and_then(move |&index| {
            self.entries[index]
                .as_ref()
                .map(|info| (index, &info.value))
        })
    }

    pub fn clear(&mut self) {
        self.free_list.clear();

        for (index, (entry, changed)) in self
            .entries
            .iter_mut()
            .zip(self.current_diff.changed_indices.iter_mut())
            .enumerate()
        {
            if entry.is_some() {
                *entry = None;
                *changed = true;
            }

            self.free_list.push_back(index);
        }

        self.index_map.clear();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.index_map.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.index_map.is_empty()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&K, usize, &V)> {
        self.entries.iter().flat_map(|entry| {
            entry
                .as_ref()
                .map(|entry| (&entry.key, entry.index, &entry.value))
        })
    }

    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.index_map.keys()
    }

    #[inline]
    pub fn keys_ixes(&self) -> impl Iterator<Item = (&K, usize)> {
        self.index_map.iter().map(|(k, &i)| (k, i))
    }

    #[allow(dead_code)]
    /// Verify that
    /// 1. every entry in `index_map` corresponds with a `Some` entry in `entries`
    /// 2. every `None` entry in `entries` corresponds with a missing key in `index_map` and an
    ///    entry in `free_list`
    /// 3. Let I be length of `index_map`. Let `E` be length of part of `entries` with is `Some` and
    ///    `M` be length of part of `entries` which is `None`. Let `F` be length of `free_list. Let
    ///    `C` be value of `capacity`.
    ///    Then:
    ///    - I = E
    ///    - E + M = C
    ///    - F = M
    pub fn check_invariant(&self) {
        // 1
        for (k, &i) in self.index_map.iter() {
            assert!(self.entries[i].is_some());
            assert!(&self.entries[i].as_ref().unwrap().key == k);
        }

        // 2
        for (i, entry) in self.entries.iter().enumerate() {
            match entry {
                None => {
                    for (_, &j) in self.index_map.iter() {
                        assert!(i != j);
                    }

                    let mut has_free = false;
                    for &j in &self.free_list {
                        if i == j {
                            has_free = true;
                        }
                    }
                    assert!(has_free);
                }
                Some(entry) => {
                    assert!(self.index_map[&entry.key] == i);
                }
            }
        }

        // 3
        let count_none = self.entries.iter().filter(|x| x.is_none()).count();
        let count_some = self.entries.iter().filter(|x| x.is_some()).count();
        assert!(self.index_map.len() == count_some);
        assert!(self.free_list.len() == count_none);
        assert!(count_none + count_some == self.capacity);
    }
}

impl<'a, K, V> DiffHandle<'a, K, V>
where
    K: Eq + Hash,
{
    #[inline]
    pub fn changed_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.parent
            .current_diff
            .changed_indices
            .iter()
            .enumerate()
            .flat_map(|(i, &changed)| if changed { Some(i) } else { None })
    }

    /// Iterate over index, keys and values of changed entries in the map. If the entry was
    /// deleted, the key/value pair is None.
    #[inline]
    pub fn changed_entries(&self) -> impl Iterator<Item = (usize, Option<(&K, &V)>)> + '_ {
        let entries = &self.parent.entries;
        self.changed_indices().map(move |ix| {
            let info = entries[ix].as_ref().map(|info| (&info.key, &info.value));
            (ix, info)
        })
    }

    #[inline]
    pub fn inner(&self) -> &StableMap<K, V> {
        self.parent
    }
}

/// The DiffHandle resets the current diff to empty when it is dropped.
impl<'a, K, V> Drop for DiffHandle<'a, K, V>
where
    K: Eq + Hash,
{
    #[inline]
    fn drop(&mut self) {
        for changed in self
            .parent
            .current_diff
            .changed_indices
            .iter_mut()
            .take(self.parent.capacity)
        {
            *changed = false;
        }
    }
}

pub struct StableMapIter<K, V> {
    entries: <Vec<Option<EntryInfo<K, V>>> as IntoIterator>::IntoIter,
}

impl<K, V> Iterator for StableMapIter<K, V> {
    type Item = (K, usize, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.entries.next() {
                None => break None,
                Some(None) => continue,
                Some(Some(entry)) => break Some((entry.key, entry.index, entry.value)),
            }
        }
    }
}

impl<K, V> IntoIterator for StableMap<K, V>
where
    K: Eq + Hash,
{
    type Item = (K, usize, V);
    type IntoIter = StableMapIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        StableMapIter {
            entries: self.entries.into_iter(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StableMap;

    #[test]
    fn test_stable_map() {
        let mut map: StableMap<i32, i32> = StableMap::new(5);

        map.update(100, 0);
        map.update(50, 1);
        map.update(2000, 2);

        {
            let diff = map.take_diff();
            let diff: Vec<(usize, Option<(&i32, &i32)>)> = diff.changed_entries().collect();

            assert_eq!(
                diff,
                vec![
                    (0, Some((&100, &0))),
                    (1, Some((&50, &1))),
                    (2, Some((&2000, &2)))
                ]
            );
        }

        map.update(2000, 3);
        map.update(51, 4);
        map.update(55, 5);
        map.remove(&50);
        map.update(60, 6);

        {
            let diff = map.take_diff();
            let diff: Vec<(usize, Option<(&i32, &i32)>)> = diff.changed_entries().collect();

            assert_eq!(
                diff,
                vec![
                    (1, Some((&60, &6))),
                    (2, Some((&2000, &3))),
                    (3, Some((&51, &4))),
                    (4, Some((&55, &5))),
                ]
            );
        }
    }
}
