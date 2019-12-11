use std::collections::{hash_map, HashMap, VecDeque};
use std::hash::Hash;

/// A fixed size key-value map that implements stable indices and supports a diff operation which
/// iterates over the keys that have changed since the last diff. Guarantees no allocation for any
/// operation except `new` (as long as K::clone, K::eq and K::hash do not allocate).
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
                let index = self
                    .free_list
                    .pop_front()
                    .expect("Too many keys in stable map at one time.");
                self.entries[index] = Some(EntryInfo { index, key, value });
                vacant.insert(index);
                (index, None)
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
            let old = self.entries[index].take().unwrap();
            old.value
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
    pub fn get(&self, key: &K) -> Option<(usize, &V)> {
        self.index_map
            .get(key)
            .and_then(move |&index| self.entries[index].as_ref().map(|info| (index, &info.value)))
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

    pub fn len(&self) -> usize {
        self.index_map.len()
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
