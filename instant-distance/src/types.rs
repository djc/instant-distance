use std::hash::Hash;
use std::ops::{Deref, Index};

use ordered_float::OrderedFloat;
use parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "serde-big-array")]
use serde_big_array::BigArray;

use crate::{Hnsw, Point, M};

pub(crate) struct Visited {
    store: Vec<u8>,
    generation: u8,
}

impl Visited {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            store: vec![0; capacity],
            generation: 1,
        }
    }

    pub(crate) fn reserve_capacity(&mut self, capacity: usize) {
        if self.store.len() != capacity {
            self.store.resize(capacity, self.generation - 1);
        }
    }

    pub(crate) fn insert(&mut self, pid: PointId) -> bool {
        let slot = &mut self.store[pid.0 as usize];
        if *slot != self.generation {
            *slot = self.generation;
            true
        } else {
            false
        }
    }

    pub(crate) fn extend(&mut self, iter: impl Iterator<Item = PointId>) {
        for pid in iter {
            self.insert(pid);
        }
    }

    pub(crate) fn clear(&mut self) {
        if self.generation < 249 {
            self.generation += 1;
            return;
        }

        let len = self.store.len();
        self.store.clear();
        self.store.resize(len, 0);
        self.generation = 1;
    }
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct UpperNode([PointId; M]);

impl UpperNode {
    pub(crate) fn from_zero(node: &ZeroNode) -> Self {
        let mut nearest = [INVALID; M];
        nearest.copy_from_slice(&node.0[..M]);
        Self(nearest)
    }
}

impl<'a> Layer for &'a [UpperNode] {
    type Slice = &'a [PointId];

    fn nearest_iter(&self, pid: PointId) -> NearestIter<Self::Slice> {
        NearestIter::new(&self[pid.0 as usize].0)
    }
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct ZeroNode(
    #[cfg_attr(feature = "serde", serde(with = "BigArray"))] pub(crate) [PointId; M * 2],
);

impl ZeroNode {
    pub(crate) fn rewrite(&mut self, mut iter: impl Iterator<Item = PointId>) {
        for slot in self.0.iter_mut() {
            if let Some(pid) = iter.next() {
                *slot = pid;
            } else if *slot != INVALID {
                *slot = INVALID;
            } else {
                break;
            }
        }
    }

    pub(crate) fn insert(&mut self, idx: usize, pid: PointId) {
        // It might be possible for all the neighbor's current neighbors to be closer to our
        // neighbor than to the new node, in which case we skip insertion of our new node's ID.
        if idx >= self.0.len() {
            return;
        }

        if self.0[idx].is_valid() {
            let end = (M * 2) - 1;
            self.0.copy_within(idx..end, idx + 1);
        }

        self.0[idx] = pid;
    }

    pub(crate) fn set(&mut self, idx: usize, pid: PointId) {
        self.0[idx] = pid;
    }
}

impl Default for ZeroNode {
    fn default() -> ZeroNode {
        ZeroNode([INVALID; M * 2])
    }
}

impl Deref for ZeroNode {
    type Target = [PointId];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> Layer for &'a [ZeroNode] {
    type Slice = &'a [PointId];

    fn nearest_iter(&self, pid: PointId) -> NearestIter<Self::Slice> {
        NearestIter::new(&self[pid.0 as usize])
    }
}

impl<'a> Layer for &'a [RwLock<ZeroNode>] {
    type Slice = MappedRwLockReadGuard<'a, [PointId]>;

    fn nearest_iter(&self, pid: PointId) -> NearestIter<Self::Slice> {
        NearestIter::new(RwLockReadGuard::map(
            self[pid.0 as usize].read(),
            Deref::deref,
        ))
    }
}

pub(crate) trait Layer {
    type Slice: Deref<Target = [PointId]>;
    fn nearest_iter(&self, pid: PointId) -> NearestIter<Self::Slice>;
}

pub(crate) struct NearestIter<T> {
    node: T,
    cur: usize,
}

impl<T> NearestIter<T>
where
    T: Deref<Target = [PointId]>,
{
    fn new(node: T) -> Self {
        Self { node, cur: 0 }
    }
}

impl<T> Iterator for NearestIter<T>
where
    T: Deref<Target = [PointId]>,
{
    type Item = PointId;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur >= self.node.len() {
            return None;
        }

        let item = self.node[self.cur];
        if !item.is_valid() {
            self.cur = self.node.len();
            return None;
        }

        self.cur += 1;
        Some(item)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct LayerId(pub usize);

impl LayerId {
    pub(crate) fn descend(&self) -> impl Iterator<Item = LayerId> {
        DescendingLayerIter { next: Some(self.0) }
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

struct DescendingLayerIter {
    next: Option<usize>,
}

impl Iterator for DescendingLayerIter {
    type Item = LayerId;

    fn next(&mut self) -> Option<Self::Item> {
        Some(LayerId(match self.next? {
            0 => {
                self.next = None;
                0
            }
            next => {
                self.next = Some(next - 1);
                next
            }
        }))
    }
}

/// A potential nearest neighbor
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Candidate {
    pub(crate) distance: OrderedFloat<f32>,
    /// The identifier for the neighboring point
    pub pid: PointId,
}

/// References a `Point` in the `Hnsw`
///
/// This can be used to index into the `Hnsw` to refer to the `Point` data.
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PointId(pub(crate) u32);

impl PointId {
    /// Whether this value represents a valid point
    pub fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }

    /// Return the identifier value
    pub fn into_inner(self) -> u32 {
        self.0
    }
}

#[doc(hidden)]
// Not part of the public API; only for use in bindings
impl From<u32> for PointId {
    fn from(id: u32) -> Self {
        PointId(id)
    }
}

impl Default for PointId {
    fn default() -> Self {
        INVALID
    }
}

impl<P> Index<PointId> for Hnsw<P> {
    type Output = P;

    fn index(&self, index: PointId) -> &Self::Output {
        &self.points[index.0 as usize]
    }
}

impl<P: Point> Index<PointId> for [P] {
    type Output = P;

    fn index(&self, index: PointId) -> &Self::Output {
        &self[index.0 as usize]
    }
}

impl Index<PointId> for [RwLock<ZeroNode>] {
    type Output = RwLock<ZeroNode>;

    fn index(&self, index: PointId) -> &Self::Output {
        &self[index.0 as usize]
    }
}

pub(crate) const INVALID: PointId = PointId(u32::MAX);
