use std::hash::Hash;
use std::ops::{Deref, Index, IndexMut};

use ordered_float::OrderedFloat;
use rand::rngs::SmallRng;
use rand::Rng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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
        let mut nearest = [PointId::invalid(); M];
        nearest.copy_from_slice(&node.0[..M]);
        Self(nearest)
    }
}

impl Layer for &Vec<UpperNode> {
    fn nearest_iter(&self, pid: PointId) -> NearestIter<'_> {
        NearestIter {
            nearest: &self[pid.0 as usize].0,
        }
    }
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Debug)]
pub(crate) struct ZeroNode(pub(crate) [PointId; M * 2]);

impl ZeroNode {
    pub(crate) fn rewrite(&mut self, mut iter: impl Iterator<Item = PointId>) {
        for slot in self.0.iter_mut() {
            if let Some(pid) = iter.next() {
                *slot = pid;
            } else if *slot != PointId::invalid() {
                *slot = PointId::invalid();
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
        ZeroNode([PointId::invalid(); M * 2])
    }
}

impl Deref for ZeroNode {
    type Target = [PointId];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Layer for &Vec<ZeroNode> {
    fn nearest_iter(&self, pid: PointId) -> NearestIter<'_> {
        NearestIter {
            nearest: &self[pid.0 as usize].0,
        }
    }
}

pub(crate) trait Layer {
    fn nearest_iter(&self, pid: PointId) -> NearestIter<'_>;
}

pub(crate) struct NearestIter<'a> {
    nearest: &'a [PointId],
}

impl<'a> Iterator for NearestIter<'a> {
    type Item = PointId;

    fn next(&mut self) -> Option<Self::Item> {
        let (&first, rest) = self.nearest.split_first()?;
        if !first.is_valid() {
            return None;
        }
        self.nearest = rest;
        Some(first)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct LayerId(pub usize);

impl LayerId {
    pub(crate) fn random(ml: f32, rng: &mut SmallRng) -> Self {
        let layer = rng.gen::<f32>();
        LayerId((-(layer.ln() * ml)).floor() as usize)
    }

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

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Candidate {
    pub(crate) distance: OrderedFloat<f32>,
    pub(crate) pid: PointId,
}

/// References a `Point` in the `Hnsw`
///
/// This can be used to index into the `Hnsw` to refer to the `Point` data.
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PointId(pub(crate) u32);

impl PointId {
    pub(crate) fn invalid() -> Self {
        PointId(u32::MAX)
    }

    /// Whether this value represents a valid point
    pub fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }
}

impl Default for PointId {
    fn default() -> Self {
        PointId::invalid()
    }
}

impl<P> Index<PointId> for Hnsw<P> {
    type Output = P;

    fn index(&self, index: PointId) -> &Self::Output {
        &self.points[index.0 as usize]
    }
}

impl<P: Point> Index<PointId> for Vec<P> {
    type Output = P;

    fn index(&self, index: PointId) -> &Self::Output {
        &self[index.0 as usize]
    }
}

impl<P: Point> Index<PointId> for [P] {
    type Output = P;

    fn index(&self, index: PointId) -> &Self::Output {
        &self[index.0 as usize]
    }
}

impl Index<PointId> for Vec<ZeroNode> {
    type Output = ZeroNode;

    fn index(&self, index: PointId) -> &Self::Output {
        &self[index.0 as usize]
    }
}

impl IndexMut<PointId> for Vec<ZeroNode> {
    fn index_mut(&mut self, index: PointId) -> &mut Self::Output {
        &mut self[index.0 as usize]
    }
}
