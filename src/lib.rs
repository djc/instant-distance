use std::cmp::{max, Ordering};
use std::ops::{Index, IndexMut};

use ahash::AHashSet as HashSet;
use ordered_float::OrderedFloat;
use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};

pub struct Hnsw<P> {
    ef_construction: usize,
    points: Vec<P>,
    zero: Vec<ZeroNode>,
    layers: Vec<Vec<UpperNode>>,
    rng: SmallRng,
}

impl<P> Hnsw<P>
where
    P: Point,
{
    pub fn new(ef_construction: usize) -> Self {
        Self {
            ef_construction,
            points: Vec::new(),
            zero: Vec::new(),
            layers: Vec::new(),
            rng: SmallRng::from_entropy(),
        }
    }

    /// Insert a point into the `Hnsw`, returning a `PointId`
    ///
    /// `PointId` implements `Hash`, `Eq` and friends, so it can be linked to some value.
    pub fn insert(&mut self, point: P, search: &mut Search) -> PointId {
        let layer = self.rng.next_u32() as f32 / u32::MAX as f32;
        let layer = LayerId((-(layer.ln() * (M as f32).ln())).floor() as usize);
        self.insert_at(point, layer, search)
    }

    /// Deterministic implementation of insertion that takes the `layer` as an argument
    ///
    /// Implements the paper's algorithm 1, although there is a slight difference in that
    /// new elements are always inserted from their selected layer, rather than delaying the
    /// addition of new layers until after the selection of a particular layer.
    fn insert_at(&mut self, point: P, layer: LayerId, search: &mut Search) -> PointId {
        let empty = self.points.is_empty();
        let pid = PointId(self.points.len());
        self.points.push(point);

        let top = LayerId(self.layers.len());
        if layer > top {
            self.layers.resize_with(layer.0, Default::default);
        }

        search.reset(1, top);
        for cur in max(top, layer).descend() {
            search.num = if cur <= layer {
                self.ef_construction
            } else {
                1
            };

            // If this layer already existed, search it for the 1 nearest neighbor
            // (this roughly corresponds to the first loop in the paper's algorithm 1).
            if cur <= top {
                debug_assert_eq!(search.layer, cur);

                // At the first layer that already existed, insert the first element as an initial
                // candidate. Because the zero-th layer always exists, also check if it was empty.
                if cur == top && !empty {
                    search.push(NodeId(0), &self[pid], self);
                }

                self.search_layer(cur, pid, search);
                // If we're still above the layer to insert at, we're going to skip the
                // insertion code below and continue to the next iteration. Before we do so,
                // we update the `Search` so it's ready for the next layer coming up.
                if cur > layer {
                    search.lower(self);
                }
            }

            // If we're above the layer to start inserting links at, skip the rest of this loop.
            if cur > layer {
                continue;
            }

            if cur.is_zero() {
                let nid = NodeId(self.zero.len());
                let mut node = ZeroNode {
                    nearest: Default::default(),
                };
                self.link(cur, (nid, &mut node.nearest), &search.nearest);
                self.zero.push(node);
            } else {
                let nid = NodeId(self.layers[cur.0 - 1].len());
                let lower = match cur.0 == 1 {
                    false => NodeId(self.layers[cur.0 - 2].len()),
                    true => NodeId(self.zero.len()),
                };

                let mut node = UpperNode {
                    pid,
                    lower,
                    nearest: Default::default(),
                };

                self.link(cur, (nid, &mut node.nearest), &search.nearest);
                self.layers[cur.0 - 1].push(node);
            }

            if search.layer == cur && !cur.is_zero() {
                search.lower(self);
            }
        }

        pid
    }

    /// Bidirectionally insert links between newly detected neighbors
    ///
    /// `layer` is the layer we're at; `new` contains the `NodeId` for the new `Node` (which has
    /// not yet been added to the layer) and its still-empty list of nearest neighbors; `found` is
    /// a slice containing the `Candidate`s found during searching (ordered from near to far).
    ///
    /// This just defers to the `Layer`'s `link()` implementation, which specializes on layer type.
    fn link(&mut self, layer: LayerId, new: (NodeId, &mut [Option<NodeId>]), found: &[Candidate]) {
        match layer.0 {
            0 => self.zero.link(new, found, &self.points),
            l => self.layers[l - 1].link(new, found, &self.points),
        }
    }

    /// Search the given `layer` for neighbors closed to the point identified by `pid`
    ///
    /// This implements the outer loop of algorithm 2 from the paper, deferring the state mutation
    /// in the inner loop to the `Search::push()` implementation.
    fn search_layer(&self, layer: LayerId, pid: PointId, search: &mut Search) {
        debug_assert_eq!(search.layer, layer);
        let point = &self[pid];
        while let Some(candidate) = search.candidates.pop() {
            if let Some(found) = search.nearest.last() {
                if candidate.distance > found.distance {
                    break;
                }
            }

            let iter = match layer.0 {
                0 => self.zero[candidate.nid].nearest_iter(),
                l => self.layers[l - 1][candidate.nid].nearest_iter(),
            };

            for nid in iter {
                search.push(nid, point, self);
            }
        }
    }
}

/// Keeps mutable state for searching a point's nearest neighbors
///
/// In particular, this contains most of the state used in algorithm 2. The structure is
/// initialized by using `push()` to add the initial enter points.
pub struct Search {
    /// Nodes visited so far (`v` in the paper)
    visited: HashSet<NodeId>,
    /// Candidates for further inspection (`C` in the paper)
    candidates: Vec<Candidate>,
    /// Nearest neighbors found so far (`W` in the paper)
    nearest: Vec<Candidate>,
    /// Maximum number of nearest neighbors to retain (`ef` in the paper)
    num: usize,
    /// Current layer
    layer: LayerId,
}

impl Search {
    /// Resets the state to be ready for a new search
    fn reset(&mut self, num: usize, layer: LayerId) {
        self.visited.clear();
        self.candidates.clear();
        self.nearest.clear();
        self.num = num;
        self.layer = layer;
    }

    /// Track node `nid` as a potential new neighbor for the given `point`
    ///
    /// Will immediately return if the node has been considered before. This implements
    /// the inner loop from the paper's algorithm 2.
    fn push<P: Point>(&mut self, nid: NodeId, point: &P, hnsw: &Hnsw<P>) {
        if !self.visited.insert(nid) {
            return;
        }

        let pid = match self.layer.0 {
            0 => hnsw.zero.pid(nid),
            l => hnsw.layers[l - 1].pid(nid),
        };

        let other = &hnsw[pid];
        let distance = OrderedFloat::from(point.distance(other));
        if self.nearest.len() >= self.num {
            if let Some(found) = self.nearest.last() {
                if distance > found.distance {
                    return;
                }
            }
        }

        if self.nearest.len() > self.num {
            self.nearest.pop();
        }

        let new = Candidate { distance, nid };
        let idx = self.candidates.binary_search(&new).unwrap_or_else(|e| e);
        self.candidates.insert(idx, new);

        let idx = self.nearest.binary_search(&new).unwrap_or_else(|e| e);
        self.nearest.insert(idx, new);
    }

    /// Lower the search to the next lower level
    ///
    /// Resets `visited`, `candidates` to match `nearest`.
    ///
    /// Panics if called while the `Search` is at level 0.
    fn lower<P: Point>(&mut self, hnsw: &Hnsw<P>) {
        debug_assert!(!self.layer.is_zero());

        self.nearest.truncate(self.num); // Limit size of the set of nearest neighbors
        let old = hnsw.layers[self.layer.0 - 1].nodes();
        for cur in self.nearest.iter_mut() {
            cur.nid = old[cur.nid].lower;
        }

        // Re-initialize the `Search`: `nearest`, the output `W` from the last round, now becomes
        // the set of enter points, which we use to initialize both `candidates` and `visited`.
        self.layer = self.layer.lower();
        self.candidates.clear();
        self.candidates.extend(&self.nearest);
        self.visited.clear();
        self.visited.extend(self.nearest.iter().map(|c| c.nid));
    }
}

impl Default for Search {
    fn default() -> Self {
        Self {
            visited: HashSet::new(),
            candidates: Vec::new(),
            nearest: Vec::new(),
            layer: LayerId(0),
            num: 1,
        }
    }
}

impl<P> Index<PointId> for Hnsw<P> {
    type Output = P;

    fn index(&self, index: PointId) -> &Self::Output {
        &self.points[index.0]
    }
}

impl<P: Point> Index<PointId> for [P] {
    type Output = P;

    fn index(&self, index: PointId) -> &Self::Output {
        &self[index.0]
    }
}

impl Index<NodeId> for Vec<UpperNode> {
    type Output = UpperNode;

    fn index(&self, index: NodeId) -> &Self::Output {
        &self[index.0]
    }
}

impl IndexMut<NodeId> for Vec<UpperNode> {
    fn index_mut(&mut self, index: NodeId) -> &mut Self::Output {
        &mut self[index.0]
    }
}

impl Index<NodeId> for [UpperNode] {
    type Output = UpperNode;

    fn index(&self, index: NodeId) -> &Self::Output {
        &self[index.0]
    }
}

impl Index<NodeId> for Vec<ZeroNode> {
    type Output = ZeroNode;

    fn index(&self, index: NodeId) -> &Self::Output {
        &self[index.0]
    }
}

impl IndexMut<NodeId> for Vec<ZeroNode> {
    fn index_mut(&mut self, index: NodeId) -> &mut Self::Output {
        &mut self[index.0]
    }
}

impl Index<NodeId> for [ZeroNode] {
    type Output = ZeroNode;

    fn index(&self, index: NodeId) -> &Self::Output {
        &self[index.0]
    }
}

impl Layer for Vec<ZeroNode> {
    const LINKS: usize = M * 2;

    type Node = ZeroNode;

    fn pid(&self, nid: NodeId) -> PointId {
        PointId(nid.0)
    }

    fn nodes(&self) -> &[Self::Node] {
        self
    }

    fn nodes_mut(&mut self) -> &mut [Self::Node] {
        self
    }
}

impl Layer for Vec<UpperNode> {
    const LINKS: usize = M;

    type Node = UpperNode;

    fn pid(&self, nid: NodeId) -> PointId {
        self.nodes()[nid].pid
    }

    fn nodes(&self) -> &[Self::Node] {
        self
    }

    fn nodes_mut(&mut self) -> &mut [Self::Node] {
        self
    }
}

trait Layer {
    const LINKS: usize;

    type Node: Node;

    fn pid(&self, nid: NodeId) -> PointId;

    fn nodes(&self) -> &[Self::Node];

    fn nodes_mut(&mut self) -> &mut [Self::Node];

    /// Bidirectionally insert links between newly detected neighbors
    ///
    /// `new` contains the `NodeId` for the new `Node` (which has not yet been added to the layer)
    /// and its still-empty list of nearest neighbors; `found` is a slice containing all
    /// `Candidate`s found during searching (ordered from near to far).
    ///
    /// Initializes both the new node's neighbors (in `new.1`) and updates the nearest neighbors
    /// for the new node's neighbors if necessary.
    fn link<P: Point>(
        &mut self,
        new: (NodeId, &mut [Option<NodeId>]),
        found: &[Candidate],
        points: &[P],
    ) {
        // Just make sure the candidates are all unique
        debug_assert_eq!(
            found.len(),
            found.iter().map(|c| c.nid).collect::<HashSet<_>>().len()
        );

        // Only use the `Self::LINKS` nearest candidates found
        for (i, candidate) in found.iter().take(Self::LINKS).enumerate() {
            // `candidate` here is the new node's neighbor
            let &Candidate { distance, nid } = candidate;
            new.1[i] = Some(nid); // Update the new node's `nearest`

            let pid = self.pid(nid);
            let old = &points[pid.0];
            let nearest = self.nodes()[nid.0].nearest();

            // Find the correct index to insert at to keep the neighbor's neighbors sorted
            let idx = nearest
                .binary_search_by(|third| {
                    // `third` here is one of the neighbors of the new node's neighbor.
                    let third = match third {
                        Some(nid) => *nid,
                        // if `third` is `None`, our new `node` is always "closer"
                        None => return Ordering::Greater,
                    };

                    let pid = self.pid(third);
                    let third_distance = OrderedFloat::from(old.distance(&points[pid.0]));
                    distance.cmp(&third_distance)
                })
                .unwrap_or_else(|e| e);

            // It might be possible for all the neighbor's current neighbors to be closer to our
            // neighbor than to the new node, in which case we skip insertion of our new node's ID.
            if idx >= nearest.len() {
                continue;
            }

            let nearest = self.nodes_mut()[nid.0].nearest_mut();
            if nearest[idx].is_none() {
                nearest[idx] = Some(new.0);
                continue;
            }

            let end = Self::LINKS - 1;
            nearest.copy_within(idx..end, idx + 1);
            nearest[idx] = Some(new.0);
        }
    }
}

#[derive(Debug)]
struct UpperNode {
    /// This node's point
    pid: PointId,
    /// The point's node on the next level down
    ///
    /// This is only used when lowering the search.
    lower: NodeId,
    /// The nearest neighbors on this layer
    ///
    /// This is always kept in sorted order (near to far).
    nearest: [Option<NodeId>; M],
}

impl Node for UpperNode {
    fn nearest(&self) -> &[Option<NodeId>] {
        &self.nearest
    }

    fn nearest_mut(&mut self) -> &mut [Option<NodeId>] {
        &mut self.nearest
    }

    fn nearest_iter(&self) -> NearestIter<'_> {
        NearestIter {
            nearest: &self.nearest,
        }
    }
}

#[derive(Debug)]
struct ZeroNode {
    /// The nearest neighbors on this layer
    ///
    /// This is always kept in sorted order (near to far).
    nearest: [Option<NodeId>; M * 2],
}

impl Node for ZeroNode {
    fn nearest(&self) -> &[Option<NodeId>] {
        &self.nearest
    }

    fn nearest_mut(&mut self) -> &mut [Option<NodeId>] {
        &mut self.nearest
    }

    fn nearest_iter(&self) -> NearestIter<'_> {
        NearestIter {
            nearest: &self.nearest,
        }
    }
}

trait Node {
    fn nearest(&self) -> &[Option<NodeId>];
    fn nearest_mut(&mut self) -> &mut [Option<NodeId>];
    fn nearest_iter(&self) -> NearestIter<'_>;
}

struct NearestIter<'a> {
    nearest: &'a [Option<NodeId>],
}

impl<'a> Iterator for NearestIter<'a> {
    type Item = NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        let (&first, rest) = self.nearest.split_first()?;
        self.nearest = rest;
        if first.is_none() {
            self.nearest = &[];
        }
        first
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct LayerId(usize);

impl LayerId {
    /// Return a `LayerId` for the layer one lower
    ///
    /// Panics when called for `LayerId(0)`.
    fn lower(&self) -> LayerId {
        LayerId(self.0 - 1)
    }

    fn descend(&self) -> DescendingLayerIter {
        DescendingLayerIter { next: Some(self.0) }
    }

    fn is_zero(&self) -> bool {
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

pub trait Point {
    fn distance(&self, other: &Self) -> f32;
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Candidate {
    distance: OrderedFloat<f32>,
    nid: NodeId,
}

/// References a node in a particular layer (usually the same layer)
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct NodeId(usize);

/// References a `Point` in the `Hnsw`
///
/// This can be used to index into the `Hnsw` to refer to the `Point` data.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PointId(usize);

/// The parameter `M` from the paper
///
/// This should become a generic argument to `Hnsw` when possible.
const M: usize = 6;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insertion() {
        let mut search = Search::default();
        let mut hnsw = Hnsw::new(100);
        hnsw.insert(Point(0.1, 0.4), &mut search);
        hnsw.insert(Point(-0.324, 0.543), &mut search);
        hnsw.insert(Point(0.87, -0.33), &mut search);
        hnsw.insert(Point(0.452, 0.932), &mut search);
    }

    struct Point(f32, f32);

    impl super::Point for Point {
        fn distance(&self, other: &Self) -> f32 {
            ((self.0 - other.0).powi(2) + (self.1 - other.1).powi(2)).sqrt()
        }
    }
}
