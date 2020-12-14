use std::cmp::{max, min, Ordering};
use std::hash::Hash;
use std::ops::Index;

use ahash::AHashSet as HashSet;
use ordered_float::OrderedFloat;
use rand::rngs::SmallRng;
use rand::{RngCore, SeedableRng};

pub struct Hnsw<P> {
    ef_search: usize,
    points: Vec<P>,
    zero: Vec<ZeroNode>,
    layers: Vec<Vec<UpperNode>>,
}

impl<P> Hnsw<P>
where
    P: Point + std::fmt::Debug,
{
    pub fn new(points: &[P], ef_construction: usize, ef_search: usize) -> (Self, Vec<PointId>) {
        if points.is_empty() {
            return (
                Self {
                    ef_search,
                    zero: Vec::new(),
                    points: Vec::new(),
                    layers: Vec::new(),
                },
                Vec::new(),
            );
        }

        // Give all points a random layer and sort the list of nodes by descending order for
        // construction. This allows us to copy higher layers to lower layers as construction
        // progresses, while preserving randomness in each point's layer and insertion order.

        let mut rng = SmallRng::from_entropy();
        let mut nodes = (0..points.len())
            .map(|i| (LayerId::random(&mut rng), i))
            .collect::<Vec<_>>();
        nodes.sort_unstable_by(|l, r| r.cmp(&l));

        // Sort the original `points` in layer order.
        // TODO: maybe optimize this? https://crates.io/crates/permutation

        let mut new_points = Vec::with_capacity(points.len());
        let mut new_nodes = Vec::with_capacity(points.len());
        let mut out = vec![PointId::invalid(); points.len()];
        for (i, &(layer, idx)) in nodes.iter().enumerate() {
            let pid = PointId(i);
            new_points.push(points[idx].clone());
            new_nodes.push((layer, pid));
            out[idx] = pid;
        }
        let (points, nodes) = (new_points, new_nodes);

        // The layer from the first node is our top layer, or the zero layer if we have no nodes.

        let top = match nodes.first() {
            Some((top, _)) => *top,
            None => LayerId(0),
        };

        // Figure out how many nodes will go on each layer. This helps us allocate memory capacity
        // for each layer in advance, and also helps enable batch insertion of points.

        let mut sizes = vec![0; top.0 + 1];
        for (layer, _) in nodes.iter().copied() {
            sizes[layer.0] += 1;
        }

        let mut start = 0;
        let mut ranges = Vec::with_capacity(top.0);
        for (i, size) in sizes.into_iter().enumerate().rev() {
            // Skip the first point, since we insert the enter point separately
            ranges.push((LayerId(i), max(start, 1)..start + size));
            start += size;
        }

        // Insert the first point so that we have an enter point to start searches with.

        let mut layers = vec![vec![]; top.0];
        let mut zero = Vec::with_capacity(points.len());
        zero.push(ZeroNode::default());

        let mut search = Search::default();
        for (layer, range) in ranges {
            let num = if layer.0 > 0 { M } else { M * 2 };
            for &(_, pid) in &nodes[range] {
                search.reset();
                let point = &points[pid];
                search.push(PointId(0), &points[pid], &points);

                for cur in top.descend() {
                    search.num = if cur <= layer { ef_construction } else { 1 };
                    zero.search(point, &mut search, &points, num);
                    match cur > layer {
                        true => search.cull(),
                        false => break,
                    }
                }

                zero.insert_node(pid, &search.nearest, &points);
            }

            // For layers above the zero layer, make a copy of the current state of the zero layer
            // with `nearest` truncated to `M` elements.
            if layer.0 > 0 {
                let mut upper = Vec::with_capacity(zero.len());
                upper.extend(zero.iter().map(|zero| {
                    let mut upper = UpperNode::default();
                    upper.nearest.copy_from_slice(&zero.nearest[..M]);
                    upper
                }));
                layers[layer.0 - 1] = upper;
            }
        }

        (
            Self {
                ef_search,
                zero,
                points,
                layers,
            },
            out,
        )
    }

    /// Search the index for the points nearest to the reference point `point`
    ///
    /// The results are returned in the `out` parameter; the number of neighbors to search for
    /// is limited by the size of the `out` parameter, and the number of results found is returned
    /// in the return value.
    ///
    /// `PointId` values can be initialized with `PointId::invalid()`.
    pub fn search(&self, point: &P, out: &mut [PointId], search: &mut Search) -> usize {
        if self.points.is_empty() {
            return 0;
        }

        search.reset();
        search.push(PointId(0), point, &self.points);
        for cur in LayerId(self.layers.len()).descend() {
            search.num = if cur.is_zero() { self.ef_search } else { 1 };

            let num = if cur.0 > 0 { M } else { M * 2 };
            match cur.0 {
                0 => self.zero.search(point, search, &self.points, num),
                l => self.layers[l - 1].search(point, search, &self.points, num),
            }

            if !cur.is_zero() {
                search.cull();
            }
        }

        let found = min(search.nearest.len(), out.len());
        for (i, candidate) in search.nearest.iter().take(found).enumerate() {
            out[i] = candidate.pid;
        }
        found
    }
}

/// Keeps mutable state for searching a point's nearest neighbors
///
/// In particular, this contains most of the state used in algorithm 2. The structure is
/// initialized by using `push()` to add the initial enter points.
pub struct Search {
    /// Nodes visited so far (`v` in the paper)
    visited: HashSet<PointId>,
    /// Candidates for further inspection (`C` in the paper)
    candidates: Vec<Candidate>,
    /// Nearest neighbors found so far (`W` in the paper)
    nearest: Vec<Candidate>,
    /// Maximum number of nearest neighbors to retain (`ef` in the paper)
    num: usize,
}

impl Search {
    /// Resets the state to be ready for a new search
    fn reset(&mut self) {
        self.visited.clear();
        self.candidates.clear();
        self.nearest.clear();
    }

    /// Track node `pid` as a potential new neighbor for the given `point`
    ///
    /// Will immediately return if the node has been considered before. This implements
    /// the inner loop from the paper's algorithm 2.
    fn push<P: Point>(&mut self, pid: PointId, point: &P, points: &[P]) {
        if !self.visited.insert(pid) {
            return;
        }

        let other = &points[pid];
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

        let new = Candidate { distance, pid };
        let idx = self.candidates.binary_search(&new).unwrap_or_else(|e| e);
        self.candidates.insert(idx, new);

        let idx = self.nearest.binary_search(&new).unwrap_or_else(|e| e);
        self.nearest.insert(idx, new);
    }

    /// Lower the search to the next lower level
    ///
    /// Re-initialize the `Search`: `nearest`, the output `W` from the last round, now becomes
    /// the set of enter points, which we use to initialize both `candidates` and `visited`.
    fn cull(&mut self) {
        self.nearest.truncate(self.num); // Limit size of the set of nearest neighbors
        self.candidates.clear();
        self.candidates.extend(&self.nearest);
        self.visited.clear();
        self.visited.extend(self.nearest.iter().map(|c| c.pid));
    }
}

impl Default for Search {
    fn default() -> Self {
        Self {
            visited: HashSet::new(),
            candidates: Vec::new(),
            nearest: Vec::new(),
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

impl<P: Point> Index<PointId> for Vec<P> {
    type Output = P;

    fn index(&self, index: PointId) -> &Self::Output {
        &self[index.0]
    }
}

impl<P: Point> Index<PointId> for [P] {
    type Output = P;

    fn index(&self, index: PointId) -> &Self::Output {
        &self[index.0]
    }
}

impl Layer for Vec<ZeroNode> {
    const LINKS: usize = M * 2;

    type Node = ZeroNode;

    fn push(&mut self, new: ZeroNode) {
        self.push(new);
    }

    fn nodes_mut(&mut self) -> &mut [Self::Node] {
        self
    }

    fn nodes(&self) -> &[Self::Node] {
        self
    }
}

impl Layer for Vec<UpperNode> {
    const LINKS: usize = M;

    type Node = UpperNode;

    fn push(&mut self, new: UpperNode) {
        self.push(new);
    }

    fn nodes_mut(&mut self) -> &mut [Self::Node] {
        self
    }

    fn nodes(&self) -> &[Self::Node] {
        self
    }
}

trait Layer {
    const LINKS: usize;

    type Node: Node;

    /// Search this layer for nodes near the given `point`
    ///
    /// This contains the loops from the paper's algorithm 2. `point` represents `q`, the query
    /// element; `search.candidates` contains the enter points `ep`. `points` contains all the
    /// points, which is required to calculate distances between two points.
    ///
    /// The `num` argument represents the number of links from each candidate to consider. This
    /// function may be called for a higher layer (with M links per node) or the zero layer (with
    /// M * 2 links per node), but for performance reasons we often call this function on the data
    /// representation matching the zero layer even when we're referring to a higher layer. In that
    /// case, we use `num` to constrain the number of per-candidate links we consider for search.
    fn search<P: Point>(&self, point: &P, search: &mut Search, points: &[P], num: usize) {
        while let Some(candidate) = search.candidates.pop() {
            if let Some(found) = search.nearest.last() {
                if candidate.distance > found.distance {
                    break;
                }
            }

            for pid in self.nodes()[candidate.pid.0].nearest_iter().take(num) {
                search.push(pid, point, points);
            }
        }
    }

    /// Insert new node in this layer
    ///
    /// `new` contains the `PointId` for the new node; `found` is a slice containing all
    /// `Candidate`s found during searching (ordered from near to far).
    ///
    /// Creates the new node, initializing its `nearest` array and updates the nearest neighbors
    /// for the new node's neighbors if necessary.
    fn insert_node<P: Point>(&mut self, new: PointId, found: &[Candidate], points: &[P]) {
        let mut node = Self::Node::default();
        let new_nearest = node.nearest_mut();

        // Just make sure the candidates are all unique
        debug_assert_eq!(
            found.len(),
            found.iter().map(|c| c.pid).collect::<HashSet<_>>().len()
        );

        // Only use the `Self::LINKS` nearest candidates found
        for (i, candidate) in found.iter().take(Self::LINKS).enumerate() {
            // `candidate` here is the new node's neighbor
            let &Candidate { distance, pid } = candidate;
            new_nearest[i] = Some(pid); // Update the new node's `nearest`

            let old = &points[pid];
            let nearest = self.nodes()[pid.0].nearest();

            // Find the correct index to insert at to keep the neighbor's neighbors sorted
            let idx = nearest
                .binary_search_by(|third| {
                    // `third` here is one of the neighbors of the new node's neighbor.
                    let third = match third {
                        Some(nid) => *nid,
                        // if `third` is `None`, our new `node` is always "closer"
                        None => return Ordering::Greater,
                    };

                    let third_distance = OrderedFloat::from(old.distance(&points[third.0]));
                    distance.cmp(&third_distance)
                })
                .unwrap_or_else(|e| e);

            // It might be possible for all the neighbor's current neighbors to be closer to our
            // neighbor than to the new node, in which case we skip insertion of our new node's ID.
            if idx >= nearest.len() {
                continue;
            }

            let nearest = self.nodes_mut()[pid.0].nearest_mut();
            if nearest[idx].is_none() {
                nearest[idx] = Some(new);
                continue;
            }

            let end = Self::LINKS - 1;
            nearest.copy_within(idx..end, idx + 1);
            nearest[idx] = Some(new);
        }

        self.push(node);
    }

    fn push(&mut self, new: Self::Node);

    fn nodes_mut(&mut self) -> &mut [Self::Node];

    fn nodes(&self) -> &[Self::Node];
}

#[derive(Clone, Copy, Debug, Default)]
struct UpperNode {
    /// The nearest neighbors on this layer
    ///
    /// This is always kept in sorted order (near to far).
    nearest: [Option<PointId>; M],
}

impl Node for UpperNode {
    fn nearest(&self) -> &[Option<PointId>] {
        &self.nearest
    }

    fn nearest_mut(&mut self) -> &mut [Option<PointId>] {
        &mut self.nearest
    }

    fn nearest_iter(&self) -> NearestIter<'_> {
        NearestIter {
            nearest: &self.nearest,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ZeroNode {
    /// The nearest neighbors on this layer
    ///
    /// This is always kept in sorted order (near to far).
    nearest: [Option<PointId>; M * 2],
}

impl Node for ZeroNode {
    fn nearest(&self) -> &[Option<PointId>] {
        &self.nearest
    }

    fn nearest_mut(&mut self) -> &mut [Option<PointId>] {
        &mut self.nearest
    }

    fn nearest_iter(&self) -> NearestIter<'_> {
        NearestIter {
            nearest: &self.nearest,
        }
    }
}

trait Node: Default {
    fn nearest(&self) -> &[Option<PointId>];
    fn nearest_mut(&mut self) -> &mut [Option<PointId>];
    fn nearest_iter(&self) -> NearestIter<'_>;
}

struct NearestIter<'a> {
    nearest: &'a [Option<PointId>],
}

impl<'a> Iterator for NearestIter<'a> {
    type Item = PointId;

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
    fn random(rng: &mut SmallRng) -> Self {
        let layer = rng.next_u32() as f32 / u32::MAX as f32;
        LayerId((-(layer.ln() * (M as f32).ln())).floor() as usize)
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

pub trait Point: Clone {
    fn distance(&self, other: &Self) -> f32;
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Candidate {
    distance: OrderedFloat<f32>,
    pid: PointId,
}

/// References a node in a particular layer (usually the same layer)
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct NodeId(usize);

/// References a `Point` in the `Hnsw`
///
/// This can be used to index into the `Hnsw` to refer to the `Point` data.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PointId(usize);

impl PointId {
    pub fn invalid() -> Self {
        PointId(usize::MAX)
    }
}

/// The parameter `M` from the paper
///
/// This should become a generic argument to `Hnsw` when possible.
const M: usize = 6;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        let (hnsw, pids) = Hnsw::new(
            &[
                Point(0.1, 0.4),
                Point(-0.324, 0.543),
                Point(0.87, -0.33),
                Point(0.452, 0.932),
            ],
            100,
            100,
        );

        let mut search = Search::default();
        let mut results = vec![PointId::invalid()];
        let p = Point(0.1, 0.35);
        let found = hnsw.search(&p, &mut results, &mut search);
        assert_eq!(found, 1);
        assert_eq!(&results, &[pids[0]]);
    }

    #[derive(Clone, Copy, Debug)]
    struct Point(f32, f32);

    impl super::Point for Point {
        fn distance(&self, other: &Self) -> f32 {
            ((self.0 - other.0).powi(2) + (self.1 - other.1).powi(2)).sqrt()
        }
    }
}
