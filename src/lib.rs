use std::cmp::{max, min, Ordering, Reverse};
use std::collections::BinaryHeap;
use std::hash::Hash;
use std::ops::{Index, IndexMut};

use ahash::AHashSet as HashSet;
#[cfg(feature = "indicatif")]
use indicatif::ProgressBar;
use ordered_float::OrderedFloat;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "serde")]
use serde_big_array::big_array;

/// Parameters for building the `Hnsw`
pub struct Builder {
    ef_search: Option<usize>,
    ef_construction: Option<usize>,
    heuristic: Option<Heuristic>,
    ml: Option<f32>,
    seed: Option<u64>,
    #[cfg(feature = "indicatif")]
    progress: Option<ProgressBar>,
}

impl Builder {
    /// Set the `efConstruction` parameter from the paper
    pub fn ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = Some(ef_construction);
        self
    }

    /// Set the `ef` parameter from the paper
    ///
    /// If the `efConstruction` parameter is not already set, it will be set
    /// to the same value as `ef` by default.
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = Some(ef);
        self
    }

    pub fn select_heuristic(mut self, params: Option<Heuristic>) -> Self {
        self.heuristic = params;
        self
    }

    /// Set the `mL` parameter from the paper
    ///
    /// If the `mL` parameter is not already set, it defaults to `ln(M)`.
    pub fn ml(mut self, ml: f32) -> Self {
        self.ml = Some(ml);
        self
    }

    /// Set the seed value for the random number generator used to generate a layer for each point
    ///
    /// If this value is left unset, a seed is generated from entropy (via `getrandom()`).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// A `ProgressBar` to track `Hnsw` construction progress
    #[cfg(feature = "indicatif")]
    pub fn progress(mut self, bar: ProgressBar) -> Self {
        self.progress = Some(bar);
        self
    }

    /// Build the `Hnsw` with the given set of points
    pub fn build<P: Point>(self, points: &[P]) -> (Hnsw<P>, Vec<PointId>) {
        Hnsw::new(points, self)
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            ef_search: None,
            ef_construction: None,
            heuristic: Some(Heuristic::default()),
            ml: None,
            seed: None,
            #[cfg(feature = "indicatif")]
            progress: None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Heuristic {
    pub extend_candidates: bool,
    pub keep_pruned: bool,
}

impl Default for Heuristic {
    fn default() -> Self {
        Heuristic {
            extend_candidates: false,
            keep_pruned: true,
        }
    }
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct Hnsw<P> {
    ef_search: usize,
    points: Vec<P>,
    zero: Vec<ZeroNode>,
    layers: Vec<Vec<UpperNode>>,
}

impl<P> Hnsw<P>
where
    P: Point,
{
    pub fn builder() -> Builder {
        Builder::default()
    }

    fn new(points: &[P], builder: Builder) -> (Self, Vec<PointId>) {
        let ef_search = builder.ef_search.unwrap_or(100);
        let ef_construction = builder.ef_construction.unwrap_or(100);
        let ml = builder.ml.unwrap_or_else(|| (M as f32).ln());
        let mut rng = match builder.seed {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_entropy(),
        };

        #[cfg(feature = "indicatif")]
        let progress = builder.progress;
        #[cfg(feature = "indicatif")]
        if let Some(bar) = &progress {
            bar.set_draw_delta(1_000);
            bar.set_length(points.len() as u64);
        }

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

        assert!(points.len() < u32::MAX as usize);
        let mut nodes = (0..points.len())
            .map(|i| (LayerId::random(ml, &mut rng), i))
            .collect::<Vec<_>>();
        nodes.sort_unstable_by_key(|&n| Reverse(n));

        // Find out how many layers are needed, so that we can discard empty layers in the next
        // step. Since layer IDs are randomly generated, there might be big gaps.

        let (mut num_layers, mut prev) = (1, nodes[0].0);
        for (layer, _) in nodes.iter() {
            if *layer != prev {
                num_layers += 1;
                prev = *layer;
            }
        }

        // Sort the original `points` in layer order.
        // TODO: maybe optimize this? https://crates.io/crates/permutation

        let mut cur_layer = LayerId(num_layers - 1);
        let mut prev_layer = nodes[0].0;
        let mut new_points = Vec::with_capacity(points.len());
        let mut new_nodes = Vec::with_capacity(points.len());
        let mut out = vec![PointId::invalid(); points.len()];
        for (i, &(layer, idx)) in nodes.iter().enumerate() {
            if prev_layer != layer {
                cur_layer = LayerId(cur_layer.0 - 1);
                prev_layer = layer;
            }

            let pid = PointId(i as u32);
            new_points.push(points[idx].clone());
            new_nodes.push((cur_layer, pid));
            out[idx] = pid;
        }
        let (points, nodes) = (new_points, new_nodes);
        debug_assert_eq!(nodes.last().unwrap().0, LayerId(0));
        debug_assert_eq!(nodes.first().unwrap().0, LayerId(num_layers - 1));

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

        let mut insertion = Search {
            ef: ef_construction,
            ..Default::default()
        };

        let mut pool = SearchPool::default();
        let mut batch = Vec::new();
        let mut done = Vec::new();
        let max_batch_len = num_cpus::get() * 4;
        for (layer, mut range) in ranges {
            let num = if layer.0 > 0 { M } else { M * 2 };
            #[cfg(feature = "indicatif")]
            if let Some(bar) = &progress {
                bar.set_message(&format!("Building index (layer {})", layer.0));
            }

            while range.start < range.end {
                let len = min(range.len(), max_batch_len);
                batch.clear();
                batch.extend(
                    nodes[range.start..(range.start + len)]
                        .iter()
                        .map(|&(_, pid)| (pid, pool.pop())),
                );

                batch.par_iter_mut().for_each(|(pid, search)| {
                    let point = &points[*pid];
                    search.push(PointId(0), point, &points);
                    for cur in top.descend() {
                        search.ef = if cur <= layer { ef_construction } else { 1 };
                        match cur > layer {
                            true => {
                                layers[cur.0 - 1].search(point, search, &points, num);
                                search.cull();
                            }
                            false => {
                                zero.search(point, search, &points, num);
                                break;
                            }
                        }
                    }
                });

                done.clear();
                for (pid, mut search) in batch.drain(..) {
                    for added in done.iter().copied() {
                        search.push(added, &points[pid], &points);
                    }

                    insert(
                        pid,
                        &mut insertion,
                        &mut search,
                        &mut zero,
                        &points,
                        &builder.heuristic,
                    );
                    done.push(pid);
                    pool.push(search);
                }

                #[cfg(feature = "indicatif")]
                if let Some(bar) = &progress {
                    bar.inc(done.len() as u64);
                }

                range.start += len;
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

        #[cfg(feature = "indicatif")]
        if let Some(bar) = progress {
            bar.finish();
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
    pub fn search(&self, point: &P, out: &mut [PointId], search: &mut Search) -> usize {
        if self.points.is_empty() {
            return 0;
        }

        search.reset();
        search.push(PointId(0), point, &self.points);
        for cur in LayerId(self.layers.len()).descend() {
            let (ef, num) = match cur.is_zero() {
                true => (self.ef_search, M * 2),
                false => (1, M),
            };

            search.ef = ef;
            match cur.0 {
                0 => self.zero.search(point, search, &self.points, num),
                l => self.layers[l - 1].search(point, search, &self.points, num),
            }

            if !cur.is_zero() {
                search.cull();
            }
        }

        let found = min(search.nearest.len(), out.len());
        for (i, candidate) in search.nearest[..found].iter().enumerate() {
            out[i] = candidate.pid;
        }
        found
    }

    /// Iterate over the keys and values in this index
    pub fn iter(&self) -> impl Iterator<Item = (PointId, &P)> {
        self.points
            .iter()
            .enumerate()
            .map(|(i, p)| (PointId(i as u32), p))
    }
}

/// Insert new node in the zero layer
///
/// * `new`: the `PointId` for the new node
/// * `insertion`: a `Search` for shrinking a neighbor set (only used with heuristic neighbor selection)
/// * `search`: the result for searching potential neighbors for the new node
/// *  `layer` contains all the nodes at the current layer
/// * `points` is a slice of all the points in the index
///
/// Creates the new node, initializing its `nearest` array and updates the nearest neighbors
/// for the new node's neighbors if necessary before appending the new node to the layer.
fn insert<P: Point>(
    new: PointId,
    insertion: &mut Search,
    search: &mut Search,
    layer: &mut Vec<ZeroNode>,
    points: &[P],
    heuristic: &Option<Heuristic>,
) {
    layer.push(ZeroNode::default());
    let found = match heuristic {
        None => search.select_simple(),
        Some(heuristic) => search.select_heuristic(&layer, M * 2, &points[new], points, *heuristic),
    };

    // Just make sure the candidates are all unique
    debug_assert_eq!(
        found.len(),
        found.iter().map(|c| c.pid).collect::<HashSet<_>>().len()
    );

    for (i, candidate) in found.iter().take(M * 2).enumerate() {
        // `candidate` here is the new node's neighbor
        let &Candidate { distance, pid } = candidate;
        if let Some(heuristic) = heuristic {
            insertion.reset();
            let candidate_point = &points[pid];
            insertion.push(new, candidate_point, points);
            for hop in layer.nearest_iter(pid) {
                insertion.push(hop, candidate_point, points);
            }

            let found =
                insertion.select_heuristic(&layer, M * 2, candidate_point, points, *heuristic);
            for (i, slot) in layer[pid].nearest.iter_mut().enumerate() {
                if let Some(&Candidate { pid, .. }) = found.get(i) {
                    *slot = pid;
                } else if *slot != PointId::invalid() {
                    *slot = PointId::invalid();
                } else {
                    break;
                }
            }

            layer[new].nearest[i] = pid;
        } else {
            // Find the correct index to insert at to keep the neighbor's neighbors sorted
            let old = &points[pid];
            let nearest = &layer[pid].nearest;
            let idx = nearest
                .binary_search_by(|third| {
                    // `third` here is one of the neighbors of the new node's neighbor.
                    let third = match third {
                        pid if pid.is_valid() => *pid,
                        // if `third` is `None`, our new `node` is always "closer"
                        _ => return Ordering::Greater,
                    };

                    distance.cmp(&old.distance(&points[third]).into())
                })
                .unwrap_or_else(|e| e);

            // It might be possible for all the neighbor's current neighbors to be closer to our
            // neighbor than to the new node, in which case we skip insertion of our new node's ID.
            if idx >= nearest.len() {
                layer[new].nearest[i] = pid;
                continue;
            }

            let nearest = &mut layer[pid].nearest;
            if nearest[idx].is_valid() {
                let end = (M * 2) - 1;
                nearest.copy_within(idx..end, idx + 1);
            }

            nearest[idx] = new;
            layer[new].nearest[i] = pid;
        }
    }
}

#[derive(Default)]
struct SearchPool {
    pool: Vec<Search>,
}

impl SearchPool {
    fn pop(&mut self) -> Search {
        match self.pool.pop() {
            Some(mut search) => {
                search.reset();
                search
            }
            None => Search::default(),
        }
    }

    fn push(&mut self, search: Search) {
        self.pool.push(search);
    }
}

impl Layer for [ZeroNode] {
    fn nearest_iter(&self, pid: PointId) -> NearestIter<'_> {
        NearestIter {
            nearest: &self[pid.0 as usize].nearest,
        }
    }
}

impl Layer for [UpperNode] {
    fn nearest_iter(&self, pid: PointId) -> NearestIter<'_> {
        NearestIter {
            nearest: &self[pid.0 as usize].nearest,
        }
    }
}

trait Layer {
    /// Search this layer for nodes near the given `point`
    ///
    /// This contains the loops from the paper's algorithm 2. `point` represents `q`, the query
    /// element; `search.candidates` contains the enter points `ep`. `points` contains all the
    /// points, which is required to calculate distances between two points.
    ///
    /// The `links` argument represents the number of links from each candidate to consider. This
    /// function may be called for a higher layer (with M links per node) or the zero layer (with
    /// M * 2 links per node), but for performance reasons we often call this function on the data
    /// representation matching the zero layer even when we're referring to a higher layer. In that
    /// case, we use `links` to constrain the number of per-candidate links we consider for search.
    fn search<P: Point>(&self, point: &P, search: &mut Search, points: &[P], links: usize) {
        while let Some(Reverse(candidate)) = search.candidates.pop() {
            if let Some(furthest) = search.nearest.last() {
                if candidate.distance > furthest.distance {
                    break;
                }
            }

            for pid in self.nearest_iter(candidate.pid).take(links) {
                search.push(pid, point, points);
            }
        }

        search.nearest.sort_unstable();
        search.nearest.truncate(search.ef);
    }

    fn nearest_iter(&self, pid: PointId) -> NearestIter<'_>;
}

/// Keeps mutable state for searching a point's nearest neighbors
///
/// In particular, this contains most of the state used in algorithm 2. The structure is
/// initialized by using `push()` to add the initial enter points.
pub struct Search {
    /// Nodes visited so far (`v` in the paper)
    visited: HashSet<PointId>,
    /// Candidates for further inspection (`C` in the paper)
    candidates: BinaryHeap<Reverse<Candidate>>,
    /// Nearest neighbors found so far (`W` in the paper)
    nearest: Vec<Candidate>,
    /// Working set for heuristic selection
    working: Vec<Candidate>,
    discarded: Vec<Candidate>,
    /// Maximum number of nearest neighbors to retain (`ef` in the paper)
    ef: usize,
}

impl Search {
    /// Resets the state to be ready for a new search
    fn reset(&mut self) {
        let Search {
            visited,
            candidates,
            nearest,
            working,
            discarded,
            ef: _,
        } = self;

        visited.clear();
        candidates.clear();
        nearest.clear();
        working.clear();
        discarded.clear();
    }

    /// Selection of neighbors for insertion (algorithm 3 from the paper)
    fn select_simple(&mut self) -> &[Candidate] {
        self.nearest.sort_unstable();
        &self.nearest
    }

    fn select_heuristic<P: Point>(
        &mut self,
        layer: &[ZeroNode],
        num: usize,
        point: &P,
        points: &[P],
        params: Heuristic,
    ) -> &[Candidate] {
        self.working.clear();
        // Get input candidates from `self.nearest` and store them in `self.working`.
        // `self.candidates` will represent `W` from the paper's algorithm 4 for now.
        for &candidate in &self.nearest {
            self.working.push(candidate);
            if params.extend_candidates {
                for hop in layer.nearest_iter(candidate.pid) {
                    if !self.visited.insert(hop) {
                        continue;
                    }

                    let other = &points[hop];
                    let distance = OrderedFloat::from(point.distance(other));
                    let new = Candidate { distance, pid: hop };
                    self.working.push(new);
                }
            }
        }

        self.working.sort_unstable();
        self.nearest.clear();
        self.discarded.clear();
        for candidate in self.working.drain(..) {
            if self.nearest.len() >= num {
                break;
            }

            // Disadvantage candidates which are closer to an existing result point than they
            // are to the query point, to facilitate bridging between clustered points.
            let candidate_point = &points[candidate.pid];
            let nearest = !self.nearest.iter().any(|result| {
                let distance = OrderedFloat::from(candidate_point.distance(&points[result.pid]));
                distance < candidate.distance
            });

            match nearest {
                true => self.nearest.push(candidate),
                false => self.discarded.push(candidate),
            }
        }

        if params.keep_pruned {
            // Add discarded connections from `working` (`Wd`) to `self.nearest` (`R`)
            for candidate in self.discarded.drain(..) {
                if self.nearest.len() >= num {
                    break;
                }
                self.nearest.push(candidate);
            }
        }

        self.nearest.sort_unstable();
        &self.nearest
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
        let new = Candidate { distance, pid };
        let idx = match self.nearest.binary_search(&new) {
            Err(idx) if idx < self.ef => idx,
            Err(_) => return,
            Ok(_) => unreachable!(),
        };

        self.nearest.insert(idx, new);
        self.candidates.push(Reverse(new));
        if self.nearest.len() > self.ef {
            self.nearest.truncate(self.ef);
        }
    }

    /// Lower the search to the next lower level
    ///
    /// Re-initialize the `Search`: `nearest`, the output `W` from the last round, now becomes
    /// the set of enter points, which we use to initialize both `candidates` and `visited`.
    ///
    /// Invariant: `nearest` should be sorted before this is called. This is generally the case
    /// because `Layer::search()` is always called right before calling `cull()`.
    fn cull(&mut self) {
        self.nearest.truncate(self.ef); // Limit size of the set of nearest neighbors

        self.candidates.clear();
        for &candidate in self.nearest.iter() {
            self.candidates.push(Reverse(candidate));
        }

        self.visited.clear();
        self.visited.extend(self.nearest.iter().map(|c| c.pid));
    }
}

impl Default for Search {
    fn default() -> Self {
        Self {
            visited: HashSet::new(),
            candidates: BinaryHeap::new(),
            nearest: Vec::new(),
            working: Vec::new(),
            discarded: Vec::new(),
            ef: 1,
        }
    }
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Debug, Default)]
struct UpperNode {
    /// The nearest neighbors on this layer
    ///
    /// This is always kept in sorted order (near to far).
    nearest: [PointId; M],
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Debug)]
struct ZeroNode {
    /// The nearest neighbors on this layer
    ///
    /// This is always kept in sorted order (near to far).
    #[cfg_attr(feature = "serde", serde(with = "BigArray"))]
    nearest: [PointId; M * 2],
}

#[cfg(feature = "serde-big-array")]
big_array! { BigArray; }

impl Default for ZeroNode {
    fn default() -> ZeroNode {
        ZeroNode {
            nearest: [PointId::invalid(); M * 2],
        }
    }
}

struct NearestIter<'a> {
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
struct LayerId(usize);

impl LayerId {
    fn random(ml: f32, rng: &mut SmallRng) -> Self {
        let layer = rng.gen::<f32>();
        LayerId((-(layer.ln() * ml)).floor() as usize)
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

pub trait Point: Clone + Sync {
    fn distance(&self, other: &Self) -> f32;
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Candidate {
    distance: OrderedFloat<f32>,
    pid: PointId,
}

/// References a `Point` in the `Hnsw`
///
/// This can be used to index into the `Hnsw` to refer to the `Point` data.
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PointId(u32);

impl PointId {
    fn invalid() -> Self {
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

/// The parameter `M` from the paper
///
/// This should become a generic argument to `Hnsw` when possible.
const M: usize = 6;
