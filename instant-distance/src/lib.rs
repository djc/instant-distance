use std::cmp::{max, Ordering, Reverse};
use std::collections::BinaryHeap;
use std::collections::HashSet;
#[cfg(feature = "indicatif")]
use std::sync::atomic::{self, AtomicUsize};

#[cfg(feature = "indicatif")]
use indicatif::ProgressBar;
use ordered_float::OrderedFloat;
use parking_lot::{Mutex, RwLock};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

mod types;
pub use types::PointId;
use types::{Candidate, Layer, LayerId, UpperNode, Visited, ZeroNode, INVALID};

#[derive(Clone)]
/// Parameters for building the `Hnsw`
pub struct Builder {
    ef_search: usize,
    ef_construction: usize,
    heuristic: Option<Heuristic>,
    ml: f32,
    seed: u64,
    #[cfg(feature = "indicatif")]
    progress: Option<ProgressBar>,
}

impl Builder {
    /// Set the `efConstruction` parameter from the paper
    pub fn ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// Set the `ef` parameter from the paper
    ///
    /// If the `efConstruction` parameter is not already set, it will be set
    /// to the same value as `ef` by default.
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    pub fn select_heuristic(mut self, params: Option<Heuristic>) -> Self {
        self.heuristic = params;
        self
    }

    /// Set the `mL` parameter from the paper
    ///
    /// If the `mL` parameter is not already set, it defaults to `1.0 / ln(M)`.
    pub fn ml(mut self, ml: f32) -> Self {
        self.ml = ml;
        self
    }

    /// Set the seed value for the random number generator used to generate a layer for each point
    ///
    /// If this value is left unset, a seed is generated from entropy (via `getrandom()`).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// A `ProgressBar` to track `Hnsw` construction progress
    #[cfg(feature = "indicatif")]
    pub fn progress(mut self, bar: ProgressBar) -> Self {
        self.progress = Some(bar);
        self
    }

    /// Build an `HnswMap` with the given sets of points and values
    pub fn build<P: Point, V: Clone>(self, points: Vec<P>, values: Vec<V>) -> HnswMap<P, V> {
        HnswMap::new(points, values, self)
    }

    /// Build the `Hnsw` with the given set of points
    pub fn build_hnsw<P: Point>(self, points: Vec<P>) -> (Hnsw<P>, Vec<PointId>) {
        Hnsw::new(points, self)
    }

    #[doc(hidden)]
    pub fn into_parts(self) -> (usize, usize, f32, u64) {
        let Self {
            ef_search,
            ef_construction,
            heuristic: _,
            ml,
            seed,
            ..
        } = self;
        (ef_search, ef_construction, ml, seed)
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            ef_search: 100,
            ef_construction: 100,
            heuristic: Some(Heuristic::default()),
            ml: 1.0 / (M as f32).ln(),
            seed: rand::random(),
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
pub struct HnswMap<P, V> {
    hnsw: Hnsw<P>,
    pub values: Vec<V>,
}

impl<P, V> HnswMap<P, V>
where
    P: Point,
    V: Clone,
{
    fn new(points: Vec<P>, values: Vec<V>, builder: Builder) -> Self {
        let (hnsw, ids) = Hnsw::new(points, builder);

        let mut sorted = ids.into_iter().enumerate().collect::<Vec<_>>();
        sorted.sort_unstable_by(|a, b| a.1.cmp(&b.1));
        let new = sorted
            .into_iter()
            .map(|(src, _)| values[src].clone())
            .collect();

        Self { hnsw, values: new }
    }

    pub fn search<'a>(
        &'a self,
        point: &P,
        search: &'a mut Search,
    ) -> impl ExactSizeIterator<Item = MapItem<'a, P, V>> + 'a {
        self.hnsw
            .search(point, search)
            .map(move |item| MapItem::from(item, self))
    }

    /// Iterate over the keys and values in this index
    pub fn iter(&self) -> impl Iterator<Item = (PointId, &P)> {
        self.hnsw.iter()
    }

    #[doc(hidden)]
    pub fn get(&self, i: usize, search: &Search) -> Option<MapItem<'_, P, V>> {
        Some(MapItem::from(self.hnsw.get(i, search)?, self))
    }

    pub fn insert(&mut self, point: P, value: V) -> Result<PointId, Box<dyn std::error::Error>> {
        let point_id = self.hnsw.insert(point, 100, Some(Heuristic::default()));
        self.values.push(value);
        Ok(point_id)
    }
}

pub struct MapItem<'a, P, V> {
    pub distance: f32,
    pub pid: PointId,
    pub point: &'a P,
    pub value: &'a V,
}

impl<'a, P, V> MapItem<'a, P, V> {
    fn from(item: Item<'a, P>, map: &'a HnswMap<P, V>) -> Self {
        MapItem {
            distance: item.distance,
            pid: item.pid,
            point: item.point,
            value: &map.values[item.pid.0 as usize],
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

    fn new(points: Vec<P>, builder: Builder) -> (Self, Vec<PointId>) {
        let ef_search = builder.ef_search;
        let ef_construction = builder.ef_construction;
        let ml = builder.ml;
        let heuristic = builder.heuristic;
        let mut rng = SmallRng::seed_from_u64(builder.seed);

        #[cfg(feature = "indicatif")]
        let progress = builder.progress;
        #[cfg(feature = "indicatif")]
        if let Some(bar) = &progress {
            bar.set_length(points.len() as u64);
            bar.set_message("Build index (preparation)");
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

        // Determine the number and size of layers.

        let mut sizes = Vec::new();
        let mut num = points.len();
        loop {
            let next = (num as f32 * ml) as usize;
            if next < M {
                break;
            }
            sizes.push((num - next, num));
            num = next;
        }
        sizes.push((num, num));
        sizes.reverse();
        let top = LayerId(sizes.len() - 1);

        // Give all points a random layer and sort the list of nodes by descending order for
        // construction. This allows us to copy higher layers to lower layers as construction
        // progresses, while preserving randomness in each point's layer and insertion order.

        assert!(points.len() < u32::MAX as usize);
        let mut shuffled = (0..points.len())
            .map(|i| (PointId(rng.gen_range(0..points.len() as u32)), i))
            .collect::<Vec<_>>();
        shuffled.sort_unstable();

        let mut out = vec![INVALID; points.len()];
        let points = shuffled
            .into_iter()
            .enumerate()
            .map(|(i, (_, idx))| {
                out[idx] = PointId(i as u32);
                points[idx].clone()
            })
            .collect::<Vec<_>>();

        // Figure out how many nodes will go on each layer. This helps us allocate memory capacity
        // for each layer in advance, and also helps enable batch insertion of points.

        let num_layers = sizes.len();
        let mut ranges = Vec::with_capacity(top.0);
        for (i, (size, cumulative)) in sizes.into_iter().enumerate() {
            let start = cumulative - size;
            // Skip the first point, since we insert the enter point separately
            ranges.push((LayerId(num_layers - i - 1), max(start, 1)..cumulative));
        }

        // Initialize data for layers

        let mut layers = vec![vec![]; top.0];
        let zero = points
            .iter()
            .map(|_| RwLock::new(ZeroNode::default()))
            .collect::<Vec<_>>();

        let state = Construction {
            zero: zero.as_slice(),
            pool: SearchPool::new(points.len()),
            top,
            points: &points,
            heuristic,
            ef_construction,
            #[cfg(feature = "indicatif")]
            progress,
            #[cfg(feature = "indicatif")]
            done: AtomicUsize::new(0),
        };

        for (layer, range) in ranges {
            #[cfg(feature = "indicatif")]
            if let Some(bar) = &state.progress {
                bar.set_message(format!("Building index (layer {})", layer.0));
            }

            let inserter = |pid| state.insert(pid, layer, &layers);

            let end = range.end;
            if layer == top {
                range.into_iter().for_each(|i| inserter(PointId(i as u32)))
            } else {
                range
                    .into_par_iter()
                    .for_each(|i| inserter(PointId(i as u32)));
            }

            // For layers above the zero layer, make a copy of the current state of the zero layer
            // with `nearest` truncated to `M` elements.
            if !layer.is_zero() {
                (&state.zero[..end])
                    .into_par_iter()
                    .map(|zero| UpperNode::from_zero(&zero.read()))
                    .collect_into_vec(&mut layers[layer.0 - 1]);
            }
        }

        #[cfg(feature = "indicatif")]
        if let Some(bar) = &state.progress {
            bar.finish();
        }

        (
            Self {
                ef_search,
                zero: zero.into_iter().map(|node| node.into_inner()).collect(),
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
    pub fn search<'a, 'b: 'a>(
        &'b self,
        point: &P,
        search: &'a mut Search,
    ) -> impl ExactSizeIterator<Item = Item<'b, P>> + 'a {
        search.reset();
        let map = move |candidate| Item::new(candidate, self);
        if self.points.is_empty() {
            return search.iter().map(map);
        }

        search.visited.reserve_capacity(self.points.len());
        search.push(PointId(0), point, &self.points);
        for cur in LayerId(self.layers.len()).descend() {
            let (ef, num) = match cur.is_zero() {
                true => (self.ef_search, M * 2),
                false => (1, M),
            };

            search.ef = ef;
            match cur.0 {
                0 => search.search(point, self.zero.as_slice(), &self.points, num),
                l => search.search(point, self.layers[l - 1].as_slice(), &self.points, num),
            }

            if !cur.is_zero() {
                search.cull();
            }
        }

        search.iter().map(map)
    }

    /// Iterate over the keys and values in this index
    pub fn iter(&self) -> impl Iterator<Item = (PointId, &P)> {
        self.points
            .iter()
            .enumerate()
            .map(|(i, p)| (PointId(i as u32), p))
    }

    #[doc(hidden)]
    pub fn get(&self, i: usize, search: &Search) -> Option<Item<'_, P>> {
        Some(Item::new(search.nearest.get(i).copied()?, self))
    }

    pub fn insert(
        &mut self,
        point: P,
        ef_construction: usize,
        heuristic: Option<Heuristic>,
    ) -> PointId {
        let new_pid = self.points.len();
        let new_point_id = PointId(new_pid as u32);

        self.points.push(point);
        self.zero.push(ZeroNode::default());

        let zeros = self
            .zero
            .iter()
            .map(|z| RwLock::new(z.clone()))
            .collect::<Vec<_>>();

        let top = if self.layers.is_empty() {
            LayerId(0)
        } else {
            LayerId(self.layers.len())
        };

        let construction = Construction {
            zero: zeros.as_slice(),
            pool: SearchPool::new(self.points.len()),
            top,
            points: self.points.as_slice(),
            heuristic,
            ef_construction,
            #[cfg(feature = "indicatif")]
            progress: None,
            #[cfg(feature = "indicatif")]
            done: AtomicUsize::new(0),
        };

        let new_layer = construction.top;
        construction.insert(new_point_id, new_layer, &self.layers);

        self.zero = construction
            .zero
            .iter()
            .map(|node| node.read().clone())
            .collect();

        new_point_id
    }
}

pub struct Item<'a, P> {
    pub distance: f32,
    pub pid: PointId,
    pub point: &'a P,
}

impl<'a, P> Item<'a, P> {
    fn new(candidate: Candidate, hnsw: &'a Hnsw<P>) -> Self {
        Self {
            distance: candidate.distance.into_inner(),
            pid: candidate.pid,
            point: &hnsw[candidate.pid],
        }
    }
}

struct Construction<'a, P: Point> {
    zero: &'a [RwLock<ZeroNode>],
    pool: SearchPool,
    top: LayerId,
    points: &'a [P],
    heuristic: Option<Heuristic>,
    ef_construction: usize,
    #[cfg(feature = "indicatif")]
    progress: Option<ProgressBar>,
    #[cfg(feature = "indicatif")]
    done: AtomicUsize,
}

impl<'a, P: Point> Construction<'a, P> {
    /// Insert new node in the zero layer
    ///
    /// * `new` is the `PointId` for the new node
    /// * `layer` contains all the nodes at the current layer
    /// * `layers` refers to the existing higher-level layers
    ///
    /// Creates the new node, initializing its `nearest` array and updates the nearest neighbors
    /// for the new node's neighbors if necessary before appending the new node to the layer.
    fn insert(&self, new: PointId, layer: LayerId, layers: &[Vec<UpperNode>]) {
        let mut node = self.zero[new].write();
        let (mut search, mut insertion) = self.pool.pop();
        insertion.ef = self.ef_construction;

        let point = &self.points[new];
        search.reset();
        search.push(PointId(0), point, self.points);
        let num = if layer.is_zero() { M * 2 } else { M };

        for cur in self.top.descend() {
            search.ef = if cur <= layer {
                self.ef_construction
            } else {
                1
            };
            match cur > layer {
                true => {
                    search.search(point, layers[cur.0 - 1].as_slice(), self.points, num);
                    search.cull();
                }
                false => {
                    search.search(point, self.zero, self.points, num);
                    break;
                }
            }
        }

        let found = match self.heuristic {
            None => {
                let candidates = search.select_simple();
                &candidates[..Ord::min(candidates.len(), M * 2)]
            }
            Some(heuristic) => {
                search.select_heuristic(&self.points[new], self.zero, self.points, heuristic)
            }
        };

        // Just make sure the candidates are all unique
        debug_assert_eq!(
            found.len(),
            found.iter().map(|c| c.pid).collect::<HashSet<_>>().len()
        );

        for (i, candidate) in found.iter().enumerate() {
            // `candidate` here is the new node's neighbor
            let &Candidate { distance, pid } = candidate;
            if let Some(heuristic) = self.heuristic {
                let found = insertion.add_neighbor_heuristic(
                    new,
                    self.zero.nearest_iter(pid),
                    self.zero,
                    &self.points[pid],
                    self.points,
                    heuristic,
                );

                self.zero[pid]
                    .write()
                    .rewrite(found.iter().map(|candidate| candidate.pid));
            } else {
                // Find the correct index to insert at to keep the neighbor's neighbors sorted
                let old = &self.points[pid];
                let idx = self.zero[pid]
                    .read()
                    .binary_search_by(|third| {
                        // `third` here is one of the neighbors of the new node's neighbor.
                        let third = match third {
                            pid if pid.is_valid() => *pid,
                            // if `third` is `None`, our new `node` is always "closer"
                            _ => return Ordering::Greater,
                        };

                        distance.cmp(&old.distance(&self.points[third]).into())
                    })
                    .unwrap_or_else(|e| e);

                self.zero[pid].write().insert(idx, new);
            }
            node.set(i, pid);
        }

        #[cfg(feature = "indicatif")]
        if let Some(bar) = &self.progress {
            let value = self.done.fetch_add(1, atomic::Ordering::Relaxed);
            if value % 1000 == 0 {
                bar.set_position(value as u64);
            }
        }

        self.pool.push((search, insertion));
    }
}

struct SearchPool {
    pool: Mutex<Vec<(Search, Search)>>,
    len: usize,
}

impl SearchPool {
    fn new(len: usize) -> Self {
        Self {
            pool: Mutex::new(Vec::new()),
            len,
        }
    }

    fn pop(&self) -> (Search, Search) {
        match self.pool.lock().pop() {
            Some(res) => res,
            None => (Search::new(self.len), Search::new(self.len)),
        }
    }

    fn push(&self, item: (Search, Search)) {
        self.pool.lock().push(item);
    }
}

/// Keeps mutable state for searching a point's nearest neighbors
///
/// In particular, this contains most of the state used in algorithm 2. The structure is
/// initialized by using `push()` to add the initial enter points.
pub struct Search {
    /// Nodes visited so far (`v` in the paper)
    visited: Visited,
    /// Candidates for further inspection (`C` in the paper)
    candidates: BinaryHeap<Reverse<Candidate>>,
    /// Nearest neighbors found so far (`W` in the paper)
    ///
    /// This must always be in sorted (nearest first) order.
    nearest: Vec<Candidate>,
    /// Working set for heuristic selection
    working: Vec<Candidate>,
    discarded: Vec<Candidate>,
    /// Maximum number of nearest neighbors to retain (`ef` in the paper)
    ef: usize,
}

impl Search {
    fn new(capacity: usize) -> Self {
        Self {
            visited: Visited::with_capacity(capacity),
            ..Default::default()
        }
    }

    /// Search the given layer for nodes near the given `point`
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
    ///
    /// Invariants: `self.nearest` should be in sorted (nearest first) order, and should be
    /// truncated to `self.ef`.
    fn search<L: Layer, P: Point>(&mut self, point: &P, layer: L, points: &[P], links: usize) {
        while let Some(Reverse(candidate)) = self.candidates.pop() {
            if let Some(furthest) = self.nearest.last() {
                if candidate.distance > furthest.distance {
                    break;
                }
            }

            for pid in layer.nearest_iter(candidate.pid).take(links) {
                self.push(pid, point, points);
            }

            // If we don't truncate here, `furthest` will be further out than necessary, making
            // us continue looping while we could have broken out.
            self.nearest.truncate(self.ef);
        }
    }

    fn add_neighbor_heuristic<L: Layer, P: Point>(
        &mut self,
        new: PointId,
        current: impl Iterator<Item = PointId>,
        layer: L,
        point: &P,
        points: &[P],
        params: Heuristic,
    ) -> &[Candidate] {
        self.reset();
        self.push(new, point, points);
        for pid in current {
            self.push(pid, point, points);
        }
        self.select_heuristic(point, layer, points, params)
    }

    /// Heuristically sort and truncate neighbors in `self.nearest`
    ///
    /// Invariant: `self.nearest` must be in sorted (nearest first) order.
    fn select_heuristic<L: Layer, P: Point>(
        &mut self,
        point: &P,
        layer: L,
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

        if params.extend_candidates {
            self.working.sort_unstable();
        }

        self.nearest.clear();
        self.discarded.clear();
        for candidate in self.working.drain(..) {
            if self.nearest.len() >= M * 2 {
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
                if self.nearest.len() >= M * 2 {
                    break;
                }
                self.nearest.push(candidate);
            }
        }

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
    }

    /// Lower the search to the next lower level
    ///
    /// Re-initialize the `Search`: `nearest`, the output `W` from the last round, now becomes
    /// the set of enter points, which we use to initialize both `candidates` and `visited`.
    ///
    /// Invariant: `nearest` should be sorted and truncated before this is called. This is generally
    /// the case because `Layer::search()` is always called right before calling `cull()`.
    fn cull(&mut self) {
        self.candidates.clear();
        for &candidate in self.nearest.iter() {
            self.candidates.push(Reverse(candidate));
        }

        self.visited.clear();
        self.visited.extend(self.nearest.iter().map(|c| c.pid));
    }

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
        &self.nearest
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = Candidate> + '_ {
        self.nearest.iter().copied()
    }
}

impl Default for Search {
    fn default() -> Self {
        Self {
            visited: Visited::with_capacity(0),
            candidates: BinaryHeap::new(),
            nearest: Vec::new(),
            working: Vec::new(),
            discarded: Vec::new(),
            ef: 1,
        }
    }
}

pub trait Point: Clone + Sync {
    fn distance(&self, other: &Self) -> f32;
}

/// The parameter `M` from the paper
///
/// This should become a generic argument to `Hnsw` when possible.
const M: usize = 32;
