// borrow_deref_ref doesn't get macro detection right, allow for now
#![allow(clippy::from_iter_instead_of_collect, clippy::borrow_deref_ref)]

use std::convert::TryFrom;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::iter::FromIterator;
use std::marker::PhantomData;

use distance_metrics::Metric;
use distance_metrics::{CosineMetric, EuclidMetric};
use instant_distance::Point;
use pyo3::conversion::IntoPy;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyList, PyModule, PyString};
use pyo3::{pyclass, pymethods, pymodule};
use pyo3::{Py, PyAny, PyErr, PyObject, PyRef, PyRefMut, PyResult, Python};
use serde::{Deserialize, Serialize};

#[pymodule]
#[pyo3(name = "instant_distance")]
fn instant_distance_py(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Neighbor>()?;
    m.add_class::<Heuristic>()?;
    m.add_class::<Config>()?;
    m.add_class::<Search>()?;
    m.add_class::<Hnsw>()?;
    m.add_class::<HnswMap>()?;
    m.add_class::<DistanceMetric>()?;
    Ok(())
}

#[pyclass]
#[derive(Copy, Clone)]
enum DistanceMetric {
    Euclid,
    Cosine,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        Self::Euclid
    }
}

// Helper macro for dispatching to inner implementation
macro_rules! impl_for_each_hnsw_with_metric {
    ($type:ident, $instance:expr, $inner:ident, $($tokens:tt)+) => {
        match $instance {
            $type::Euclid($inner) => {
                $($tokens)+
            }
            $type::Cosine($inner) => {
                $($tokens)+
            }
        }
    };
}

#[pyclass]
struct HnswMap {
    inner: HnswMapWithMetric,
}

#[derive(Deserialize, Serialize)]
enum HnswMapWithMetric {
    Euclid(instant_distance::HnswMap<FloatArray<EuclidMetric>, MapValue>),
    Cosine(instant_distance::HnswMap<FloatArray<CosineMetric>, MapValue>),
}

#[pymethods]
impl HnswMap {
    /// Build the index
    #[staticmethod]
    fn build(points: &PyList, values: &PyList, config: &Config) -> PyResult<Self> {
        let values = values
            .into_iter()
            .map(MapValue::try_from)
            .collect::<Result<Vec<_>, PyErr>>()?;
        let builder = instant_distance::Builder::from(config);
        let inner = match config.distance_metric {
            DistanceMetric::Euclid => {
                let points = FloatArray::try_from_pylist(points)?;
                HnswMapWithMetric::Euclid(builder.build(points, values))
            }
            DistanceMetric::Cosine => {
                let points = FloatArray::try_from_pylist(points)?;
                HnswMapWithMetric::Cosine(builder.build(points, values))
            }
        };
        Ok(Self { inner })
    }

    /// Load an index from the given file name
    #[staticmethod]
    fn load(fname: &str) -> PyResult<Self> {
        let hnsw_map = bincode::deserialize_from::<_, HnswMapWithMetric>(BufReader::with_capacity(
            32 * 1024 * 1024,
            File::open(fname)?,
        ))
        .map_err(|e| PyValueError::new_err(format!("deserialization error: {e:?}")))?;
        Ok(Self { inner: hnsw_map })
    }

    /// Dump the index to the given file name
    fn dump(&self, fname: &str) -> PyResult<()> {
        let f = BufWriter::with_capacity(32 * 1024 * 1024, File::create(fname)?);
        bincode::serialize_into(f, &self.inner)
            .map_err(|e| PyValueError::new_err(format!("serialization error: {e:?}")))?;
        Ok(())
    }

    /// Search the index for points neighboring the given point
    ///
    /// The `search` object contains buffers used for searching. When the search completes,
    /// iterate over the `Search` to get the results. The number of results should be equal
    /// to the `ef_search` parameter set in the index's `config`.
    ///
    /// For best performance, reusing `Search` objects is recommended.
    fn search(slf: Py<Self>, point: &PyAny, search: &mut Search, py: Python<'_>) -> PyResult<()> {
        impl_for_each_hnsw_with_metric!(HnswMapWithMetric, &slf.try_borrow(py)?.inner, hnsw, {
            let point = FloatArray::try_from(point)?;
            let _ = hnsw.search(&point, &mut search.inner);
        });
        search.cur = Some((HnswType::Map(slf.clone_ref(py)), 0));
        Ok(())
    }
}

/// An instance of hierarchical navigable small worlds
#[pyclass]
struct Hnsw {
    inner: HnswWithMetric,
}

#[derive(Deserialize, Serialize)]
enum HnswWithMetric {
    Euclid(instant_distance::Hnsw<FloatArray<EuclidMetric>>),
    Cosine(instant_distance::Hnsw<FloatArray<CosineMetric>>),
}

#[pymethods]
impl Hnsw {
    /// Build the index
    #[staticmethod]
    fn build(input: &PyList, config: &Config) -> PyResult<(Self, Vec<u32>)> {
        let builder = instant_distance::Builder::from(config);
        let (inner, ids) = match config.distance_metric {
            DistanceMetric::Euclid => {
                let points = FloatArray::try_from_pylist(input)?;
                let (hnsw, ids) = builder.build_hnsw(points);
                (HnswWithMetric::Euclid(hnsw), ids)
            }
            DistanceMetric::Cosine => {
                let points = FloatArray::try_from_pylist(input)?;
                let (hnsw, ids) = builder.build_hnsw(points);
                (HnswWithMetric::Cosine(hnsw), ids)
            }
        };
        let ids = Vec::from_iter(ids.into_iter().map(|pid| pid.into_inner()));
        Ok((Self { inner }, ids))
    }

    /// Load an index from the given file name
    #[staticmethod]
    fn load(fname: &str) -> PyResult<Self> {
        let hnsw = bincode::deserialize_from::<_, HnswWithMetric>(BufReader::with_capacity(
            32 * 1024 * 1024,
            File::open(fname)?,
        ))
        .map_err(|e| PyValueError::new_err(format!("deserialization error: {e:?}")))?;
        Ok(Self { inner: hnsw })
    }

    /// Dump the index to the given file name
    fn dump(&self, fname: &str) -> PyResult<()> {
        let f = BufWriter::with_capacity(32 * 1024 * 1024, File::create(fname)?);
        bincode::serialize_into(f, &self.inner)
            .map_err(|e| PyValueError::new_err(format!("serialization error: {e:?}")))?;
        Ok(())
    }

    /// Search the index for points neighboring the given point
    ///
    /// The `search` object contains buffers used for searching. When the search completes,
    /// iterate over the `Search` to get the results. The number of results should be equal
    /// to the `ef_search` parameter set in the index's `config`.
    ///
    /// For best performance, reusing `Search` objects is recommended.
    fn search(slf: Py<Self>, point: &PyAny, search: &mut Search, py: Python<'_>) -> PyResult<()> {
        impl_for_each_hnsw_with_metric!(HnswWithMetric, &slf.try_borrow(py)?.inner, hnsw, {
            let point = FloatArray::try_from(point)?;
            let _ = hnsw.search(&point, &mut search.inner);
        });
        search.cur = Some((HnswType::Hnsw(slf.clone_ref(py)), 0));
        Ok(())
    }
}

/// Search buffer and result set
#[pyclass]
struct Search {
    inner: instant_distance::Search,
    cur: Option<(HnswType, usize)>,
}

#[pymethods]
impl Search {
    /// Initialize an empty search buffer
    #[new]
    fn new() -> Self {
        Self {
            inner: instant_distance::Search::default(),
            cur: None,
        }
    }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    /// Return the next closest point
    fn __next__(mut slf: PyRefMut<Self>) -> Option<Neighbor> {
        let (index, idx) = match slf.cur.take() {
            Some(x) => x,
            None => return None,
        };

        let py = slf.py();
        let neighbor = match &index {
            HnswType::Hnsw(hnsw) => {
                let hnsw = hnsw.as_ref(py).borrow();
                impl_for_each_hnsw_with_metric!(HnswWithMetric, &hnsw.inner, hnsw, {
                    let item = hnsw.get(idx, &slf.inner);
                    item.map(|item| Neighbor {
                        distance: item.distance,
                        pid: item.pid.into_inner(),
                        value: py.None(),
                    })
                })
            }
            HnswType::Map(map) => {
                let map = map.as_ref(py).borrow();
                impl_for_each_hnsw_with_metric!(HnswMapWithMetric, &map.inner, map, {
                    let item = map.get(idx, &slf.inner);
                    item.map(|item| Neighbor {
                        distance: item.distance,
                        pid: item.pid.into_inner(),
                        value: item.value.into_py(py),
                    })
                })
            }
        };

        slf.cur = neighbor.as_ref().map(|_| (index, idx + 1));
        neighbor
    }
}

enum HnswType {
    Hnsw(Py<Hnsw>),
    Map(Py<HnswMap>),
}

#[pyclass]
#[derive(Copy, Clone, Default)]
struct Config {
    /// Number of nearest neighbors to cache during the search
    #[pyo3(get, set)]
    ef_search: usize,
    /// Number of nearest neighbors to cache during construction
    #[pyo3(get, set)]
    ef_construction: usize,
    /// Parameter to control the number of layers
    #[pyo3(get, set)]
    ml: f32,
    /// Random seed used to randomize the order of points
    ///
    /// This can be useful if you want to have fully deterministic results.
    #[pyo3(get, set)]
    seed: u64,
    /// Whether to use the heuristic search algorithm
    ///
    /// This will prioritize neighbors that are farther away from other, closer neighbors,
    /// in order to get better results on clustered data points.
    #[pyo3(get, set)]
    heuristic: Option<Heuristic>,
    /// Distance metric to use
    ///
    /// Defaults to Euclidean distance
    #[pyo3(get, set)]
    distance_metric: DistanceMetric,
}

#[pymethods]
impl Config {
    #[new]
    fn new() -> Self {
        let builder = instant_distance::Builder::default();
        let (ef_search, ef_construction, ml, seed) = builder.into_parts();
        let heuristic = Some(Heuristic::default());
        let distance_metric = DistanceMetric::default();
        Self {
            ef_search,
            ef_construction,
            ml,
            seed,
            heuristic,
            distance_metric,
        }
    }
}

impl From<&Config> for instant_distance::Builder {
    fn from(py: &Config) -> Self {
        let Config {
            ef_search,
            ef_construction,
            ml,
            seed,
            heuristic,
            distance_metric: _,
        } = *py;
        Self::default()
            .ef_search(ef_search)
            .ef_construction(ef_construction)
            .ml(ml)
            .seed(seed)
            .select_heuristic(heuristic.map(|h| h.into()))
    }
}

#[pyclass]
#[derive(Copy, Clone)]
struct Heuristic {
    /// Whether to extend the candidate set before selecting results
    ///
    /// This is only useful only for extremely clustered data.
    #[pyo3(get, set)]
    extend_candidates: bool,
    /// Whether to keep pruned neighbors to make the neighbor set size constant
    #[pyo3(get, set)]
    keep_pruned: bool,
}

#[pymethods]
impl Heuristic {
    #[new]
    fn new() -> Self {
        let default = instant_distance::Heuristic::default();
        let instant_distance::Heuristic {
            extend_candidates,
            keep_pruned,
        } = default;
        Self {
            extend_candidates,
            keep_pruned,
        }
    }
}

impl Default for Heuristic {
    fn default() -> Self {
        Self {
            extend_candidates: false,
            keep_pruned: true,
        }
    }
}

impl From<Heuristic> for instant_distance::Heuristic {
    fn from(py: Heuristic) -> Self {
        let Heuristic {
            extend_candidates,
            keep_pruned,
        } = py;
        Self {
            extend_candidates,
            keep_pruned,
        }
    }
}

/// Item found by the nearest neighbor search
#[pyclass]
struct Neighbor {
    /// Distance to the neighboring point
    #[pyo3(get)]
    distance: f32,
    /// Identifier for the neighboring point
    #[pyo3(get)]
    pid: u32,
    /// Value for the neighboring point (only set for `HnswMap` results)
    #[pyo3(get)]
    value: PyObject,
}

#[pymethods]
impl Neighbor {
    fn __repr__(&self) -> PyResult<String> {
        match Python::with_gil(|py| self.value.is_none(py)) {
            false => Ok(format!(
                "instant_distance.Neighbor(distance={}, pid={}, value={})",
                self.distance,
                self.pid,
                Python::with_gil(|py| self.value.as_ref(py).repr().map(|s| s.to_string()))?,
            )),
            true => Ok(format!(
                "instant_distance.Item(distance={}, pid={})",
                self.distance, self.pid,
            )),
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
struct FloatArray<M> {
    array: Vec<f32>,
    phantom: PhantomData<M>,
}

impl<M: Metric> FloatArray<M> {
    fn try_from_pylist(list: &PyList) -> Result<Vec<Self>, PyErr> {
        list.into_iter().map(FloatArray::try_from).collect()
    }
}

impl<M: Metric> From<Vec<f32>> for FloatArray<M> {
    fn from(mut array: Vec<f32>) -> Self {
        M::preprocess(&mut array);
        Self {
            array,
            phantom: PhantomData,
        }
    }
}

impl<M: Metric> TryFrom<&PyAny> for FloatArray<M> {
    type Error = PyErr;

    fn try_from(value: &PyAny) -> Result<Self, Self::Error> {
        let array: Vec<f32> = value
            .iter()?
            .map(|val| val.and_then(|v| v.extract::<f32>()))
            .collect::<Result<_, _>>()?;
        Ok(Self::from(array))
    }
}

impl<M: Metric + Clone + Sync> Point for FloatArray<M> {
    fn distance(&self, rhs: &Self) -> f32 {
        M::distance(&self.array, &rhs.array)
    }
}

#[derive(Clone, Deserialize, Serialize)]
enum MapValue {
    String(String),
}

impl TryFrom<&PyAny> for MapValue {
    type Error = PyErr;

    fn try_from(value: &PyAny) -> Result<Self, Self::Error> {
        Ok(MapValue::String(value.extract::<String>()?))
    }
}

impl IntoPy<Py<PyAny>> for &'_ MapValue {
    fn into_py(self, py: Python<'_>) -> Py<PyAny> {
        match self {
            MapValue::String(s) => PyString::new(py, s).into(),
        }
    }
}
