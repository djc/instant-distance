use std::marker::PhantomData;

use rand::{rngs::SmallRng, Rng, SeedableRng};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::types::{LayerId, Meta, INVALID};
use crate::{Builder, Element, Point, PointId};

pub struct PointIter<'a, E, P: Point<Element = E>> {
    values: &'a [P::Element],
    index: usize,
}

impl<'a, E, P: Point<Element = E>> Iterator for PointIter<'a, E, P> {
    type Item = PointRef<'a, E, P>;

    fn next(&mut self) -> Option<Self::Item> {
        let start = self.index * P::STRIDE;
        let end = start + P::STRIDE;
        self.index += 1;
        self.values.get(start..end).map(PointRef)
    }
}

pub trait Storage<E, P: Point<Element = E>> {
    fn iter(&self) -> PointIter<'_, E, P>;
    fn get(&self, index: usize) -> Option<PointRef<'_, E, P>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}

pub struct PointRef<'a, E, P: Point<Element = E>>(pub &'a [P::Element]);

impl<'a, E, P: Point<Element = E>> PointRef<'a, E, P> {
    pub fn from_data(values: &'a [P::Element]) -> Self {
        Self(values)
    }
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Default)]
pub struct ContiguousStorage<E: Element, P: Point<Element = E>> {
    pub values: Vec<E>,
    _phantom: PhantomData<P>,
}

impl<E: Element, P: Point<Element = E>> ContiguousStorage<E, P> {
    pub(crate) fn empty() -> Self {
        Self {
            values: Vec::new(),
            _phantom: PhantomData,
        }
    }
    pub(crate) fn new(
        points: Vec<P>,
        meta: &Meta,
        builder: Builder,
    ) -> (Self, Vec<PointId>, Vec<(LayerId, PointId)>) {
        let mut rng = SmallRng::seed_from_u64(builder.seed);
        let points_len = points.len();
        assert!(points.len() < u32::MAX as usize);
        let mut shuffled = (0..points_len)
            .map(|i| (PointId(rng.gen_range(0..points_len as u32)), i))
            .zip(points)
            .collect::<Vec<_>>();
        shuffled.sort_unstable_by_key(|(pid, _)| *pid);

        let mut new_points = Vec::with_capacity(points_len);
        let mut layer_assignments = Vec::with_capacity(points_len);
        let mut out = vec![INVALID; points_len];
        let mut at_layer = meta.next_lower(None).unwrap();
        for (i, ((_, idx), point)) in shuffled.into_iter().enumerate() {
            let pid = PointId(layer_assignments.len() as u32);
            if i == at_layer.1 {
                at_layer = meta.next_lower(Some(at_layer.0)).unwrap();
            }

            new_points.push(point);
            layer_assignments.push((at_layer.0, pid));
            out[idx] = pid;
        }

        debug_assert_eq!(
            layer_assignments.first().unwrap().0,
            LayerId(meta.len() - 1)
        );
        debug_assert_eq!(layer_assignments.last().unwrap().0, LayerId(0));

        (
            Self {
                values: new_points
                    .iter()
                    .flat_map(Point::as_slice)
                    .copied()
                    .collect::<Vec<_>>(),
                _phantom: PhantomData,
            },
            out,
            layer_assignments,
        )
    }
}

impl<'a, E: Element + 'a, P: Point<Element = E>> Storage<E, P> for ContiguousStorage<E, P> {
    fn iter(&self) -> PointIter<'_, E, P> {
        PointIter {
            values: &self.values,
            index: 0,
        }
    }

    fn get(&self, index: usize) -> Option<PointRef<'_, E, P>> {
        self.values
            .get(index * P::STRIDE..(index + 1) * P::STRIDE)
            .map(PointRef)
    }

    fn len(&self) -> usize {
        self.values.len().saturating_div(P::STRIDE)
    }

    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// blanket impl
impl<E: Element, P: Point<Element = E>> Point for PointRef<'_, E, P> {
    const STRIDE: usize = P::STRIDE;
    type Element = E;

    fn as_slice(&self) -> &[Self::Element] {
        self.0
    }

    fn distance(&self, other: &Self) -> f32 {
        Element::distance(self.0, other.0)
    }
}
