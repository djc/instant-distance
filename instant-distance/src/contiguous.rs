use rand::{rngs::SmallRng, Rng, SeedableRng};

use crate::types::{LayerId, Meta, INVALID};
use crate::{Builder, Point, PointId};

/// Give all points a random layer and sort the list of nodes by descending order for
/// construction. This allows us to copy higher layers to lower layers as construction
/// progresses, while preserving randomness in each point's layer and insertion order.
pub(crate) fn shuffle_points_for_layer_assignment<P: Point>(
    points: Vec<P>,
    meta: &Meta,
    builder: Builder,
) -> (Vec<P>, Vec<PointId>, Vec<(LayerId, PointId)>) {
    let mut rng = SmallRng::seed_from_u64(builder.seed);
    assert!(points.len() < u32::MAX as usize);
    let mut shuffled = (0..points.len())
        .map(|i| (PointId(rng.gen_range(0..points.len() as u32)), i))
        .collect::<Vec<_>>();
    shuffled.sort_unstable();

    let mut out = vec![INVALID; points.len()];
    let mut new_points = Vec::with_capacity(points.len());
    let mut new_nodes = Vec::with_capacity(points.len());
    let mut at_layer = meta.next_lower(None).unwrap();
    for (i, (_, idx)) in shuffled.into_iter().enumerate() {
        let pid = PointId(new_nodes.len() as u32);
        if i == at_layer.1 {
            at_layer = meta.next_lower(Some(at_layer.0)).unwrap();
        }
        new_points.push(points[idx].clone());
        new_nodes.push((at_layer.0, pid));
        out[idx] = pid;
    }
    debug_assert_eq!(new_nodes.first().unwrap().0, LayerId(meta.len() - 1));
    debug_assert_eq!(new_nodes.last().unwrap().0, LayerId(0));
    (new_points, out, new_nodes)
}
