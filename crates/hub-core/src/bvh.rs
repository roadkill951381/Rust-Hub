//! A bounding volume hierarchy over triangles, for ray casts and
//! closest-point queries on meshes with hundreds of thousands of faces.
//!
//! Wall-thickness checks cast one ray per face and hollowing asks for the
//! nearest surface point at every voxel, so both need sub-linear queries.

use glam::Vec3;

use crate::mesh::Mesh;

const LEAF_SIZE: usize = 4;

/// A triangle BVH in whatever space the input mesh is in.
#[derive(Debug, Clone)]
pub struct Bvh {
    nodes: Vec<Node>,
    /// Triangles in leaf order.
    tris: Vec<[Vec3; 3]>,
    /// Original triangle index for each entry of `tris`.
    ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
struct Node {
    min: Vec3,
    max: Vec3,
    /// Leaf: first triangle. Internal: index of the left child (right is `start + 1`).
    start: u32,
    /// Triangles in a leaf, 0 for internal nodes.
    count: u32,
}

/// A ray hit: distance along the (unit) ray and the original triangle index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub t: f32,
    pub triangle: u32,
}

/// The nearest surface point to a query point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nearest {
    pub distance: f32,
    pub point: Vec3,
    pub triangle: u32,
}

impl Bvh {
    /// Builds a BVH over every triangle of `mesh`.
    pub fn new(mesh: &Mesh) -> Self {
        let tris: Vec<[Vec3; 3]> = mesh
            .triangles()
            .map(|t| t.map(|i| mesh.positions[i as usize]))
            .collect();
        Self::from_triangles(tris)
    }

    pub fn from_triangles(tris: Vec<[Vec3; 3]>) -> Self {
        let n = tris.len();
        let centroids: Vec<Vec3> = tris.iter().map(|t| (t[0] + t[1] + t[2]) / 3.0).collect();
        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut nodes = Vec::with_capacity(2 * n / LEAF_SIZE + 1);
        if n > 0 {
            nodes.push(Node {
                min: Vec3::ZERO,
                max: Vec3::ZERO,
                start: 0,
                count: 0,
            });
            build(&mut nodes, 0, &tris, &centroids, &mut order, 0);
        }
        let sorted = order.iter().map(|&i| tris[i as usize]).collect();
        Self {
            nodes,
            tris: sorted,
            ids: order,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tris.is_empty()
    }

    /// The first triangle hit by the ray `origin + t·dir` with `t_min < t < t_max`.
    /// Both sides of a triangle count. `dir` need not be normalized; `t` is in its units.
    pub fn ray_first(&self, origin: Vec3, dir: Vec3, t_min: f32, t_max: f32) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        let mut limit = t_max;
        self.walk_ray(origin, dir, t_min, |t, triangle| {
            if t < limit {
                limit = t;
                best = Some(Hit { t, triangle });
            }
            limit
        });
        best
    }

    /// Every hit along the ray with `t > t_min`, sorted by distance.
    pub fn ray_all(&self, origin: Vec3, dir: Vec3, t_min: f32) -> Vec<Hit> {
        let mut hits = Vec::new();
        self.walk_ray(origin, dir, t_min, |t, triangle| {
            hits.push(Hit { t, triangle });
            f32::INFINITY
        });
        hits.sort_by(|a, b| a.t.total_cmp(&b.t));
        hits
    }

    /// Visits triangles the ray hits; `visit` returns the current far limit.
    fn walk_ray(&self, origin: Vec3, dir: Vec3, t_min: f32, mut visit: impl FnMut(f32, u32) -> f32) {
        if self.nodes.is_empty() {
            return;
        }
        let inv = dir.recip();
        let mut limit = f32::INFINITY;
        let mut stack = vec![0_u32];
        while let Some(i) = stack.pop() {
            let node = self.nodes[i as usize];
            if !slab(origin, inv, node.min, node.max, t_min, limit) {
                continue;
            }
            if node.count > 0 {
                let range = node.start as usize..(node.start + node.count) as usize;
                for k in range {
                    if let Some(t) = intersect(origin, dir, &self.tris[k])
                        && t > t_min
                        && t < limit
                    {
                        limit = visit(t, self.ids[k]);
                    }
                }
            } else {
                stack.push(node.start);
                stack.push(node.start + 1);
            }
        }
    }

    /// The closest surface point within `max_distance` of `p`.
    pub fn nearest(&self, p: Vec3, max_distance: f32) -> Option<Nearest> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best_d2 = max_distance * max_distance;
        let mut best: Option<(Vec3, u32)> = None;
        let mut stack = vec![0_u32];
        while let Some(i) = stack.pop() {
            let node = self.nodes[i as usize];
            if box_distance2(p, node.min, node.max) >= best_d2 {
                continue;
            }
            if node.count > 0 {
                let range = node.start as usize..(node.start + node.count) as usize;
                for k in range {
                    let q = closest_on_triangle(p, &self.tris[k]);
                    let d2 = q.distance_squared(p);
                    if d2 < best_d2 {
                        best_d2 = d2;
                        best = Some((q, self.ids[k]));
                    }
                }
            } else {
                // Visit the nearer child first so the bound tightens sooner.
                let (l, r) = (node.start, node.start + 1);
                let dl = box_distance2(p, self.nodes[l as usize].min, self.nodes[l as usize].max);
                let dr = box_distance2(p, self.nodes[r as usize].min, self.nodes[r as usize].max);
                if dl < dr {
                    stack.push(r);
                    stack.push(l);
                } else {
                    stack.push(l);
                    stack.push(r);
                }
            }
        }
        best.map(|(point, triangle)| Nearest {
            distance: best_d2.sqrt(),
            point,
            triangle,
        })
    }
}

fn build(
    nodes: &mut Vec<Node>,
    index: usize,
    tris: &[[Vec3; 3]],
    centroids: &[Vec3],
    order: &mut [u32],
    offset: usize,
) {
    let (mut min, mut max) = (Vec3::INFINITY, Vec3::NEG_INFINITY);
    let (mut cmin, mut cmax) = (Vec3::INFINITY, Vec3::NEG_INFINITY);
    for &i in order.iter() {
        for v in tris[i as usize] {
            min = min.min(v);
            max = max.max(v);
        }
        cmin = cmin.min(centroids[i as usize]);
        cmax = cmax.max(centroids[i as usize]);
    }
    let extent = cmax - cmin;
    if order.len() <= LEAF_SIZE || extent.max_element() <= 0.0 {
        nodes[index] = Node {
            min,
            max,
            start: offset as u32,
            count: order.len() as u32,
        };
        return;
    }
    let axis = extent.max_position();
    let mid = order.len() / 2;
    order.select_nth_unstable_by(mid, |&a, &b| {
        centroids[a as usize][axis].total_cmp(&centroids[b as usize][axis])
    });
    let left = nodes.len();
    let empty = Node {
        min: Vec3::ZERO,
        max: Vec3::ZERO,
        start: 0,
        count: 0,
    };
    nodes.push(empty);
    nodes.push(empty);
    nodes[index] = Node {
        min,
        max,
        start: left as u32,
        count: 0,
    };
    let (lo, hi) = order.split_at_mut(mid);
    build(nodes, left, tris, centroids, lo, offset);
    build(nodes, left + 1, tris, centroids, hi, offset + mid);
}

fn slab(origin: Vec3, inv: Vec3, min: Vec3, max: Vec3, t_min: f32, t_max: f32) -> bool {
    let t0 = (min - origin) * inv;
    let t1 = (max - origin) * inv;
    // NaN (0 · ∞ on an axis-parallel ray through a slab face) must not reject the box.
    let lo = t0.min(t1);
    let hi = t0.max(t1);
    let near = lo.max_element().max(t_min);
    let far = hi.min_element().min(t_max);
    near <= far || near.is_nan() || far.is_nan()
}

/// Möller–Trumbore, both faces.
fn intersect(origin: Vec3, dir: Vec3, [a, b, c]: &[Vec3; 3]) -> Option<f32> {
    let e1 = *b - *a;
    let e2 = *c - *a;
    let p = dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < f32::MIN_POSITIVE {
        return None;
    }
    let inv = 1.0 / det;
    let s = origin - *a;
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = dir.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    Some(e2.dot(q) * inv)
}

fn box_distance2(p: Vec3, min: Vec3, max: Vec3) -> f32 {
    let d = (min - p).max(p - max).max(Vec3::ZERO);
    d.length_squared()
}

/// Closest point on a triangle (Ericson, Real-Time Collision Detection, 5.1.5).
pub fn closest_on_triangle(p: Vec3, [a, b, c]: &[Vec3; 3]) -> Vec3 {
    let (a, b, c) = (*a, *b, *c);
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn ray_through_a_cube_hits_both_sides() {
        let bvh = Bvh::new(&primitives::cube(2.0));
        let hits = bvh.ray_all(Vec3::new(-5.0, 0.1, 0.2), Vec3::X, 0.0);
        assert_eq!(hits.len(), 2, "{hits:?}");
        assert!((hits[0].t - 4.0).abs() < 1e-5);
        assert!((hits[1].t - 6.0).abs() < 1e-5);
        let first = bvh
            .ray_first(Vec3::new(-5.0, 0.1, 0.2), Vec3::X, 0.0, f32::INFINITY)
            .unwrap();
        assert!((first.t - 4.0).abs() < 1e-5);
        assert!(
            bvh.ray_first(Vec3::new(-5.0, 3.0, 0.0), Vec3::X, 0.0, f32::INFINITY)
                .is_none()
        );
    }

    #[test]
    fn nearest_matches_brute_force_on_a_sphere() {
        let mesh = primitives::uv_sphere(1.0, 32, 16);
        let bvh = Bvh::new(&mesh);
        let tris: Vec<[Vec3; 3]> = mesh
            .triangles()
            .map(|t| t.map(|i| mesh.positions[i as usize]))
            .collect();
        for p in [
            Vec3::ZERO,
            Vec3::new(0.3, -0.2, 0.1),
            Vec3::new(2.0, 1.0, -0.5),
            Vec3::new(-0.49, 0.0, 0.0),
        ] {
            let brute = tris
                .iter()
                .map(|t| closest_on_triangle(p, t).distance(p))
                .fold(f32::INFINITY, f32::min);
            let fast = bvh.nearest(p, f32::INFINITY).unwrap().distance;
            assert!((brute - fast).abs() < 1e-6, "{p}: {brute} vs {fast}");
        }
        assert!(bvh.nearest(Vec3::ZERO, 0.1).is_none());
    }
}
