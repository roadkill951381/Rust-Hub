//! Hollowing: give a solid model an inner cavity so it prints with walls of a
//! set thickness, which saves resin and weight.
//!
//! The cavity is the level set "distance to the surface = wall thickness"
//! inside the model. Distances come from a BVH on a voxel grid, inside/outside
//! from one ray per grid row, and the surface from marching tetrahedra, which
//! (unlike classic marching cubes) never leaves cracks or non-manifold edges.

use std::collections::HashMap;

use glam::Vec3;

use crate::bvh::Bvh;
use crate::mesh::Mesh;

/// Most grid nodes a hollow may use (about 160³), to bound time and memory.
const MAX_NODES: usize = 4_000_000;

/// Measurements of a computed cavity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cavity {
    /// Grid spacing used, in the mesh's units.
    pub voxel: f32,
    /// Volume removed, positive, in cubic mesh units.
    pub volume: f64,
}

/// Computes the inner surface of a hollowed `mesh` (closed, outward-facing):
/// a closed shell `wall` inside the outer surface, wound to face into the void.
/// Append it to the mesh to hollow it.
///
/// `voxel` is the grid spacing; `None` picks `wall / 2`, coarsened as needed to
/// stay within a fixed node budget. Fails when the model is too thin to hold a
/// cavity with that wall thickness.
pub fn cavity(mesh: &Mesh, wall: f32, voxel: Option<f32>) -> Result<(Mesh, Cavity), String> {
    if !(wall.is_finite() && wall > 0.0) {
        return Err("wall thickness must be positive".into());
    }
    let bounds = mesh.bounds().ok_or("the mesh is empty")?;
    let size = bounds.size();
    let mut h = voxel.unwrap_or(wall * 0.5);
    if !(h.is_finite() && h > 0.0) {
        return Err("voxel size must be positive".into());
    }
    let nodes_for = |h: f32| {
        let n = (size / h).ceil() + 4.0;
        (n.x * n.y * n.z) as usize
    };
    while nodes_for(h) > MAX_NODES {
        h *= 1.1;
    }
    let dims = ((size / h).ceil() + 4.0).as_uvec3();
    let (nx, ny, nz) = (dims.x as usize, dims.y as usize, dims.z as usize);
    // Offset the grid by an odd fraction of a voxel so rows rarely run exactly
    // along a face or through a vertex.
    let origin = bounds.min - Vec3::splat(h * 1.5) + Vec3::new(0.137, 0.281, 0.352) * h;
    let at = |i: usize, j: usize, k: usize| origin + Vec3::new(i as f32, j as f32, k as f32) * h;

    let bvh = Bvh::new(mesh);
    let band = wall + 2.0 * h;
    let mut field = vec![0.0_f32; nx * ny * nz];

    // Rows are independent, so split the z layers across threads.
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let layer = nx * ny;
    let per_thread = nz.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in field.chunks_mut(layer * per_thread).enumerate() {
            let bvh = &bvh;
            scope.spawn(move || {
                let k0 = chunk_index * per_thread;
                for (dk, slab) in chunk.chunks_mut(layer).enumerate() {
                    let k = k0 + dk;
                    for j in 0..ny {
                        let row = &mut slab[j * nx..(j + 1) * nx];
                        fill_row(bvh, row, at(0, j, k), h, wall, band);
                    }
                }
            });
        }
    });

    let mesh = march(&field, [nx, ny, nz], &at);
    if mesh.triangle_count() == 0 {
        return Err(format!(
            "the model is too thin to hollow with {:.2} mm walls",
            wall * 1000.0
        ));
    }
    let volume = -mesh.signed_volume();
    Ok((mesh, Cavity { voxel: h, volume }))
}

/// Fills one grid row along +X: inside nodes get `distance - wall` (clamped
/// deep inside), outside nodes `-wall`.
fn fill_row(bvh: &Bvh, row: &mut [f32], start: Vec3, h: f32, wall: f32, band: f32) {
    let hits = bvh.ray_all(start - Vec3::X * h, Vec3::X, 0.0);
    let mut crossings: Vec<f32> = Vec::with_capacity(hits.len());
    for hit in hits {
        // A ray through a shared edge hits both faces at the same distance.
        if crossings
            .last()
            .is_none_or(|&t| hit.t - t > 1e-6 * h.max(hit.t))
        {
            crossings.push(hit.t);
        }
    }
    let mut next = 0;
    for (i, value) in row.iter_mut().enumerate() {
        let t = (i as f32 + 1.0) * h;
        while next < crossings.len() && crossings[next] < t {
            next += 1;
        }
        let inside = next % 2 == 1;
        *value = if inside {
            let p = start + Vec3::X * (i as f32 * h);
            match bvh.nearest(p, band) {
                Some(n) => n.distance - wall,
                None => band - wall,
            }
        } else {
            -wall
        };
        if *value == 0.0 {
            *value = -f32::EPSILON * h; // keep the level set off grid nodes
        }
    }
}

/// The six tetrahedra of a cube around its 0–7 diagonal (Kuhn triangulation),
/// corners numbered x + 2y + 4z. Every cube splits its faces the same way, so
/// neighbouring cubes share edges and the surface has no cracks.
const TETS: [[usize; 4]; 6] = [
    [0, 1, 3, 7],
    [0, 3, 2, 7],
    [0, 2, 6, 7],
    [0, 6, 4, 7],
    [0, 4, 5, 7],
    [0, 5, 1, 7],
];

/// Marching tetrahedra over `field`, emitting the zero level set wound to face
/// toward positive values.
fn march(field: &[f32], [nx, ny, nz]: [usize; 3], at: &dyn Fn(usize, usize, usize) -> Vec3) -> Mesh {
    let id = |i: usize, j: usize, k: usize| (i + nx * (j + ny * k)) as u32;
    let mut mesh = Mesh::default();
    let mut edge_vertex: HashMap<(u32, u32), u32> = HashMap::new();
    let mut vertex = |mesh: &mut Mesh, a: (u32, Vec3, f32), b: (u32, Vec3, f32)| -> u32 {
        let key = if a.0 < b.0 { (a.0, b.0) } else { (b.0, a.0) };
        *edge_vertex.entry(key).or_insert_with(|| {
            let t = a.2 / (a.2 - b.2);
            mesh.positions.push(a.1 + (b.1 - a.1) * t);
            (mesh.positions.len() - 1) as u32
        })
    };

    for k in 0..nz - 1 {
        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                let corner = |c: usize| {
                    let (di, dj, dk) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
                    let (ci, cj, ck) = (i + di, j + dj, k + dk);
                    let g = id(ci, cj, ck);
                    (g, at(ci, cj, ck), field[g as usize])
                };
                let corners: [(u32, Vec3, f32); 8] = std::array::from_fn(corner);
                if corners.iter().all(|c| c.2 > 0.0) || corners.iter().all(|c| c.2 <= 0.0) {
                    continue;
                }
                for tet in TETS {
                    let v = tet.map(|c| corners[c]);
                    let (pos, neg): (Vec<&(u32, Vec3, f32)>, Vec<_>) =
                        v.iter().partition(|c| c.2 > 0.0);
                    let tris: Vec<[u32; 3]> = match (pos.len(), neg.len()) {
                        (1, 3) => {
                            let p = *pos[0];
                            vec![[0, 1, 2].map(|n| vertex(&mut mesh, p, *neg[n]))]
                        }
                        (3, 1) => {
                            let n = *neg[0];
                            vec![[0, 1, 2].map(|p| vertex(&mut mesh, *pos[p], n))]
                        }
                        (2, 2) => {
                            // The four crossing edges form a quad in this cyclic order.
                            let q = [
                                vertex(&mut mesh, *pos[0], *neg[0]),
                                vertex(&mut mesh, *pos[0], *neg[1]),
                                vertex(&mut mesh, *pos[1], *neg[1]),
                                vertex(&mut mesh, *pos[1], *neg[0]),
                            ];
                            vec![[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
                        }
                        _ => continue,
                    };
                    let toward_positive = pos.iter().map(|c| c.1).sum::<Vec3>() / pos.len() as f32
                        - neg.iter().map(|c| c.1).sum::<Vec3>() / neg.len() as f32;
                    for mut t in tris {
                        let [a, b, c] = t.map(|x| mesh.positions[x as usize]);
                        if (b - a).cross(c - a).dot(toward_positive) < 0.0 {
                            t.swap(1, 2);
                        }
                        if t[0] != t[1] && t[1] != t[2] && t[0] != t[2] {
                            mesh.indices.extend_from_slice(&t);
                        }
                    }
                }
            }
        }
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::topology;
    use crate::primitives;

    #[test]
    fn hollow_cube_has_the_requested_walls() {
        let cube = primitives::cube(0.02); // 20 mm
        let (inner, info) = cavity(&cube, 0.002, Some(0.0005)).unwrap();
        let t = topology(&inner);
        assert!(t.is_watertight(), "{t:?}");
        assert_eq!(t.flipped_edges, 0);
        assert_eq!(t.shells, 1);
        // The cavity faces inward and is a 16 mm cube with slightly rounded edges.
        assert!(inner.signed_volume() < 0.0);
        let b = inner.bounds().unwrap();
        assert!((b.size().x - 0.016).abs() < 0.0005, "{:?}", b.size());
        assert!((info.volume - 0.016_f64.powi(3)).abs() < 0.1 * 0.016_f64.powi(3));
    }

    #[test]
    fn thin_models_cannot_be_hollowed() {
        let plate = primitives::cube(1.0).transformed(glam::Mat4::from_scale(Vec3::new(
            0.02, 0.002, 0.02,
        )));
        assert!(cavity(&plate, 0.002, None).is_err());
    }
}
