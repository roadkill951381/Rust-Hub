//! Printability checks and the matching repairs.
//!
//! All checks weld vertices by position first, because game meshes split
//! vertices along UV and normal seams and would otherwise look full of holes.

use std::collections::HashMap;

use glam::Vec3;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::io::to_print_space;
use crate::mesh::{Mesh, default_weld_tolerance, weld};
use crate::scene::{Object, ObjectId};

/// Edge and orientation statistics for one welded mesh.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Topology {
    /// Vertices after welding by position.
    pub welded_vertices: usize,
    /// Triangles with zero area after welding.
    pub degenerate_triangles: usize,
    /// Edges used by exactly one triangle: the borders of holes.
    pub boundary_edges: usize,
    /// Edges shared by three or more triangles.
    pub non_manifold_edges: usize,
    /// Edges shared by two triangles that wind the same way (a flipped face).
    pub flipped_edges: usize,
    /// Separate connected pieces.
    pub shells: usize,
}

impl Topology {
    pub fn is_watertight(&self) -> bool {
        self.boundary_edges == 0 && self.non_manifold_edges == 0
    }
}

/// The result of a printability check, sizes in millimetres, print axes Z-up.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PrintReport {
    pub object: ObjectId,
    pub name: String,
    pub triangles: usize,
    /// Width (X), depth (Y) and height (Z) on the print bed, in mm.
    pub size_mm: [f32; 3],
    pub volume_mm3: f64,
    pub surface_area_mm2: f64,
    pub topology: Topology,
    pub watertight: bool,
    /// True when every surface faces outward.
    pub outward: bool,
    /// True when nothing below blocks a slicer.
    pub printable: bool,
    /// Plain-language problems, empty when printable.
    pub issues: Vec<String>,
}

/// Welded topology of a mesh with a tolerance scaled to its size.
pub fn topology(mesh: &Mesh) -> Topology {
    Analysis::new(mesh).topology
}

/// Checks one object in world space.
pub fn check_printable(object: &Object) -> PrintReport {
    let world = object.world_mesh();
    let analysis = Analysis::new(&world);
    let t = &analysis.topology;

    let size_mm = world
        .bounds()
        .map_or(Vec3::ZERO, |b| to_print_space(b.size()).abs() * 1000.0);
    let volume_mm3 = world.signed_volume() * 1e9;
    let watertight = t.is_watertight() && world.triangle_count() > 0;
    let outward = t.flipped_edges == 0 && analysis.shell_volumes.iter().all(|&v| v > 0.0);

    let mut issues = Vec::new();
    if world.triangle_count() == 0 {
        issues.push("the mesh has no triangles".to_owned());
    }
    if t.boundary_edges > 0 {
        issues.push(format!(
            "{} open edges: the surface has holes, so it does not enclose a solid",
            t.boundary_edges
        ));
    }
    if t.non_manifold_edges > 0 {
        issues.push(format!(
            "{} non-manifold edges are shared by three or more faces",
            t.non_manifold_edges
        ));
    }
    if t.flipped_edges > 0 {
        issues.push(format!(
            "{} edges join faces with opposite winding; run repair to unify them",
            t.flipped_edges
        ));
    } else if watertight && analysis.shell_volumes.iter().any(|&v| v <= 0.0) {
        issues.push("some shells face inward; run repair to flip them".to_owned());
    }
    if t.degenerate_triangles > 0 {
        issues.push(format!(
            "{} zero-area triangles; run repair to remove them",
            t.degenerate_triangles
        ));
    }
    if size_mm.min_element() < 0.4 && world.triangle_count() > 0 {
        issues.push(format!(
            "the model is only {:.2} mm thick on its thinnest axis",
            size_mm.min_element()
        ));
    }

    PrintReport {
        object: object.id,
        name: object.name.clone(),
        triangles: world.triangle_count(),
        size_mm: size_mm.to_array(),
        volume_mm3,
        surface_area_mm2: world.surface_area() * 1e6,
        topology: analysis.topology.clone(),
        watertight,
        outward,
        printable: issues.is_empty(),
        issues,
    }
}

/// What [`repair`] changed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RepairSummary {
    pub removed_degenerate: usize,
    pub removed_duplicates: usize,
    pub flipped_triangles: usize,
}

/// Removes zero-area and duplicate triangles, makes winding consistent across
/// each shell, and turns inward-facing shells outward.
///
/// Vertices, UVs and normals are kept (only flipped faces' normals change), so
/// the result is still good for games. Holes are reported, not filled.
pub fn repair(mesh: &Mesh) -> (Mesh, RepairSummary) {
    let tolerance = default_weld_tolerance(mesh.bounds());
    let welded = weld(&mesh.positions, tolerance);
    let key = |i: u32| welded.remap[i as usize];
    let mut summary = RepairSummary::default();

    // 1. Drop degenerate and duplicate triangles.
    let mut seen = std::collections::HashSet::new();
    let mut tris: Vec<[u32; 3]> = Vec::with_capacity(mesh.triangle_count());
    for t in mesh.triangles() {
        let w = t.map(key);
        let area2 = (welded.positions[w[1] as usize] - welded.positions[w[0] as usize])
            .cross(welded.positions[w[2] as usize] - welded.positions[w[0] as usize])
            .length_squared();
        if w[0] == w[1] || w[1] == w[2] || w[0] == w[2] || area2 <= f32::EPSILON * tolerance.powi(4)
        {
            summary.removed_degenerate += 1;
            continue;
        }
        let mut sorted = w;
        sorted.sort_unstable();
        if !seen.insert(sorted) {
            summary.removed_duplicates += 1;
            continue;
        }
        tris.push(t);
    }

    // 2. Unify winding across manifold edges, shell by shell (breadth-first).
    let welded_tris: Vec<[u32; 3]> = tris.iter().map(|t| t.map(key)).collect();
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (f, t) in welded_tris.iter().enumerate() {
        for (a, b) in edges(*t) {
            edge_faces.entry(undirected(a, b)).or_default().push(f);
        }
    }
    let mut flip = vec![false; tris.len()];
    let mut shell_of = vec![usize::MAX; tris.len()];
    let mut shells = 0;
    for start in 0..tris.len() {
        if shell_of[start] != usize::MAX {
            continue;
        }
        shell_of[start] = shells;
        let mut queue = std::collections::VecDeque::from([start]);
        while let Some(f) = queue.pop_front() {
            let t = oriented(welded_tris[f], flip[f]);
            for (a, b) in edges(t) {
                let neighbours = &edge_faces[&undirected(a, b)];
                if neighbours.len() != 2 {
                    continue;
                }
                for &g in neighbours {
                    if g == f || shell_of[g] != usize::MAX {
                        continue;
                    }
                    // A consistent neighbour traverses the shared edge as b → a.
                    let same_direction = edges(welded_tris[g]).any(|e| e == (a, b));
                    flip[g] = same_direction;
                    shell_of[g] = shells;
                    queue.push_back(g);
                }
            }
        }
        shells += 1;
    }

    // 3. Flip any shell whose enclosed volume is negative.
    let mut shell_volume = vec![0.0_f64; shells];
    for (f, t) in welded_tris.iter().enumerate() {
        let [a, b, c] = oriented(*t, flip[f]).map(|i| welded.positions[i as usize].as_dvec3());
        shell_volume[shell_of[f]] += a.dot(b.cross(c)) / 6.0;
    }
    for f in 0..tris.len() {
        if shell_volume[shell_of[f]] < 0.0 {
            flip[f] = !flip[f];
        }
    }

    let mut out = mesh.clone();
    out.indices.clear();
    let mut flipped_vertices = vec![false; mesh.positions.len()];
    for (f, t) in tris.iter().enumerate() {
        let t = oriented(*t, flip[f]);
        if flip[f] {
            summary.flipped_triangles += 1;
            for i in t {
                flipped_vertices[i as usize] = true;
            }
        }
        out.indices.extend_from_slice(&t);
    }
    if summary.flipped_triangles > 0 && !out.normals.is_empty() {
        // Flipped faces had inward normals; rebuild normals for the vertices they touch.
        let mut recomputed = out.clone();
        recomputed.compute_normals();
        for (i, flipped) in flipped_vertices.into_iter().enumerate() {
            if flipped {
                out.normals[i] = recomputed.normals[i];
            }
        }
    }
    (out, summary)
}

struct Analysis {
    topology: Topology,
    shell_volumes: Vec<f64>,
}

impl Analysis {
    fn new(mesh: &Mesh) -> Self {
        let tolerance = default_weld_tolerance(mesh.bounds());
        let welded = weld(&mesh.positions, tolerance);
        let mut topology = Topology {
            welded_vertices: welded.positions.len(),
            ..Topology::default()
        };

        // Undirected edge → (use count, net direction).
        let mut edge_use: HashMap<(u32, u32), (u32, i32)> = HashMap::new();
        let mut parent: Vec<u32> = (0..welded.positions.len() as u32).collect();
        let mut tris = Vec::with_capacity(mesh.triangle_count());
        for t in mesh.triangles() {
            let w = t.map(|i| welded.remap[i as usize]);
            if w[0] == w[1] || w[1] == w[2] || w[0] == w[2] {
                topology.degenerate_triangles += 1;
                continue;
            }
            tris.push(w);
            for (a, b) in edges(w) {
                let e = edge_use.entry(undirected(a, b)).or_insert((0, 0));
                e.0 += 1;
                e.1 += if a < b { 1 } else { -1 };
                union(&mut parent, a, b);
            }
        }
        for &(count, net) in edge_use.values() {
            match count {
                1 => topology.boundary_edges += 1,
                2 if net != 0 => topology.flipped_edges += 1,
                2 => {}
                _ => topology.non_manifold_edges += 1,
            }
        }

        let mut shell_index = HashMap::new();
        let mut shell_volumes = Vec::new();
        for w in &tris {
            let root = find(&mut parent, w[0]);
            let next = shell_index.len();
            let s = *shell_index.entry(root).or_insert(next);
            if s == shell_volumes.len() {
                shell_volumes.push(0.0);
            }
            let [a, b, c] = w.map(|i| welded.positions[i as usize].as_dvec3());
            shell_volumes[s] += a.dot(b.cross(c)) / 6.0;
        }
        topology.shells = shell_volumes.len();
        Self {
            topology,
            shell_volumes,
        }
    }
}

fn edges([a, b, c]: [u32; 3]) -> impl Iterator<Item = (u32, u32)> {
    [(a, b), (b, c), (c, a)].into_iter()
}

fn undirected(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

fn oriented([a, b, c]: [u32; 3], flip: bool) -> [u32; 3] {
    if flip { [a, c, b] } else { [a, b, c] }
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

fn union(parent: &mut [u32], a: u32, b: u32) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        parent[ra as usize] = rb;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::primitives::{self, Primitive};
    use crate::scene::{Material, Transform};

    fn object(mesh: Mesh) -> Object {
        Object {
            id: ObjectId(0),
            name: "test".into(),
            mesh: Arc::new(mesh),
            transform: Transform::IDENTITY,
            material: Material::default(),
        }
    }

    #[test]
    fn primitives_are_printable() {
        for p in [Primitive::Cube, Primitive::Sphere, Primitive::Cylinder] {
            let report = check_printable(&object(p.build(0.02)));
            assert!(report.printable, "{p:?}: {:?}", report.issues);
            assert_eq!(report.topology.shells, 1);
        }
    }

    #[test]
    fn cube_report_is_in_millimetres() {
        let report = check_printable(&object(primitives::cube(0.02)));
        for s in report.size_mm {
            assert!((s - 20.0).abs() < 1e-3);
        }
        assert!((report.volume_mm3 - 8000.0).abs() < 1e-2);
    }

    #[test]
    fn open_mesh_is_not_watertight() {
        let mut cube = primitives::cube(1.0);
        cube.indices.truncate(cube.indices.len() - 6); // remove one face
        let report = check_printable(&object(cube));
        assert!(!report.watertight);
        assert_eq!(report.topology.boundary_edges, 4);
        assert!(!report.printable);
    }

    #[test]
    fn repair_fixes_flipped_faces_and_inside_out_shells() {
        let mut cube = primitives::cube(1.0);
        cube.indices[..3].swap(1, 2); // flip one triangle
        assert!(topology(&cube).flipped_edges > 0);
        let (fixed, summary) = repair(&cube);
        assert_eq!(summary.flipped_triangles, 1);
        assert!(check_printable(&object(fixed)).printable);

        let mut inside_out = primitives::uv_sphere(1.0, 16, 8);
        inside_out.flip_all();
        assert!(!check_printable(&object(inside_out.clone())).outward);
        let (fixed, _) = repair(&inside_out);
        let report = check_printable(&object(fixed));
        assert!(report.outward && report.printable, "{:?}", report.issues);
    }

    #[test]
    fn repair_removes_degenerate_and_duplicate_triangles() {
        let mut cube = primitives::cube(1.0);
        let first: Vec<u32> = cube.indices[..3].to_vec();
        cube.indices.extend_from_slice(&first); // duplicate
        cube.indices.extend_from_slice(&[0, 0, 1]); // degenerate
        let (fixed, summary) = repair(&cube);
        assert_eq!(summary.removed_duplicates, 1);
        assert_eq!(summary.removed_degenerate, 1);
        assert_eq!(fixed.triangle_count(), 12);
    }

    #[test]
    fn two_cubes_are_two_shells() {
        let mut m = primitives::cube(1.0);
        let other = primitives::cube(1.0).transformed(glam::Mat4::from_translation(Vec3::X * 3.0));
        m.append(&other);
        let report = check_printable(&object(m));
        assert_eq!(report.topology.shells, 2);
        assert!(report.printable);
    }
}
