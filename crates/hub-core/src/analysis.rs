//! Printability checks and the matching repairs.
//!
//! All checks weld vertices by position first, because game meshes split
//! vertices along UV and normal seams and would otherwise look full of holes.

use std::collections::{HashMap, HashSet, VecDeque};

use glam::Vec3;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::bvh::Bvh;
use crate::io::to_print_space;
use crate::mesh::{Mesh, Welded, default_weld_tolerance, weld};
use crate::scene::{Object, ObjectId};

/// Thinnest wall a typical FDM printer with a 0.4 mm nozzle prints reliably
/// (two perimeters).
pub const DEFAULT_MIN_WALL_MM: f32 = 0.8;

/// Holes with more boundary vertices than this are filled with a fan around a
/// new centre vertex instead of the minimum-area triangulation, which is cubic.
const MAX_DP_HOLE: usize = 128;

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
    /// Separate connected pieces, including inner cavity surfaces.
    pub shells: usize,
    /// Inward-facing shells enclosed by another shell: the voids of a hollowed model.
    pub cavities: usize,
}

impl Topology {
    pub fn is_watertight(&self) -> bool {
        self.boundary_edges == 0 && self.non_manifold_edges == 0
    }
}

/// Settings for [`check_printable_with`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CheckOptions {
    /// Walls thinner than this many millimetres are reported as too thin.
    pub min_wall_mm: f32,
}

impl Default for CheckOptions {
    fn default() -> Self {
        Self {
            min_wall_mm: DEFAULT_MIN_WALL_MM,
        }
    }
}

/// Wall thickness measured by casting a ray inward from every face.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct WallReport {
    /// The threshold the check used.
    pub min_wall_mm: f32,
    /// Thinnest wall found, in mm, leaving out the thinnest
    /// [`IGNORED_THIN_FRACTION`] of the surface.
    pub thinnest_mm: f32,
    /// Surface area whose wall is thinner than `min_wall_mm`.
    pub thin_area_mm2: f64,
    /// That area as a fraction of the whole surface, 0 to 1.
    pub thin_fraction: f64,
}

/// The result of a printability check, sizes in millimetres, print axes Z-up.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PrintReport {
    pub object: ObjectId,
    pub name: String,
    pub triangles: usize,
    /// Width (X), depth (Y) and height (Z) on the print bed, in mm.
    pub size_mm: [f32; 3],
    /// Printed material volume (cavities subtracted).
    pub volume_mm3: f64,
    pub surface_area_mm2: f64,
    pub topology: Topology,
    pub watertight: bool,
    /// True when every surface faces away from the material (outward, or into a cavity).
    pub outward: bool,
    /// Wall thickness, measured only on watertight, outward meshes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall: Option<WallReport>,
    /// True when nothing in `issues` blocks a slicer.
    pub printable: bool,
    /// Plain-language problems that block printing, empty when printable.
    pub issues: Vec<String>,
    /// Problems that print but may fail or look wrong, such as thin walls.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// Welded topology of a mesh with a tolerance scaled to its size.
pub fn topology(mesh: &Mesh) -> Topology {
    Analysis::new(mesh).topology
}

/// Checks one object in world space with the default options.
pub fn check_printable(object: &Object) -> PrintReport {
    check_printable_with(object, &CheckOptions::default())
}

/// Checks one object in world space.
pub fn check_printable_with(object: &Object, options: &CheckOptions) -> PrintReport {
    let world = object.world_mesh();
    let analysis = Analysis::new(&world);
    let t = &analysis.topology;

    let size_mm = world
        .bounds()
        .map_or(Vec3::ZERO, |b| to_print_space(b.size()).abs() * 1000.0);
    let volume_mm3 = world.signed_volume() * 1e9;
    let watertight = t.is_watertight() && world.triangle_count() > 0;
    let outward = t.flipped_edges == 0 && analysis.inward_shells == 0;

    let mut issues = Vec::new();
    let mut warnings = Vec::new();
    if world.triangle_count() == 0 {
        issues.push("the mesh has no triangles".to_owned());
    }
    if t.boundary_edges > 0 {
        issues.push(format!(
            "{} open edges: the surface has holes, so it does not enclose a solid; \
             run repair to fill them",
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
    } else if watertight && analysis.inward_shells > 0 {
        issues.push(format!(
            "{} shell(s) face inward; run repair to flip them",
            analysis.inward_shells
        ));
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

    let wall = (watertight && outward)
        .then(|| wall_thickness(&world, options.min_wall_mm * 0.001))
        .flatten();
    if let Some(w) = &wall
        && w.thin_fraction > IGNORED_THIN_FRACTION
    {
        warnings.push(format!(
            "{:.1}% of the surface ({:.0} mm²) has walls thinner than {} mm, the thinnest {:.2} mm; \
             thicken or scale up those parts",
            w.thin_fraction * 100.0,
            w.thin_area_mm2,
            options.min_wall_mm,
            w.thinnest_mm
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
        wall,
        printable: issues.is_empty(),
        issues,
        warnings,
    }
}

/// Share of the surface, thinnest first, left out of
/// [`WallReport::thinnest_mm`], and the share of thin surface below which
/// thin walls are not worth a warning.
pub const IGNORED_THIN_FRACTION: f64 = 0.001;

/// Measures wall thickness (in the mesh's units) by casting a ray from each
/// face centre straight into the material and timing its exit.
///
/// The mesh must be closed and outward-facing; faces whose ray never exits are
/// skipped. Returns `None` for an empty mesh.
///
/// The reported thinnest wall ignores the thinnest [`IGNORED_THIN_FRACTION`]
/// of the surface: sharp creases and remeshing noise give a few tiny faces
/// absurdly short rays that say nothing about how the part prints.
pub fn wall_thickness(mesh: &Mesh, min_wall: f32) -> Option<WallReport> {
    let bvh = Bvh::new(mesh);
    let scale = mesh.bounds()?.size().max_element();
    let eps = scale * 1e-5;
    let mut samples: Vec<(f32, f64)> = Vec::with_capacity(mesh.triangle_count());
    let (mut total, mut thin) = (0.0_f64, 0.0_f64);
    for tri in mesh.triangles() {
        let [a, b, c] = tri.map(|i| mesh.positions[i as usize]);
        let cross = (b - a).cross(c - a);
        let area = f64::from(cross.length()) * 0.5;
        let Some(n) = cross.try_normalize() else {
            continue;
        };
        total += area;
        let start = (a + b + c) / 3.0;
        let Some(hit) = bvh.ray_first(start, -n, eps, f32::INFINITY) else {
            continue;
        };
        samples.push((hit.t, area));
        if hit.t < min_wall {
            thin += area;
        }
    }
    samples.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let ignore = total * IGNORED_THIN_FRACTION;
    let mut skipped = 0.0;
    let thinnest = samples
        .iter()
        .find(|&&(_, area)| {
            skipped += area;
            skipped > ignore
        })
        .or(samples.last())?
        .0;
    Some(WallReport {
        min_wall_mm: min_wall * 1000.0,
        thinnest_mm: thinnest * 1000.0,
        thin_area_mm2: thin * 1e6,
        thin_fraction: if total > 0.0 { thin / total } else { 0.0 },
    })
}

/// Settings for [`repair_with`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RepairOptions {
    /// Close holes with new triangles. On by default.
    #[serde(default = "yes")]
    pub fill_holes: bool,
    /// Delete separate pieces smaller than this fraction of the largest piece's
    /// volume, such as the floating specks AI generators leave behind.
    /// Off by default; 0.01 is a good value for generated models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remove_small_shells: Option<f32>,
}

fn yes() -> bool {
    true
}

impl Default for RepairOptions {
    fn default() -> Self {
        Self {
            fill_holes: true,
            remove_small_shells: None,
        }
    }
}

/// What [`repair`] changed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RepairSummary {
    pub removed_degenerate: usize,
    pub removed_duplicates: usize,
    pub flipped_triangles: usize,
    #[serde(default)]
    pub holes_filled: usize,
    #[serde(default)]
    pub fill_triangles: usize,
    #[serde(default)]
    pub removed_shells: usize,
}

impl RepairSummary {
    /// Number of individual fixes, for summaries.
    pub fn fixes(&self) -> usize {
        self.removed_degenerate
            + self.removed_duplicates
            + self.flipped_triangles
            + self.holes_filled
            + self.removed_shells
    }
}

/// [`repair_with`] using the default options.
pub fn repair(mesh: &Mesh) -> (Mesh, RepairSummary) {
    repair_with(mesh, &RepairOptions::default())
}

/// Removes zero-area and duplicate triangles, makes winding consistent across
/// each shell, fills holes, and turns inside-out shells outward (leaving the
/// inner surfaces of cavities facing into the cavity).
///
/// Vertices, UVs and normals are kept (only flipped faces' normals change), so
/// the result is still good for games.
pub fn repair_with(mesh: &Mesh, options: &RepairOptions) -> (Mesh, RepairSummary) {
    let mut summary = RepairSummary::default();
    let welded = weld(&mesh.positions, default_weld_tolerance(mesh.bounds()));

    // 1. Drop degenerate and duplicate triangles, then unify winding.
    let tris = drop_bad_triangles(mesh, &welded, &mut summary);
    let flips = unify_winding(&tris, &welded);
    let mut out = mesh.clone();
    out.indices.clear();
    for (t, &flip) in tris.iter().zip(&flips) {
        out.indices.extend_from_slice(&oriented(*t, flip));
    }
    // Whether each output triangle is reversed relative to the input.
    let mut net_flip = flips;

    // 2. Close the holes, now that neighbouring faces agree on a winding.
    if options.fill_holes {
        let (filled, holes, added) = fill_holes(&out);
        out = filled;
        summary.holes_filled = holes;
        summary.fill_triangles = added;
        net_flip.resize(out.triangle_count(), false);
    }

    // 3. Turn inside-out shells outward, keeping cavities pointing into the void.
    let analysis = Analysis::new(&out);
    let mut shell_flip = vec![false; analysis.shell_kinds.len()];
    for (s, kind) in analysis.shell_kinds.iter().enumerate() {
        shell_flip[s] = *kind == ShellKind::Inverted;
    }
    let mut remove = vec![false; analysis.shell_kinds.len()];
    if let Some(fraction) = options.remove_small_shells {
        let largest = analysis
            .shell_volumes
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        for (s, v) in analysis.shell_volumes.iter().enumerate() {
            if v.abs() < largest * f64::from(fraction) {
                remove[s] = true;
                summary.removed_shells += 1;
            }
        }
    }
    if shell_flip.iter().any(|&f| f) || summary.removed_shells > 0 {
        let source: Vec<[u32; 3]> = out.triangles().collect();
        let flips = std::mem::take(&mut net_flip);
        out.indices.clear();
        for (f, t) in source.into_iter().enumerate() {
            let (flip, keep) = match analysis.shell_of_triangle[f] {
                Some(s) => (shell_flip[s], !remove[s]),
                // Degenerate after welding; step 1 already dropped those.
                None => (false, true),
            };
            if keep {
                out.indices.extend_from_slice(&oriented(t, flip));
                net_flip.push(flips[f] != flip);
            }
        }
    }

    let mut flipped_vertices = vec![false; out.positions.len()];
    for (t, &flip) in out.triangles().zip(&net_flip) {
        if flip {
            summary.flipped_triangles += 1;
            for i in t {
                flipped_vertices[i as usize] = true;
            }
        }
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

fn drop_bad_triangles(mesh: &Mesh, welded: &Welded, summary: &mut RepairSummary) -> Vec<[u32; 3]> {
    let tolerance = default_weld_tolerance(mesh.bounds());
    let key = |i: u32| welded.remap[i as usize];
    let mut seen = HashSet::new();
    let mut tris = Vec::with_capacity(mesh.triangle_count());
    for t in mesh.triangles() {
        let w = t.map(key);
        let p = |i: usize| welded.positions[w[i] as usize];
        let area2 = (p(1) - p(0)).cross(p(2) - p(0)).length_squared();
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
    tris
}

/// Breadth-first over manifold edges: which triangles to flip so every shell
/// winds one way.
fn unify_winding(tris: &[[u32; 3]], welded: &Welded) -> Vec<bool> {
    let welded_tris: Vec<[u32; 3]> = tris
        .iter()
        .map(|t| t.map(|i| welded.remap[i as usize]))
        .collect();
    let mut edge_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (f, t) in welded_tris.iter().enumerate() {
        for (a, b) in edges(*t) {
            edge_faces.entry(undirected(a, b)).or_default().push(f);
        }
    }
    let mut flip = vec![false; tris.len()];
    let mut visited = vec![false; tris.len()];
    for start in 0..tris.len() {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        let mut queue = VecDeque::from([start]);
        while let Some(f) = queue.pop_front() {
            let t = oriented(welded_tris[f], flip[f]);
            for (a, b) in edges(t) {
                let neighbours = &edge_faces[&undirected(a, b)];
                if neighbours.len() != 2 {
                    continue;
                }
                for &g in neighbours {
                    if g == f || visited[g] {
                        continue;
                    }
                    // A consistent neighbour traverses the shared edge as b → a.
                    flip[g] = edges(welded_tris[g]).any(|e| e == (a, b));
                    visited[g] = true;
                    queue.push_back(g);
                }
            }
        }
    }
    flip
}

/// Closes every hole (loop of open edges) with triangles wound to match their
/// neighbours. Returns the new mesh, the number of holes and of triangles added.
///
/// Holes of up to 128 edges get a minimum-area triangulation; larger ones a fan
/// around a new centre vertex.
pub fn fill_holes(mesh: &Mesh) -> (Mesh, usize, usize) {
    let welded = weld(&mesh.positions, default_weld_tolerance(mesh.bounds()));
    let mut representative = vec![u32::MAX; welded.positions.len()];
    for (i, &w) in welded.remap.iter().enumerate() {
        if representative[w as usize] == u32::MAX {
            representative[w as usize] = i as u32;
        }
    }
    let tris: Vec<[u32; 3]> = mesh
        .triangles()
        .map(|t| t.map(|i| welded.remap[i as usize]))
        .filter(|w| w[0] != w[1] && w[1] != w[2] && w[0] != w[2])
        .collect();
    let mut count: HashMap<(u32, u32), u32> = HashMap::new();
    for t in &tris {
        for (a, b) in edges(*t) {
            *count.entry(undirected(a, b)).or_default() += 1;
        }
    }
    // Each open half-edge a → b becomes the hole edge b → a, so holes wind
    // the opposite way to the faces around them, as the filling must.
    let mut next: HashMap<u32, Vec<u32>> = HashMap::new();
    for t in &tris {
        for (a, b) in edges(*t) {
            if count[&undirected(a, b)] == 1 {
                next.entry(b).or_default().push(a);
            }
        }
    }

    let mut out = mesh.clone();
    let (mut holes, mut added) = (0, 0);
    let mut starts: Vec<u32> = next.keys().copied().collect();
    starts.sort_unstable();
    for start in starts {
        while let Some(first) = next.get_mut(&start).and_then(Vec::pop) {
            let mut ring = vec![start];
            let mut current = first;
            let mut closed = false;
            while ring.len() <= tris.len() {
                if current == start {
                    closed = true;
                    break;
                }
                ring.push(current);
                match next.get_mut(&current).and_then(Vec::pop) {
                    Some(n) => current = n,
                    None => break,
                }
            }
            if !closed || ring.len() < 3 {
                continue;
            }
            let points: Vec<Vec3> = ring.iter().map(|&w| welded.positions[w as usize]).collect();
            let local = triangulate_hole(&points);
            let index = |k: usize, out: &mut Mesh| -> u32 {
                if k < ring.len() {
                    representative[ring[k] as usize]
                } else {
                    push_vertex(
                        out,
                        points.iter().copied().sum::<Vec3>() / points.len() as f32,
                    )
                }
            };
            let mut centre = None;
            for [a, b, c] in local {
                let tri = [a, b, c].map(|k| {
                    if k < ring.len() {
                        index(k, &mut out)
                    } else {
                        *centre.get_or_insert_with(|| index(k, &mut out))
                    }
                });
                out.indices.extend_from_slice(&tri);
                added += 1;
            }
            holes += 1;
        }
    }
    (out, holes, added)
}

fn push_vertex(mesh: &mut Mesh, p: Vec3) -> u32 {
    let i = mesh.positions.len() as u32;
    mesh.positions.push(p);
    if !mesh.normals.is_empty() {
        mesh.normals.push(Vec3::ZERO);
    }
    if !mesh.uvs.is_empty() {
        mesh.uvs.push(glam::Vec2::ZERO);
    }
    i
}

/// Triangles over ring indices (index `points.len()` means a new centre vertex),
/// each wound in ring order.
fn triangulate_hole(points: &[Vec3]) -> Vec<[usize; 3]> {
    let n = points.len();
    if n == 3 {
        return vec![[0, 1, 2]];
    }
    if n > MAX_DP_HOLE {
        return (0..n).map(|i| [i, (i + 1) % n, n]).collect();
    }
    // Minimum total area over all triangulations of the ring (O(n³) dynamic programming).
    // Slivers are penalized so collinear runs of boundary vertices do not
    // become zero-area triangles.
    let scale = points
        .iter()
        .fold(0.0_f32, |m, p| m.max(p.distance(points[0])));
    let sliver = f64::from(scale * scale) * 1e-10;
    let penalty = f64::from(scale * scale) * 10.0;
    let area = |i: usize, k: usize, j: usize| -> f64 {
        let a = f64::from(
            (points[k] - points[i])
                .cross(points[j] - points[i])
                .length()
                * 0.5,
        );
        if a < sliver { a + penalty } else { a }
    };
    let mut cost = vec![0.0_f64; n * n];
    let mut split = vec![0_usize; n * n];
    for gap in 2..n {
        for i in 0..n - gap {
            let j = i + gap;
            let mut best = f64::INFINITY;
            for k in i + 1..j {
                let c = cost[i * n + k] + cost[k * n + j] + area(i, k, j);
                if c < best {
                    best = c;
                    split[i * n + j] = k;
                }
            }
            cost[i * n + j] = best;
        }
    }
    let mut out = Vec::with_capacity(n - 2);
    let mut stack = vec![(0, n - 1)];
    while let Some((i, j)) = stack.pop() {
        if j < i + 2 {
            continue;
        }
        let k = split[i * n + j];
        out.push([i, k, j]);
        stack.push((i, k));
        stack.push((k, j));
    }
    out
}

/// How a closed shell sits relative to the material.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShellKind {
    /// Encloses positive volume: a solid part.
    Solid,
    /// Faces inward but sits inside another shell: the wall of a cavity.
    Cavity,
    /// Faces inward with nothing around it: an inside-out part.
    Inverted,
}

struct Analysis {
    topology: Topology,
    shell_volumes: Vec<f64>,
    shell_kinds: Vec<ShellKind>,
    /// Shell index per triangle, `None` for degenerate triangles.
    shell_of_triangle: Vec<Option<usize>>,
    inward_shells: usize,
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
                tris.push(None);
                continue;
            }
            tris.push(Some(w));
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
        let mut shell_of_triangle = Vec::with_capacity(tris.len());
        for w in &tris {
            let Some(w) = w else {
                shell_of_triangle.push(None);
                continue;
            };
            let root = find(&mut parent, w[0]);
            let next = shell_index.len();
            let s = *shell_index.entry(root).or_insert(next);
            if s == shell_volumes.len() {
                shell_volumes.push(0.0);
            }
            shell_of_triangle.push(Some(s));
            let [a, b, c] = w.map(|i| welded.positions[i as usize].as_dvec3());
            shell_volumes[s] += a.dot(b.cross(c)) / 6.0;
        }
        topology.shells = shell_volumes.len();

        let shell_kinds = classify_shells(mesh, &shell_of_triangle, &shell_volumes);
        topology.cavities = shell_kinds
            .iter()
            .filter(|&&k| k == ShellKind::Cavity)
            .count();
        let inward_shells = shell_kinds
            .iter()
            .filter(|&&k| k == ShellKind::Inverted)
            .count();
        Self {
            topology,
            shell_volumes,
            shell_kinds,
            shell_of_triangle,
            inward_shells,
        }
    }
}

/// Decides which inward-facing shells are cavities: those with a point inside
/// some other shell, by counting crossings of a ray against the other shells.
fn classify_shells(mesh: &Mesh, shell_of: &[Option<usize>], volumes: &[f64]) -> Vec<ShellKind> {
    let mut kinds: Vec<ShellKind> = volumes
        .iter()
        .map(|&v| {
            if v > 0.0 {
                ShellKind::Solid
            } else {
                ShellKind::Inverted
            }
        })
        .collect();
    if volumes.len() < 2 || kinds.iter().all(|&k| k == ShellKind::Solid) {
        return kinds;
    }
    let bvh = Bvh::new(mesh);
    let tris: Vec<[u32; 3]> = mesh.triangles().collect();
    // An irrational-looking direction, so rays rarely graze edges exactly.
    let dir = Vec3::new(0.577_215_7, 0.618_034, 0.531_329_3).normalize();
    let candidates: Vec<usize> = (0..kinds.len())
        .filter(|&s| kinds[s] != ShellKind::Solid)
        .collect();
    for s in candidates {
        let Some(f) = shell_of.iter().position(|&x| x == Some(s)) else {
            continue;
        };
        let [a, b, c] = tris[f].map(|i| mesh.positions[i as usize]);
        let origin = (a + b + c) / 3.0;
        let mut last = f32::NEG_INFINITY;
        let mut crossings = 0;
        for hit in bvh.ray_all(origin, dir, 0.0) {
            if shell_of[hit.triangle as usize].is_none_or(|x| x == s) {
                continue;
            }
            // A ray through a shared edge hits both faces at the same distance.
            if hit.t - last > 1e-7 * (1.0 + hit.t.abs()) {
                crossings += 1;
            }
            last = hit.t;
        }
        if crossings % 2 == 1 {
            kinds[s] = ShellKind::Cavity;
        }
    }
    kinds
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
            assert!(report.warnings.is_empty(), "{p:?}: {:?}", report.warnings);
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
        let wall = report.wall.unwrap();
        assert!((wall.thinnest_mm - 20.0).abs() < 1e-3, "{wall:?}");
        assert_eq!(wall.thin_area_mm2, 0.0);
    }

    #[test]
    fn thin_plates_get_a_wall_warning() {
        // 40 × 40 × 0.5 mm: printable, but below the 0.8 mm default.
        let plate = primitives::cube(1.0)
            .transformed(glam::Mat4::from_scale(Vec3::new(0.04, 0.0005, 0.04)));
        let report = check_printable(&object(plate));
        assert!(report.printable, "{:?}", report.issues);
        let wall = report.wall.as_ref().unwrap();
        assert!((wall.thinnest_mm - 0.5).abs() < 1e-3, "{wall:?}");
        assert!(wall.thin_fraction > 0.9, "{wall:?}");
        assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
    }

    #[test]
    fn open_mesh_is_not_watertight() {
        let mut cube = primitives::cube(1.0);
        cube.indices.truncate(cube.indices.len() - 6); // remove one face
        let report = check_printable(&object(cube));
        assert!(!report.watertight);
        assert_eq!(report.topology.boundary_edges, 4);
        assert!(!report.printable);
        assert!(report.wall.is_none());
    }

    #[test]
    fn repair_fills_a_missing_face() {
        let mut cube = primitives::cube(1.0);
        cube.indices.truncate(cube.indices.len() - 6);
        let (fixed, summary) = repair(&cube);
        assert_eq!(summary.holes_filled, 1);
        assert_eq!(summary.fill_triangles, 2);
        let report = check_printable(&object(fixed.clone()));
        assert!(report.printable, "{:?}", report.issues);
        assert!((fixed.signed_volume() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn repair_fills_a_large_ragged_hole() {
        // Cut the top cap off a fine sphere: a 200-edge hole takes the fan path,
        // a 64-edge one the minimum-area path.
        for segments in [64, 200] {
            let mut sphere = primitives::uv_sphere(1.0, segments, 32);
            let top: HashSet<u32> = sphere
                .positions
                .iter()
                .enumerate()
                .filter(|(_, p)| p.y > 0.9)
                .map(|(i, _)| i as u32)
                .collect();
            let kept: Vec<u32> = sphere
                .triangles()
                .filter(|t| !t.iter().all(|i| top.contains(i)))
                .filter(|t| !t.iter().any(|i| top.contains(i)))
                .flatten()
                .collect();
            sphere.indices = kept;
            assert!(!check_printable(&object(sphere.clone())).watertight);
            let (fixed, summary) = repair(&sphere);
            assert_eq!(summary.holes_filled, 1, "{segments}");
            let report = check_printable(&object(fixed));
            assert!(report.printable, "{segments}: {:?}", report.issues);
        }
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
    fn cavities_are_outward_and_survive_repair() {
        // A 20 mm cube with a 10 mm inward-facing cube inside it: a hollow part.
        let mut part = primitives::cube(0.02);
        let mut void = primitives::cube(0.01);
        void.flip_all();
        part.append(&void);
        let report = check_printable(&object(part.clone()));
        assert!(report.printable && report.outward, "{:?}", report.issues);
        assert_eq!(report.topology.cavities, 1);
        assert!((report.volume_mm3 - 7000.0).abs() < 1e-2);
        let wall = report.wall.unwrap();
        assert!((wall.thinnest_mm - 5.0).abs() < 1e-3, "{wall:?}");

        let (fixed, summary) = repair(&part);
        assert_eq!(summary.flipped_triangles, 0);
        assert_eq!(fixed.indices, part.indices);

        // The same inverted cube on its own is inside out, not a cavity.
        let report = check_printable(&object(void));
        assert!(!report.outward && !report.printable);
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
    fn repair_can_drop_floating_specks() {
        let mut m = primitives::cube(1.0);
        m.append(&primitives::cube(0.05).transformed(glam::Mat4::from_translation(Vec3::X * 3.0)));
        let (kept, summary) = repair(&m);
        assert_eq!(summary.removed_shells, 0);
        assert_eq!(topology(&kept).shells, 2);
        let options = RepairOptions {
            remove_small_shells: Some(0.01),
            ..RepairOptions::default()
        };
        let (cleaned, summary) = repair_with(&m, &options);
        assert_eq!(summary.removed_shells, 1);
        assert_eq!(topology(&cleaned).shells, 1);
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
