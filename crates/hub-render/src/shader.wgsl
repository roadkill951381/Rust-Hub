// Rust-Hub viewport shader: lit meshes and a ground grid.

struct Camera {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    // xyz: direction toward the key light; w: unused.
    light: vec4<f32>,
};

struct Object {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
    color: vec4<f32>,
    // x: metallic, y: roughness, z: 1 = flat shading, w: 1 = selected.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> object: Object;

struct MeshOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_mesh(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>) -> MeshOut {
    let world = object.model * vec4<f32>(position, 1.0);
    var out: MeshOut;
    out.clip = camera.view_proj * world;
    out.world = world.xyz;
    out.normal = (object.normal_matrix * vec4<f32>(normal, 0.0)).xyz;
    return out;
}

@fragment
fn fs_mesh(in: MeshOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let to_eye = normalize(camera.eye.xyz - in.world);

    // Derivatives must run in uniform control flow, so compute the face normal first.
    var face = normalize(cross(dpdx(in.world), dpdy(in.world)));
    if dot(face, to_eye) < 0.0 {
        face = -face;
    }
    var n = face;
    if object.params.z < 0.5 && dot(in.normal, in.normal) > 1e-8 {
        n = normalize(in.normal);
        if !front {
            n = -n;
        }
    }

    var base = object.color.rgb;
    if !front {
        // The inside of a surface is showing: a hole or a flipped face. Tint it.
        base = vec3<f32>(0.85, 0.22, 0.18);
    }

    let metallic = object.params.x;
    let roughness = clamp(object.params.y, 0.05, 1.0);
    let l = normalize(camera.light.xyz);
    let fill = normalize(vec3<f32>(-l.x, 0.3, -l.z));
    let h = normalize(l + to_eye);

    let sky = mix(vec3<f32>(0.10, 0.09, 0.085), vec3<f32>(0.24, 0.27, 0.32), n.y * 0.5 + 0.5);
    let diffuse = base * (1.0 - metallic) * (max(dot(n, l), 0.0) * 0.75 + max(dot(n, fill), 0.0) * 0.18);
    let shininess = 2.0 / (roughness * roughness * roughness * roughness) - 2.0;
    let f0 = mix(vec3<f32>(0.04), base, metallic);
    let spec = f0 * pow(max(dot(n, h), 0.0), max(shininess, 1.0)) * (shininess + 8.0) / 25.0 * max(dot(n, l), 0.0);
    var color = base * sky + diffuse + spec;

    if object.params.w > 0.5 {
        let rim = pow(1.0 - max(dot(n, to_eye), 0.0), 2.5);
        color = color + vec3<f32>(1.0, 0.55, 0.15) * rim * 0.9;
    }
    // Exposure tone map keeps white materials from clipping.
    return vec4<f32>(vec3<f32>(1.0) - exp(-color * 1.6), 1.0);
}

struct LineOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_line(@location(0) position: vec3<f32>, @location(1) color: vec4<f32>) -> LineOut {
    var out: LineOut;
    out.clip = camera.view_proj * vec4<f32>(position, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    return in.color;
}
