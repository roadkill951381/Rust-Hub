use std::collections::HashMap;
use std::sync::Arc;

use glam::{Mat3, Mat4, Vec3};
use hub_core::{Mesh, ObjectId, Scene};
use wgpu::util::DeviceExt;

use crate::camera::Camera;

/// Colour format of the offscreen image the renderer draws into.
pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DISPLAY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SAMPLES: u32 = 4;
const BACKGROUND: wgpu::Color = wgpu::Color {
    r: 0.035,
    g: 0.04,
    b: 0.05,
    a: 1.0,
};

/// Per-frame options.
#[derive(Debug, Clone, Copy, Default)]
pub struct FrameOptions {
    pub selected: Option<ObjectId>,
    pub hide_grid: bool,
}

/// An offscreen colour image plus its multisampled and depth attachments.
#[derive(Debug)]
pub struct RenderTarget {
    size: (u32, u32),
    msaa: wgpu::TextureView,
    depth: wgpu::TextureView,
    resolve: wgpu::Texture,
    resolve_view: wgpu::TextureView,
    display_view: wgpu::TextureView,
}

impl RenderTarget {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let size = (width.max(1), height.max(1));
        let extent = wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        };
        let texture = |label, format, samples, usage, view_formats| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: extent,
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats,
            })
        };
        let msaa = texture(
            "hub msaa",
            COLOR_FORMAT,
            SAMPLES,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            &[],
        );
        let depth = texture(
            "hub depth",
            DEPTH_FORMAT,
            SAMPLES,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
            &[],
        );
        let resolve = texture(
            "hub color",
            COLOR_FORMAT,
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            &[DISPLAY_FORMAT],
        );
        Self {
            size,
            msaa: msaa.create_view(&wgpu::TextureViewDescriptor::default()),
            depth: depth.create_view(&wgpu::TextureViewDescriptor::default()),
            resolve_view: resolve.create_view(&wgpu::TextureViewDescriptor::default()),
            display_view: resolve.create_view(&wgpu::TextureViewDescriptor {
                format: Some(DISPLAY_FORMAT),
                ..Default::default()
            }),
            resolve,
        }
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// The finished, single-sampled image.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.resolve
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.resolve_view
    }

    /// The same image viewed as plain `Rgba8Unorm`, for UI toolkits such as
    /// egui that composite in gamma space and must not decode sRGB again.
    pub fn display_view(&self) -> &wgpu::TextureView {
        &self.display_view
    }
}

struct GpuObject {
    mesh: Arc<Mesh>,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

struct Grid {
    spacing: f32,
    buffer: wgpu::Buffer,
    count: u32,
}

/// Draws a [`Scene`] into a [`RenderTarget`].
///
/// GPU buffers are cached per object and rebuilt only when the object's mesh
/// `Arc` changes, so transforms and selection update without re-uploading.
pub struct Renderer {
    mesh_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    object_layout: wgpu::BindGroupLayout,
    objects: HashMap<ObjectId, GpuObject>,
    grid: Option<Grid>,
}

impl std::fmt::Debug for Renderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Renderer")
            .field("cached_objects", &self.objects.len())
            .finish_non_exhaustive()
    }
}

const CAMERA_SIZE: u64 = 96;
const OBJECT_SIZE: u64 = 160;

impl Renderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let uniform_layout = |label| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            })
        };
        let camera_layout = uniform_layout("hub camera layout");
        let object_layout = uniform_layout("hub object layout");

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hub camera"),
            size: CAMERA_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hub camera"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let depth = |write| wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(write),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };
        let multisample = wgpu::MultisampleState {
            count: SAMPLES,
            ..Default::default()
        };

        let mesh_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hub mesh"),
            bind_group_layouts: &[Some(&camera_layout), Some(&object_layout)],
            immediate_size: 0,
        });
        let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hub mesh"),
            layout: Some(&mesh_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_mesh"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 24,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_mesh"),
                targets: &[Some(COLOR_FORMAT.into())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None, // back faces are drawn, tinted, to reveal holes
                ..Default::default()
            },
            depth_stencil: Some(depth(true)),
            multisample,
            multiview_mask: None,
            cache: None,
        });

        let line_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hub lines"),
            bind_group_layouts: &[Some(&camera_layout)],
            immediate_size: 0,
        });
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hub lines"),
            layout: Some(&line_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_line"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 28,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_line"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: COLOR_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(depth(false)),
            multisample,
            multiview_mask: None,
            cache: None,
        });

        Self {
            mesh_pipeline,
            line_pipeline,
            camera_buffer,
            camera_bind_group,
            object_layout,
            objects: HashMap::new(),
            grid: None,
        }
    }

    /// Records and submits one frame.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        camera: &Camera,
        target: &RenderTarget,
        options: FrameOptions,
    ) {
        let (w, h) = target.size;
        let aspect = w as f32 / h as f32;
        let eye = camera.eye();
        // Key light from over the viewer's left shoulder.
        let view = camera.view();
        let light = (eye - camera.target).normalize_or_zero() + view.row(1).truncate() * 0.8
            - view.row(0).truncate() * 0.6;
        let mut camera_data = [0f32; 24];
        camera_data[..16].copy_from_slice(&camera.view_projection(aspect).to_cols_array());
        camera_data[16..19].copy_from_slice(&eye.to_array());
        camera_data[20..23].copy_from_slice(&light.normalize_or_zero().to_array());
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&camera_data));

        self.sync_objects(device, queue, scene, options.selected);
        if !options.hide_grid {
            self.sync_grid(device, scene);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hub frame"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hub frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.msaa,
                    depth_slice: None,
                    resolve_target: Some(&target.resolve_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(BACKGROUND),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&self.mesh_pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            for object in scene.objects() {
                let Some(gpu) = self.objects.get(&object.id) else {
                    continue;
                };
                pass.set_bind_group(1, &gpu.bind_group, &[]);
                pass.set_vertex_buffer(0, gpu.vertices.slice(..));
                pass.set_index_buffer(gpu.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..gpu.index_count, 0, 0..1);
            }

            if let (false, Some(grid)) = (options.hide_grid, &self.grid) {
                pass.set_pipeline(&self.line_pipeline);
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                pass.set_vertex_buffer(0, grid.buffer.slice(..));
                pass.draw(0..grid.count, 0..1);
            }
        }
        queue.submit([encoder.finish()]);
    }

    fn sync_objects(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        selected: Option<ObjectId>,
    ) {
        self.objects
            .retain(|id, _| scene.objects().iter().any(|o| o.id == *id));
        for object in scene.objects() {
            let stale = self
                .objects
                .get(&object.id)
                .is_none_or(|g| !Arc::ptr_eq(&g.mesh, &object.mesh));
            if stale && object.mesh.triangle_count() > 0 {
                let gpu = self.upload(device, &object.mesh);
                self.objects.insert(object.id, gpu);
            }
            let Some(gpu) = self.objects.get(&object.id) else {
                continue;
            };
            let model = object.transform.matrix();
            let normal = Mat4::from_mat3(Mat3::from_mat4(model).inverse().transpose());
            let m = object.material;
            let flat = object.mesh.normals.is_empty();
            let mut data = [0f32; 40];
            data[..16].copy_from_slice(&model.to_cols_array());
            data[16..32].copy_from_slice(&normal.to_cols_array());
            data[32..36].copy_from_slice(&m.base_color);
            data[36] = m.metallic;
            data[37] = m.roughness;
            data[38] = if flat { 1.0 } else { 0.0 };
            data[39] = if selected == Some(object.id) {
                1.0
            } else {
                0.0
            };
            queue.write_buffer(&gpu.uniforms, 0, bytemuck::cast_slice(&data));
        }
    }

    fn upload(&self, device: &wgpu::Device, mesh: &Arc<Mesh>) -> GpuObject {
        let has_normals = mesh.normals.len() == mesh.positions.len();
        let vertices: Vec<f32> = mesh
            .positions
            .iter()
            .enumerate()
            .flat_map(|(i, p)| {
                let n = if has_normals {
                    mesh.normals[i]
                } else {
                    Vec3::ZERO
                };
                [p.x, p.y, p.z, n.x, n.y, n.z]
            })
            .collect();
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hub vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hub indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hub object"),
            size: OBJECT_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hub object"),
            layout: &self.object_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });
        GpuObject {
            mesh: Arc::clone(mesh),
            vertices,
            indices,
            index_count: mesh.indices.len() as u32,
            uniforms,
            bind_group,
        }
    }

    /// Rebuilds the floor grid when the scene's scale changes by a power of ten.
    fn sync_grid(&mut self, device: &wgpu::Device, scene: &Scene) {
        let extent = scene
            .bounds()
            .map_or(0.1, |b| b.min.abs().max(b.max.abs()).max_element() * 2.0)
            .max(0.01);
        let spacing = 10f32.powf((extent / 10.0).log10().floor()).max(0.001);
        if self
            .grid
            .as_ref()
            .is_some_and(|g| (g.spacing - spacing).abs() < spacing * 1e-3)
        {
            return;
        }
        let lines = 20i32;
        let half = spacing * lines as f32;
        let mut data: Vec<f32> = Vec::new();
        let mut line = |a: Vec3, b: Vec3, color: [f32; 4]| {
            for p in [a, b] {
                data.extend_from_slice(&p.to_array());
                data.extend_from_slice(&color);
            }
        };
        for i in -lines..=lines {
            let t = i as f32 * spacing;
            let color = if i == 0 {
                None
            } else if i % 10 == 0 {
                Some([0.5, 0.52, 0.56, 0.55])
            } else {
                Some([0.4, 0.42, 0.46, 0.22])
            };
            if let Some(c) = color {
                line(Vec3::new(t, 0.0, -half), Vec3::new(t, 0.0, half), c);
                line(Vec3::new(-half, 0.0, t), Vec3::new(half, 0.0, t), c);
            }
        }
        // X axis red, Z axis blue (Z is depth on the print bed).
        line(
            Vec3::new(-half, 0.0, 0.0),
            Vec3::new(half, 0.0, 0.0),
            [0.85, 0.3, 0.3, 0.9],
        );
        line(
            Vec3::new(0.0, 0.0, -half),
            Vec3::new(0.0, 0.0, half),
            [0.3, 0.5, 0.9, 0.9],
        );
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("hub grid"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.grid = Some(Grid {
            spacing,
            count: (data.len() / 7) as u32,
            buffer,
        });
    }
}
