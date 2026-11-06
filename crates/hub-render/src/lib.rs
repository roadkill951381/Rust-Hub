//! Rust-Hub renderer.
//!
//! One wgpu renderer serves both front ends: the desktop app draws into a
//! [`RenderTarget`] it shows through egui, and [`Headless`] draws the same
//! frame off-screen and encodes a PNG, so Claude can look at its own work
//! through the MCP `render_preview` tool.

mod camera;
mod renderer;

pub use camera::Camera;
pub use renderer::{COLOR_FORMAT, FrameOptions, RenderTarget, Renderer};

use hub_core::Scene;

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("no GPU adapter is available: {0}")]
    NoAdapter(String),
    #[error("could not open the GPU device: {0}")]
    Device(String),
    #[error("reading the image back from the GPU failed: {0}")]
    Readback(String),
    #[error("PNG encoding failed: {0}")]
    Encode(String),
}

/// A self-contained GPU context for rendering without a window.
pub struct Headless {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
    adapter_name: String,
}

impl std::fmt::Debug for Headless {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Headless")
            .field("adapter", &self.adapter_name)
            .finish_non_exhaustive()
    }
}

/// Picks a camera that frames the whole scene from the default angle.
pub fn framing_camera(scene: &Scene) -> Camera {
    let mut camera = Camera::default();
    if let Some(bounds) = scene.bounds() {
        camera.frame(bounds);
    }
    camera
}

impl Headless {
    /// Opens the best available adapter, falling back to a software one.
    pub fn new() -> Result<Self, RenderError> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, RenderError> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let mut last_error = String::new();
        let mut adapter = None;
        for force_fallback_adapter in [false, true] {
            match instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter,
                    compatible_surface: None,
                    apply_limit_buckets: false,
                })
                .await
            {
                Ok(a) => {
                    adapter = Some(a);
                    break;
                }
                Err(e) => last_error = e.to_string(),
            }
        }
        let adapter = adapter.ok_or(RenderError::NoAdapter(last_error))?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("hub headless"),
                required_limits: wgpu::Limits::downlevel_defaults()
                    .using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .map_err(|e| RenderError::Device(e.to_string()))?;
        let renderer = Renderer::new(&device);
        tracing::info!(adapter = %adapter_name, "headless renderer ready");
        Ok(Self {
            device,
            queue,
            renderer,
            adapter_name,
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Renders the scene and returns PNG bytes.
    pub fn render_png(
        &mut self,
        scene: &Scene,
        camera: &Camera,
        width: u32,
        height: u32,
        options: FrameOptions,
    ) -> Result<Vec<u8>, RenderError> {
        let target = RenderTarget::new(&self.device, width, height);
        self.renderer
            .render(&self.device, &self.queue, scene, camera, &target, options);
        let (w, h) = target.size();
        let rgba = self.read_back(target.texture(), w, h)?;
        let mut png = Vec::new();
        image::write_buffer_with_format(
            &mut std::io::Cursor::new(&mut png),
            &rgba,
            w,
            h,
            image::ExtendedColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .map_err(|e| RenderError::Encode(e.to_string()))?;
        Ok(png)
    }

    fn read_back(&self, texture: &wgpu::Texture, w: u32, h: u32) -> Result<Vec<u8>, RenderError> {
        let unpadded = w * 4;
        let padded = unpadded.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hub readback"),
            size: u64::from(padded * h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let (tx, rx) = std::sync::mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, .., move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| RenderError::Readback(e.to_string()))?;
        rx.recv()
            .map_err(|e| RenderError::Readback(e.to_string()))?
            .map_err(|e| RenderError::Readback(e.to_string()))?;
        let view = buffer
            .get_mapped_range(..)
            .map_err(|e| RenderError::Readback(e.to_string()))?;
        let mut rgba = Vec::with_capacity((unpadded * h) as usize);
        for row in view.chunks_exact(padded as usize) {
            rgba.extend_from_slice(&row[..unpadded as usize]);
        }
        Ok(rgba)
    }
}
