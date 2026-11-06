//! The 3D viewport: renders the scene off-screen with `hub-render` and shows
//! the image as an egui texture, handling orbit, pan, zoom and picking.

use egui::{PointerButton, Sense, Ui};
use glam::Vec2;
use hub_core::{ObjectId, Scene};
use hub_render::{Camera, FrameOptions, RenderTarget, Renderer};

pub struct Viewport {
    pub camera: Camera,
    gpu: Option<Gpu>,
}

struct Gpu {
    renderer: Renderer,
    target: RenderTarget,
    texture: egui::TextureId,
}

/// A click in the viewport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Click {
    Object(ObjectId),
    Empty,
}

/// What the user did in the viewport this frame.
#[derive(Debug, Default)]
pub struct ViewportResponse {
    pub clicked: Option<Click>,
}

impl Viewport {
    pub fn new() -> Self {
        Self {
            camera: Camera::default(),
            gpu: None,
        }
    }

    pub fn frame(&mut self, scene: &Scene, only: Option<ObjectId>) {
        let bounds = match only {
            Some(id) => scene.get(id).ok().and_then(hub_core::Object::world_bounds),
            None => scene.bounds(),
        };
        if let Some(b) = bounds {
            self.camera.frame(b);
        }
    }

    pub fn show(
        &mut self,
        ui: &mut Ui,
        frame: &eframe::Frame,
        scene: &Scene,
        selected: Option<ObjectId>,
    ) -> ViewportResponse {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let mut out = ViewportResponse::default();

        // Camera input.
        let delta = response.drag_delta();
        if response.dragged_by(PointerButton::Primary) && !ui.input(|i| i.modifiers.shift) {
            self.camera.orbit(Vec2::new(delta.x, delta.y));
        } else if response.dragged_by(PointerButton::Secondary)
            || response.dragged_by(PointerButton::Middle)
            || (response.dragged_by(PointerButton::Primary) && ui.input(|i| i.modifiers.shift))
        {
            self.camera.pan(Vec2::new(delta.x, delta.y), rect.height());
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.camera.zoom(scroll / 50.0);
            }
        }
        if response.clicked()
            && let Some(pos) = response.interact_pointer_pos()
        {
            let ndc = Vec2::new(
                (pos.x - rect.left()) / rect.width() * 2.0 - 1.0,
                1.0 - (pos.y - rect.top()) / rect.height() * 2.0,
            );
            let (origin, dir) = self.camera.ray(ndc, rect.width() / rect.height());
            out.clicked = Some(scene.pick(origin, dir).map_or(Click::Empty, Click::Object));
        }

        // Render at physical resolution.
        let Some(state) = frame.wgpu_render_state() else {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "The 3D viewport needs the wgpu renderer.",
                egui::FontId::proportional(16.0),
                ui.visuals().text_color(),
            );
            return out;
        };
        let ppp = ui.ctx().pixels_per_point();
        let size = (
            (rect.width() * ppp).round().max(1.0) as u32,
            (rect.height() * ppp).round().max(1.0) as u32,
        );
        let gpu = self.gpu.get_or_insert_with(|| {
            let target = RenderTarget::new(&state.device, size.0, size.1);
            let texture = state.renderer.write().register_native_texture(
                &state.device,
                target.display_view(),
                wgpu::FilterMode::Linear,
            );
            Gpu {
                renderer: Renderer::new(&state.device),
                target,
                texture,
            }
        });
        if gpu.target.size() != size {
            gpu.target = RenderTarget::new(&state.device, size.0, size.1);
            state
                .renderer
                .write()
                .update_egui_texture_from_wgpu_texture(
                    &state.device,
                    gpu.target.display_view(),
                    wgpu::FilterMode::Linear,
                    gpu.texture,
                );
        }
        gpu.renderer.render(
            &state.device,
            &state.queue,
            scene,
            &self.camera,
            &gpu.target,
            FrameOptions {
                selected,
                hide_grid: false,
            },
        );
        ui.painter().image(
            gpu.texture,
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        if scene.is_empty() {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Drop a .glb, .gltf, .stl or .3mf file here, or use File > Import",
                egui::FontId::proportional(16.0),
                ui.visuals().weak_text_color(),
            );
        }
        out
    }
}
