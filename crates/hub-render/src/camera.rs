use glam::{Mat4, Vec2, Vec3};
use hub_core::Aabb;

/// An orbit camera around a target point, Y-up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub target: Vec3,
    pub distance: f32,
    /// Rotation about +Y in radians; 0 looks toward −Z.
    pub yaw: f32,
    /// Elevation in radians, positive looks down on the target.
    pub pitch: f32,
    pub fov_y: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 0.25,
            yaw: 0.6,
            pitch: 0.45,
            fov_y: 40_f32.to_radians(),
        }
    }
}

impl Camera {
    pub fn eye(&self) -> Vec3 {
        let dir = Vec3::new(
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.cos(),
        );
        self.target + dir * self.distance
    }

    pub fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    /// Near and far planes scaled to the orbit distance, so millimetre parts
    /// and metre-scale scenes both render without depth fighting.
    pub fn clip_planes(&self) -> (f32, f32) {
        (self.distance * 0.01, self.distance * 100.0)
    }

    pub fn projection(&self, aspect: f32) -> Mat4 {
        let (near, far) = self.clip_planes();
        // wgpu clip space: Z in [0, 1], Y up — the DirectX convention.
        glam::camera::rh::proj::directx::perspective(self.fov_y, aspect.max(1e-3), near, far)
    }

    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection(aspect) * self.view()
    }

    /// Points the camera at `bounds` and backs off until it fits the view.
    pub fn frame(&mut self, bounds: Aabb) {
        let radius = (bounds.size().length() * 0.5).max(1e-3);
        self.target = bounds.center();
        self.distance = radius / (self.fov_y * 0.5).sin() * 1.1;
    }

    /// Orbits by a drag of `delta` pixels.
    pub fn orbit(&mut self, delta: Vec2) {
        self.yaw -= delta.x * 0.008;
        self.pitch = (self.pitch + delta.y * 0.008).clamp(-1.55, 1.55);
    }

    /// Pans by a drag of `delta` pixels in a viewport `height` pixels tall.
    pub fn pan(&mut self, delta: Vec2, height: f32) {
        let view = self.view();
        let right = view.row(0).truncate();
        let up = view.row(1).truncate();
        let scale = 2.0 * self.distance * (self.fov_y * 0.5).tan() / height.max(1.0);
        self.target += (-right * delta.x + up * delta.y) * scale;
    }

    /// Zooms toward the target; `steps` > 0 moves closer.
    pub fn zoom(&mut self, steps: f32) {
        self.distance = (self.distance * (-steps * 0.1).exp()).clamp(1e-4, 1e5);
    }

    /// A world-space ray through a point given in normalized device coordinates.
    pub fn ray(&self, ndc: Vec2, aspect: f32) -> (Vec3, Vec3) {
        let inv = self.view_projection(aspect).inverse();
        let near = inv.project_point3(ndc.extend(0.0));
        let far = inv.project_point3(ndc.extend(1.0));
        (near, (far - near).normalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_puts_target_at_center() {
        let mut cam = Camera::default();
        cam.frame(Aabb {
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
        });
        let clip = cam.view_projection(1.0) * Vec3::ZERO.extend(1.0);
        assert!((clip.x / clip.w).abs() < 1e-5 && (clip.y / clip.w).abs() < 1e-5);
    }

    #[test]
    fn center_ray_hits_target() {
        let cam = Camera::default();
        let (origin, dir) = cam.ray(Vec2::ZERO, 1.5);
        let to_target = (cam.target - origin).normalize();
        assert!(dir.abs_diff_eq(to_target, 1e-4));
    }
}
