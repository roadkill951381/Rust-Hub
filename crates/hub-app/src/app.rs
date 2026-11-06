//! Application state and panels. Every change goes through
//! [`Session::execute`], the same path the CLI and the MCP server use.

use std::path::PathBuf;

use egui::{Color32, Key, KeyboardShortcut, Modifiers, RichText, Ui};
use hub_core::analysis::{PrintReport, check_printable};
use hub_core::primitives::Primitive;
use hub_core::{Command, Material, ObjectId, Outcome, Session, SizeAxis};

use crate::viewport::{Click, Viewport};

const MODEL_EXTENSIONS: [&str; 4] = ["glb", "gltf", "stl", "3mf"];

pub struct HubApp {
    session: Session,
    viewport: Viewport,
    selected: Option<ObjectId>,
    status: Status,
    /// Cached print report for the selection, keyed by session revision.
    report: Option<(u64, ObjectId, PrintReport)>,
    target_size_mm: f32,
    size_axis: SizeAxis,
    /// Increments for every new drag gesture so each is one undo step.
    gesture: u64,
    rename_buffer: Option<(ObjectId, String)>,
}

struct Status {
    text: String,
    error: bool,
}

impl HubApp {
    pub fn new(cc: &eframe::CreationContext<'_>, files: Vec<PathBuf>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let mut app = Self {
            session: Session::new(),
            viewport: Viewport::new(),
            selected: None,
            status: Status {
                text: "Ready. Import a model or add a primitive to start.".into(),
                error: false,
            },
            report: None,
            target_size_mm: 80.0,
            size_axis: SizeAxis::Longest,
            gesture: 0,
            rename_buffer: None,
        };
        for path in files {
            app.import(path);
        }
        app
    }

    fn execute(&mut self, command: Command) -> Option<Outcome> {
        let result = self.session.execute(command);
        self.finish(result)
    }

    fn execute_drag(&mut self, command: Command) {
        let result = self.session.execute_coalescing(command, self.gesture);
        self.finish(result);
    }

    fn finish(&mut self, result: hub_core::Result<Outcome>) -> Option<Outcome> {
        match result {
            Ok(outcome) => {
                self.status = Status {
                    text: outcome.summary.clone(),
                    error: false,
                };
                if let Some(&id) = outcome.created.last() {
                    self.selected = Some(id);
                }
                if self
                    .selected
                    .is_some_and(|id| self.session.scene().get(id).is_err())
                {
                    self.selected = None;
                }
                Some(outcome)
            }
            Err(e) => {
                tracing::warn!("{e}");
                self.status = Status {
                    text: e.to_string(),
                    error: true,
                };
                None
            }
        }
    }

    fn import(&mut self, path: PathBuf) {
        let was_empty = self.session.scene().is_empty();
        if self.execute(Command::Import { path }).is_some() && was_empty {
            self.viewport.frame(self.session.scene(), None);
        }
    }

    fn import_dialog(&mut self) {
        if let Some(paths) = rfd::FileDialog::new()
            .add_filter("3D models", &MODEL_EXTENSIONS)
            .pick_files()
        {
            for path in paths {
                self.import(path);
            }
        }
    }

    fn export_dialog(&mut self, only_selected: bool) {
        let objects = if only_selected {
            self.selected.map(|id| vec![id])
        } else {
            None
        };
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Binary glTF (games)", &["glb"])
            .add_filter("glTF + bin (games)", &["gltf"])
            .add_filter("STL (printing)", &["stl"])
            .add_filter("3MF (printing)", &["3mf"])
            .set_file_name("model.glb")
            .save_file()
        {
            self.execute(Command::Export { path, objects });
        }
    }

    fn selection(&self) -> Option<Vec<ObjectId>> {
        self.selected.map(|id| vec![id])
    }

    fn shortcuts(&mut self, ui: &mut Ui) {
        let pressed =
            |mods, key| ui.input_mut(|i| i.consume_shortcut(&KeyboardShortcut::new(mods, key)));
        let redo = pressed(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z)
            || pressed(Modifiers::COMMAND, Key::Y);
        let undo = pressed(Modifiers::COMMAND, Key::Z);
        let open = pressed(Modifiers::COMMAND, Key::O);
        let export = pressed(Modifiers::COMMAND, Key::E);
        let typing = ui.ctx().memory(|m| m.focused().is_some());
        let delete = !typing
            && (pressed(Modifiers::NONE, Key::Delete) || pressed(Modifiers::NONE, Key::Backspace));
        let frame = !typing && pressed(Modifiers::NONE, Key::F);
        if redo {
            self.execute(Command::Redo);
        } else if undo {
            self.execute(Command::Undo);
        }
        if open {
            self.import_dialog();
        }
        if export {
            self.export_dialog(false);
        }
        if delete && let Some(id) = self.selected {
            self.execute(Command::Delete { objects: vec![id] });
        }
        if frame {
            self.viewport.frame(self.session.scene(), self.selected);
        }
    }

    fn menu_bar(&mut self, ui: &mut Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Import…   Ctrl+O").clicked() {
                    self.import_dialog();
                }
                ui.separator();
                if ui.button("Export all…   Ctrl+E").clicked() {
                    self.export_dialog(false);
                }
                if ui
                    .add_enabled(
                        self.selected.is_some(),
                        egui::Button::new("Export selected…"),
                    )
                    .clicked()
                {
                    self.export_dialog(true);
                }
                ui.separator();
                if ui.button("Quit").clicked() {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
            ui.menu_button("Edit", |ui| {
                if ui
                    .add_enabled(self.session.can_undo(), egui::Button::new("Undo   Ctrl+Z"))
                    .clicked()
                {
                    self.execute(Command::Undo);
                }
                if ui
                    .add_enabled(
                        self.session.can_redo(),
                        egui::Button::new("Redo   Ctrl+Shift+Z"),
                    )
                    .clicked()
                {
                    self.execute(Command::Redo);
                }
                ui.separator();
                if ui
                    .add_enabled(self.selected.is_some(), egui::Button::new("Delete   Del"))
                    .clicked()
                    && let Some(id) = self.selected
                {
                    self.execute(Command::Delete { objects: vec![id] });
                }
            });
            ui.menu_button("Add", |ui| {
                for shape in [Primitive::Cube, Primitive::Sphere, Primitive::Cylinder] {
                    if ui.button(format!("{} (20 mm)", shape.name())).clicked() {
                        let was_empty = self.session.scene().is_empty();
                        self.execute(Command::AddPrimitive {
                            shape,
                            size_mm: 20.0,
                            name: None,
                        });
                        if was_empty {
                            self.viewport.frame(self.session.scene(), None);
                        }
                    }
                }
            });
            ui.menu_button("View", |ui| {
                if ui.button("Frame all").clicked() {
                    self.viewport.frame(self.session.scene(), None);
                }
                if ui
                    .add_enabled(
                        self.selected.is_some(),
                        egui::Button::new("Frame selected   F"),
                    )
                    .clicked()
                {
                    self.viewport.frame(self.session.scene(), self.selected);
                }
            });
        });
    }

    fn objects_panel(&mut self, ui: &mut Ui) {
        ui.heading("Objects");
        ui.separator();
        if self.session.scene().is_empty() {
            ui.weak("No objects yet.");
            return;
        }
        egui::ScrollArea::vertical().show(ui, |ui| {
            for o in self.session.scene().objects() {
                let selected = self.selected == Some(o.id);
                let label = format!("{}  ·  {} tris", o.name, o.mesh.triangle_count());
                if ui.selectable_label(selected, label).clicked() {
                    self.selected = if selected { None } else { Some(o.id) };
                }
            }
        });
    }

    fn inspector(&mut self, ui: &mut Ui) {
        ui.heading("Inspector");
        ui.separator();
        let Some(id) = self.selected else {
            ui.weak("Select an object to edit it.");
            ui.add_space(12.0);
            self.print_tools(ui, None);
            return;
        };
        let Ok(object) = self.session.scene().get(id) else {
            self.selected = None;
            return;
        };
        let (name, transform, material) = (object.name.clone(), object.transform, object.material);
        let info = hub_core::session::info(object);

        // Name, committed on Enter or focus loss.
        let buffer = match &mut self.rename_buffer {
            Some((bid, text)) if *bid == id => text,
            slot => &mut slot.insert((id, name.clone())).1,
        };
        let response = ui.add(egui::TextEdit::singleline(buffer).desired_width(f32::INFINITY));
        if response.lost_focus() && *buffer != name && !buffer.trim().is_empty() {
            let new_name = buffer.trim().to_owned();
            self.execute(Command::Rename {
                object: id,
                name: new_name,
            });
        } else if !response.has_focus() && *buffer != name {
            buffer.clone_from(&name);
        }

        let [w, d, h] = info.size_mm;
        ui.label(format!("{w:.2} × {d:.2} × {h:.2} mm  (W × D × H)"));
        ui.label(format!(
            "{} triangles, {} vertices",
            info.triangles, info.vertices
        ));
        ui.add_space(8.0);

        egui::CollapsingHeader::new("Transform")
            .default_open(true)
            .show(ui, |ui| {
                let mut position = (transform.translation * 1000.0).to_array();
                let mut rotation = transform.euler_degrees().to_array();
                let mut scale = transform.scale.to_array();
                let mut changed = false;
                egui::Grid::new("transform").num_columns(4).show(ui, |ui| {
                    for (label, values, speed, suffix, decimals) in [
                        ("Position", &mut position, 0.5, " mm", 3),
                        ("Rotation", &mut rotation, 1.0, "°", 2),
                        ("Scale", &mut scale, 0.01, "", 6),
                    ] {
                        ui.label(label);
                        for v in values.iter_mut() {
                            // Scale drags proportionally so tiny and huge factors both move smoothly.
                            let speed = if suffix.is_empty() {
                                (v.abs() * speed).max(1e-6)
                            } else {
                                speed
                            };
                            let r = ui.add(
                                egui::DragValue::new(v)
                                    .speed(speed)
                                    .suffix(suffix)
                                    .max_decimals(decimals),
                            );
                            changed |= r.changed();
                            if r.drag_started() || r.gained_focus() {
                                self.gesture += 1;
                            }
                        }
                        ui.end_row();
                    }
                });
                if changed {
                    self.execute_drag(Command::SetTransform {
                        object: id,
                        position_mm: Some(position),
                        rotation_deg: Some(rotation),
                        scale: Some(scale),
                    });
                }
                ui.horizontal(|ui| {
                    if ui
                        .button("Apply transform")
                        .on_hover_text("Bake into the vertices")
                        .clicked()
                    {
                        self.execute(Command::ApplyTransform { object: id });
                    }
                    if ui.button("Reset").clicked() {
                        self.execute(Command::SetTransform {
                            object: id,
                            position_mm: Some([0.0; 3]),
                            rotation_deg: Some([0.0; 3]),
                            scale: Some([1.0; 3]),
                        });
                    }
                });
            });

        egui::CollapsingHeader::new("Material")
            .default_open(false)
            .show(ui, |ui| {
                let mut m: Material = material;
                let mut changed = false;
                ui.horizontal(|ui| {
                    ui.label("Base colour");
                    let r = ui.color_edit_button_rgba_unmultiplied(&mut m.base_color);
                    if r.clicked() {
                        self.gesture += 1;
                    }
                    changed |= r.changed();
                });
                for (label, value) in [
                    ("Metallic", &mut m.metallic),
                    ("Roughness", &mut m.roughness),
                ] {
                    let r = ui.add(egui::Slider::new(value, 0.0..=1.0).text(label));
                    if r.drag_started() {
                        self.gesture += 1;
                    }
                    changed |= r.changed();
                }
                if changed {
                    self.execute_drag(Command::SetMaterial {
                        object: id,
                        material: m,
                    });
                }
            });

        ui.add_space(8.0);
        self.print_tools(ui, Some(id));
    }

    fn print_tools(&mut self, ui: &mut Ui, id: Option<ObjectId>) {
        egui::CollapsingHeader::new("3D printing")
            .default_open(true)
            .show(ui, |ui| {
                let scope = if id.is_some() {
                    "selection"
                } else {
                    "all objects"
                };
                ui.weak(format!("Applies to {scope}."));
                ui.horizontal(|ui| {
                    ui.label("Size");
                    ui.add(
                        egui::DragValue::new(&mut self.target_size_mm)
                            .range(0.1..=10_000.0)
                            .suffix(" mm"),
                    );
                    egui::ComboBox::from_id_salt("axis")
                        .selected_text(axis_name(self.size_axis))
                        .show_ui(ui, |ui| {
                            for axis in [
                                SizeAxis::Longest,
                                SizeAxis::X,
                                SizeAxis::Depth,
                                SizeAxis::Height,
                            ] {
                                ui.selectable_value(&mut self.size_axis, axis, axis_name(axis));
                            }
                        });
                    if ui.button("Scale").clicked()
                        && self
                            .execute(Command::ScaleToSize {
                                objects: self.selection(),
                                size_mm: self.target_size_mm,
                                axis: self.size_axis,
                            })
                            .is_some()
                    {
                        self.viewport.frame(self.session.scene(), id);
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Place on bed").clicked()
                        && self
                            .execute(Command::PlaceOnFloor {
                                objects: self.selection(),
                            })
                            .is_some()
                    {
                        self.viewport.frame(self.session.scene(), id);
                    }
                    if ui.button("Repair").clicked() {
                        self.execute(Command::Repair {
                            objects: self.selection(),
                        });
                    }
                });

                let Some(id) = id else { return };
                let revision = self.session.revision();
                let fresh = self
                    .report
                    .as_ref()
                    .is_some_and(|(rev, rid, _)| *rev == revision && *rid == id);
                if !fresh && let Ok(o) = self.session.scene().get(id) {
                    self.report = Some((revision, id, check_printable(o)));
                }
                let Some((_, _, report)) = &self.report else {
                    return;
                };
                ui.add_space(6.0);
                let (text, color) = if report.printable {
                    ("Ready to print", Color32::from_rgb(110, 200, 120))
                } else {
                    ("Not ready to print", Color32::from_rgb(235, 110, 90))
                };
                ui.label(RichText::new(text).strong().color(color));
                check_row(ui, "Watertight", report.watertight);
                check_row(ui, "Faces point outward", report.outward);
                check_row(
                    ui,
                    "No zero-area faces",
                    report.topology.degenerate_triangles == 0,
                );
                ui.label(format!(
                    "Volume {:.2} cm³ · {} shell(s)",
                    report.volume_mm3 / 1000.0,
                    report.topology.shells
                ));
                for issue in &report.issues {
                    ui.label(RichText::new(format!("• {issue}")).small());
                }
            });
    }

    fn status_bar(&self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            let color = if self.status.error {
                Color32::from_rgb(235, 110, 90)
            } else {
                ui.visuals().text_color()
            };
            ui.label(RichText::new(&self.status.text).color(color));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let triangles: usize = self
                    .session
                    .scene()
                    .objects()
                    .iter()
                    .map(|o| o.mesh.triangle_count())
                    .sum();
                ui.weak(format!(
                    "{} objects · {triangles} triangles",
                    self.session.scene().objects().len()
                ));
            });
        });
    }
}

fn check_row(ui: &mut Ui, label: &str, ok: bool) {
    let (mark, color) = if ok {
        ("yes", Color32::from_rgb(110, 200, 120))
    } else {
        ("no", Color32::from_rgb(235, 110, 90))
    };
    ui.horizontal(|ui| {
        ui.label(label);
        ui.label(RichText::new(mark).strong().color(color));
    });
}

fn axis_name(axis: SizeAxis) -> &'static str {
    match axis {
        SizeAxis::Longest => "longest side",
        SizeAxis::X => "width",
        SizeAxis::Depth => "depth",
        SizeAxis::Height => "height",
    }
}

impl eframe::App for HubApp {
    fn ui(&mut self, ui: &mut Ui, frame: &mut eframe::Frame) {
        self.shortcuts(ui);

        let dropped: Vec<PathBuf> = ui.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf())
                .filter(|p| !p.as_os_str().is_empty())
                .collect()
        });
        for path in dropped {
            self.import(path);
        }

        egui::Panel::top("menu").show(ui, |ui| self.menu_bar(ui));
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));
        egui::Panel::left("objects")
            .resizable(true)
            .default_size(220.0)
            .show(ui, |ui| self.objects_panel(ui));
        egui::Panel::right("inspector")
            .resizable(true)
            .default_size(300.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.inspector(ui));
            });
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| {
                let response = self
                    .viewport
                    .show(ui, frame, self.session.scene(), self.selected);
                match response.clicked {
                    Some(Click::Object(id)) => self.selected = Some(id),
                    Some(Click::Empty) => self.selected = None,
                    None => {}
                }
            });
    }
}
