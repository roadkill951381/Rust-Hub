//! Application state and panels. Every change goes through
//! [`Session::execute`], the same path the CLI and the MCP server use.

use std::path::PathBuf;

use egui::{Color32, Key, KeyboardShortcut, Modifiers, RichText, Ui};
use hub_core::analysis::{DEFAULT_MIN_WALL_MM, IGNORED_THIN_FRACTION, PrintReport};
use hub_core::library::{Asset, AssetSource};
use hub_core::primitives::Primitive;
use hub_core::{Command, Material, ObjectId, Outcome, Session, SizeAxis, Step};

use crate::viewport::{Click, Viewport};
use crate::work::{ReportCache, ReportKey, Running};

const MODEL_EXTENSIONS: [&str; 4] = ["glb", "gltf", "stl", "3mf"];
const IMAGE_EXTENSIONS: [&str; 4] = ["png", "jpg", "jpeg", "webp"];

/// Shape models offered in the Generate panel: service name and label.
const GENERATORS: [(&str, &str); 5] = [
    ("", "Service default"),
    ("hunyuan3d-2", "Hunyuan3D 2 (best detail)"),
    ("hunyuan3d-2mini", "Hunyuan3D 2 mini (less memory)"),
    ("triposg", "TripoSG"),
    ("demo", "Demo shapes (no GPU)"),
];

const OK: Color32 = Color32::from_rgb(110, 200, 120);
const BAD: Color32 = Color32::from_rgb(235, 110, 90);
const WARN: Color32 = Color32::from_rgb(230, 180, 80);

pub struct HubApp {
    session: Session,
    ctx: egui::Context,
    viewport: Viewport,
    selected: Option<ObjectId>,
    status: Status,
    /// The long command in progress, if any. One at a time.
    running: Option<Running>,
    reports: ReportCache,
    print: PrintSettings,
    generate: GenerateForm,
    library: LibraryView,
    /// Increments for every new drag gesture so each is one undo step.
    gesture: u64,
    rename_buffer: Option<(ObjectId, String)>,
}

struct PrintSettings {
    target_size_mm: f32,
    size_axis: SizeAxis,
    min_wall_mm: f32,
    fill_holes: bool,
    remove_specks: bool,
    /// Percent of the largest piece's volume below which pieces are removed.
    speck_percent: f32,
    hollow_wall_mm: f32,
    remesh_voxel_mm: f32,
    decimate_ratio: f32,
}

impl Default for PrintSettings {
    fn default() -> Self {
        Self {
            target_size_mm: 80.0,
            size_axis: SizeAxis::Longest,
            min_wall_mm: DEFAULT_MIN_WALL_MM,
            fill_holes: true,
            remove_specks: true,
            speck_percent: 1.0,
            hollow_wall_mm: 2.0,
            remesh_voxel_mm: 0.5,
            decimate_ratio: 0.5,
        }
    }
}

struct GenerateForm {
    prompt: String,
    image: Option<PathBuf>,
    /// Index into [`GENERATORS`].
    model: usize,
    texture: bool,
    fixed_seed: bool,
    seed: u64,
    size_mm: f32,
}

impl Default for GenerateForm {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            image: None,
            model: 0,
            texture: false,
            fixed_seed: false,
            seed: 1234,
            size_mm: 80.0,
        }
    }
}

struct LibraryView {
    query: String,
    assets: Vec<Asset>,
    error: Option<String>,
    /// Re-read the library before the next draw.
    stale: bool,
    save_name: String,
    save_note: String,
}

impl Default for LibraryView {
    fn default() -> Self {
        Self {
            query: String::new(),
            assets: Vec::new(),
            error: None,
            stale: true,
            save_name: String::new(),
            save_note: String::new(),
        }
    }
}

struct Status {
    text: String,
    error: bool,
}

impl HubApp {
    pub fn new(cc: &eframe::CreationContext<'_>, files: Vec<PathBuf>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let mut app = Self {
            session: Session::with_engines(hub_engines::from_env()),
            ctx: cc.egui_ctx.clone(),
            viewport: Viewport::new(),
            selected: None,
            status: Status {
                text: "Ready. Import a model, add a primitive or generate one to start.".into(),
                error: false,
            },
            running: None,
            reports: ReportCache::default(),
            print: PrintSettings::default(),
            generate: GenerateForm::default(),
            library: LibraryView::default(),
            gesture: 0,
            rename_buffer: None,
        };
        for path in files {
            app.import(path);
        }
        app
    }

    /// Runs a command. Long ones (generate, hollow, Blender jobs) start on a
    /// worker thread and return `None`; their result arrives in [`Self::poll_job`].
    fn execute(&mut self, command: Command) -> Option<Outcome> {
        if !command.is_job() {
            let result = self.session.execute(command);
            return self.finish(result);
        }
        if let Some(running) = &self.running {
            self.status = Status {
                text: format!(
                    "Wait for {} to finish first.",
                    running.label().to_lowercase()
                ),
                error: true,
            };
            return None;
        }
        match self.session.begin(command) {
            Ok(Step::Done(outcome)) => self.finish(Ok(outcome)),
            Ok(Step::Job(job)) => {
                self.status = Status {
                    text: format!("{}…", job.label()),
                    error: false,
                };
                self.running = Some(Running::start(job, self.ctx.clone()));
                None
            }
            Err(e) => self.finish(Err(e)),
        }
    }

    /// Commits a finished job. Edits made meanwhile are kept: the session
    /// refuses a result for an object that changed while the job ran.
    fn poll_job(&mut self) {
        let Some(result) = self.running.as_ref().and_then(Running::poll) else {
            return;
        };
        self.running = None;
        let was_empty = self.session.scene().objects().len() <= 1;
        let result = result.and_then(|output| self.session.finish(output));
        if let Some(outcome) = self.finish(result)
            && outcome.generation.is_some()
        {
            self.library.stale = true;
            let focus = if was_empty { None } else { self.selected };
            self.viewport.frame(self.session.scene(), focus);
        }
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

    fn left_panel(&mut self, ui: &mut Ui) {
        // Fill the panel instead of sizing to the content, or widgets that
        // take the available width would widen the panel every frame.
        egui::ScrollArea::vertical()
            .auto_shrink([false, true])
            .show(ui, |ui| {
                egui::CollapsingHeader::new(RichText::new("Generate").heading())
                    .default_open(true)
                    .show(ui, |ui| self.generate_panel(ui));
                ui.separator();
                egui::CollapsingHeader::new(RichText::new("Objects").heading())
                    .default_open(true)
                    .show(ui, |ui| self.objects_panel(ui));
                ui.separator();
                egui::CollapsingHeader::new(RichText::new("Library").heading())
                    .default_open(true)
                    .show(ui, |ui| self.library_panel(ui));
            });
    }

    fn generate_panel(&mut self, ui: &mut Ui) {
        let form = &mut self.generate;
        ui.add(
            egui::TextEdit::multiline(&mut form.prompt)
                .hint_text("A knight in plate armour, full body, standing")
                .desired_rows(3)
                .desired_width(ui.available_width()),
        );
        ui.horizontal(|ui| {
            let picture = form.image.as_ref().and_then(|p| p.file_name()).map_or_else(
                || "no picture".to_owned(),
                |n| n.to_string_lossy().into_owned(),
            );
            ui.label(format!("Picture: {picture}"))
                .on_hover_text("Optional. A picture of one object, used instead of the prompt.");
            if ui.small_button("Choose…").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("Images", &IMAGE_EXTENSIONS)
                    .pick_file()
            {
                form.image = Some(path);
            }
            if form.image.is_some() && ui.small_button("Clear").clicked() {
                form.image = None;
            }
        });
        egui::Grid::new("generate").num_columns(2).show(ui, |ui| {
            ui.label("Model");
            egui::ComboBox::from_id_salt("generator")
                .selected_text(GENERATORS[form.model].1)
                .show_ui(ui, |ui| {
                    for (i, (_, label)) in GENERATORS.iter().enumerate() {
                        ui.selectable_value(&mut form.model, i, *label);
                    }
                });
            ui.end_row();
            ui.label("Size");
            ui.add(
                egui::DragValue::new(&mut form.size_mm)
                    .range(1.0..=2000.0)
                    .suffix(" mm"),
            )
            .on_hover_text("Longest side of the result");
            ui.end_row();
            ui.checkbox(&mut form.fixed_seed, "Seed")
                .on_hover_text("Fix the seed to get the same model again");
            ui.add_enabled(form.fixed_seed, egui::DragValue::new(&mut form.seed));
            ui.end_row();
        });
        ui.checkbox(&mut form.texture, "Paint colour texture")
            .on_hover_text(
                "Slower and needs more GPU memory. The texture is kept in the library copy.",
            );
        let ready = !form.prompt.trim().is_empty() || form.image.is_some();
        let idle = self.running.is_none();
        let clicked = ui
            .add_enabled(ready && idle, egui::Button::new("Generate"))
            .on_disabled_hover_text(if idle {
                "Write a prompt or choose a picture first"
            } else {
                "Another job is running"
            })
            .clicked();
        if let Some(generator) = &self.session.engines().generator {
            ui.weak(format!("Runs on {}", generator.describe()));
        }
        if clicked {
            let form = &self.generate;
            let prompt = form.prompt.trim();
            let command = Command::Generate {
                prompt: (!prompt.is_empty()).then(|| prompt.to_owned()),
                image: form.image.clone(),
                model: Some(GENERATORS[form.model].0)
                    .filter(|m| !m.is_empty())
                    .map(str::to_owned),
                texture: form.texture,
                seed: form.fixed_seed.then_some(form.seed),
                faces: None,
                name: None,
                size_mm: Some(form.size_mm),
            };
            self.execute(command);
        }
    }

    fn objects_panel(&mut self, ui: &mut Ui) {
        if self.session.scene().is_empty() {
            ui.weak("No objects yet.");
            return;
        }
        for o in self.session.scene().objects() {
            let selected = self.selected == Some(o.id);
            let label = format!("{}  ·  {} tris", o.name, o.mesh.triangle_count());
            if ui.selectable_label(selected, label).clicked() {
                self.selected = if selected { None } else { Some(o.id) };
            }
        }
    }

    fn refresh_library(&mut self) {
        self.library.stale = false;
        let query = self.library.query.trim();
        let command = Command::ListAssets {
            query: (!query.is_empty()).then(|| query.to_owned()),
        };
        match self.session.execute(command) {
            Ok(outcome) => {
                self.library.assets = outcome.assets;
                self.library.error = None;
            }
            Err(e) => {
                self.library.assets.clear();
                self.library.error = Some(e.to_string());
            }
        }
    }

    fn library_panel(&mut self, ui: &mut Ui) {
        if self.library.stale {
            self.refresh_library();
        }
        if let Some(library) = &self.session.engines().library {
            ui.weak(library.root().display().to_string());
        }
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("Reload").clicked() {
                    self.library.stale = true;
                }
                let search = ui.add(
                    egui::TextEdit::singleline(&mut self.library.query)
                        .hint_text("Search")
                        .desired_width(ui.available_width()),
                );
                if search.changed() {
                    self.library.stale = true;
                }
            });
        });
        if let Some(error) = &self.library.error {
            ui.label(RichText::new(error).color(BAD).small());
        } else if self.library.assets.is_empty() {
            ui.weak("Nothing saved yet. Generated models are saved here automatically.");
        }

        let mut load = None;
        for asset in &self.library.assets {
            let count = asset.versions.len();
            let plural = if count == 1 { "" } else { "s" };
            let title = format!("{}  ·  {count} version{plural}", asset.name);
            egui::CollapsingHeader::new(title)
                .id_salt(&asset.slug)
                .show(ui, |ui| {
                    for v in asset.versions.iter().rev() {
                        ui.horizontal(|ui| {
                            if ui.small_button("Load").clicked() {
                                load = Some((asset.name.clone(), v.version));
                            }
                            let how = match &v.source {
                                AssetSource::Saved => "saved".to_owned(),
                                AssetSource::Generated(g) => format!("generated, {}", g.model),
                            };
                            let mut line = format!("v{} · {} tris · {how}", v.version, v.triangles);
                            if let Some(note) = &v.note {
                                line.push_str(" · ");
                                line.push_str(note);
                            }
                            ui.label(RichText::new(line).small());
                        });
                    }
                });
        }
        if let Some((name, version)) = load {
            let was_empty = self.session.scene().is_empty();
            if self
                .execute(Command::LoadAsset {
                    name,
                    version: Some(version),
                })
                .is_some()
            {
                let focus = if was_empty { None } else { self.selected };
                self.viewport.frame(self.session.scene(), focus);
            }
        }

        ui.add_space(6.0);
        let scope = if self.selected.is_some() {
            "selection"
        } else {
            "scene"
        };
        if self.library.save_name.is_empty()
            && let Some(o) = self
                .selected
                .and_then(|id| self.session.scene().get(id).ok())
        {
            self.library.save_name.clone_from(&o.name);
        }
        egui::Grid::new("save_asset").num_columns(2).show(ui, |ui| {
            ui.label("Name");
            ui.text_edit_singleline(&mut self.library.save_name);
            ui.end_row();
            ui.label("Note");
            ui.add(egui::TextEdit::singleline(&mut self.library.save_note).hint_text("optional"));
            ui.end_row();
        });
        let can_save =
            !self.session.scene().is_empty() && !self.library.save_name.trim().is_empty();
        if ui
            .add_enabled(
                can_save,
                egui::Button::new(format!("Save {scope} as new version")),
            )
            .clicked()
        {
            let note = self.library.save_note.trim();
            let command = Command::SaveAsset {
                name: self.library.save_name.trim().to_owned(),
                objects: self.selection(),
                note: (!note.is_empty()).then(|| note.to_owned()),
            };
            if self.execute(command).is_some() {
                self.library.save_note.clear();
                self.library.stale = true;
            }
        }
    }

    fn inspector(&mut self, ui: &mut Ui) {
        ui.heading("Inspector");
        ui.separator();
        let Some(id) = self.selected else {
            ui.weak("Select an object to edit it.");
            ui.add_space(12.0);
            self.print_tools(ui, None);
            self.mesh_tools(ui);
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
                // Adding zero turns -0 into 0, which reads better.
                let mut rotation = transform.euler_degrees().to_array().map(|v| v + 0.0);
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
        self.mesh_tools(ui);
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
                    let p = &mut self.print;
                    ui.label("Size");
                    ui.add(
                        egui::DragValue::new(&mut p.target_size_mm)
                            .range(0.1..=10_000.0)
                            .suffix(" mm"),
                    );
                    egui::ComboBox::from_id_salt("axis")
                        .selected_text(axis_name(p.size_axis))
                        .show_ui(ui, |ui| {
                            for axis in [
                                SizeAxis::Longest,
                                SizeAxis::X,
                                SizeAxis::Depth,
                                SizeAxis::Height,
                            ] {
                                ui.selectable_value(&mut p.size_axis, axis, axis_name(axis));
                            }
                        });
                    if ui.button("Scale").clicked()
                        && self
                            .execute(Command::ScaleToSize {
                                objects: self.selection(),
                                size_mm: self.print.target_size_mm,
                                axis: self.print.size_axis,
                            })
                            .is_some()
                    {
                        self.viewport.frame(self.session.scene(), id);
                    }
                });
                if ui.button("Place on bed").clicked()
                    && self
                        .execute(Command::PlaceOnFloor {
                            objects: self.selection(),
                        })
                        .is_some()
                {
                    self.viewport.frame(self.session.scene(), id);
                }

                ui.add_space(4.0);
                let p = &mut self.print;
                ui.checkbox(&mut p.fill_holes, "Fill holes");
                ui.horizontal(|ui| {
                    ui.checkbox(&mut p.remove_specks, "Remove pieces under");
                    ui.add_enabled(
                        p.remove_specks,
                        egui::DragValue::new(&mut p.speck_percent)
                            .range(0.01..=50.0)
                            .speed(0.05)
                            .suffix(" %"),
                    )
                    .on_hover_text(
                        "Of the largest piece's volume. Removes the specks AI models leave.",
                    );
                });
                if ui
                    .button("Repair")
                    .on_hover_text("Fix face direction, weld seams, drop zero-area faces")
                    .clicked()
                {
                    let p = &self.print;
                    self.execute(Command::Repair {
                        objects: self.selection(),
                        fill_holes: Some(p.fill_holes),
                        remove_small_shells: p.remove_specks.then_some(p.speck_percent / 100.0),
                    });
                }

                ui.add_space(4.0);
                let idle = self.running.is_none();
                ui.horizontal(|ui| {
                    let p = &mut self.print;
                    ui.label("Walls");
                    ui.add(
                        egui::DragValue::new(&mut p.hollow_wall_mm)
                            .range(0.5..=20.0)
                            .speed(0.05)
                            .suffix(" mm"),
                    );
                    if ui
                        .add_enabled(idle, egui::Button::new("Hollow"))
                        .on_hover_text(
                            "For resin printing: saves material. Add drain holes in your slicer.",
                        )
                        .clicked()
                    {
                        self.execute(Command::Hollow {
                            objects: self.selection(),
                            wall_mm: self.print.hollow_wall_mm,
                            voxel_mm: None,
                        });
                    }
                });

                let Some(id) = id else { return };
                self.print_report(ui, id);
            });
    }

    fn print_report(&mut self, ui: &mut Ui, id: ObjectId) {
        let key = ReportKey {
            revision: self.session.revision(),
            object: id,
            min_wall_mm: self.print.min_wall_mm,
        };
        let Ok(object) = self.session.scene().get(id) else {
            return;
        };
        ui.add_space(6.0);
        let Some((report, fresh)) = self.reports.get(key, object, &self.ctx) else {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.weak("Checking…");
            });
            return;
        };
        ui.horizontal(|ui| {
            let (text, color) = match (report.printable, report.warnings.is_empty()) {
                (true, true) => ("Ready to print", OK),
                (true, false) => ("Printable, with warnings", WARN),
                (false, _) => ("Not ready to print", BAD),
            };
            ui.label(RichText::new(text).strong().color(color));
            if !fresh {
                ui.spinner();
            }
        });
        show_report(ui, report);
        ui.horizontal(|ui| {
            ui.label("Thinnest wall to accept");
            ui.add(
                egui::DragValue::new(&mut self.print.min_wall_mm)
                    .range(0.1..=10.0)
                    .speed(0.02)
                    .suffix(" mm"),
            );
        });
    }

    fn mesh_tools(&mut self, ui: &mut Ui) {
        egui::CollapsingHeader::new("Mesh tools")
            .default_open(false)
            .show(ui, |ui| {
                let label = self
                    .session
                    .engines()
                    .mesh_tools
                    .as_ref()
                    .map_or_else(|| "Blender".to_owned(), |t| t.describe());
                ui.weak(format!("Runs in {label} in the background."));
                let idle = self.running.is_none();
                let mut command = None;
                egui::Grid::new("mesh_tools").num_columns(2).show(ui, |ui| {
                    let p = &mut self.print;
                    ui.add(
                        egui::DragValue::new(&mut p.remesh_voxel_mm)
                            .range(0.05..=50.0)
                            .speed(0.01)
                            .prefix("voxel ")
                            .suffix(" mm"),
                    );
                    if ui
                        .add_enabled(idle, egui::Button::new("Remesh"))
                        .on_hover_text(
                            "Rebuild the surface: closes holes and fuses overlapping parts. Loses UVs.",
                        )
                        .clicked()
                    {
                        command = Some(Command::Remesh {
                            objects: None,
                            voxel_mm: p.remesh_voxel_mm,
                        });
                    }
                    ui.end_row();
                    ui.add(
                        egui::Slider::new(&mut p.decimate_ratio, 0.01..=1.0)
                            .text("keep")
                            .custom_formatter(|v, _| format!("{:.0} %", v * 100.0))
                            .custom_parser(|s| {
                                s.trim_end_matches('%').trim().parse::<f64>().ok().map(|v| v / 100.0)
                            }),
                    );
                    if ui
                        .add_enabled(idle, egui::Button::new("Decimate"))
                        .on_hover_text("Fewer triangles, same shape")
                        .clicked()
                    {
                        command = Some(Command::Decimate {
                            objects: None,
                            ratio: Some(p.decimate_ratio),
                            triangles: None,
                        });
                    }
                    ui.end_row();
                });
                if ui
                    .add_enabled(idle, egui::Button::new("UV unwrap"))
                    .on_hover_text("Cut seams and lay out UVs so the model can be textured")
                    .clicked()
                {
                    command = Some(Command::UvUnwrap {
                        objects: None,
                        angle_deg: None,
                        margin: None,
                    });
                }
                if let Some(mut command) = command {
                    set_objects(&mut command, self.selection());
                    self.execute(command);
                }
            });
    }

    fn status_bar(&self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            let color = ui.visuals().text_color();
            if let Some(running) = &self.running {
                let (fraction, stage) = running.progress();
                ui.spinner();
                ui.add(
                    egui::ProgressBar::new(fraction)
                        .desired_width(160.0)
                        .show_percentage(),
                );
                ui.label(stage);
            } else {
                let color = if self.status.error { BAD } else { color };
                ui.label(RichText::new(&self.status.text).color(color));
            }
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

fn show_report(ui: &mut Ui, report: &PrintReport) {
    check_row(ui, "Watertight", report.watertight);
    check_row(ui, "Faces point outward", report.outward);
    check_row(
        ui,
        "No zero-area faces",
        report.topology.degenerate_triangles == 0,
    );
    let cavities = match report.topology.cavities {
        0 => String::new(),
        1 => ", 1 cavity".to_owned(),
        n => format!(", {n} cavities"),
    };
    ui.label(format!(
        "Volume {:.2} cm³ · {} shell(s){cavities}",
        report.volume_mm3 / 1000.0,
        report
            .topology
            .shells
            .saturating_sub(report.topology.cavities),
    ));
    if let Some(wall) = &report.wall {
        let color = if wall.thin_fraction > IGNORED_THIN_FRACTION {
            WARN
        } else {
            ui.visuals().text_color()
        };
        ui.label(
            RichText::new(format!(
                "Thinnest wall {:.2} mm ({:.1} % of the surface under {:.1} mm)",
                wall.thinnest_mm,
                wall.thin_fraction * 100.0,
                wall.min_wall_mm
            ))
            .color(color),
        );
    }
    for issue in &report.issues {
        ui.label(RichText::new(format!("• {issue}")).small().color(BAD));
    }
    for warning in &report.warnings {
        ui.label(RichText::new(format!("• {warning}")).small().color(WARN));
    }
}

/// Points a command that takes `objects` at `objects`.
fn set_objects(command: &mut Command, objects: Option<Vec<ObjectId>>) {
    match command {
        Command::Remesh { objects: o, .. }
        | Command::Decimate { objects: o, .. }
        | Command::UvUnwrap { objects: o, .. } => *o = objects,
        _ => {}
    }
}

fn check_row(ui: &mut Ui, label: &str, ok: bool) {
    let (mark, color) = if ok { ("yes", OK) } else { ("no", BAD) };
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
        self.poll_job();
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
            .default_size(260.0)
            .show(ui, |ui| self.left_panel(ui));
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
