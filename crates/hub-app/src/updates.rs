//! The "Check for updates" window: every piece's installed and newest
//! version, and a button that runs each upgrade in a window of its own.

use std::collections::HashSet;
use std::sync::mpsc::{self, Receiver};

use egui::{Color32, RichText, Ui};
use hub_updates::{Item, LaunchOptions, Report, Status};

enum State {
    Idle,
    Checking(Receiver<Report>),
    Done(Report),
}

pub struct UpdatesWindow {
    pub open: bool,
    state: State,
    show_all: bool,
    /// Items whose upgrade was started in this session.
    started: HashSet<String>,
    /// The hub upgrade waiting for the person to agree to close the hub.
    confirm_restart: Option<Item>,
    message: Option<String>,
}

impl Default for UpdatesWindow {
    fn default() -> Self {
        Self {
            open: false,
            state: State::Idle,
            show_all: false,
            started: HashSet::new(),
            confirm_restart: None,
            message: None,
        }
    }
}

impl UpdatesWindow {
    /// Opens the window and checks everything again, off the UI thread.
    pub fn check(&mut self, ctx: &egui::Context) {
        self.open = true;
        self.message = None;
        self.started.clear();
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        std::thread::Builder::new()
            .name("update check".into())
            .spawn(move || {
                let report = hub_updates::check(&hub_updates::Setup::from_env());
                let _ = tx.send(report);
                ctx.request_repaint();
            })
            .expect("spawn update check");
        self.state = State::Checking(rx);
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        if let State::Checking(rx) = &self.state
            && let Ok(report) = rx.try_recv()
        {
            self.state = State::Done(report);
        }
        let mut open = self.open;
        egui::Window::new("Updates")
            .open(&mut open)
            .default_width(680.0)
            .default_height(640.0)
            .resizable(true)
            .show(ctx, |ui| self.contents(ui));
        self.open = open;
        self.restart_dialog(ctx);
    }

    fn contents(&mut self, ui: &mut Ui) {
        let mut recheck = false;
        ui.horizontal(|ui| {
            match &self.state {
                State::Checking(_) => {
                    ui.spinner();
                    ui.label("Checking the hub, hub-gen, models, ComfyUI, Blender and tools…");
                }
                State::Done(report) => {
                    ui.strong(report.summary());
                }
                State::Idle => {}
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let idle = !matches!(self.state, State::Checking(_));
                recheck = ui
                    .add_enabled(idle, egui::Button::new("Check again"))
                    .clicked();
                ui.checkbox(&mut self.show_all, "Show everything");
            });
        });
        if let Some(m) = &self.message {
            ui.label(RichText::new(m).color(ui.visuals().hyperlink_color));
        }
        ui.separator();
        if recheck {
            self.check(ui.ctx());
            return;
        }
        let State::Done(report) = &self.state else {
            return;
        };
        let mut launch: Option<Item> = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for piece in &report.pieces {
                    let hidden = hidden_text(&piece.items);
                    ui.horizontal(|ui| {
                        ui.heading(&piece.name);
                        if let Some(s) = &piece.summary {
                            ui.label(RichText::new(s).weak());
                        }
                    });
                    for item in piece
                        .items
                        .iter()
                        .filter(|i| self.show_all || i.worth_showing())
                    {
                        if let Some(chosen) = self.item(ui, item) {
                            launch = Some(chosen);
                        }
                    }
                    if !hidden.is_empty() && !self.show_all {
                        ui.label(RichText::new(hidden).weak());
                    }
                    ui.add_space(8.0);
                }
            });
        if let Some(item) = launch {
            self.start(ui.ctx(), item);
        }
    }

    /// One row; returns the item when its Update button was pressed.
    fn item(&self, ui: &mut Ui, item: &Item) -> Option<Item> {
        let mut pressed = None;
        ui.group(|ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                let (text, color) = badge(ui, item.status);
                ui.label(RichText::new(text).color(color).strong());
                ui.label(&item.name);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.started.contains(&item.id) {
                        ui.label(RichText::new("started").weak());
                    } else if item.can_update() {
                        let verb = if item.status == Status::NotInstalled {
                            "Install"
                        } else {
                            "Update"
                        };
                        if ui.button(verb).clicked() {
                            pressed = Some(item.clone());
                        }
                    }
                    if let Some(u) = &item.upgrade
                        && item.status != Status::Current
                    {
                        if !u.steps.is_empty()
                            && ui
                                .small_button("Copy commands")
                                .on_hover_text(u.script())
                                .clicked()
                        {
                            ui.ctx().copy_text(u.script());
                        }
                        if let Some(link) = &u.link {
                            ui.hyperlink_to("Open page", link);
                        }
                    }
                });
            });
            let versions = match (&item.installed, &item.latest) {
                (Some(a), Some(b)) if a != b => format!("{a}  →  {b}"),
                (Some(a), _) => a.clone(),
                (None, Some(b)) => format!("newest: {b}"),
                (None, None) => String::new(),
            };
            if !versions.is_empty() {
                ui.label(RichText::new(versions).monospace());
            }
            if let Some(d) = &item.detail {
                ui.label(RichText::new(d).small());
            }
            if let Some(why) = &item.blocked {
                ui.label(RichText::new(why).small().color(ui.visuals().warn_fg_color));
            }
            if item.status != Status::Current
                && let Some(note) = item.upgrade.as_ref().and_then(|u| u.note.as_ref())
            {
                ui.label(RichText::new(note).small().weak());
            }
        });
        pressed
    }

    fn start(&mut self, ctx: &egui::Context, item: Item) {
        let Some(upgrade) = &item.upgrade else {
            return;
        };
        if upgrade.rebuilds_hub {
            self.confirm_restart = Some(item);
            return;
        }
        self.message = Some(
            match hub_updates::launch(&item.id, &item.name, upgrade, &LaunchOptions::default()) {
                Ok(launched) => {
                    self.started.insert(item.id.clone());
                    match launched.log {
                        Some(log) => format!(
                            "Updating {} in the background; its output goes to {}. Check again when it's done.",
                            item.name,
                            log.display()
                        ),
                        None => format!(
                            "Updating {} in a new window. Check again when it says Done.",
                            item.name
                        ),
                    }
                }
                Err(e) => format!("Could not start the update: {e}"),
            },
        );
        ctx.request_repaint();
    }

    fn restart_dialog(&mut self, ctx: &egui::Context) {
        let Some(item) = self.confirm_restart.clone() else {
            return;
        };
        let mut keep = true;
        egui::Window::new("Update Rust-Hub")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.label("Rust-Hub closes so it can rebuild itself, and opens again when the build is done.");
                ui.label("Save or export anything in the scene you want to keep first.");
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui.button("Update and restart").clicked() {
                        keep = false;
                        let options = LaunchOptions {
                            wait_for: Some(std::process::id()),
                            then_start: std::env::current_exe().ok(),
                        };
                        let upgrade = item.upgrade.clone().unwrap_or_default();
                        match hub_updates::launch(&item.id, &item.name, &upgrade, &options) {
                            Ok(_) => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
                            Err(e) => self.message = Some(format!("Could not start the update: {e}")),
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        keep = false;
                    }
                });
            });
        if !keep {
            self.confirm_restart = None;
        }
    }
}

/// e.g. "8 up to date, 5 downloaded when first used".
fn hidden_text(items: &[Item]) -> String {
    let count = |s| {
        items
            .iter()
            .filter(|i| i.status == s && !i.worth_showing())
            .count()
    };
    let mut parts = Vec::new();
    match count(Status::Current) {
        0 => {}
        n => parts.push(format!("{n} up to date")),
    }
    match count(Status::NotInstalled) {
        0 => {}
        n => parts.push(format!(
            "{n} not downloaded yet (downloaded when first used)"
        )),
    }
    parts.join(", ")
}

fn badge(ui: &Ui, status: Status) -> (&'static str, Color32) {
    let v = ui.visuals();
    match status {
        Status::Current => ("up to date", Color32::from_rgb(90, 170, 90)),
        Status::Available => ("update available", v.hyperlink_color),
        Status::HeldBack => ("held back", v.warn_fg_color),
        Status::NotInstalled => ("not installed", v.weak_text_color()),
        Status::Unknown => ("not checked", v.weak_text_color()),
    }
}
