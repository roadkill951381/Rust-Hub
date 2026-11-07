//! `hub`: the Rust-Hub command line.
//!
//! Every subcommand is a thin wrapper over [`hub_core::Command`], the same
//! commands the desktop app and the MCP server run.

use std::io::Read as _;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context as _, Result, bail};
use clap::{Parser, Subcommand};
use hub_core::{Command, Outcome, Session, SizeAxis, Step};

#[derive(Debug, Parser)]
#[command(
    name = "hub",
    version,
    about = "Rust-Hub: 3D model tools for games and printing"
)]
struct Cli {
    /// Print results as JSON instead of text.
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// List the objects in a model file with sizes in millimetres.
    Info { file: PathBuf },
    /// Check a model for 3D printing. Exits with status 2 when it is not printable.
    Check {
        file: PathBuf,
        /// Thinnest acceptable wall in millimetres.
        #[arg(long, default_value_t = 0.8)]
        min_wall_mm: f32,
    },
    /// Convert between .glb, .gltf, .stl and .3mf, optionally preparing for print.
    Convert {
        input: PathBuf,
        output: PathBuf,
        /// Scale uniformly so the longest side is this many millimetres.
        #[arg(long)]
        size_mm: Option<f32>,
        /// Repair first: winding, zero-area and duplicate faces, holes.
        #[arg(long)]
        repair: bool,
        /// With --repair, also delete loose pieces smaller than this fraction
        /// of the largest one's volume (0.01 suits AI-generated models).
        #[arg(long, requires = "repair")]
        remove_small_shells: Option<f32>,
        /// Hollow for resin printing with walls this many millimetres thick.
        #[arg(long)]
        hollow: Option<f32>,
        /// Centre on the origin and rest on the floor.
        #[arg(long)]
        place_on_floor: bool,
    },
    /// Generate a model from a prompt or picture on the local hub-gen service
    /// (your own GPU), save it to the asset library, and optionally export it.
    Generate {
        /// What to make: one object.
        prompt: Option<String>,
        /// A picture of the object to build from, instead of a prompt.
        #[arg(long)]
        image: Option<PathBuf>,
        /// Shape model: hunyuan3d-2, hunyuan3d-2mini, triposg (service default if
        /// omitted), or comfyui[:WORKFLOW] to run the whole job in ComfyUI.
        #[arg(long)]
        model: Option<String>,
        /// Text-to-image model for the prompt: hunyuan-dit, flux-schnell, or
        /// comfyui[:WORKFLOW] to make the picture in ComfyUI.
        #[arg(long)]
        image_model: Option<String>,
        /// Also paint a colour texture.
        #[arg(long)]
        texture: bool,
        #[arg(long)]
        seed: Option<u64>,
        /// Target triangle count.
        #[arg(long)]
        faces: Option<u32>,
        /// Name for the library asset.
        #[arg(long)]
        name: Option<String>,
        /// Scale so the longest side is this many millimetres.
        #[arg(long)]
        size_mm: Option<f32>,
        /// A portrait whose face the character keeps (PuLID in ComfyUI).
        #[arg(long)]
        face: Option<PathBuf>,
        /// Draw front, side and back views and build the shape from all of them.
        #[arg(long)]
        multiview: bool,
        /// A rigged character (.glb) to rig the result like, keeping its
        /// skeleton and animations. Describe its rest pose in the prompt.
        #[arg(long, value_name = "GLB")]
        rig_like: Option<PathBuf>,
        /// Start of the material names when rigging, e.g. `surv_mei`.
        #[arg(long, requires = "rig_like")]
        materials: Option<String>,
        /// Draw over the reference as it looks instead of a grey figure, to
        /// remake that same character.
        #[arg(long, requires = "rig_like")]
        keep_look: bool,
        /// How much of the reference the picture repaints, 0 to 1 (default
        /// 0.88, or 0.7 with --keep-look).
        #[arg(long, requires = "rig_like")]
        repaint: Option<f32>,
        /// Also write the result here (.glb, .gltf, .stl or .3mf). A rigged
        /// character is copied as the rigged .glb.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Rig a model like an existing rigged character with Blender, keeping
    /// the character's skeleton, rest pose and animations.
    Rig {
        /// The model to rig (.glb).
        model: PathBuf,
        /// The rigged character to copy the skeleton from (.glb).
        #[arg(long)]
        like: PathBuf,
        /// Where to write the rigged model (.glb).
        #[arg(short, long)]
        output: PathBuf,
        /// Reduce the model to about this many triangles first.
        #[arg(long)]
        triangles: Option<u32>,
        /// Start of the material names, e.g. `surv_mei` (the reference's by default).
        #[arg(long)]
        materials: Option<String>,
    },
    /// Paint a sharp face onto a rigged character (.glb, as `rig` writes it):
    /// Blender renders the head close up, ComfyUI redraws it with the face in
    /// a portrait (FLUX with PuLID), and Blender paints it onto the texture.
    SharpenFace {
        /// The rigged, textured character (.glb).
        model: PathBuf,
        /// The portrait whose face to keep (PNG or JPEG).
        #[arg(long)]
        face: PathBuf,
        /// Where to write the result (.glb).
        #[arg(short, long)]
        output: PathBuf,
        /// Who it is, e.g. the prompt the character was made from.
        #[arg(long)]
        prompt: Option<String>,
        /// How much of the close-up to redraw, 0 to 1 (0.65 by default):
        /// more is sharper and more like the portrait.
        #[arg(long)]
        repaint: Option<f32>,
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Bring a rigged character's colours to a game character's levels: its
    /// skin and clothes are each darkened or lightened, keeping their hues, to
    /// the brightness of the same parts of LIKE. Can also shrink the texture.
    MatchColours {
        /// The rigged, textured character (.glb).
        model: PathBuf,
        /// The game's own character (.glb), usually the one it was rigged like.
        #[arg(long)]
        like: PathBuf,
        /// Where to write the result (.glb).
        #[arg(short, long)]
        output: PathBuf,
        /// How far to go, 0 to 1 (1 by default matches the brightness).
        #[arg(long)]
        strength: Option<f32>,
        /// Shrink the texture to at most this many pixels a side, e.g. 1024.
        #[arg(long)]
        max_texture: Option<u32>,
        /// Write the texture as a JPEG, several times smaller than a PNG.
        #[arg(long)]
        jpeg: bool,
    },
    /// List the asset library, optionally only names containing QUERY.
    Assets { query: Option<String> },
    /// Render a model to a PNG with the viewport renderer.
    Render {
        file: PathBuf,
        output: PathBuf,
        #[arg(long, default_value_t = 1024)]
        width: u32,
        #[arg(long, default_value_t = 768)]
        height: u32,
    },
    /// Run a JSON array of commands from a file, or stdin with "-".
    Run { script: PathBuf },
    /// Check the hub, hub-gen, model weights, ComfyUI, Blender and tools for
    /// newer versions, and show how to upgrade each.
    Updates {
        /// Run the upgrade of this item, by the id the check shows.
        #[arg(long, value_name = "ID")]
        apply: Option<String>,
        /// Also list what is up to date.
        #[arg(long)]
        all: bool,
    },
    /// Print the JSON Schema of a command, for scripts and agents.
    Schema,
    /// Serve the hub to Claude over MCP on stdin/stdout.
    Mcp,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    // Logs go to stderr so stdout stays clean for JSON and MCP.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("HUB_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    match run(cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    let json = cli.json;
    let mut session = Session::with_engines(hub_engines::from_env());
    match cli.command {
        Cmd::Info { file } => {
            session.execute(Command::Import { path: file })?;
            print(json, &session.execute(Command::ListObjects)?);
        }
        Cmd::Check { file, min_wall_mm } => {
            session.execute(Command::Import { path: file })?;
            let outcome = session.execute(Command::CheckPrintable {
                objects: None,
                min_wall_mm: Some(min_wall_mm),
            })?;
            print(json, &outcome);
            if outcome.reports.iter().any(|r| !r.printable) {
                return Ok(ExitCode::from(2));
            }
        }
        Cmd::Convert {
            input,
            output,
            size_mm,
            repair,
            remove_small_shells,
            hollow,
            place_on_floor,
        } => {
            session.execute(Command::Import { path: input })?;
            if repair {
                session.execute(Command::Repair {
                    objects: None,
                    fill_holes: None,
                    remove_small_shells,
                })?;
            }
            if let Some(size_mm) = size_mm {
                session.execute(Command::ScaleToSize {
                    objects: None,
                    size_mm,
                    axis: SizeAxis::Longest,
                })?;
            }
            if let Some(wall_mm) = hollow {
                print(
                    json,
                    &run_with_progress(
                        &mut session,
                        Command::Hollow {
                            objects: None,
                            wall_mm,
                            voxel_mm: None,
                        },
                    )?,
                );
            }
            if place_on_floor {
                session.execute(Command::PlaceOnFloor { objects: None })?;
            }
            print(
                json,
                &session.execute(Command::Export {
                    path: output,
                    objects: None,
                })?,
            );
        }
        Cmd::Render {
            file,
            output,
            width,
            height,
        } => {
            session.execute(Command::Import { path: file })?;
            let mut renderer = hub_render::Headless::new()?;
            let camera = hub_render::framing_camera(session.scene());
            let png = renderer.render_png(
                session.scene(),
                &camera,
                width,
                height,
                hub_render::FrameOptions::default(),
            )?;
            std::fs::write(&output, png)
                .with_context(|| format!("writing {}", output.display()))?;
            eprintln!("rendered with {}", renderer.adapter_name());
            print(
                json,
                &Outcome::summary(format!("Rendered {}", output.display())),
            );
        }
        Cmd::Generate {
            prompt,
            image,
            model,
            image_model,
            texture,
            seed,
            faces,
            name,
            size_mm,
            face,
            multiview,
            rig_like,
            materials,
            keep_look,
            repaint,
            output,
        } => {
            let rigged = rig_like.is_some();
            let outcome = run_with_progress(
                &mut session,
                Command::Generate {
                    prompt,
                    image,
                    model,
                    image_model,
                    texture,
                    seed,
                    faces,
                    name,
                    size_mm,
                    face,
                    multiview,
                    rig_like,
                    materials,
                    keep_look,
                    repaint,
                },
            )?;
            print(json, &outcome);
            match (output, &outcome.path) {
                // The scene has no skeletons, so copy the rigged file itself.
                (Some(path), Some(file)) if rigged => {
                    if !path
                        .extension()
                        .is_some_and(|e| e.eq_ignore_ascii_case("glb"))
                    {
                        bail!("a rigged character is written as .glb; use a .glb output");
                    }
                    std::fs::copy(file, &path)?;
                    eprintln!("wrote {}", path.display());
                }
                (Some(path), _) => print(
                    json,
                    &session.execute(Command::Export {
                        path,
                        objects: None,
                    })?,
                ),
                (None, _) => {}
            }
        }
        Cmd::Rig {
            model,
            like,
            output,
            triangles,
            materials,
        } => print(
            json,
            &run_with_progress(
                &mut session,
                Command::Rig {
                    model,
                    reference: like,
                    output,
                    triangles,
                    materials,
                },
            )?,
        ),
        Cmd::SharpenFace {
            model,
            face,
            output,
            prompt,
            repaint,
            seed,
        } => print(
            json,
            &run_with_progress(
                &mut session,
                Command::SharpenFace {
                    model,
                    face,
                    output,
                    prompt,
                    repaint,
                    seed,
                },
            )?,
        ),
        Cmd::MatchColours {
            model,
            like,
            output,
            strength,
            max_texture,
            jpeg,
        } => print(
            json,
            &run_with_progress(
                &mut session,
                Command::MatchColours {
                    model,
                    like,
                    output,
                    strength,
                    max_texture,
                    jpeg,
                },
            )?,
        ),
        Cmd::Assets { query } => print(json, &session.execute(Command::ListAssets { query })?),
        Cmd::Run { script } => return run_script(&mut session, &script, json),
        Cmd::Updates { apply, all } => return updates(apply.as_deref(), all, json),
        Cmd::Schema => {
            let schema = schemars::schema_for!(Command);
            println!("{}", serde_json::to_string_pretty(&schema)?);
        }
        Cmd::Mcp => {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            runtime
                .block_on(hub_mcp::serve_stdio())
                .map_err(|e| anyhow::anyhow!(e))?;
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn updates(apply: Option<&str>, all: bool, json: bool) -> Result<ExitCode> {
    use hub_updates::Status;

    eprintln!("checking for updates...");
    let report = hub_updates::check(&hub_updates::Setup::from_env());
    if let Some(id) = apply {
        let Some(item) = report.find(id) else {
            bail!("no item {id}; run `hub updates --all` to see every id");
        };
        let Some(upgrade) = item.upgrade.as_ref().filter(|u| !u.steps.is_empty()) else {
            bail!(
                "{} has nothing to run{}",
                item.name,
                item.upgrade
                    .as_ref()
                    .and_then(|u| u.link.as_deref())
                    .map(|l| format!("; see {l}"))
                    .unwrap_or_default()
            );
        };
        if let Some(why) = &item.blocked {
            bail!("{why}");
        }
        if let Some(note) = &upgrade.note {
            eprintln!("{note}");
        }
        if upgrade.rebuilds_hub {
            // This program is one of the files the build replaces, so the
            // steps wait for it to exit.
            let options = hub_updates::LaunchOptions {
                wait_for: Some(std::process::id()),
                then_start: None,
            };
            let launched = hub_updates::launch(&item.id, &item.name, upgrade, &options)
                .map_err(|e| anyhow::anyhow!(e))?;
            match launched.log {
                Some(log) => eprintln!("updating in the background; output in {}", log.display()),
                None => eprintln!("updating in a new window"),
            }
            return Ok(ExitCode::SUCCESS);
        }
        hub_updates::run_inline(upgrade).map_err(|e| anyhow::anyhow!(e))?;
        eprintln!("{} updated", item.name);
        return Ok(ExitCode::SUCCESS);
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(ExitCode::SUCCESS);
    }
    for piece in &report.pieces {
        let shown: Vec<_> = piece
            .items
            .iter()
            .filter(|i| all || i.worth_showing())
            .collect();
        let hidden = piece.items.len() - shown.len();
        match &piece.summary {
            Some(s) => println!("{}: {s}", piece.name),
            None => println!("{}", piece.name),
        }
        for item in shown {
            let mark = match item.status {
                Status::Current => "ok  ",
                Status::Available => "NEW ",
                Status::HeldBack => "held",
                Status::NotInstalled => "--  ",
                Status::Unknown => "?   ",
            };
            let versions = match (&item.installed, &item.latest) {
                (Some(a), Some(b)) if a != b => format!("{a} -> {b}"),
                (Some(a), _) => a.clone(),
                (None, Some(b)) => format!("newest {b}"),
                (None, None) => String::new(),
            };
            println!("  [{mark}] {}  {versions}", item.name);
            if let Some(d) = &item.detail {
                println!("         {d}");
            }
            if let Some(why) = &item.blocked {
                println!("         {why}");
            }
            if item.status != Status::Current
                && let Some(u) = &item.upgrade
            {
                if !u.steps.is_empty() && item.blocked.is_none() {
                    println!("         run: hub updates --apply {}", item.id);
                }
                if let Some(link) = &u.link {
                    println!("         see: {link}");
                }
            }
        }
        if hidden > 0 {
            println!("  {hidden} more up to date or not needed yet (--all shows them)");
        }
    }
    println!("\n{}", report.summary());
    Ok(ExitCode::SUCCESS)
}

/// Runs a command, showing a job's progress on stderr.
fn run_with_progress(session: &mut Session, command: Command) -> Result<Outcome> {
    match session.begin(command)? {
        Step::Done(outcome) => Ok(outcome),
        Step::Job(job) => {
            let interactive = std::io::IsTerminal::is_terminal(&std::io::stderr());
            let last = std::sync::Mutex::new(String::new());
            let output = job.run(&|fraction, stage| {
                let mut last = last
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if interactive {
                    eprint!("\r{:5.1}%  {stage:<48}", fraction * 100.0);
                } else if *last != stage {
                    eprintln!("{:5.1}%  {stage}", fraction * 100.0);
                }
                stage.clone_into(&mut last);
            });
            if interactive {
                eprintln!();
            }
            Ok(session.finish(output?)?)
        }
    }
}

fn run_script(session: &mut Session, script: &PathBuf, json: bool) -> Result<ExitCode> {
    let text = if script.as_os_str() == "-" {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        s
    } else {
        std::fs::read_to_string(script).with_context(|| format!("reading {}", script.display()))?
    };
    let commands: Vec<Command> =
        serde_json::from_str(&text).context("the script must be a JSON array of commands")?;
    if commands.is_empty() {
        bail!("the script contains no commands");
    }
    let mut outcomes = Vec::new();
    for (i, command) in commands.into_iter().enumerate() {
        match session.execute(command) {
            Ok(outcome) => {
                if !json {
                    print(false, &outcome);
                }
                outcomes.push(outcome);
            }
            Err(e) => {
                if json {
                    println!(
                        "{}",
                        serde_json::json!({ "outcomes": outcomes, "error": format!("command {i} failed: {e}") })
                    );
                }
                bail!("command {i} failed: {e}");
            }
        }
    }
    if json {
        println!("{}", serde_json::json!({ "outcomes": outcomes }));
    }
    Ok(ExitCode::SUCCESS)
}

fn print(json: bool, outcome: &Outcome) {
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(outcome).expect("outcomes serialize")
        );
        return;
    }
    println!("{}", outcome.summary);
    for o in &outcome.objects {
        let [w, d, h] = o.size_mm;
        println!(
            "  [{}] {}: {w:.2} × {d:.2} × {h:.2} mm (W × D × H), {} triangles",
            o.id, o.name, o.triangles
        );
    }
    for r in &outcome.reports {
        let [w, d, h] = r.size_mm;
        let verdict = if r.printable {
            "printable"
        } else {
            "NOT printable"
        };
        println!(
            "  [{}] {}: {verdict}; {w:.2} × {d:.2} × {h:.2} mm, {:.1} cm³, {} shell(s)",
            r.object,
            r.name,
            r.volume_mm3 / 1000.0,
            r.topology.shells
        );
        if let Some(w) = &r.wall {
            println!(
                "      walls: thinnest {:.2} mm; {:.1}% thinner than {} mm",
                w.thinnest_mm,
                w.thin_fraction * 100.0,
                w.min_wall_mm
            );
        }
        for issue in &r.issues {
            println!("      - {issue}");
        }
        for warning in &r.warnings {
            println!("      ! {warning}");
        }
    }
    for (i, r) in outcome.repairs.iter().enumerate() {
        println!(
            "  repair {i}: removed {} zero-area and {} duplicate triangles, flipped {}, \
             filled {} hole(s), removed {} small piece(s)",
            r.removed_degenerate,
            r.removed_duplicates,
            r.flipped_triangles,
            r.holes_filled,
            r.removed_shells
        );
    }
    for a in &outcome.assets {
        let latest = a.latest();
        println!(
            "  {} ({} version(s)){}",
            a.name,
            a.versions.len(),
            latest.map_or(String::new(), |v| {
                let [w, d, h] = v.size_mm;
                format!(
                    ", latest {w:.0} × {d:.0} × {h:.0} mm, {} triangles",
                    v.triangles
                )
            })
        );
    }
}
