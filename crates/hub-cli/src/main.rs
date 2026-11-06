//! `hub`: the Rust-Hub command line.
//!
//! Every subcommand is a thin wrapper over [`hub_core::Command`], the same
//! commands the desktop app and the MCP server run.

use std::io::Read as _;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context as _, Result, bail};
use clap::{Parser, Subcommand};
use hub_core::{Command, Outcome, Session, SizeAxis};

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
    Check { file: PathBuf },
    /// Convert between .glb, .gltf, .stl and .3mf, optionally preparing for print.
    Convert {
        input: PathBuf,
        output: PathBuf,
        /// Scale uniformly so the longest side is this many millimetres.
        #[arg(long)]
        size_mm: Option<f32>,
        /// Repair winding, zero-area and duplicate faces first.
        #[arg(long)]
        repair: bool,
        /// Centre on the origin and rest on the floor.
        #[arg(long)]
        place_on_floor: bool,
    },
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
    let mut session = Session::new();
    match cli.command {
        Cmd::Info { file } => {
            session.execute(Command::Import { path: file })?;
            print(json, &session.execute(Command::ListObjects)?);
        }
        Cmd::Check { file } => {
            session.execute(Command::Import { path: file })?;
            let outcome = session.execute(Command::CheckPrintable { objects: None })?;
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
            place_on_floor,
        } => {
            session.execute(Command::Import { path: input })?;
            if repair {
                session.execute(Command::Repair { objects: None })?;
            }
            if let Some(size_mm) = size_mm {
                session.execute(Command::ScaleToSize {
                    objects: None,
                    size_mm,
                    axis: SizeAxis::Longest,
                })?;
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
        Cmd::Run { script } => return run_script(&mut session, &script, json),
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
        for issue in &r.issues {
            println!("      - {issue}");
        }
    }
    for (i, r) in outcome.repairs.iter().enumerate() {
        println!(
            "  repair {i}: removed {} zero-area and {} duplicate triangles, flipped {}",
            r.removed_degenerate, r.removed_duplicates, r.flipped_triangles
        );
    }
}
