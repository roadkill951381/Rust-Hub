//! Running the programs that report versions, and the upgrade steps.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::{Step, Upgrade};

/// Runs a program without a console window and returns its trimmed output
/// (stdout, else stderr) when it succeeds within `timeout`.
pub fn output(
    program: &Path,
    args: &[&str],
    dir: Option<&Path>,
    timeout: Duration,
) -> Option<String> {
    capture(program, args, dir, timeout, false)
}

/// As [`output`], but keeps the output of a failing exit too (`rustup check`
/// exits with 100 when an update is available).
pub fn output_any_exit(
    program: &Path,
    args: &[&str],
    dir: Option<&Path>,
    timeout: Duration,
) -> Option<String> {
    capture(program, args, dir, timeout, true)
}

fn capture(
    program: &Path,
    args: &[&str],
    dir: Option<&Path>,
    timeout: Duration,
    any_exit: bool,
) -> Option<String> {
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        // A check must never stop to ask for a password.
        .env("GIT_TERMINAL_PROMPT", "0");
    if let Some(dir) = dir {
        command.current_dir(dir);
    }
    hide_window(&mut command);
    let mut child = command.spawn().ok()?;
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if start.elapsed() < timeout => std::thread::sleep(Duration::from_millis(20)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                tracing::warn!(program = %program.display(), ?args, "timed out");
                return None;
            }
        }
    }
    let out = child.wait_with_output().ok()?;
    if !out.status.success() && !any_exit {
        tracing::debug!(
            program = %program.display(),
            ?args,
            stderr = %String::from_utf8_lossy(&out.stderr).trim(),
            "failed"
        );
        return None;
    }
    let text = if out.stdout.iter().all(u8::is_ascii_whitespace) {
        out.stderr
    } else {
        out.stdout
    };
    Some(String::from_utf8_lossy(&text).trim().to_owned())
}

fn hide_window(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    let _ = command;
}

/// How the upgrade's script was started.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Launched {
    pub script: PathBuf,
    /// Where its output goes when it has no window of its own.
    pub log: Option<PathBuf>,
}

/// Options for [`launch`].
#[derive(Debug, Clone, Default)]
pub struct LaunchOptions {
    /// Wait for this process (the hub) to exit before the first step.
    pub wait_for: Option<u32>,
    /// Start this program when every step succeeded (the rebuilt hub).
    pub then_start: Option<PathBuf>,
}

/// Writes the upgrade's steps to a script and starts it: in its own console
/// window on Windows, so the person sees each step and can read errors; in
/// the background with a log elsewhere.
pub fn launch(
    id: &str,
    name: &str,
    upgrade: &Upgrade,
    options: &LaunchOptions,
) -> Result<Launched, String> {
    let dir = std::env::temp_dir().join("rust-hub").join("updates");
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    let file: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let windows = cfg!(windows);
    let script = dir.join(format!("{file}.{}", if windows { "cmd" } else { "sh" }));
    let text = if windows {
        cmd_script(name, upgrade, options)
    } else {
        sh_script(name, upgrade, options)
    };
    std::fs::write(&script, text)
        .map_err(|e| format!("could not write {}: {e}", script.display()))?;

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        const CREATE_NEW_CONSOLE: u32 = 0x0000_0010;
        Command::new("cmd")
            .arg("/c")
            .arg(&script)
            .creation_flags(CREATE_NEW_CONSOLE)
            .spawn()
            .map_err(|e| format!("could not start the update: {e}"))?;
        Ok(Launched { script, log: None })
    }
    #[cfg(not(windows))]
    {
        let log = script.with_extension("log");
        let out = std::fs::File::create(&log)
            .map_err(|e| format!("could not create {}: {e}", log.display()))?;
        let err = out.try_clone().map_err(|e| e.to_string())?;
        Command::new("sh")
            .arg(&script)
            .stdin(Stdio::null())
            .stdout(out)
            .stderr(err)
            .spawn()
            .map_err(|e| format!("could not start the update: {e}"))?;
        Ok(Launched {
            script,
            log: Some(log),
        })
    }
}

/// Runs the steps here, with their output on this terminal.
pub fn run_inline(upgrade: &Upgrade) -> Result<(), String> {
    for step in &upgrade.steps {
        eprintln!("> {step}");
        let mut command = Command::new(&step.program);
        command.args(&step.args);
        if let Some(dir) = &step.dir {
            command.current_dir(dir);
        }
        let status = command
            .status()
            .map_err(|e| format!("could not run {}: {e}", step.program))?;
        if !status.success() {
            return Err(format!("`{step}` failed ({status})"));
        }
    }
    Ok(())
}

fn quote_cmd(arg: &str) -> String {
    if !arg.is_empty() && !arg.contains([' ', '&', '(', ')', '^', '|', '<', '>', '"']) {
        arg.to_owned()
    } else {
        format!("\"{}\"", arg.replace('"', "\"\""))
    }
}

fn quote_sh(arg: &str) -> String {
    if !arg.is_empty()
        && arg
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_./:=@+,".contains(c))
    {
        arg.to_owned()
    } else {
        format!("'{}'", arg.replace('\'', r"'\''"))
    }
}

/// The step as a shell line, with arguments quoted for this platform.
pub fn render(step: &Step) -> String {
    let quote = if cfg!(windows) { quote_cmd } else { quote_sh };
    std::iter::once(step.program.as_str())
        .chain(step.args.iter().map(String::as_str))
        .map(quote)
        .collect::<Vec<_>>()
        .join(" ")
}

fn cmd_script(name: &str, upgrade: &Upgrade, options: &LaunchOptions) -> String {
    let mut s = String::from("@echo off\r\n");
    let _ = write!(
        s,
        "title Rust-Hub update: {name}\r\necho Updating {name}\r\n"
    );
    if let Some(note) = &upgrade.note {
        let _ = write!(s, "echo {note}\r\n");
    }
    if let Some(pid) = options.wait_for {
        let _ = write!(
            s,
            "echo Waiting for Rust-Hub to close...\r\n\
             powershell -NoProfile -Command \"Wait-Process -Id {pid} -ErrorAction SilentlyContinue\"\r\n"
        );
    }
    for step in &upgrade.steps {
        s.push_str("echo.\r\n");
        if let Some(dir) = &step.dir {
            let dir = quote_cmd(&dir.display().to_string());
            let _ = write!(s, "cd /d {dir} || goto :failed\r\n");
        }
        let line = render(step);
        let shown: String = line
            .chars()
            .flat_map(|c| "&|<>()^".contains(c).then_some('^').into_iter().chain([c]))
            .collect();
        let _ = write!(s, "echo ^> {shown}\r\ncall {line} || goto :failed\r\n");
    }
    s.push_str("echo.\r\necho Done.\r\n");
    if let Some(program) = &options.then_start {
        let _ = write!(
            s,
            "start \"\" {}\r\n",
            quote_cmd(&program.display().to_string())
        );
    }
    s.push_str(
        "pause\r\nexit /b 0\r\n:failed\r\necho.\r\necho The update stopped because the step above failed.\r\npause\r\nexit /b 1\r\n",
    );
    s
}

fn sh_script(name: &str, upgrade: &Upgrade, options: &LaunchOptions) -> String {
    let mut s = String::from("#!/bin/sh\nset -e\n");
    let _ = writeln!(s, "echo {}", quote_sh(&format!("Updating {name}")));
    if let Some(pid) = options.wait_for {
        let _ = writeln!(s, "while kill -0 {pid} 2>/dev/null; do sleep 1; done");
    }
    for step in &upgrade.steps {
        if let Some(dir) = &step.dir {
            let _ = writeln!(s, "cd {}", quote_sh(&dir.display().to_string()));
        }
        let line = render(step);
        let _ = writeln!(s, "echo {}\n{line}", quote_sh(&format!("> {line}")));
    }
    s.push_str("echo Done.\n");
    if let Some(program) = &options.then_start {
        let _ = writeln!(
            s,
            "{} >/dev/null 2>&1 &",
            quote_sh(&program.display().to_string())
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scripts_stop_at_the_first_failure() {
        let upgrade = Upgrade {
            steps: vec![
                Step::new(
                    Some(Path::new("/src/Rust Hub")),
                    "git",
                    ["pull", "--ff-only"],
                ),
                Step::new(None, "cargo", ["build", "--release"]),
            ],
            note: None,
            link: None,
            rebuilds_hub: true,
        };
        let options = LaunchOptions {
            wait_for: Some(42),
            then_start: Some(PathBuf::from("/src/hub-app")),
        };
        let cmd = cmd_script("Rust-Hub", &upgrade, &options);
        assert!(
            cmd.contains("cd /d \"/src/Rust Hub\" || goto :failed"),
            "{cmd}"
        );
        assert!(
            cmd.contains("call git pull --ff-only || goto :failed"),
            "{cmd}"
        );
        assert!(cmd.contains("Wait-Process -Id 42"), "{cmd}");
        assert!(cmd.contains("start \"\" /src/hub-app"), "{cmd}");
        let sh = sh_script("Rust-Hub", &upgrade, &options);
        assert!(sh.starts_with("#!/bin/sh\nset -e\n"));
        assert!(sh.contains("cd '/src/Rust Hub'\n"), "{sh}");
        assert!(sh.contains("while kill -0 42"), "{sh}");
    }
}
