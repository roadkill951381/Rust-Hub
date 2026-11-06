//! Version numbers as Python packages, ComfyUI, Blender and uv write them,
//! with Python's version ranges (`>=4.48,<5`), as far as ordering needs.

use std::cmp::Ordering;

/// A version such as `2.11.0+cu128`, `1.0.0rc1` or `v0.42.1`. The local part
/// after `+` is ignored, so a CUDA build of a release equals the release.
#[derive(Debug, Clone)]
pub struct Version {
    release: Vec<u64>,
    /// Development (0), alpha (1), beta (2), release candidate (3), final (4)
    /// or post release (5), and its number.
    phase: (u8, u64),
}

impl Version {
    pub fn parse(text: &str) -> Option<Self> {
        let text = text.trim().to_ascii_lowercase();
        let text = text.strip_prefix('v').unwrap_or(&text);
        let text = text.split('+').next().unwrap_or_default();
        let text = text.rsplit('!').next().unwrap_or_default();
        let end = text
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap_or(text.len());
        let release: Vec<u64> = text[..end]
            .trim_end_matches('.')
            .split('.')
            .map(str::parse)
            .collect::<Result<_, _>>()
            .ok()?;
        let rest = text[end..].trim_start_matches(['.', '-', '_']);
        let label = rest.trim_end_matches(|c: char| c.is_ascii_digit());
        let number = rest[label.len()..].parse().unwrap_or(0);
        let label = label.trim_end_matches(['.', '-', '_']);
        let phase = match label {
            "" if rest.is_empty() => (4, 0),
            "dev" => (0, number),
            "a" | "alpha" => (1, number),
            "b" | "beta" => (2, number),
            "rc" | "c" | "pre" | "preview" => (3, number),
            "post" | "rev" | "r" | "" => (5, number),
            // e.g. "a1.dev2": count it as the pre-release it starts with.
            other if other.starts_with('a') => (1, 0),
            other if other.starts_with('b') => (2, 0),
            other if other.starts_with("rc") => (3, 0),
            _ => (4, 0),
        };
        Some(Self { release, phase })
    }

    /// Alphas, betas, release candidates and development builds.
    pub fn is_prerelease(&self) -> bool {
        self.phase.0 < 4
    }

    fn part(&self, i: usize) -> u64 {
        self.release.get(i).copied().unwrap_or(0)
    }

    fn cmp_release(&self, other: &Self) -> Ordering {
        let n = self.release.len().max(other.release.len());
        (0..n)
            .map(|i| self.part(i).cmp(&other.part(i)))
            .find(|o| o.is_ne())
            .unwrap_or(Ordering::Equal)
    }

    /// Whether `self` starts with `prefix`'s numbers (`5.2.3` starts with `5.2`).
    fn starts_with(&self, prefix: &Self) -> bool {
        prefix
            .release
            .iter()
            .enumerate()
            .all(|(i, p)| self.part(i) == *p)
    }
}

impl PartialEq for Version {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Version {}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp_release(other).then(self.phase.cmp(&other.phase))
    }
}

/// Whether `version` is inside a Python range such as `>=4.48,<5`. An empty
/// range allows anything; a clause this does not understand is ignored.
pub fn allows(range: &str, version: &Version) -> bool {
    range
        .split(',')
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .all(|clause| {
            let op_len = clause
                .find(|c: char| !matches!(c, '<' | '>' | '=' | '!' | '~'))
                .unwrap_or(clause.len());
            let (op, bound) = clause.split_at(op_len);
            let bound = bound.trim();
            if let Some(prefix) = bound.strip_suffix(".*") {
                let Some(prefix) = Version::parse(prefix) else {
                    return true;
                };
                return match op {
                    "==" => version.starts_with(&prefix),
                    "!=" => !version.starts_with(&prefix),
                    _ => true,
                };
            }
            let Some(bound) = Version::parse(bound) else {
                return true;
            };
            match op {
                ">=" => *version >= bound,
                ">" => *version > bound,
                "<=" => *version <= bound,
                // `<5` leaves out 5's pre-releases too.
                "<" => version.cmp_release(&bound).is_lt(),
                "==" | "===" => *version == bound,
                "!=" => *version != bound,
                "~=" => {
                    let mut prefix = bound.clone();
                    if prefix.release.len() > 1 {
                        prefix.release.pop();
                    }
                    prefix.phase = (4, 0);
                    *version >= bound && version.starts_with(&prefix)
                }
                _ => true,
            }
        })
}

/// A version as written, and parsed.
pub type Found = Option<(String, Version)>;

/// The newest final release among `versions`, and the newest that `range`
/// allows.
pub fn newest<'a>(versions: impl IntoIterator<Item = &'a str>, range: &str) -> (Found, Found) {
    let mut any: Found = None;
    let mut allowed: Found = None;
    for text in versions {
        let Some(v) = Version::parse(text) else {
            continue;
        };
        if v.is_prerelease() {
            continue;
        }
        if any.as_ref().is_none_or(|(_, best)| v > *best) {
            any = Some((text.to_owned(), v.clone()));
        }
        if allows(range, &v) && allowed.as_ref().is_none_or(|(_, best)| v > *best) {
            allowed = Some((text.to_owned(), v));
        }
    }
    (any, allowed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> Version {
        Version::parse(s).unwrap()
    }

    #[test]
    fn versions_order_like_python() {
        assert!(v("4.57.6") < v("5.0.0"));
        assert!(v("5.0.0rc1") < v("5.0.0"));
        assert!(v("5.0.0.dev3") < v("5.0.0a1"));
        assert!(v("5.0.0") < v("5.0.0.post1"));
        assert!(v("1.10") > v("1.9.9"));
        assert_eq!(v("2.11.0+cu128"), v("2.11.0"));
        assert_eq!(v("v0.42.1"), v("0.42.1"));
        assert_eq!(v("5.2"), v("5.2.0"));
        assert!(v("1.0.0rc1").is_prerelease());
        assert!(!v("2.11.0+cu128").is_prerelease());
        assert!(Version::parse("main").is_none());
    }

    #[test]
    fn ranges_hold_back_capped_packages() {
        assert!(allows(">=4.48,<5", &v("4.57.6")));
        assert!(!allows(">=4.48,<5", &v("5.22.1")));
        assert!(!allows("<5", &v("5.0.0rc1")));
        assert!(allows("", &v("9")));
        assert!(allows("==2.*", &v("2.11.0")));
        assert!(!allows("!=2.11.0", &v("2.11.0+cu128")));
        assert!(allows("~=1.4.2", &v("1.4.9")));
        assert!(!allows("~=1.4.2", &v("1.5.0")));
        assert!(allows("~=1.4", &v("1.9")));
    }

    #[test]
    fn newest_picks_final_releases() {
        let all = ["4.48.0", "4.57.6", "5.22.1", "5.23.0rc1", "junk"];
        let (any, allowed) = newest(all, ">=4.48,<5");
        assert_eq!(any.unwrap().0, "5.22.1");
        assert_eq!(allowed.unwrap().0, "4.57.6");
    }
}
