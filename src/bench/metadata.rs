use serde::Serialize;

const GIT_COMMIT_HEX_LEN: usize = 40;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Metadata {
    pub git_commit: Option<String>,
    pub test_time: String,
    pub device: String,
    pub device_name: String,
    pub model_id: String,
    pub model_bytes: u64,
    pub model_parameter_count: u64,
}

pub fn git_commit(stdout: &[u8]) -> Option<String> {
    let commit = std::str::from_utf8(stdout).ok()?.trim();
    let is_commit = commit.len() == GIT_COMMIT_HEX_LEN
        && commit
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'));
    is_commit.then(|| commit.to_owned())
}

#[cfg(test)]
mod tests {
    use crate::bench::metadata::git_commit;

    #[test]
    fn git_commit_accepts_only_a_full_hash() {
        struct StdoutExpectedCase<T0, T1> {
            stdout: T0,
            expected: T1,
        }
        [
            StdoutExpectedCase {
                stdout: "1a507290862799c21d9795b56a2c5909de82b3ae\n",
                expected: Some("1a507290862799c21d9795b56a2c5909de82b3ae".to_owned()),
            },
            StdoutExpectedCase {
                stdout: "",
                expected: None,
            },
            StdoutExpectedCase {
                stdout: "1a50729\n",
                expected: None,
            },
            StdoutExpectedCase {
                stdout: "1A507290862799C21D9795B56A2C5909DE82B3AE\n",
                expected: None,
            },
            StdoutExpectedCase {
                stdout: "HEAD\nfatal: not a git repository\n",
                expected: None,
            },
        ]
        .into_iter()
        .for_each(
            |StdoutExpectedCase {
                 stdout,
                 expected,
             }| {
                assert_eq!(git_commit(stdout.as_bytes()), expected, "stdout {stdout:?}");
            },
        );
    }
}
