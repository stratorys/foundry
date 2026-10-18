use serde::Serialize;

const GIT_COMMIT_HEX_LEN: usize = 40;
const SECONDS_PER_DAY: u64 = 86_400;
const SECONDS_PER_HOUR: u64 = 3_600;
const SECONDS_PER_MINUTE: u64 = 60;
const DAYS_FROM_0000_03_01_TO_EPOCH: u64 = 719_468;
const DAYS_PER_ERA: u64 = 146_097;
const YEARS_PER_ERA: u64 = 400;

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

pub fn rfc3339_utc(unix_s: u64) -> Option<String> {
    let second_of_day = unix_s.checked_rem(SECONDS_PER_DAY)?;
    let day_shifted = unix_s
        .checked_div(SECONDS_PER_DAY)?
        .checked_add(DAYS_FROM_0000_03_01_TO_EPOCH)?;
    let era = day_shifted.checked_div(DAYS_PER_ERA)?;
    let day_of_era = day_shifted.checked_rem(DAYS_PER_ERA)?;
    let year_of_era = day_of_era
        .checked_sub(day_of_era.checked_div(1_460)?)?
        .checked_add(day_of_era.checked_div(36_524)?)?
        .checked_sub(day_of_era.checked_div(146_096)?)?
        .checked_div(365)?;
    let day_of_year = day_of_era.checked_sub(
        year_of_era
            .checked_mul(365)?
            .checked_add(year_of_era.checked_div(4)?)?
            .checked_sub(year_of_era.checked_div(100)?)?,
    )?;
    let month_from_march = day_of_year
        .checked_mul(5)?
        .checked_add(2)?
        .checked_div(153)?;
    let day = day_of_year
        .checked_sub(
            month_from_march
                .checked_mul(153)?
                .checked_add(2)?
                .checked_div(5)?,
        )?
        .checked_add(1)?;
    let (month, year_offset) = if month_from_march < 10 {
        (month_from_march.checked_add(3)?, 0)
    } else {
        (month_from_march.checked_sub(9)?, 1)
    };
    let year = era
        .checked_mul(YEARS_PER_ERA)?
        .checked_add(year_of_era)?
        .checked_add(year_offset)?;
    let hour = second_of_day.checked_div(SECONDS_PER_HOUR)?;
    let minute = second_of_day
        .checked_rem(SECONDS_PER_HOUR)?
        .checked_div(SECONDS_PER_MINUTE)?;
    let second = second_of_day.checked_rem(SECONDS_PER_MINUTE)?;
    Some(format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z"
    ))
}

#[cfg(test)]
mod tests {
    use crate::bench::metadata::{
        git_commit,
        rfc3339_utc,
    };

    #[test]
    fn rfc3339_utc_formats_unix_seconds() {
        struct UnixExpectedCase {
            unix_s: u64,
            expected: &'static str,
        }
        [
            UnixExpectedCase {
                unix_s: 0,
                expected: "1970-01-01T00:00:00Z",
            },
            UnixExpectedCase {
                unix_s: 951_782_400,
                expected: "2000-02-29T00:00:00Z",
            },
            UnixExpectedCase {
                unix_s: 1_709_164_800,
                expected: "2024-02-29T00:00:00Z",
            },
            UnixExpectedCase {
                unix_s: 1_760_521_949,
                expected: "2025-10-15T09:52:29Z",
            },
            UnixExpectedCase {
                unix_s: 4_102_444_799,
                expected: "2099-12-31T23:59:59Z",
            },
        ]
        .into_iter()
        .for_each(
            |UnixExpectedCase {
                 unix_s,
                 expected,
             }| {
                assert_eq!(
                    rfc3339_utc(unix_s),
                    Some(expected.to_owned()),
                    "{unix_s} seconds after the epoch"
                );
            },
        );
    }

    #[test]
    fn git_commit_accepts_only_a_full_hash() {
        struct StdoutExpectedCase {
            stdout: &'static str,
            expected: Option<String>,
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
