use std::collections::BTreeMap;
use std::fmt;
use std::time::{
    SystemTime,
    UNIX_EPOCH,
};

use serde::{
    Deserialize,
    Serialize,
};

pub(crate) const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Available<T> {
    pub(crate) value: Option<T>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) reason: Option<String>,
}

impl<T> Available<T> {
    pub(crate) const fn known(value: T) -> Self {
        Self {
            value: Some(value),
            reason: None,
        }
    }

    pub(crate) fn unknown(reason: impl fmt::Display) -> Self {
        Self {
            value: None,
            reason: Some(reason.to_string()),
        }
    }

    pub(crate) fn from_result<E: fmt::Display>(result: Result<T, E>) -> Self {
        match result {
            Ok(value) => Self::known(value),
            Err(reason) => Self::unknown(reason),
        }
    }

    pub(crate) fn from_option(
        value: Option<T>,
        reason: &str,
    ) -> Self {
        match value {
            Some(value) => Self::known(value),
            None => Self::unknown(reason),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Status {
    Succeeded,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Instrumentation {
    Untraced,
    Traced,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PhaseDto {
    Warmup,
    Measured,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExecutionOutcomeDto {
    Completed,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ChecksumStatus {
    Verified,
    Mismatch,
    NotVerified,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Report {
    pub(crate) schema_version: u32,
    pub(crate) run: Run,
    pub(crate) foundry: Foundry,
    pub(crate) runtime_repository: RepositoryProvenance,
    pub(crate) invocation: Invocation,
    pub(crate) environment: Environment,
    pub(crate) workload: Available<Workload>,
    pub(crate) protocol: Protocol,
    pub(crate) executions: Vec<Execution>,
    pub(crate) statistics: Available<Statistics>,
    pub(crate) memory: Available<Memory>,
    pub(crate) gpu_timing: Available<GpuTiming>,
    pub(crate) trace: Available<TraceInfo>,
    pub(crate) limitations: Vec<String>,
    pub(crate) failure: Option<Failure>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Run {
    pub(crate) id: String,
    pub(crate) started_at_utc: String,
    pub(crate) status: Status,
    pub(crate) instrumentation: Instrumentation,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Foundry {
    pub(crate) version: String,
    pub(crate) build: BuildProvenance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RepositoryState {
    pub(crate) dirty: bool,
    pub(crate) tracked_modifications: u64,
    pub(crate) untracked_files: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct BuildProvenance {
    pub(crate) foundry_version: String,
    pub(crate) commit: Available<String>,
    pub(crate) repository_state: Available<RepositoryState>,
    pub(crate) rustc: Available<String>,
    pub(crate) target: Available<String>,
    pub(crate) host: Available<String>,
    pub(crate) profile: Available<String>,
    pub(crate) opt_level: Available<String>,
    pub(crate) debug: Available<String>,
    pub(crate) debug_assertions: bool,
    pub(crate) panic: Available<String>,
    pub(crate) features: Vec<String>,
    pub(crate) rustflags: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RepositoryProvenance {
    pub(crate) directory: Available<String>,
    pub(crate) commit: Available<String>,
    pub(crate) repository_state: Available<RepositoryState>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Invocation {
    pub(crate) arguments: Vec<String>,
    pub(crate) working_directory: Available<String>,
    pub(crate) parameters: Parameters,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Parameters {
    pub(crate) backend: String,
    pub(crate) layers: u32,
    pub(crate) weight_mib: u64,
    pub(crate) iterations: u32,
    pub(crate) warmup: u32,
    pub(crate) bootstrap_resamples: u32,
    pub(crate) seed: u64,
    pub(crate) report: Option<String>,
    pub(crate) trace: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Environment {
    pub(crate) machine_model: Available<String>,
    pub(crate) cpu: Available<String>,
    pub(crate) gpu: Available<String>,
    pub(crate) physical_memory_bytes: Available<u64>,
    pub(crate) os_name: Available<String>,
    pub(crate) os_version: Available<String>,
    pub(crate) os_build: Available<String>,
    pub(crate) architecture: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Workload {
    pub(crate) layers: u32,
    pub(crate) weight_bytes: u64,
    pub(crate) payload_bytes: u64,
    pub(crate) alignment_bytes: u64,
    pub(crate) slot_bytes: u64,
    pub(crate) slot_budget_bytes: u64,
    pub(crate) configurations: Vec<ConfigurationPlan>,
    pub(crate) device: DeviceLimits,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ConfigurationPlan {
    pub(crate) buffers: u32,
    pub(crate) commands: u64,
    pub(crate) planned_slot_peak_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DeviceLimits {
    pub(crate) recommended_working_set_bytes: u64,
    pub(crate) max_buffer_length_bytes: u64,
    pub(crate) unified_memory: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Protocol {
    pub(crate) configurations: Vec<u32>,
    pub(crate) configuration_order: String,
    pub(crate) warmup_rounds: u32,
    pub(crate) measured_rounds: u32,
    pub(crate) timer: String,
    pub(crate) timed_region: String,
    pub(crate) untimed: Vec<String>,
    pub(crate) units: Units,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Units {
    pub(crate) durations: String,
    pub(crate) sizes: String,
    pub(crate) throughput: String,
    pub(crate) trace_timestamps: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Execution {
    pub(crate) sequence: u32,
    pub(crate) phase: PhaseDto,
    pub(crate) round: u32,
    pub(crate) position: u32,
    pub(crate) buffers: u32,
    pub(crate) elapsed_ns: Option<u64>,
    pub(crate) outcome: ExecutionOutcomeDto,
    pub(crate) checksums: ChecksumStatus,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct IntervalDto {
    pub(crate) lower: f64,
    pub(crate) upper: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Statistics {
    pub(crate) method: Method,
    pub(crate) configurations: Vec<ConfigurationStatistics>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Method {
    pub(crate) samples: String,
    pub(crate) median: String,
    pub(crate) mean: String,
    pub(crate) standard_deviation: String,
    pub(crate) confidence_level: f64,
    pub(crate) interval: String,
    pub(crate) quantile: String,
    pub(crate) probabilities: [f64; 2],
    pub(crate) resamples: u32,
    pub(crate) seed: u64,
    pub(crate) generator: String,
    pub(crate) bounded_draws: String,
    pub(crate) traversal: String,
    pub(crate) speedup: String,
    pub(crate) recalculation_tolerance: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct ConfigurationStatistics {
    pub(crate) buffers: u32,
    pub(crate) samples: u32,
    pub(crate) median_ns: u64,
    pub(crate) min_ns: u64,
    pub(crate) max_ns: u64,
    pub(crate) mean_ns: Available<f64>,
    pub(crate) standard_deviation_ns: Available<f64>,
    pub(crate) median_ci_ns: Available<IntervalDto>,
    pub(crate) mean_ci_ns: Available<IntervalDto>,
    pub(crate) throughput_gib_s: Available<f64>,
    pub(crate) speedup: Available<f64>,
    pub(crate) speedup_ci: Available<IntervalDto>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Memory {
    pub(crate) cpu_payload_bytes: u64,
    pub(crate) cpu_payload_lifetime: String,
    pub(crate) slot_budget_bytes: u64,
    pub(crate) accounting: Vec<String>,
    pub(crate) configurations: Vec<ConfigurationMemory>,
    pub(crate) executions: Vec<ExecutionMemory>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ConfigurationMemory {
    pub(crate) buffers: u32,
    pub(crate) executions: u32,
    pub(crate) planned_slot_peak_bytes: u64,
    pub(crate) planned_peak_within_budget: bool,
    pub(crate) observed_private_slot_peak_requested_bytes_max: u64,
    pub(crate) observed_private_slot_peak_allocated_bytes_max: u64,
    pub(crate) retained_private_slot_bytes_beyond_plan_max: u64,
    pub(crate) categories: Vec<CategoryPeak>,
    pub(crate) device_current_allocated_sampled_max_bytes: u64,
    pub(crate) diagnostic_storage_bytes_max: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CategoryPeak {
    pub(crate) category: String,
    pub(crate) peak_requested_bytes_max: u64,
    pub(crate) peak_allocated_bytes_max: u64,
    pub(crate) allocations_max: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ExecutionMemory {
    pub(crate) sequence: u32,
    pub(crate) buffers: u32,
    pub(crate) planned_slot_peak_bytes: u64,
    pub(crate) observed_private_slot_peak_requested_bytes: u64,
    pub(crate) retained_private_slot_bytes_beyond_plan: u64,
    pub(crate) categories: Vec<CategoryUsage>,
    pub(crate) tracked_peak_requested_bytes: u64,
    pub(crate) tracked_peak_allocated_bytes: u64,
    pub(crate) snapshots: Vec<Snapshot>,
    pub(crate) live_allocations_at_extraction: u64,
    pub(crate) device_current_allocated_sampled_max_bytes: u64,
    pub(crate) device_samples: u64,
    pub(crate) diagnostic_storage_bytes: u64,
    pub(crate) diagnostic_records_truncated: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CategoryUsage {
    pub(crate) category: String,
    pub(crate) allocations: u64,
    pub(crate) frees: u64,
    pub(crate) peak_requested_bytes: u64,
    pub(crate) peak_allocated_bytes: u64,
    pub(crate) live_requested_bytes_at_extraction: u64,
    pub(crate) live_allocated_bytes_at_extraction: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Snapshot {
    pub(crate) boundary: String,
    pub(crate) at_ns: Option<u64>,
    pub(crate) live_allocations: u64,
    pub(crate) live_requested_bytes: BTreeMap<String, u64>,
    pub(crate) live_allocated_bytes: BTreeMap<String, u64>,
    pub(crate) device_current_allocated_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct GpuTiming {
    pub(crate) granularity: String,
    pub(crate) semantics: String,
    pub(crate) valid: u64,
    pub(crate) invalid: u64,
    pub(crate) truncated: u64,
    pub(crate) pending_at_extraction: u64,
    pub(crate) invalid_reasons: BTreeMap<String, u64>,
    pub(crate) gate: String,
    pub(crate) executions: Vec<GpuExecution>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct GpuExecution {
    pub(crate) sequence: u32,
    pub(crate) buffers: u32,
    pub(crate) copy_command_buffers: u64,
    pub(crate) copy_busy_ns: u64,
    pub(crate) compute_command_buffers: u64,
    pub(crate) compute_busy_ns: u64,
    pub(crate) overlapping_command_buffers: u64,
    pub(crate) first_start_to_last_end_ns: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TraceInfo {
    pub(crate) file: String,
    pub(crate) format: String,
    pub(crate) clock: Clock,
    pub(crate) tracks: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Clock {
    pub(crate) source: String,
    pub(crate) timebase_numer: u32,
    pub(crate) timebase_denom: u32,
    pub(crate) origin_host_ns: u64,
    pub(crate) gpu_conversion: String,
    pub(crate) trace_unit: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Failure {
    pub(crate) stage: String,
    pub(crate) message: String,
    pub(crate) phase: Option<PhaseDto>,
    pub(crate) round: Option<u32>,
    pub(crate) position: Option<u32>,
    pub(crate) buffers: Option<u32>,
    pub(crate) completed_executions: u32,
}

fn civil(days: i64) -> Option<(i64, u32, u32)> {
    let shifted = days.checked_add(719_468)?;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.checked_sub(era.checked_mul(146_097)?)?;
    let year_of_era = day_of_era
        .checked_sub(day_of_era.checked_div(1460)?)?
        .checked_add(day_of_era.checked_div(36_524)?)?
        .checked_sub(day_of_era.checked_div(146_096)?)?
        .checked_div(365)?;
    let day_of_year = day_of_era.checked_sub(
        year_of_era
            .checked_mul(365)?
            .checked_add(year_of_era.checked_div(4)?)?
            .checked_sub(year_of_era.checked_div(100)?)?,
    )?;
    let month_index = day_of_year
        .checked_mul(5)?
        .checked_add(2)?
        .checked_div(153)?;
    let day = day_of_year
        .checked_sub(
            month_index
                .checked_mul(153)?
                .checked_add(2)?
                .checked_div(5)?,
        )?
        .checked_add(1)?;
    let month = if month_index < 10 {
        month_index.checked_add(3)?
    } else {
        month_index.checked_sub(9)?
    };
    let year = year_of_era
        .checked_add(era.checked_mul(400)?)?
        .checked_add(i64::from(month <= 2))?;
    Some((year, u32::try_from(month).ok()?, u32::try_from(day).ok()?))
}

pub(crate) fn utc_timestamp(time: SystemTime) -> Option<String> {
    let since = time.duration_since(UNIX_EPOCH).ok()?;
    let seconds = i64::try_from(since.as_secs()).ok()?;
    let days = seconds.div_euclid(86_400);
    let of_day = seconds.rem_euclid(86_400);
    let (year, month, day) = civil(days)?;
    Some(format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:06}Z",
        of_day.checked_div(3600)?,
        of_day.rem_euclid(3600).checked_div(60)?,
        of_day.rem_euclid(60),
        since.subsec_micros()
    ))
}

pub(crate) fn run_id(
    time: SystemTime,
    process: u32,
) -> String {
    let compact = utc_timestamp(time)
        .map(|stamp| stamp.replace(['-', ':', '.'], ""))
        .unwrap_or_else(|| "unknown-time".to_owned());
    format!("{compact}-{process}")
}

#[cfg(test)]
mod tests {
    use std::time::{
        Duration,
        UNIX_EPOCH,
    };

    use super::{
        Available,
        civil,
        run_id,
        utc_timestamp,
    };

    #[test]
    fn utc_timestamps_follow_the_civil_calendar() {
        assert_eq!(civil(0), Some((1970, 1, 1)), "the epoch");
        assert_eq!(civil(11_016), Some((2000, 2, 29)), "a leap day");
        assert_eq!(civil(20_731), Some((2026, 10, 5)), "a recent date");
        assert_eq!(
            utc_timestamp(UNIX_EPOCH + Duration::new(1_791_201_845, 123_456_789)).as_deref(),
            Some("2026-10-05T12:04:05.123456Z"),
            "seconds and microseconds"
        );
        assert_eq!(
            run_id(UNIX_EPOCH + Duration::from_secs(1_791_201_845), 42),
            "20261005T120405000000Z-42",
            "the run id is compact and names the process"
        );
    }

    #[test]
    fn unavailable_values_carry_a_reason() -> Result<(), Box<dyn std::error::Error>> {
        let known = serde_json::to_value(Available::known(3_u64))?;
        assert_eq!(known, serde_json::json!({ "value": 3 }), "known values");
        let unknown = serde_json::to_value(Available::<u64>::unknown("no sensor"))?;
        assert_eq!(
            unknown,
            serde_json::json!({ "value": null, "reason": "no sensor" }),
            "unknown values are null with a reason"
        );
        Ok(())
    }
}
