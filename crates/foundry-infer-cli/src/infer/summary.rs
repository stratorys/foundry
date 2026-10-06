use std::collections::BTreeMap;

use crate::bench::stats::lossy;
use crate::infer::error::InferError;
use crate::infer::report::{
    Consistency,
    Execution,
    MetricSummary,
    Mismatch,
};

const NANOS_PER_SECOND: f64 = 1e9;

pub(crate) fn decode_tokens_per_s(token_available_ns: &[u64]) -> Option<f64> {
    let first = *token_available_ns.first()?;
    let last = *token_available_ns.last()?;
    let intervals = token_available_ns.len().checked_sub(1)?;
    let span_ns = last.checked_sub(first).filter(|span| *span > 0)?;
    let intervals = lossy(u128::try_from(intervals).ok()?);
    Some(intervals / (lossy(u128::from(span_ns)) / NANOS_PER_SECOND))
}

fn summarize(
    unit: &'static str,
    values: &[f64],
) -> Option<MetricSummary> {
    let count = values.len();
    if count < 2 {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = lossy(u128::try_from(count).ok()?);
    let middle = count.checked_div(2)?;
    let upper = *sorted.get(middle)?;
    let median = if count.checked_rem(2)? == 0 {
        (sorted.get(middle.checked_sub(1)?)? + upper) / 2.0
    } else {
        upper
    };
    let mean = sorted.iter().sum::<f64>() / n;
    let squares = sorted
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>();
    Some(MetricSummary {
        unit,
        count,
        median,
        mean,
        min: *sorted.first()?,
        max: *sorted.last()?,
        stdev_sample: (squares / (n - 1.0)).sqrt(),
    })
}

pub(crate) fn statistics(
    executions: &[Execution]
) -> Result<BTreeMap<&'static str, MetricSummary>, InferError> {
    let measured: Vec<&Execution> = executions
        .iter()
        .filter(|execution| execution.phase == "measured")
        .collect();
    let ns = |value: u64| lossy(u128::from(value));
    let column = |pick: &dyn Fn(&Execution) -> Option<f64>| -> Result<Vec<f64>, InferError> {
        measured
            .iter()
            .map(|execution| pick(execution).ok_or(InferError::Statistics("a metric is missing")))
            .collect()
    };
    let elapsed = column(&|execution| Some(ns(execution.elapsed_ns)))?;
    let first = column(&|execution| execution.first_token_ns.map(ns))?;
    let decode = column(&|execution| execution.decode_tokens_per_s)?;
    let summary = |unit, values: &[f64]| {
        summarize(unit, values).ok_or(InferError::Statistics(
            "at least two measured executions are required",
        ))
    };
    Ok(BTreeMap::from([
        ("elapsed_ns", summary("ns", &elapsed)?),
        ("first_token_ns", summary("ns", &first)?),
        ("decode_tokens_per_s", summary("tokens/s", &decode)?),
    ]))
}

pub(crate) fn consistency(executions: &[Execution]) -> Consistency {
    let Some(reference) = executions.first() else {
        return Consistency {
            consistent: None,
            compared_executions: 0,
            mismatches: Vec::new(),
        };
    };
    let mismatches: Vec<Mismatch> = executions
        .iter()
        .skip(1)
        .filter(|execution| execution.token_ids != reference.token_ids)
        .map(|execution| Mismatch {
            phase: execution.phase,
            index: execution.index,
            first_divergent_position: reference
                .token_ids
                .iter()
                .zip(&execution.token_ids)
                .position(|(left, right)| left != right)
                .unwrap_or_else(|| reference.token_ids.len().min(execution.token_ids.len())),
        })
        .collect();
    Consistency {
        consistent: Some(mismatches.is_empty()),
        compared_executions: executions.len(),
        mismatches,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        consistency,
        decode_tokens_per_s,
        statistics,
    };
    use crate::infer::report::Execution;

    fn execution(
        phase: &'static str,
        index: u32,
        elapsed_ns: u64,
    ) -> Execution {
        let token_available_ns: Vec<u64> = (0..128_u64)
            .map(|position| position.saturating_mul(10).saturating_add(1_000))
            .collect();
        Execution {
            phase,
            index,
            generated_count: 128,
            token_ids: vec![7; 128],
            elapsed_ns,
            first_token_ns: token_available_ns.first().copied(),
            decode_tokens_per_s: decode_tokens_per_s(&token_available_ns),
            token_available_ns,
            device_allocated_bytes: 0,
        }
    }

    #[test]
    fn decode_throughput_spans_the_first_to_last_token() {
        assert_eq!(decode_tokens_per_s(&[]), None, "no tokens");
        assert_eq!(decode_tokens_per_s(&[5, 5]), None, "no elapsed time");
        assert_eq!(
            decode_tokens_per_s(&[0, 500_000_000, 1_000_000_000]),
            Some(2.0),
            "two intervals in one second"
        );
    }

    #[test]
    fn statistics_cover_measured_executions_only() -> Result<(), Box<dyn std::error::Error>> {
        let executions = vec![
            execution("warmup", 0, 1_000_000),
            execution("measured", 0, 10),
            execution("measured", 1, 30),
            execution("measured", 2, 20),
        ];
        let summary = statistics(&executions)?;
        let elapsed = summary.get("elapsed_ns").ok_or("elapsed summary")?;
        assert_eq!(elapsed.count, 3, "warmup excluded");
        assert_eq!(elapsed.median, 20.0, "median");
        assert_eq!(elapsed.max, 30.0, "max");
        assert_eq!(elapsed.stdev_sample, 10.0, "sample standard deviation");
        assert!(
            statistics(executions.get(..2).unwrap_or_default()).is_err(),
            "one measured execution is not enough"
        );
        Ok(())
    }

    #[test]
    fn consistency_reports_the_first_divergence() {
        let mut executions = vec![execution("warmup", 0, 1), execution("measured", 0, 1)];
        if let Some(token) = executions
            .get_mut(1)
            .and_then(|run| run.token_ids.get_mut(9))
        {
            *token = 3;
        }
        let result = consistency(&executions);
        assert_eq!(result.consistent, Some(false), "a mismatch is detected");
        assert_eq!(
            result
                .mismatches
                .first()
                .map(|mismatch| mismatch.first_divergent_position),
            Some(9),
            "first divergent position"
        );
    }
}
