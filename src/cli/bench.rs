use std::num::NonZeroU32;

use foundry::bench::{
    BenchConfig,
    Report,
    bench,
};
use foundry::nn::Device;
use tracing::{
    error,
    info,
};

use crate::cli::error::CliError;
use crate::cli::platform::{
    DEVICE,
    DeviceTask,
    run_on_platform,
};
use crate::cli::report::{
    BENCH_REPORT_FILE,
    BenchRecord,
    ModelSize,
    ReportStamp,
    metadata,
    write_report,
};
use crate::cli::snapshot::{
    Snapshot,
    load_llama,
};
use crate::cli::{
    BenchArgs,
    ModelId,
};

struct BenchTask<'bench> {
    model: &'bench ModelId,
    snapshot: &'bench Snapshot,
    vocab_size: NonZeroU32,
    bench_config: &'bench BenchConfig,
}

pub fn bench_command(args: &BenchArgs) -> Result<(), CliError> {
    let stamp = ReportStamp::capture()?;
    let snapshot = Snapshot::open(&args.model)?;
    let size = ModelSize::measure(&args.model, &snapshot.weights)?;
    let vocab_size = u32::try_from(snapshot.config.vocab_size())
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or_else(|| {
            error!(
                message = "Vocabulary size is zero or does not fit in u32.",
                vocab_size = snapshot.config.vocab_size()
            );
            CliError::VocabSize
        })?;
    let bench_config = BenchConfig {
        prompt_token_count: args.prompt_tokens,
        gen_token_count: args.gen_tokens,
        depth_token_count: args.depth,
        warmup_run_count: args.warmup_runs,
        warmup_ms: args.warmup_ms,
        run_count: args.runs,
        batch_size: args.batch_size,
        seed: args.seed,
    };
    let (device_name, report) = run_on_platform(BenchTask {
        model: &args.model,
        snapshot: &snapshot,
        vocab_size,
        bench_config: &bench_config,
    })?;
    let record = BenchRecord {
        metadata: metadata(stamp, &args.model, size, device_name),
        report,
    };
    let report_path = write_report(&args.out, BENCH_REPORT_FILE, &record)?;
    info!(
        message = "Benchmarked the model.",
        device = DEVICE,
        device_name = %record.metadata.device_name,
        git_commit = ?record.metadata.git_commit,
        mode = ?record.report.mode,
        llama_cpp_avg_ts = record.report.metrics.llama_cpp.avg_ts,
        mlx_prompt_tps = ?record.report.metrics.mlx.as_ref().map(|mlx| mlx.prompt_tps),
        mlx_generation_tps = ?record.report.metrics.mlx.as_ref().and_then(|mlx| mlx.generation_tps),
        peak_memory_bytes = ?record.report.peak_memory_bytes,
        peak_memory_source = ?record.report.peak_memory_source,
        report = %report_path.display(),
    );
    Ok(())
}

impl DeviceTask for BenchTask<'_> {
    type Output = (String, Report);

    fn run<D: Device>(
        self,
        device: &mut D,
    ) -> Result<(String, Report), CliError> {
        let max_cache_len = [
            self.bench_config.depth_token_count,
            self.bench_config.prompt_token_count,
            self.bench_config.gen_token_count,
        ]
        .into_iter()
        .try_fold(0_usize, |max_cache_len, token_count| {
            usize::try_from(token_count)
                .ok()
                .and_then(|token_count| max_cache_len.checked_add(token_count))
        })
        .ok_or_else(|| {
            error!(
                message = "Benchmark token counts overflow usize.",
                bench_config = ?self.bench_config
            );
            CliError::SeqLenOverflow
        })?;
        let device_name = device.device_name();
        info!(message = "Selected the device.", %device_name);
        let llama = load_llama(device, self.model, self.snapshot, max_cache_len)?;
        let report =
            bench(device, &llama, self.vocab_size, self.bench_config).map_err(|error| {
                error!(message = "Benchmark failed.", model = %self.model, %error);
                CliError::Bench
            })?;
        Ok((device_name, report))
    }
}
