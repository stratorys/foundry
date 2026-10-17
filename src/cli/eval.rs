use foundry::eval::{
    Reference,
    ReferenceReport,
    baselines,
    eval_reference,
    max_cache_len,
    read_reference,
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
    EVAL_REPORT_FILE,
    EvalRecord,
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
    EvalArgs,
    ModelId,
};

struct EvalTask<'eval> {
    model: &'eval ModelId,
    snapshot: &'eval Snapshot,
    references: &'eval [Reference],
    max_cache_len: usize,
}

pub fn eval_command(args: &EvalArgs) -> Result<(), CliError> {
    let stamp = ReportStamp::capture()?;
    let references = args
        .references
        .iter()
        .map(|dir| {
            read_reference(dir).map_err(|error| {
                error!(message = "Reading a reference failed.", dir = %dir.display(), %error);
                CliError::Reference
            })
        })
        .collect::<Result<Vec<Reference>, CliError>>()?;
    if references
        .iter()
        .all(|reference| reference.activations.is_none())
    {
        error!(message = "No reference holds layer activations.");
        return Err(CliError::NoActivations);
    }
    let max_cache_len = references
        .iter()
        .map(|reference| max_cache_len(&reference.generation))
        .try_fold(0, |max_cache_len, cache_len| {
            Some(max_cache_len.max(cache_len?))
        })
        .ok_or_else(|| {
            error!(message = "Prompt and generated token counts overflow usize.");
            CliError::SeqLenOverflow
        })?;
    let baseline = baselines(&references).map_err(|error| {
        error!(message = "Comparing the references failed.", %error);
        CliError::Eval
    })?;
    let snapshot = Snapshot::open(&args.model)?;
    let size = ModelSize::measure(&args.model, &snapshot.weights)?;
    let (device_name, reference_reports) = run_on_platform(EvalTask {
        model: &args.model,
        snapshot: &snapshot,
        references: &references,
        max_cache_len,
    })?;
    let record = EvalRecord {
        metadata: metadata(stamp, &args.model, size, device_name),
        references: reference_reports,
        baseline,
    };
    let report_path = write_report(&args.out, EVAL_REPORT_FILE, &record)?;
    let failed_engines: Vec<&str> = record
        .references
        .iter()
        .filter(|reference| reference.is_failure())
        .map(|reference| reference.engine.as_str())
        .collect();
    info!(
        message = "Evaluated the model against the references.",
        device = DEVICE,
        device_name = %record.metadata.device_name,
        reference_count = record.references.len(),
        ?failed_engines,
        report = %report_path.display(),
    );
    if !failed_engines.is_empty() {
        return Err(CliError::EvalMismatch);
    }
    Ok(())
}

impl DeviceTask for EvalTask<'_> {
    type Output = (String, Vec<ReferenceReport>);

    fn run<D: Device>(
        self,
        device: &mut D,
    ) -> Result<(String, Vec<ReferenceReport>), CliError> {
        let device_name = device.device_name();
        info!(message = "Selected the device.", %device_name);
        let llama = load_llama(device, self.model, self.snapshot, self.max_cache_len)?;
        let modules = llama.named_modules();
        let eos_token_id = self.snapshot.config.eos_token_id();
        let reports = self
            .references
            .iter()
            .map(|reference| {
                eval_reference(device, &llama, &modules, eos_token_id, reference).map_err(|error| {
                    error!(
                        message = "Evaluation failed.",
                        model = %self.model,
                        dir = %reference.dir.display(),
                        %error,
                    );
                    CliError::Eval
                })
            })
            .collect::<Result<Vec<ReferenceReport>, CliError>>()?;
        Ok((device_name, reports))
    }
}
