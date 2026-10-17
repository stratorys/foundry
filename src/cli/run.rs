use std::io::Write;
use std::time::Duration;

use foundry::generate::{
    Generation,
    generate,
};
use foundry::nn::Device;
use tracing::{
    error,
    info,
};

use crate::cli::ModelId;
use crate::cli::error::CliError;
use crate::cli::platform::{
    DEVICE,
    DeviceTask,
    run_on_platform,
};
use crate::cli::snapshot::{
    Snapshot,
    load_llama,
};

struct RunTask<'run> {
    model: &'run ModelId,
    snapshot: &'run Snapshot,
    prompt_ids: &'run [u32],
    max_new_tokens: usize,
}

pub fn run(
    model: &ModelId,
    prompt_ids: &[u32],
    max_new_tokens: usize,
) -> Result<(), CliError> {
    let snapshot = Snapshot::open(model)?;
    let generation = run_on_platform(RunTask {
        model,
        snapshot: &snapshot,
        prompt_ids,
        max_new_tokens,
    })?;
    let token_ids = generation.token_ids.concat();
    info!(
        message = "Generated tokens.",
        device = DEVICE,
        prompt_tokens = prompt_ids.len(),
        generated_tokens = token_ids.len(),
        prefill_ms = %generation.prefill.as_millis(),
        decode_steps = generation.decode_steps.len(),
        decode_ms_total = %generation.decode_steps.iter().sum::<Duration>().as_millis(),
    );
    write_token_ids(&mut std::io::stdout().lock(), &token_ids).map_err(|error| {
        error!(message = "Writing the token IDs to stdout failed.", %error);
        CliError::Output
    })
}

impl DeviceTask for RunTask<'_> {
    type Output = Generation;

    fn run<D: Device>(
        self,
        device: &mut D,
    ) -> Result<Generation, CliError> {
        let max_cache_len = self
            .prompt_ids
            .len()
            .checked_add(self.max_new_tokens)
            .ok_or_else(|| {
                error!(
                    message = "Prompt and generated token counts overflow usize.",
                    prompt_tokens = self.prompt_ids.len(),
                    max_new_tokens = self.max_new_tokens,
                );
                CliError::SeqLenOverflow
            })?;
        let llama = load_llama(device, self.model, self.snapshot, max_cache_len)?;
        generate(
            device,
            &llama,
            &[self.prompt_ids],
            self.max_new_tokens,
            self.snapshot.config.eos_token_id(),
        )
        .map_err(|error| {
            error!(message = "Generation failed.", model = %self.model, %error);
            CliError::Generate
        })
    }
}

fn write_token_ids(
    out: &mut impl Write,
    token_ids: &[u32],
) -> std::io::Result<()> {
    let token_ids = token_ids
        .iter()
        .map(u32::to_string)
        .collect::<Vec<String>>()
        .join(",");
    writeln!(out, "[{token_ids}]")
}

#[cfg(test)]
mod tests {
    use crate::cli::run::write_token_ids;

    #[test]
    fn write_token_ids_prints_a_json_array() {
        let mut out = Vec::new();
        write_token_ids(&mut out, &[1, 2, 3]).expect("writing to a vector succeeds");
        assert_eq!(
            String::from_utf8(out),
            Ok("[1,2,3]\n".to_owned()),
            "token IDs as a JSON array"
        );
    }
}
