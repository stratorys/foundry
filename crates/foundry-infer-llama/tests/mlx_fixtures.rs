#![cfg(target_os = "macos")]
#![expect(
    clippy::print_stderr,
    reason = "the validation test reports its per-probe measurements"
)]

use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{
    Read,
    Seek,
    SeekFrom,
};
use std::path::{
    Path,
    PathBuf,
};

use foundry_infer_llama::metal::{
    LlamaModel,
    Stage,
};
use foundry_infer_llama::safetensors::{
    Dtype,
    Header,
    TensorEntry,
    read_header,
};
use serde_json::{
    Value,
    json,
};

type TestResult = Result<(), Box<dyn Error>>;

const FIXTURE_FILE: &str = "fixtures.safetensors";
const FLOOR: f64 = 1e-3;

struct Fixtures {
    path: PathBuf,
    header: Header,
    entries: BTreeMap<String, TensorEntry>,
}

impl Fixtures {
    fn open(directory: &Path) -> Result<Self, Box<dyn Error>> {
        let path = directory.join(FIXTURE_FILE);
        let header = read_header(&path)?;
        let entries = header
            .tensors
            .iter()
            .map(|entry| (entry.name.clone(), entry.clone()))
            .collect();
        Ok(Self {
            path,
            header,
            entries,
        })
    }

    fn contains(
        &self,
        name: &str,
    ) -> bool {
        self.entries.contains_key(name)
    }

    fn bytes(
        &self,
        name: &str,
    ) -> Result<(Dtype, Vec<u8>), Box<dyn Error>> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| format!("fixture tensor {name} is missing"))?;
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(
            self.header
                .data_start
                .checked_add(entry.start)
                .ok_or("fixture offset overflow")?,
        ))?;
        let mut bytes = vec![0_u8; usize::try_from(entry.len)?];
        file.read_exact(&mut bytes)?;
        Ok((entry.dtype, bytes))
    }

    fn floats(
        &self,
        name: &str,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        let (dtype, bytes) = self.bytes(name)?;
        Ok(match dtype {
            Dtype::F16 => bytes
                .chunks_exact(2)
                .map(|pair| half::f16::from_le_bytes(pair.try_into().unwrap_or_default()).to_f32())
                .collect(),
            Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap_or_default()))
                .collect(),
            Dtype::U32 => return Err(format!("{name} is not a float tensor").into()),
        })
    }

    fn tokens(
        &self,
        name: &str,
    ) -> Result<Vec<u32>, Box<dyn Error>> {
        let (dtype, bytes) = self.bytes(name)?;
        if dtype != Dtype::U32 {
            return Err(format!("{name} is not a token tensor").into());
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap_or_default()))
            .collect())
    }
}

fn probe_name(stage: Stage) -> Option<String> {
    let layer = |index: u32, probe: &str| Some(format!("layers.{index}.{probe}"));
    match stage {
        Stage::Embedding => Some("embedding".to_owned()),
        Stage::AttentionNorm(index) => layer(index, "attention_norm"),
        Stage::Query(index) => layer(index, "query"),
        Stage::Key(index) => layer(index, "key"),
        Stage::Value(index) => layer(index, "value"),
        Stage::Attention(index) => layer(index, "attention"),
        Stage::AttentionResidual(index) => layer(index, "attention_residual"),
        Stage::MlpNorm(index) => layer(index, "mlp_norm"),
        Stage::Gate(index) => layer(index, "gate"),
        Stage::Up(index) => layer(index, "up"),
        Stage::Swiglu(index) => layer(index, "swiglu"),
        Stage::Output(index) => layer(index, "output"),
        Stage::FinalNorm => Some("final_norm".to_owned()),
        Stage::Logits => None,
    }
}

fn relative_l2(
    actual: &[f32],
    expected: &[f32],
) -> f64 {
    if actual.len() != expected.len() {
        return f64::INFINITY;
    }
    let (error, norm) =
        actual
            .iter()
            .zip(expected)
            .fold((0.0_f64, 0.0_f64), |(error, norm), (&a, &e)| {
                let difference = f64::from(a) - f64::from(e);
                (
                    difference.mul_add(difference, error),
                    f64::from(e).mul_add(f64::from(e), norm),
                )
            });
    (error / norm.max(f64::MIN_POSITIVE)).sqrt()
}

fn max_abs(
    left: &[f32],
    right: &[f32],
) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0.0, f64::max)
}

struct Check {
    name: String,
    foundry_vs_reference: f64,
    mlx_vs_reference: f64,
    foundry_vs_mlx: f64,
}

impl Check {
    fn bound(&self) -> f64 { (2.0 * self.mlx_vs_reference).max(FLOOR) }

    fn passed(&self) -> bool { self.foundry_vs_reference <= self.bound() }

    fn json(&self) -> Value {
        json!({
            "probe": self.name,
            "foundry_vs_reference": self.foundry_vs_reference,
            "mlx_vs_reference": self.mlx_vs_reference,
            "foundry_vs_mlx": self.foundry_vs_mlx,
            "bound": self.bound(),
            "passed": self.passed(),
        })
    }
}

fn check(
    fixtures: &Fixtures,
    name: String,
    foundry: &[f32],
    mlx_name: &str,
    reference_name: &str,
) -> Result<Check, Box<dyn Error>> {
    let mlx = fixtures.floats(mlx_name)?;
    let reference = fixtures.floats(reference_name)?;
    Ok(Check {
        name,
        foundry_vs_reference: relative_l2(foundry, &reference),
        mlx_vs_reference: relative_l2(&mlx, &reference),
        foundry_vs_mlx: relative_l2(foundry, &mlx),
    })
}

fn row(
    values: &[f32],
    index: usize,
    width: usize,
) -> Vec<f32> {
    values
        .chunks(width)
        .nth(index)
        .map(<[f32]>::to_vec)
        .unwrap_or_default()
}

struct Paths {
    snapshot: PathBuf,
    fixtures: PathBuf,
    manifest: PathBuf,
}

fn paths() -> Result<Option<Paths>, Box<dyn Error>> {
    let snapshot = env::var_os("FOUNDRY_LLAMA_SNAPSHOT");
    let fixtures = env::var_os("FOUNDRY_LLAMA_FIXTURES");
    let (Some(snapshot), Some(fixtures)) = (snapshot, fixtures) else {
        if env::var_os("FOUNDRY_REQUIRE_LLAMA").is_some() {
            return Err("FOUNDRY_LLAMA_SNAPSHOT and FOUNDRY_LLAMA_FIXTURES are required".into());
        }
        eprintln!("llama fixtures: SKIPPED, full-model validation not run");
        return Ok(None);
    };
    let manifest = env::var_os("FOUNDRY_LLAMA_MANIFEST").map_or_else(
        || {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../.private-data/models/llama-3.2-3b-instruct-4bit/manifest.json")
        },
        PathBuf::from,
    );
    Ok(Some(Paths {
        snapshot: snapshot.into(),
        fixtures: fixtures.into(),
        manifest,
    }))
}

#[test]
fn full_model_matches_mlx_within_documented_tolerances() -> TestResult {
    let Some(paths) = paths()? else {
        return Ok(());
    };
    let fixtures = Fixtures::open(&paths.fixtures)?;
    let model = LlamaModel::load(&paths.snapshot, &paths.manifest)?;
    eprintln!("llama fixtures: running on {}", model.device_name());
    let vocab = usize::try_from(model.config().vocab)?;
    let input = fixtures.tokens("tokens.input")?;
    let continuation = fixtures.tokens("tokens.continuation")?;
    let mut session = model.session()?;
    let mut checks = Vec::new();
    let mut foundry_logits: Vec<Vec<f32>> = Vec::new();
    let mut foundry_tokens = Vec::new();
    let steps = std::iter::once(input.clone()).chain(
        continuation
            .iter()
            .take(continuation.len().saturating_sub(1))
            .map(|&token| vec![token]),
    );
    for (step, tokens) in steps.enumerate() {
        let phase = if step == 0 {
            "prefill".to_owned()
        } else {
            format!("decode{}", step.saturating_sub(1))
        };
        let mut probes: Vec<(String, Vec<f32>)> = Vec::new();
        let mut logits = Vec::new();
        let token = session.forward_probed(&tokens, &mut |stage, values| {
            let widened = || values.iter().map(|value| value.to_f32()).collect();
            if stage == Stage::Logits {
                logits = widened();
            } else if let Some(name) = probe_name(stage)
                && fixtures.contains(&format!("mlx.{phase}.{name}"))
            {
                probes.push((name, widened()));
            }
        })?;
        for (name, values) in probes {
            checks.push(check(
                &fixtures,
                format!("{phase}.{name}"),
                &values,
                &format!("mlx.{phase}.{name}"),
                &format!("reference.{phase}.{name}"),
            )?);
        }
        foundry_logits.push(logits);
        foundry_tokens.push(token);
    }
    let mlx_logits = fixtures.floats("mlx.logits")?;
    let reference_logits = fixtures.floats("reference.logits")?;
    let mut argmax = Vec::new();
    for (position, logits) in foundry_logits.iter().enumerate() {
        let mlx = row(&mlx_logits, position, vocab);
        let reference = row(&reference_logits, position, vocab);
        checks.push(Check {
            name: format!("logits.{position}"),
            foundry_vs_reference: relative_l2(logits, &reference),
            mlx_vs_reference: relative_l2(&mlx, &reference),
            foundry_vs_mlx: relative_l2(logits, &mlx),
        });
        let expected = continuation.get(position).copied().unwrap_or(u32::MAX);
        let actual = foundry_tokens.get(position).copied().unwrap_or(u32::MAX);
        let delta = 2.0 * max_abs(&mlx, &reference);
        let value = |token: u32| {
            usize::try_from(token)
                .ok()
                .and_then(|index| reference.get(index))
                .map_or(f64::NAN, |value| f64::from(*value))
        };
        let gap = (value(expected) - value(actual)).abs();
        argmax.push((position, expected, actual, gap, delta));
    }
    let mut generated_session = model.session()?;
    generated_session.prepare(&input)?;
    let generated = generated_session.generate(continuation.len(), |_, _| {})?;
    let divergence = generated
        .iter()
        .zip(&continuation)
        .position(|(left, right)| left != right);
    let near_tie = |position: usize| {
        argmax
            .get(position)
            .is_some_and(|(_, _, _, gap, delta)| gap < delta)
    };
    let failed_probes: Vec<&Check> = checks.iter().filter(|check| !check.passed()).collect();
    let failed_argmax: Vec<_> = argmax
        .iter()
        .filter(|(_, expected, actual, gap, delta)| expected != actual && gap >= delta)
        .collect();
    let sequence_ok = divergence.is_none_or(near_tie);
    let worst_ratio = checks
        .iter()
        .map(|check| check.foundry_vs_reference / check.bound())
        .fold(0.0, f64::max);
    let foundry_vs_mlx_max = checks
        .iter()
        .filter(|check| check.name.starts_with("logits."))
        .map(|check| check.foundry_vs_mlx)
        .fold(0.0, f64::max);
    eprintln!(
        "llama fixtures: {} probes, worst e_foundry/bound = {worst_ratio:.3}, logits \
         foundry-vs-mlx max = {foundry_vs_mlx_max:.3e}, argmax mismatches = {}, greedy divergence \
         = {divergence:?}",
        checks.len(),
        argmax.iter().filter(|(_, e, a, ..)| e != a).count(),
    );
    if let Some(report) = env::var_os("FOUNDRY_LLAMA_VALIDATION_REPORT") {
        let document = json!({
            "rule": "e_foundry(p) <= max(2 * e_mlx(p), 1e-3) against the f32 reference; argmax equal unless gap < 2 * max|mlx - reference|",
            "checks": checks.iter().map(Check::json).collect::<Vec<_>>(),
            "argmax": argmax.iter().map(|(position, expected, actual, gap, delta)| json!({
                "position": position, "mlx": expected, "foundry": actual, "reference_gap": gap, "near_tie_bound": delta
            })).collect::<Vec<_>>(),
            "greedy": {"foundry": generated, "mlx": continuation, "first_divergence": divergence},
            "worst_ratio_to_bound": worst_ratio,
        });
        std::fs::write(report, serde_json::to_string_pretty(&document)?)?;
    }
    assert!(
        failed_probes.is_empty(),
        "first probe outside tolerance: {}",
        failed_probes
            .first()
            .map(|check| check.json().to_string())
            .unwrap_or_default()
    );
    assert!(
        failed_argmax.is_empty(),
        "argmax mismatches outside near ties: {failed_argmax:?}"
    );
    assert!(
        sequence_ok,
        "greedy output diverges from MLX at {divergence:?} without a near tie"
    );
    Ok(())
}
