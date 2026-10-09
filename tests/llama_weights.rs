use foundry::backend::cpu::CpuBackend;
use foundry::models::llama::{
    LlamaConfig,
    LlamaWeights,
};
use foundry::weights::Weights;
use hf_hub::HFClientSync;

#[test]
#[ignore = "needs meta-llama/Llama-3.2-3B-Instruct in the Hugging Face cache"]
fn llama_weights_load_the_official_snapshot() {
    let directory = HFClientSync::new()
        .expect("the Hugging Face client is created")
        .model("meta-llama", "Llama-3.2-3B-Instruct")
        .snapshot_download()
        .local_files_only(true)
        .send()
        .expect("meta-llama/Llama-3.2-3B-Instruct is in the Hugging Face cache");
    let config = LlamaConfig::open(&directory).expect("the config is valid");
    let weights = Weights::open(&directory).expect("the safetensors files are valid");

    let llama =
        LlamaWeights::load(&mut CpuBackend::new(), &config, &weights).expect("the weights load");
    assert_eq!(llama.layers().len(), 28, "one entry per layer");
    assert_eq!(
        u64::try_from(llama.bytes_uploaded()).ok(),
        weights.bytes_declared(),
        "the uploaded bytes equal the total size declared by the index"
    );
}
