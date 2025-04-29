import json
import pandas as pd
import evaluate
import whisper_normalizer.basic as whisper_basic
import whisper_normalizer.english as whisper_english

from tqdm import tqdm

def _read_jsonl(file_path: str) -> list[dict]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def main(cap_len=None):
    models = [
        # "ultravox-70b", 
        # "ultravox-8b", 
        "from_wandb/ultravox-70b", 
        "from_wandb/ultravox-8b", 
        "ultravox-0_4_1-70b",
        "ultravox-0_4_1-8b",
        "deepgram-nova-3", 
        "deepgram-nova-2",
        "deepgram-nova-2-en",
        "whisper", 
        "scribe", 
        "scribe-eng", 
        "openai-gpt-4o-transcribe", 
        "openai-gpt-4o-transcribe-en", 
        "openai-gpt-4o-mini-transcribe", 
        "openai-gpt-4o-mini-transcribe-en"
        ]
    datasets = [
        "fleurs-en-transcription.test.v1",
        "librispeech-clean-transcription.test.v1",
        "librispeech-other-transcription.test.v1",
        "commonvoice-en-transcription.test.v1",
        "ami-ihm-transcription.test.v1",
        "ami-ihm-transcription-short.test.v1",
        "ultravox-calls-transcription.test.v1",
        "ultravox-calls-prekrisp-transcription.test.v1",
        "ultravox-calls-telephony-transcription.test.v1",
        "ultravox-calls-telephony-prekrisp-transcription.test.v1",
    ]

    res = []

    for model in tqdm(models, desc="Models"):
        for dataset in tqdm(datasets, desc="Datasets"):
            results_file = f"results/{model}/{dataset}.log"
            results = _read_jsonl(results_file)
            results = [r for r in results if "type" in r and r["type"] == "match"]

            results = [
                {
                    "sample_id": r["sample_id"],
                    "expected": r["data"]["expected"],
                    "predicted": r["data"]["sampled"],
                    "wer": r["data"]["wer"],
                }
                for r in results
            ]
            df = pd.DataFrame(results)
            df["model"] = model
            df["dataset"] = dataset
            res.append(df)

    res = pd.concat(res)
    res_pvt = res.pivot(
        index=["sample_id", "dataset"], columns="model", values=["wer", "expected", "predicted"]
    ).reset_index(level="dataset")
    res_pvt.columns = res_pvt.columns.map(lambda x: "_".join(x) if isinstance(x, tuple) else x)

    additional_results = []

    for model in tqdm(models, desc="Models"):
        for dataset in tqdm(datasets, desc="Datasets"):

            res_no_null = res_pvt[res_pvt[f"expected_{model}"].str.len() >= 0]
            res_no_null = res_no_null[res_no_null.dataset_ == dataset]

            expected = res_no_null[f"expected_{model}"].tolist()
            predicted = res_no_null[f"predicted_{model}"].tolist()

            # Replace API errors with empty strings
            errors = ["Error:" in p for p in predicted]
            if any(errors):
                print(f"WARNING: Errors in {sum(errors)}/{len(predicted)} samples for {model} on {dataset}. Replacing with empty string(s).")
                predicted = [p if "Error:" not in p else "" for p in predicted]

            wer = _compute_wer(expected, predicted, cap_len=cap_len)

            # print(f"{model}, {dataset}: {wer}")
            additional_results.append({
                "model": model,
                "dataset": dataset,
                "wer": wer
            }) 

    additional_results = pd.DataFrame(additional_results).pivot(
        index="dataset", columns="model", values="wer"
    )[models].loc[datasets]
    
    if cap_len is not None:
        additional_results.to_csv(f"transcription_results_{int(cap_len)}.csv")
        res_pvt.to_csv(f"transcription_examples_{int(cap_len)}.csv")
    else:
        additional_results.to_csv(f"transcription_results.csv")
        res_pvt.to_csv(f"transcription_examples.csv")

def _compute_wer(expected, sampled, cap_len=200):
    expected = [expected] if not isinstance(expected, list) else expected
    sampled = [sampled] if not isinstance(sampled, list) else sampled

    # Initialize the appropriate text normalizer
    normalizer = whisper_english.EnglishTextNormalizer()

    # Normalize both reference and hypothesis
    expected = [normalizer(e) for e in expected]
    sampled = [normalizer(s) for s in sampled]

    # Handle empty strings
    expected = [e if e.strip() else "<silence>" for e in expected]
    sampled = [s if s.strip() else "<silence>" for s in sampled]

    if cap_len is not None:
        pair = [(e, s[:int((cap_len/100)*len(e))]) for e, s in zip(expected, sampled)]
        expected, sampled = zip(*pair)

    wer_metric = evaluate.load("wer")
    wer_score = wer_metric.compute(predictions=sampled, references=expected) *100.0
    return wer_score


if __name__ == "__main__":
    main(cap_len=None)
    main(cap_len=200)