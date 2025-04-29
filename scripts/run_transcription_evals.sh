# List of models
models=("direct/ultravox-70b" "direct/ultravox-8b" "scribe" "scribe-eng" "deepgram/nova-3" "deepgram/nova-2" "deepgram/nova-2-en" "whisper" "openai/gpt-4o-transcribe" "openai/gpt-4o-mini-transcribe" "openai/gpt-4o-transcribe-en" "openai/gpt-4o-mini-transcribe-en")
# Base directory for results
base_dir="results"

# Iterate over each model
for model in "${models[@]}"; do
    # Replace '/' with '-' and strip 'direct-' from the model name for directory and log file paths
    safe_model_name="${model//\//-}"
    safe_model_name="${safe_model_name#direct-}"

    # Set EVALS_THREADS based on the model
    if [[ "$safe_model_name" == "ultravox-70b" || "$safe_model_name" == "ultravox-8b" ]]; then
        export EVALS_THREADS=2
    elif [[ "$safe_model_name" == "openai/gpt-4o-transcribe" || "$safe_model_name" == "openai/gpt-4o-mini-transcribe" || "$safe_model_name" == "openai/gpt-4o-transcribe-en" || "$safe_model_name" == "openai/gpt-4o-mini-transcribe-en" ]]; then
        export EVALS_THREADS=24
    else
        export EVALS_THREADS=5
    fi

    # Create the directory for the model if it doesn't exist
    model_dir="$base_dir/$safe_model_name"
    mkdir -p "$model_dir"

    # Construct the log file path
    log_file="$model_dir/audio_transcription.txt"

    # Run the evaluation command
    EVALS_THREADS=$EVALS_THREADS oaievalset generation/$model audio-transcription-set --log_to_file "$log_file" --record_dir "$model_dir/" --max_samples 5000
done
