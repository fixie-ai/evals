"""
Modal-based remote execution of evals with GPU support.
"""
import argparse
import os
import sys
from typing import Optional, cast

import modal

# Create a Modal app
app = modal.App("oaieval-remote")

# Assumes you've set up this secret in Modal
secrets = [
    modal.Secret.from_name("openai-secret"),  
    modal.Secret.from_name("huggingface-secret"),
]

# Create a base image with evals and all its dependencies
evals_base_image = (
    modal.Image.debian_slim(python_version="3.12.2")
    .apt_install("git")
    .pip_install("git+https://github.com/fixie-ai/evals.git@3cccec3a0478822b0de419ed0e5bc062b1ec488f", "accelerate")
)

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evals remotely through Modal")
    parser.add_argument(
        "completion_fn",
        type=str,
        help="One or more CompletionFn URLs, separated by commas (,)",
    )
    parser.add_argument("eval", type=str, help="Name of an eval. See registry.")
    parser.add_argument("--commit_hash", type=str, help="Git commit hash to use for the eval code")
    parser.add_argument(
        "--completion_args",
        type=str,
        default="",
        help="Specify additional parameters for completion_fn (e.g., 'key1=value1,key2=value2')",
    )
    parser.add_argument("--extra_eval_params", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20220722)
    parser.add_argument("--registry_path", type=str, default=None, action="append")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--visible", action=argparse.BooleanOptionalAction, default=None,
                       help="Whether to show samples during evaluation")
    parser.add_argument("--user", type=str, default=os.getenv("USER", "unknown"),
                       help="User running the eval")
    parser.add_argument("--log_to_file", type=str, default=None,
                       help="Log to a file instead of stdout")
    return parser

# Define the function with base image and environment variables
@app.function(
    timeout=21600,
    container_idle_timeout=60,
    image=evals_base_image,
    secrets=secrets,
    gpu=modal.gpu.H100(count=4),
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name("hf-cache", create_if_missing=True)
    }
)
def run_eval(args_dict: dict) -> None:
    import os
    import sys
    print("Python path:", sys.path)
    print("HF cache contents:", os.listdir("/root/.cache/huggingface"))

    from evals.cli.oaieval import run, OaiEvalArguments

    # Remove Modal-specific args before passing to oaieval
    args_dict.pop("commit_hash", None)
    args_dict.pop("gpu", None)

    # Ensure all required fields from OaiEvalArguments exist with defaults
    defaults = {
        "user": "",  # Default empty string as per oaieval.py
        "record_path": None,
        "log_to_file": None,
        "local_run": True,
        "http_run": False,
        "http_run_url": None,
        "http_batch_size": 100,
        "http_fail_percent_threshold": 5,
        "dry_run": False,
        "dry_run_logging": True,
    }

    # Update args with defaults for any missing fields
    for key, value in defaults.items():
        if key not in args_dict:
            args_dict[key] = value

    # Convert dict back to namespace
    args = argparse.Namespace(**args_dict)

    # Use the original oaieval run function
    run(cast(OaiEvalArguments, args))

def main() -> None:
    parser = get_parser()
    args = parser.parse_args()

    args_dict = vars(args)

    # Run the function with output enabled
    with modal.enable_output():
        with app.run():
            run_eval.remote(args_dict)

if __name__ == "__main__":
    main()



#python evals/cli/oaieval_remote.py generation/gpu/ultravox text-bigbench

