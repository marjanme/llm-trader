import argparse
from pathlib import Path

from experiment_logic.experiment_runner import run_experiment
from experiment_logic.llm_client import build_llm_client
from experiment_logic.load_data import load_input_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one LLM prediction experiment.")
    parser.add_argument("--market-id", required=True, help="Market identifier to run.")
    parser.add_argument("--configuration-id", required=True, help="Configuration identifier to run.")
    parser.add_argument(
        "--provider",
        required=True,
        choices=["mock"],
        help="LLM provider to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_data_dir = Path("input_data")
    results_dir = Path("results")

    loaded_input_data = load_input_data(input_data_dir)
    llm_client = build_llm_client(provider=args.provider)

    run_dir = run_experiment(
        loaded_input_data=loaded_input_data,
        market_id=args.market_id,
        configuration_id=args.configuration_id,
        llm_client=llm_client,
        results_dir=results_dir,
    )

    print(f"Experiment completed. Results written to {run_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
