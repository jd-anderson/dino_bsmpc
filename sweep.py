import subprocess
import json
import os
import argparse


def extract_planning_result_dir_alternative(output_string):
    lines = output_string.split('\n')
    for line in lines:
        if "Planning result saved dir:" in line:
            path = line.split("Planning result saved dir:")[-1].strip()
            return path
    return None


def parse_logs_json(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()

        lines = content.strip().split('\n')

        final_success_rate = None
        max_step = 0
        total_steps = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                if "final_eval/success_rate" in data:
                    final_success_rate = data["final_eval/success_rate"]

                if "step" in data:
                    step_num = data["step"]
                    max_step = max(max_step, step_num)
                    total_steps = max_step

            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line: {line[:50]}... Error: {e}")
                continue

        return {
            "final_success_rate": final_success_rate,
            "total_steps": total_steps,
            "file_path": file_path
        }

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def parse_logs_from_directory(directory_path):
    logs_path = os.path.join(directory_path, "logs.json")
    return parse_logs_json(logs_path)


def write_results_to_json(filename, results):
    with open(filename, 'w') as file:
        json.dump(results, file, indent=2)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run planning experiments with different model epochs"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="mod1_bisim_1024_coef_1",
        help="Name of the model to use (default: mod1_bisim_1024_coef_1)"
    )

    parser.add_argument(
        "--n-evals",
        type=int,
        default=5,
        help="Number of evaluations (default: 5)"
    )

    parser.add_argument(
        "--planner",
        type=str,
        default="cem",
        help="Planner type (default: cem)"
    )

    parser.add_argument(
        "--goal-h",
        type=int,
        default=5,
        help="Goal horizon (default: 5)"
    )

    parser.add_argument(
        "--goal-source",
        type=str,
        default="random_state",
        help="Goal source (default: random_state)"
    )

    parser.add_argument(
        "--opt-steps",
        type=int,
        default=30,
        help="Optimization steps for planner (default: 30)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=6000,
        help="Timeout for subprocess in seconds (default: 6000)"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    model_epochs = list(range(5, 55, 5))  # 5, 10, ..., 50
    backgrounds = ['default', 'slight_change', 'gradient']

    base_args = [
        "python", "plan.py",
        f"model_name={args.model_name}",
        f"n_evals={args.n_evals}",
        f"planner={args.planner}",
        f"goal_H={args.goal_h}",
        f"goal_source='{args.goal_source}'",
        f"planner.opt_steps={args.opt_steps}"
    ]

    for background in backgrounds:
        result_logs_list = []
        output_filename = f'{args.model_name}_sweep_{background}.json'
        print("=" * 75)
        print(f"\n{'=' * 75}")
        print(f"Running sweep for background: {background}")
        print(f"Output file: {output_filename}")
        print(f"{'=' * 75}\n")
        
        for model_epoch in model_epochs:
            cmd = base_args + [
                f"model_epoch={model_epoch}",
                f"point_maze_env.background={background}",
            ]

            print(f"\nRunning planning for epoch {model_epoch} with background {background}...")
            print(f"Command: {' '.join(cmd)}")

            # run planning
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)

            # check if subprocess succeeded
            if result.returncode != 0:
                print(f"ERROR: Planning failed with return code {result.returncode}")
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")
                result_logs = {
                    'model_epoch': model_epoch,
                    'background': background,
                    'error': f'Planning failed with return code {result.returncode}',
                    'stdout': result.stdout[:500] if result.stdout else None,
                    'stderr': result.stderr[:500] if result.stderr else None
                }
                result_logs_list.append(result_logs)
                print(f"Result: {result_logs}")
                write_results_to_json(output_filename, result_logs_list)
                continue

            # select the result directory
            result_dir = extract_planning_result_dir_alternative(result.stdout)

            if result_dir is None:
                print(f"ERROR: Could not extract result directory from stdout")
                print(f"STDOUT (last 1000 chars):\n{result.stdout[-1000:]}")
                result_logs = {
                    'model_epoch': model_epoch,
                    'background': background,
                    'error': 'Could not extract result directory from stdout',
                    'stdout_preview': result.stdout[-500:] if result.stdout else None
                }
                result_logs_list.append(result_logs)
                print(f"Result: {result_logs}")
                write_results_to_json(output_filename, result_logs_list)
                continue

            print(f"Result directory: {result_dir}")

            if not os.path.exists(result_dir):
                print(f"ERROR: Result directory does not exist: {result_dir}")
                result_logs = {
                    'model_epoch': model_epoch,
                    'background': background,
                    'error': f'Result directory does not exist: {result_dir}',
                    'result_dir': result_dir
                }
                result_logs_list.append(result_logs)
                print(f"Result: {result_logs}")
                write_results_to_json(output_filename, result_logs_list)
                continue

            result_logs = {
                'model_epoch': model_epoch,
                'background': background
            }

            logs_data = parse_logs_from_directory(result_dir)
            if logs_data is None:
                print(f"WARNING: Could not parse logs.json from {result_dir}")
                result_logs['error'] = 'Could not parse logs.json'
                result_logs['result_dir'] = result_dir
            else:
                result_logs.update(logs_data)

            result_logs_list.append(result_logs)

            print(f"Result: {result_logs}")
            write_results_to_json(output_filename, result_logs_list)

        print(f"\nCompleted sweep for background: {background}")
        print(f"Final results saved to {output_filename}")
