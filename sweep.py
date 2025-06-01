import subprocess
import itertools
import time
import json
import os

# Define parameters
MODEL_NAME = "mod1_bisim_1024_coef_1"

def extract_planning_result_dir_alternative(output_string):
    """
    Alternative method using string splitting (more robust for this specific case)
    
    Args:
        output_string (str): The output string containing the planning result
        
    Returns:
        str: The extracted directory path, or None if not found
    """
    lines = output_string.split('\n')
    for line in lines:
        if "Planning result saved dir:" in line:
            # Split by the marker and take the path part
            path = line.split("Planning result saved dir:")[-1].strip()
            return path
    return None


def parse_logs_json(file_path):
    """
    Parse logs.json file to extract final success rate and total steps.
    
    Args:
        file_path (str): Path to the logs.json file
        
    Returns:
        dict: Dictionary containing final_success_rate and total_steps
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
        
        # Split by lines and parse each JSON object
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
                
                # Check for final_eval/success_rate
                if "final_eval/success_rate" in data:
                    final_success_rate = data["final_eval/success_rate"]
                
                # Track the maximum step number
                if "step" in data:
                    step_num = data["step"]
                    max_step = max(max_step, step_num)
                    total_steps = max_step  # Update total steps
                    
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
    """
    Parse logs.json from a specific directory.
    
    Args:
        directory_path (str): Path to directory containing logs.json
        
    Returns:
        dict: Dictionary containing final_success_rate and total_steps
    """
    logs_path = os.path.join(directory_path, "logs.json")
    return parse_logs_json(logs_path)


if __name__=="__main__":

    # Define parameter ranges
    bisim_weights = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5]
    model_epochs = list(range(5, 55, 5))  # 5, 10, ..., 50

    base_args = [
        "python", "plan.py",
        "model_name="+MODEL_NAME,  # Replace with actual model name
        "n_evals=5",
        "planner=cem",
        "goal_H=5",
        "goal_source='random_state'",
        "planner.opt_steps=30"
    ]

    result_logs_list = []

    for bisim_weight in bisim_weights:
        for model_epoch in model_epochs:
            
            # Create commands
            cmd = base_args + [
                f"objective.bisim_weight={bisim_weight}",
                f"model_epoch={model_epoch}"
            ]

            # run planning
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=6000)

            # select the result directory from result
            result_dir = extract_planning_result_dir_alternative(result.stdout)

            result_logs = {'bisim_weight': bisim_weight,
                            'model_epoch': model_epoch}

            result_logs.update(parse_logs_from_directory(result_dir))

            result_logs_list.append(result_logs)

            print(result_logs)


    for result_log in result_logs_list:
        print(result_log)
