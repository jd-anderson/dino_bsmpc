import subprocess
import os
import argparse
import json
from typing import List, Dict, Any, Tuple
import time
import sys
import signal
import psutil  # pip install psutil
import pynvml as nvml  # pip install nvidia-ml-py3


def wait_for_available_gpu(self, timeout: int = 600) -> int:
    """
    Wait for a GPU to become available.

    Returns:
        Available GPU ID or raises TimeoutError
    """
    start_time = time.time()
    check_interval = 60  # check every minute

    while time.time() - start_time < timeout:
        # check for completed processes
        self.check_completed_processes()

        # check if any GPU is available
        gpu_id = self.get_available_gpu()
        if gpu_id != -1:
            # wait
            if self.gpu_monitoring:
                wait_for_gpu_memory_release(gpu_id, threshold_mb=1000)

            return gpu_id

        # check timeout if specified
        if timeout is not None:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"No GPU became available within {timeout} seconds")

        time.sleep(check_interval)


def init_gpu_monitoring():
    """Initialize NVIDIA GPU monitoring."""
    try:
        nvml.nvmlInit()
        return True
    except:
        print("Warning: Could not initialize GPU monitoring")
        return False


def get_gpu_memory_usage(gpu_id: int) -> Tuple[float, float]:
    """
    Get GPU memory usage.
    
    Returns:
        Tuple of (used_memory_mb, total_memory_mb)
    """
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**2, info.total / 1024**2
    except:
        return 0, 0


def wait_for_gpu_memory_release(gpu_id: int, threshold_mb: float = 1000, timeout: int = 30):
    """
    Wait for GPU memory to be released below threshold.
    
    Args:
        gpu_id: GPU device ID
        threshold_mb: Memory threshold in MB
        timeout: Maximum wait time in seconds
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        used, total = get_gpu_memory_usage(gpu_id)
        if used < threshold_mb:
            return True
        time.sleep(1)
    
    print(f"Warning: GPU {gpu_id} memory not fully released (using {used:.0f}MB)")
    return False


def kill_process_tree(pid):
    """Kill a process and all its children."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # terminate children first
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        
        # then terminate parent
        parent.terminate()
        
        # wait and force kill if needed
        gone, alive = psutil.wait_procs(children + [parent], timeout=5)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        print(f"Error killing process tree: {e}")


def generate_model_name(hyperparams: Dict[str, Any]) -> str:
    """Generate a unique and interpretable model name from hyperparameters."""
    name_parts = []
    
    # Handle common parameters with specific naming conventions
    if 'bisim_latent_dim' in hyperparams:
        name_parts.append(f"bisim_{hyperparams['bisim_latent_dim']}")
    
    if 'bisim_hidden_dim' in hyperparams:
        name_parts.append(f"hidden_{hyperparams['bisim_hidden_dim']}")
    
    if 'model.bypass_dinov2' in hyperparams and hyperparams['model.bypass_dinov2']:
        name_parts.append("no_dino")
    
    if 'bisim_coef' in hyperparams:
        name_parts.append(f"coef_{hyperparams['bisim_coef']}")
    
    if 'training.bisim_lr' in hyperparams:
        lr_str = f"{hyperparams['training.bisim_lr']:.0e}".replace("e-0", "e-")
        name_parts.append(f"lr_{lr_str}")
    
    if 'var_loss_coef' in hyperparams:
        name_parts.append(f"varloss_coef_{hyperparams['var_loss_coef']}")
    
    if 'num_pcs' in hyperparams:
        name_parts.append(f"numpcs_{hyperparams['num_pcs']}")
    
    if 'VC_target' in hyperparams:
        name_parts.append(f"vctarget_{hyperparams['VC_target']}")
    
    if 'PCA1_loss_target' in hyperparams:
        name_parts.append(f"pca1target_{hyperparams['PCA1_loss_target']}")
    
    if 'PCAloss_epoch' in hyperparams:
        name_parts.append(f"pcaepoch_{hyperparams['PCAloss_epoch']}")
    
    if 'env' in hyperparams:
        name_parts.append(hyperparams['env'])
    
    if 'frameskip' in hyperparams:
        name_parts.append(f"fs_{hyperparams['frameskip']}")
    
    if 'num_hist' in hyperparams:
        name_parts.append(f"hist_{hyperparams['num_hist']}")
    
    if 'training.epochs' in hyperparams:
        name_parts.append(f"epochs_{hyperparams['training.epochs']}")
    
    return "_".join(name_parts)


class ProcessManager:
    """Manages training processes."""
    
    def __init__(self, gpus: List[int], gpu_monitoring: bool = True, 
                 max_processes_per_gpu: int = 1, memory_threshold_gb: float = 40.0):
        self.gpus = gpus
        self.gpu_monitoring = gpu_monitoring and init_gpu_monitoring()
        self.running_processes = []
        self.results = []
        self.max_processes_per_gpu = max_processes_per_gpu
        self.memory_threshold_gb = memory_threshold_gb  # max memory usage before considering GPU full
        
        # setup signal handlers for cleanup
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interruption signals."""
        print("\nReceived interrupt signal. Cleaning up processes...")
        self.cleanup_all_processes()
        sys.exit(1)
    
    def cleanup_all_processes(self):
        """Terminate all running processes."""
        for proc_info in self.running_processes:
            process = proc_info['process']
            try:
                kill_process_tree(process.pid)
            except:
                pass
            
            # close file handles
            for f in ['stdout_file', 'stderr_file']:
                if f in proc_info and proc_info[f]:
                    try:
                        proc_info[f].close()
                    except:
                        pass
    
    def get_available_gpu(self) -> int:
        """
        Get the GPU with the least memory usage.
        
        Returns:
            GPU ID with lowest memory usage, or -1 if none available
        """
        if not self.gpu_monitoring:
            # round-robin if monitoring not available
            min_processes = float('inf')
            best_gpu = -1
            
            for gpu_id in self.gpus:
                gpu_process_count = sum(1 for p in self.running_processes if p['gpu_id'] == gpu_id)
                if gpu_process_count < self.max_processes_per_gpu and gpu_process_count < min_processes:
                    min_processes = gpu_process_count
                    best_gpu = gpu_id
            
            return best_gpu
        
        best_gpu = -1
        min_usage = float('inf')
        
        for gpu_id in self.gpus:
            # check if gpu is already running max processes
            gpu_process_count = sum(1 for p in self.running_processes if p['gpu_id'] == gpu_id)
            if gpu_process_count >= self.max_processes_per_gpu:
                continue
            
            used, total = get_gpu_memory_usage(gpu_id)
            used_gb = used / 1024  # mb to gb
            
            # check if gpu has enough free memory
            if used_gb > self.memory_threshold_gb:
                continue
            
            if used < min_usage:
                min_usage = used
                best_gpu = gpu_id
        
        return best_gpu
    
    def wait_for_available_gpu(self, timeout: int = None) -> int:
        """
        Wait for a GPU to become available.
        
        Args:
            timeout: Maximum wait time in seconds (None for infinite wait)
        
        Returns:
            Available GPU ID
        """
        start_time = time.time()
        check_interval = 10  # Check every 10 seconds
        
        print(f"Waiting for GPU to become available... (Current: {len(self.running_processes)}/{len(self.gpus)} GPUs in use)")
        
        while True:
            # check for completed processes
            self.check_completed_processes()
            
            # check if any gpu is available
            gpu_id = self.get_available_gpu()
            if gpu_id != -1:
                # wait
                if self.gpu_monitoring:
                    wait_for_gpu_memory_release(gpu_id, threshold_mb=1000)
                
                return gpu_id
            
            # check timeout if specified
            if timeout is not None:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"No GPU became available within {timeout} seconds")
            
            time.sleep(check_interval)
    
    def launch_training(self, hyperparams: Dict[str, Any], model_name: str, gpu_id: int):
        """Launch a training process on specified GPU."""
        # setup output directory and log files
        out_dir = os.path.join("./outputs", model_name)
        os.makedirs(out_dir, exist_ok=True)
        
        stdout_path = os.path.join(out_dir, "train_stdout.log")
        stderr_path = os.path.join(out_dir, "train_stderr.log")
        
        # log files
        stdout_f = open(stdout_path, "w", buffering=1)
        stderr_f = open(stderr_path, "w", buffering=1)
        
        # build command
        cmd = [sys.executable, "train.py", "--config-name", "train.yaml"]
        for k, v in hyperparams.items():
            if isinstance(v, bool):
                cmd.append(f"{k}={str(v).lower()}")
            else:
                cmd.append(f"{k}={v}")
        
        cmd.extend([
            f"hydra.run.dir={out_dir}",
            f"hydra.sweep.dir={out_dir}"
        ])
        
        # environment setup
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["WANDB_MODE"] = "disabled"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        print(f"\n{'='*80}")
        print(f"Launching {model_name} on GPU {gpu_id}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*80}")
        
        # launch process
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=stdout_f,
            stderr=stderr_f,
            text=True,
            preexec_fn=os.setsid  # create new process group for better cleanup
        )
        
        # track process info
        self.running_processes.append({
            'process': process,
            'model_name': model_name,
            'hyperparams': hyperparams,
            'gpu_id': gpu_id,
            'start_time': time.time(),
            'stdout_file': stdout_f,
            'stderr_file': stderr_f,
            'stdout_path': stdout_path,
            'stderr_path': stderr_path
        })
    
    def check_completed_processes(self):
        """Check for and handle completed processes."""
        completed = []
        
        for proc_info in self.running_processes:
            process = proc_info['process']
            
            if process.poll() is not None:  # process completed
                # calc duration
                duration = time.time() - proc_info['start_time']
                success = process.returncode == 0
                
                # close file handles
                proc_info['stdout_file'].close()
                proc_info['stderr_file'].close()
                
                # read tail of logs
                tail_stdout = ""
                tail_stderr = ""
                try:
                    with open(proc_info['stdout_path'], "r") as f:
                        lines = f.readlines()
                        tail_stdout = "".join(lines[-50:])
                except:
                    pass
                
                try:
                    with open(proc_info['stderr_path'], "r") as f:
                        lines = f.readlines()
                        tail_stderr = "".join(lines[-100:])
                except:
                    pass
                
                # store result
                self.results.append({
                    "model_name": proc_info['model_name'],
                    "hyperparams": proc_info['hyperparams'],
                    "success": success,
                    "returncode": process.returncode,
                    "duration_seconds": duration,
                    "gpu_id": proc_info['gpu_id'],
                    "stdout": tail_stdout,
                    "stderr": tail_stderr,
                })
                
                completed.append(proc_info)
                
                status = "SUCCESS" if success else "FAILURE"
                print(f"\n[{status}] {proc_info['model_name']} completed on GPU {proc_info['gpu_id']} "
                      f"in {duration/60:.1f} minutes")
                
                # force cleanup of subprocess resources
                try:
                    kill_process_tree(process.pid)
                except:
                    pass
                
                # clear gpu memory
                if self.gpu_monitoring:
                    wait_for_gpu_memory_release(proc_info['gpu_id'])
        
        # remove completed processes
        for proc_info in completed:
            self.running_processes.remove(proc_info)
    
    def run_training_queue(self, hyperparam_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run all training configurations."""
        total_configs = len(hyperparam_configs)
        
        for i, hyperparams in enumerate(hyperparam_configs):
            model_name = generate_model_name(hyperparams)
            
            print(f"\n[{i+1}/{total_configs}] Preparing to launch {model_name}")
            
            # wait for available gpu
            try:
                gpu_id = self.wait_for_available_gpu(timeout=None)  # no timeout, wait indefinitely
                print(f"GPU {gpu_id} is available, launching model...")
            except KeyboardInterrupt:
                print(f"\nInterrupted while waiting for GPU. Stopping...")
                break
            
            # launch training
            self.launch_training(hyperparams, model_name, gpu_id)
            
            # small delay to ensure process starts properly
            time.sleep(5)
        
        # wait for all remaining processes to complete
        print(f"\n{'='*80}")
        print("All models queued. Waiting for remaining processes to complete...")
        print(f"{'='*80}")
        while self.running_processes:
            self.check_completed_processes()
            time.sleep(30)  # check every 30 seconds
        
        return self.results


def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save training results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # print summary
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"\nTraining Summary:")
    print(f"Total models: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        avg_duration = sum(r["duration_seconds"] for r in successful) / len(successful)
        print(f"Average training time: {avg_duration/60:.1f} minutes")
    
    if failed:
        print(f"\nFailed models:")
        for result in failed:
            print(f"  - {result['model_name']} (GPU {result.get('gpu_id', 'N/A')}): "
                  f"{result.get('stderr', 'Unknown error')[:100]}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sweep over hyperparameters and train multiple models"
    )
    
    parser.add_argument(
        "--config-file",
        type=str,
        default="train_sweep_config.json",
        help="JSON file containing hyperparameter sweep configurations"
    )
    
    parser.add_argument(
        "--gpus",
        nargs='+',
        type=int,
        required=True,
        help="List of GPU IDs to use for training"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="train_sweep_results.json",
        help="Output file for results (default: train_sweep_results.json)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=36000,
        help="Timeout per training job in seconds (default: 36000 = 10 hours)"
    )
    
    parser.add_argument(
        "--max-processes-per-gpu",
        type=int,
        default=1,
        help="Maximum number of processes per GPU (default: 1)"
    )
    
    parser.add_argument(
        "--memory-threshold-gb",
        type=float,
        default=40.0,
        help="GPU memory threshold in GB before considering it full (default: 40.0)"
    )
    parser.add_argument(
        "--disable-gpu-monitoring",
        action="store_true",
        help="Disable GPU memory monitoring"
    )
    
    return parser.parse_args()


def main():
    """Main function to run hyperparameter sweep."""
    args = parse_arguments()
    
    # load hyperparameter configurations
    if not os.path.exists(args.config_file):
        print(f"Error: Config file {args.config_file} not found.")
        print("Please create a JSON file with hyperparameter configurations.")
        print("\nExample config file format:")
        example_config = [
            {
                "bisim_latent_dim": 512,
                "bisim_hidden_dim": 1024,
                "model.bypass_dinov2": True,
                "training.bisim_lr": 5e-6,
                "var_loss_coef": 1.0,
            },
            {
                "bisim_latent_dim": 256,
                "bisim_hidden_dim": 512,
                "model.bypass_dinov2": False,
                "training.bisim_lr": 1e-5,
                "var_loss_coef": 2.0,
            }
        ]
        print(json.dumps(example_config, indent=2))
        return
    
    with open(args.config_file, 'r') as f:
        hyperparam_configs = json.load(f)
    
    print(f"Loaded {len(hyperparam_configs)} hyperparameter configurations")
    print(f"Using GPUs: {args.gpus}")
    
    # init process manager
    manager = ProcessManager(
        gpus=args.gpus,
        gpu_monitoring=not args.disable_gpu_monitoring,
        max_processes_per_gpu=args.max_processes_per_gpu,
        memory_threshold_gb=args.memory_threshold_gb
    )
    
    try:
        # run training queue
        results = manager.run_training_queue(hyperparam_configs)
        
        # save results
        save_results(results, args.output_file)
        
        print(f"\nAll training completed! Results saved to {args.output_file}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial results...")
        save_results(manager.results, args.output_file + ".partial")
        
    finally:
        # ensure cleanup
        manager.cleanup_all_processes()
        if manager.gpu_monitoring:
            nvml.nvmlShutdown()


if __name__ == "__main__":
    main()
