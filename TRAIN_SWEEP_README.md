# Train Sweep Script

The `train_sweep.py` script allows you to sweep over different hyperparameters and train multiple models automatically. It loads configurations from a JSON file and uses an intelligent GPU queue system to train models in parallel across multiple GPUs.

## Dependencies

The script requires the following Python packages:
- `psutil`: For process management and cleanup
- `nvidia-ml-py3`: For GPU memory monitoring (optional, can be disabled with `--disable-gpu-monitoring`)

Install with:
```bash
pip install psutil nvidia-ml-py3
```

## Usage

### Basic Usage

```bash
python train_sweep.py --config-file train_sweep_config.json --gpus 0 1 2 3
```

### Configuration File Format

The configuration file should be a JSON array of objects, where each object contains hyperparameter key-value pairs:

```json
[
  {
    "env": "point_maze",
    "frameskip": 5,
    "num_hist": 3,
    "training.epochs": 50,
    "bisim_latent_dim": 512,
    "bisim_hidden_dim": 1024,
    "model.bypass_dinov2": true,
    "training.bisim_lr": 5e-6,
    "var_loss_coef": 1.0,
    "num_pcs": 10,
    "VC_target": 1,
    "PCA1_loss_target": 0.01,
    "PCAloss_epoch": 50
  }
]
```

## Supported Hyperparameters

You can specify any hyperparameter from the `train.yaml` config file. Common ones include:

- `env`: Environment name (e.g., "point_maze", "pusht")
- `frameskip`: Frame skip value
- `num_hist`: Number of history frames
- `training.epochs`: Number of training epochs
- `bisim_latent_dim`: Bisimulation latent dimension
- `bisim_hidden_dim`: Bisimulation hidden dimension
- `model.bypass_dinov2`: Whether to bypass DINOv2
- `bisim_coef`: Bisimulation loss coefficient
- `training.bisim_lr`: Learning rate
- `var_loss_coef`: Variance loss coefficient
- `num_pcs`: Number of principal components
- `VC_target`: Variance covariance target
- `PCA1_loss_target`: PCA1 loss target
- `PCAloss_epoch`: PCA loss epoch (should be less than total number of training epochs)

## Model Naming

The script automatically generates interpretable model names based on the hyperparameters:

```text
bisim_512_hidden_1024_no_dino_lr_5e-06_varloss_coef_1_numpcs_10_vctarget_1_pca1target_0.01_pcaepoch_50_point_maze_fs_5_hist_3_epochs_50
```

## Command Line Arguments

- `--config-file`: JSON file containing hyperparameter configurations (default: train_sweep_config.json)
- `--gpus`: List of GPU IDs to use for training (required)
- `--timeout`: Timeout per training job in seconds (default: 36000 = 10 hours)
- `--output-file`: Output file for results (default: train_sweep_results.json)
- `--max-processes-per-gpu`: Maximum number of processes per GPU (default: 1)
- `--memory-threshold-gb`: GPU memory threshold in GB before considering it full (default: 40.0)
- `--disable-gpu-monitoring`: Disable GPU memory monitoring

## GPU Queue System

The script uses an intelligent queue system to manage GPU resources:

1. **GPU Monitoring**: Monitors GPU memory usage and process count to determine availability
2. **Smart Assignment**: Assigns models to GPUs with the lowest memory usage and process count
3. **Memory Management**: Waits for GPU memory to be released before starting new processes
4. **Process Management**: Automatically cleans up completed processes and their resources
5. **Resource Utilization**: Maximizes GPU utilization by keeping all specified GPUs busy until all models are trained
6. **Fault Tolerance**: Handles process cleanup on interruption (Ctrl+C) and saves partial results

## Output

The script generates:

1. **Model directories**: Each trained model is saved in `./outputs/{model_name}/`
2. **Log files**: Training logs are saved as `train_stdout.log` and `train_stderr.log` in each model directory
3. **Results file**: Training results are saved to the specified output file (default: `train_sweep_results.json`)
4. **Partial results**: If interrupted, partial results are saved to `{output_file}.partial`

## Example Commands

### Multi-GPU Training
```bash
python train_sweep.py --config-file train_sweep_config.json --gpus 0 1 2 3
```

### Single GPU Training
```bash
python train_sweep.py --config-file train_sweep_config.json --gpus 0
```

### Custom Timeout and Output
```bash
python train_sweep.py --config-file train_sweep_config.json --gpus 0 1 --timeout 7200 --output-file my_results.json
```

### Advanced GPU Management
```bash
python train_sweep.py --config-file train_sweep_config.json --gpus 0 1 2 3 --max-processes-per-gpu 2 --memory-threshold-gb 30
```

### Disable GPU Monitoring
```bash
python train_sweep.py --config-file train_sweep_config.json --gpus 0 1 --disable-gpu-monitoring
```

## Results Format

The results file contains a JSON array with information about each training run:

```json
[
  {
    "model_name": "bisim_512_hidden_1024_no_dino_lr_5e-06_varloss_coef_1_point_maze_fs_5_hist_3_epochs_50",
    "hyperparams": {
      "env": "point_maze",
      "frameskip": 5,
      "num_hist": 3,
      "training.epochs": 50,
      "bisim_latent_dim": 512,
      "bisim_hidden_dim": 1024,
      "model.bypass_dinov2": true,
      "training.bisim_lr": 5e-6,
      "var_loss_coef": 1.0,
      "PCA1_loss_target": 0.01,
      "PCAloss_epoch": 50
    },
    "success": true,
    "returncode": 0,
    "duration_seconds": 3600,
    "gpu_id": 0,
    "stdout": "...",
    "stderr": "..."
  }
]
```

## Notes

- The script uses a GPU queue system
- Each model is trained independently on a single GPU
- Models are assigned to GPUs based on memory usage and process count (not round-robin)
- GPU memory monitoring requires `nvidia-ml-py3` and `psutil` packages
- The script provides real-time progress updates and final summary statistics
- All models run in parallel across the specified GPUs until completion
- Process cleanup is handled automatically on interruption (Ctrl+C)
- Training logs are captured and stored for each model
- The script waits for GPU memory to be released before starting new processes