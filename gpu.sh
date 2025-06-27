srun -p dev_gpu_h100  --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=24 --mem=193gb --time=00:30:00 --pty bash
