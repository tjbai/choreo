#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <script>"
    exit 1
fi

python_file="$1"
file_name="${python_file%.*}"

cat > "${file_name}.slurm" << EOL
#!/bin/bash
#SBATCH --job-name=${file_name}
#SBATCH -A jeisner1_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:0
#SBATCH --output=${file_name}.out

uv run ${python_file}
EOL

chmod +x "${file_name}.slurm"
echo "Generated ${file_name}.slurm"
