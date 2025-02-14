#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <script>"
    exit 1
fi

python_file="$1"
full_file_name="${python_file%.*}"
base_file_name=$(basename "$full_file_name")

cat > "${full_file_name}.slurm" << EOL
#!/bin/bash
#SBATCH --job-name=${base_file_name}
#SBATCH -A jeisner1_gpu
#SBATCH --partition=ica100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=${full_file_name}.out

uv run ${python_file}
EOL

chmod +x "${full_file_name}.slurm"
echo "Generated ${full_file_name}.slurm"

