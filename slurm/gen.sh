#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <script>"
    exit 1
fi

python_file="$1"
full_file_name="${python_file%.*}"

# Get the last two elements of the path
path_parts=( $(echo "$full_file_name" | tr '/' ' ') )
num_parts=${#path_parts[@]}

if [ $num_parts -ge 2 ]; then
    last_two_dirs="${path_parts[$num_parts-2]}/${path_parts[$num_parts-1]}"
else
    last_two_dirs="$full_file_name"
fi

base_file_name=$(basename "$full_file_name")

cat > "${full_file_name}.slurm" << EOL
#!/bin/bash
#SBATCH --job-name=${last_two_dirs}
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
