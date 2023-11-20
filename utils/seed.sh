#!/bin/bash

output_csv="/mnt/home/mjhajaria/transforms/seed_data.csv"

# file_count=$(find /mnt/home/mjhajaria/ceph/stan_output/simplex -type f -name "*.csv" | wc -l)
# processed=0

# # Create CSV header
# echo "Filepath,Seed" > "$output_csv"

# # Process CSV files and track progress
# find /mnt/home/mjhajaria/ceph/stan_output/simplex -type f -name "*.csv" -print0 |
# while IFS= read -r -d '' file; do
#     ((processed++))
#     seed=$(head -n 50 "$file" | grep -i 'seed' | awk '{print $NF}')
#     echo "\"$file\",\"$seed\"" >> "$output_csv"
#     percentage=$((processed * 100 / file_count))
#     echo -ne "Progress: $percentage% ($processed/$file_count)\r"
# done

# echo -e "\nProcessing complete!"
#!/bin/bash


# Create CSV header
