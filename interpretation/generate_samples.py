import os
import numpy as np

file_path = '/MBL/y.npz'
data = np.load(file_path)

os.makedirs('methylation_files', exist_ok=True)
os.makedirs('unmethylation_files', exist_ok=True)

for i in range(0, 30):
    # Using chromosome 1 as a representative example
    indices_ones = np.where(data['chr1'][:, i] == 1)[0]
    indices_zeros = np.where(data['chr1'][:, i] == 0)[0]

    # Randomly selecting 1000 samples per single cell
    num_samples = 1000
    random_indices_ones = np.random.choice(indices_ones, size=num_samples, replace=False)
    random_indices_zeros = np.random.choice(indices_zeros, size=num_samples, replace=False)

    output_file_ones = os.path.join('methylation_files', f'1{i}.txt')
    output_file_zeros = os.path.join('unmethylation_files', f'0{i}.txt')

    with open(output_file_ones, 'w') as f:
        for idx in random_indices_ones:
            line = f"chr1,{i},{idx},{data['chr1'][idx, i]}\n"
            f.write(line)

    with open(output_file_zeros, 'w') as f:
        for idx in random_indices_zeros:
            line = f"chr1,{i},{idx},{data['chr1'][idx, i]}\n"
            f.write(line)

    print(f"{i} Done")
