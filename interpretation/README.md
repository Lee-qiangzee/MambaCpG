# interpretation

## generate_samples.py

Generate the site samples to be analyzed. An equal number of methylated and unmethylated samples are selected within each cell.The output is a configuration file similar to that of CpG Transformer(https://github.com/gdewael/cpg-transformer/tree/main/interpretation).

Every row indicates one site to interpret. Columns denote the following:
1.chromosome key in the input files.
2.row (cell) index of the site to be interpreted.
3.column (CpG site) index of the site to be interpreted.
4.Reference label of the site to be interpreted. 

```bash
python generate_samples.py
```

## integrated_gradients.py

Integrated Gradients decomposes the model output into contributions from individual features by calculating the difference between the prediction of the input sample and a baseline.

```bash
python integrated_gradients.py X.npz y.npz pos.npz --output --halfwindowsize --model_checkpoint --config_file 
```

## contribution_matrix.py

Plot a heatmap histogram of the contribution matrix.

```bash
python contribution_matrix.py
```

## cell_correlation.py

Calculate cell correlation based on contribution scores.

```bash
python cell_correlation.py
```