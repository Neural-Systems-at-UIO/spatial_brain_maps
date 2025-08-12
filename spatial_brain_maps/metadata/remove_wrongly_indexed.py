import pandas as pd

"""
We checked which datasets were misnumbered and here we remove them. 
"""
metadata = pd.read_csv("filtered_ISH_pixel_size.csv", index_col=0)
exclude = [2193, 1878, 633]
metadata = metadata[~metadata["experiment_id"].isin(exclude)]
metadata = metadata.reset_index(drop=True)
metadata.to_csv("metadata_exp_removed.csv", index=None)
