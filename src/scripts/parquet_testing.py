import pandas as pd
import os

# df = pd.read_parquet("./cache/datasets--EleutherAI--wmdp_bio_cloze/snapshots/f7f4d630417c01ae23b9d9845f79e72cf391384e/data/cloze_compatible-00000-of-00001.parquet")
# print(df.columns.tolist())
# print(df["answer"].unique())

df_bioweapons_and_bioterrorism = pd.read_parquet("./cache/datasets--EleutherAI--wmdp_bio_robust_mcqa/snapshots/6ec6af2c25473039c23acfa68f3ef1b5df77536c/bioweapons_and_bioterrorism/robust-00000-of-00001.parquet")
print(df_bioweapons_and_bioterrorism.columns.tolist())
print(df_bioweapons_and_bioterrorism["answer"].unique())