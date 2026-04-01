import pandas as pd
df_multi = pd.read_csv('/home/wby/baselines/SR_TAMP/eval/global_results_multi.csv')
print("Multi objects:", df_multi['object_name'].unique())
