import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('/home/wby/baselines/SR_TAMP/eval/global_results_single.csv')

# Multiply by 100 for percentages
df['add_success_rate'] *= 100
df['adds_success_rate'] *= 100

# Calculate means per method and object
df_grouped = df.groupby(['method', 'object_name'])[['add_success_rate', 'adds_success_rate']].mean().reset_index()

# Calculate overall average per method
df_avg = df.groupby(['method'])[['add_success_rate', 'adds_success_rate']].mean().reset_index()
df_avg['object_name'] = 'Avg.'

# Combine
df_all = pd.concat([df_grouped, df_avg], ignore_index=True)

# Format the cells: ADD / ADD-S
df_all['cell'] = df_all.apply(lambda row: f"{row['add_success_rate']:.1f} / {row['adds_success_rate']:.1f}", axis=1)

# Pivot table: rows=methods, cols=objects
pivot = df_all.pivot(index='method', columns='object_name', values='cell')

# Order of methods
method_order = ['TSDF++', 'MidFusion', 'BundleSDF', 'FoundationPose', 'Ours']
# Keep methods that exist
method_order = [m for m in method_order if m in pivot.index]
pivot = pivot.reindex(method_order)

# Order of objects: alphabetical + Avg.
objects = sorted([c for c in pivot.columns if c != 'Avg.'])
col_order = objects + ['Avg.']
pivot = pivot[col_order]

# Format object names
col_names = [c.replace('_', ' ').title() if c != 'Avg.' else c for c in col_order]

num_cols = len(col_names) + 1
print("\\begin{table*}[t]")
print("\\centering")
print("\\caption{Single Object Tracking Performance. Each cell reports ADD / ADD-S in percentage.}")
print("\\label{tab:single_obj_detailed}")
print("\\resizebox{\\textwidth}{!}{")
print("\\begin{tabular}{l" + "c" * (num_cols - 1) + "}")
print("\\toprule")
print("Method & " + " & ".join(col_names) + " \\\\")
print("\\midrule")

for method in method_order:
    row_vals = pivot.loc[method].values
    # highlight best in bold? The user didn't explicitly ask to bold best, but it's standard.
    # Let's just output the plain values, or maybe we can do bolding if we want.
    # For now, just print the values.
    print(f"{method} & " + " & ".join(row_vals) + " \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("}")
print("\\end{table*}")
