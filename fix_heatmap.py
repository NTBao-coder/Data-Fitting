import json

notebook_path = 'part2_notebook.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "numeric_cols_after = pipeline.numeric_features" in source:
            new_source = []
            for line in cell['source']:
                if "numeric_cols_after = pipeline.numeric_features" in line:
                    new_source.append("numeric_cols_after = pipeline.vif_passed_features\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Fixed heatmap code!")
