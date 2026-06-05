import json

with open('part2_notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'y_test_raw = test_df' in line and 'Salary' in line:
                new_source.append(line)
                new_source.append('if "Log_Salary" in X_train_processed.columns:\n')
                new_source.append('    y_train = X_train_processed.pop("Log_Salary")\n')
                new_source.append('else:\n')
                new_source.append('    y_train = train_df["Log_Salary"]\n')
                new_source.append('if "Log_Salary" in X_test_processed.columns:\n')
                new_source.append('    X_test_processed.pop("Log_Salary")\n')
            else:
                new_source.append(line)
        cell['source'] = new_source

with open('part2_notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Notebook fixed!')
