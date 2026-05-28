import json

notebook_path = 'part2_notebook.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

pipeline_index = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # 1. Modify the first heatmap
        if "corr_matrix = df.select_dtypes" in source:
            # We add the drop command before corr_matrix calculation
            new_source = []
            for line in cell['source']:
                if "corr_matrix = df.select_dtypes" in line:
                    new_source.append("df = df.drop(columns=['Unnamed: 0'], errors='ignore')\n")
                new_source.append(line)
            cell['source'] = new_source
            
        # 2. Find the pipeline cell
        if "pipeline = NBADataPipeline()" in source and "process_pipeline" in source:
            pipeline_index = i

if pipeline_index != -1:
    # Insert new cells after pipeline_index
    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Khảo sát lại ma trận tương quan SAU KHI qua Pipeline\n",
            "Như bạn có thể thấy ở output trên, Pipeline đã loại bỏ các biến có độ tương quan thấp và các biến bị đa cộng tuyến cao bằng VIF. Chúng ta hãy vẽ lại Heatmap để kiểm chứng:"
        ]
    }
    
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Lọc ra các cột số (không tính cột phân loại - Dummy) trong X_train để vẽ\n",
            "numeric_cols_after = pipeline.numeric_features\n",
            "corr_matrix_after = X_train[numeric_cols_after].corr()\n",
            "\n",
            "plt.figure(figsize=(16, 12))\n",
            "sns.heatmap(corr_matrix_after, annot=True, annot_kws={'size': 10}, cmap='coolwarm', fmt=\".2f\", linewidths=0.5, vmin=-1, vmax=1)\n",
            "plt.xticks(rotation=45, ha='right', fontsize=11)\n",
            "plt.yticks(fontsize=11)\n",
            "plt.title(f'Ma trận tương quan SAU KHI lọc VIF (chỉ còn lại {len(numeric_cols_after)} biến số)')\n",
            "plt.show()"
        ]
    }
    
    nb['cells'].insert(pipeline_index + 1, code_cell)
    nb['cells'].insert(pipeline_index + 1, markdown_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook fixed successfully!")
