import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.misc import load_jsonl, load_json
from loguru import logger

def draw(dataset=None, output_path=None):
    # Iterating through each file and extract the 3 columns we need
    if dataset is None:
        if output_path.endswith('.json'):
            dataset = load_json(output_path)
        elif output_path.endswith('.jsonl'):
            dataset = load_jsonl(output_path)
        else:
            raise ValueError(f"Unsupported file format: {output_path}")
    
    acc_list = []
    for entry in dataset:
            # Extracting the required fields
            document_depth = entry.get("depth_percent", None)
            context_length = entry.get("context_length", None)
            score = entry.get("score", None)
            # Appending to the list
            acc_list.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "score": score
            })
    # Creating a DataFrame
    df = pd.DataFrame(acc_list)
    # df['Context Length1'] = [int(i[:-1]) for i in df['Context Length']]
    vmin, vmax = 0.00, 2.00
    from matplotlib.colors import LinearSegmentedColormap
    # Create the pivot table
    pivot_table = pd.pivot_table(df, values='score', index=['Document Depth'], columns=['Context Length'],
                                 aggfunc='mean')
    # Create a custom colormap for the heatmap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with the custom normalization
    plt.figure(figsize=(11, 4))  # Adjust these dimensions as needed
    sns.heatmap(
        pivot_table,
        cmap=cmap,
        vmin=vmin,  # Use the TwoSlopeNorm object
        cbar_kws={'label': 'score'}
        # Uncomment the following line if you want to annotate cells with their numerical value
        # annot=True
    )
    # More aesthetics
    # title = output_path.split('/')[-1].split('.')[0]
    # plt.title(title)  # Adds a title, uncomment to use
    plt.xlabel('Token Limit / K')  # X-axis label
    plt.ylabel('Depth Percent / %')  # Y-axis label
    plt.xticks(rotation=0)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # Show the plot
    if output_path is not None:
        output_path = output_path.replace("jsonl", "png") if output_path.endswith(".jsonl") else output_path.replace("json", "png")
        plt.savefig(output_path)
        logger.info(f'save to: {output_path}')
    else:
        plt.show()
# unit test
if __name__ == "__main__":
    import os    
    fnames = os.listdir("output/needle_haystack/San Francisco/")
    for fname in fnames:
        if fname.endswith(".jsonl"):
            print(fname)
            input_file_name = f"output/needle_haystack/San Francisco/{fname}"
            draw(output_path=input_file_name, title=fname)
