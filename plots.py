import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Take results of the form q1=0.5851980447769165, q2=0.7368029356002808, q3=0.8716562390327454, min=-0.1974795013666153, max=0.9958550930023193 and return just the numbers
def get_results(line):
    results = line.split(", ")
    final_results = [float(re.findall(r'[\-]*[0-9]+\.[0-9]+', r)[0]) for r in results]
    return final_results

def process_data(file, num_classes, sub_set_classes = None):
    # Remove empty line
    file.readline()
    # Remove Total Stat Header
    file.readline()
    # Add total to dictionary
    total_results = get_results(file.readline().strip())
    # Get results for the classes
    class_results = []
    for i in range(num_classes):
       file.readline()
       if sub_set_classes is None or i in sub_set_classes:
            class_results.append(get_results(file.readline().strip()))  
       else:
            file.readline()
    return class_results, total_results, file

def create_plot_sns(data, title, filename, x_val, hue_val, distance=False, jaccard=False):
    # fig, ax = plt.subplots()
    # # "GraphSAGE":"#00549F", "node2vec":"#57AB27"
    color_dict = {"FastRP":"#E30066", "GATV2":"#612158", "GraphSAGE":"#F6A800", "GIN":"#00549F", "Node2Vec":"#57AB27"}

    light_gray = ".8"
    dark_gray =".15"
    sns.set(context="notebook", style="whitegrid", font_scale=1,
        rc={"axes.edgecolor": light_gray, "xtick.color": dark_gray,
            "ytick.color": dark_gray, "xtick.bottom": True,
            "font.size":8,"axes.titlesize":6,"axes.labelsize":6, "xtick.labelsize":15, "legend.fontsize":6, 
            "ytick.labelsize":15, "axes.linewidth":1, 
            "xtick.minor.width":0.5, "xtick.major.width":0.5,
            "ytick.minor.width":0.5, "ytick.major.width":0.5, "lines.linewidth": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "grid.linewidth":0.5
           })

    g = sns.catplot(data=data, x=x_val, y="Similarity", kind="box", hue=hue_val, legend=False,
            palette=color_dict, aspect=2, whis=1000000)
            #height=width/2, 
    g.set_ylabels(title, fontsize='17')
    g.set_xlabels("")
    g.axes[0,0].legend(loc='upper center', bbox_to_anchor=(0.45, -0.16), fancybox=False, shadow=False, ncol=5, fontsize='17')
    if distance:
        g.axes[0,0].set_ylim(0.0, 180)
    elif jaccard:
        g.axes[0,0].set_ylim(0.0, 1.0)
    else:
        g.axes[0,0].set_ylim(-1, 1)
    
    g.set_titles("")
    g.savefig("plots/"+filename+".pdf", bbox_inches="tight")


directory = "./similarity_results/"
# Iterate over files in directory

total_stats = {}
class_stats_cora = {}
class_stats_coauthor = {}
class_stats_arvix = {}

cora_classes = [0,1,2,3,4,5,6]
cora_classes_order = [6,1,5,0,2,4,3]
coauthor_classes = [2,5,6,9,12,13,14]
coauthor_classes_order = [9,6,12,14,2,5,13]
arvix_classes = [12, 16, 20, 21, 24, 28, 35]
arvix_classes_order = [12,35,21,20,28,24,16]
class_graphs_labels = ["A", "B", "C", "D", "E", "F", "G"]
arvix_metrics = ["aligned cosine", "unaligned cosine", "distance"]

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        model_dataset = filename.split("-")
        model = model_dataset[0]
        dataset = model_dataset[1][:(len(model_dataset[1])-4)]

        if dataset == "Cora":
            NUM_CLASSES = 7
            include_classes = cora_classes
            sorted_classes = cora_classes_order
        elif dataset == "Coauthor":
            NUM_CLASSES = 15
            include_classes = coauthor_classes
            sorted_classes = coauthor_classes_order
        else:
            NUM_CLASSES = 40
            include_classes = arvix_classes
            sorted_classes = arvix_classes_order
        # NUM_CLASSES = 7 if dataset == "Cora" else 15
        # coauthor_classes =  if dataset == "Coauthor" else [0,1,2,3,4,5,6]

        print("Reading plot data for", model, "with dataset", dataset)

        with open(directory+filename) as file:
            while True:

                # Get next line from file
                line = file.readline()

                # if line is empty
                # end of file is reached
                if not line:
                    break
                
                line = line.strip()
                if line == "aligned cosine" or line == "jaccard" or line == "2nd order cosine" or line == "unaligned cosine" or line == "distance":
                    key_val = line
                    print("Reading in data for metric", key_val)
                    if key_val not in total_stats:
                        print("Adding new key to table")
                        total_stats[key_val] = []
                        class_stats_cora[key_val] = []
                        class_stats_coauthor[key_val] = []
                        if key_val in arvix_metrics:
                            print("Adding new key to arvix table")
                            class_stats_arvix[key_val] = []

                    class_results, total_results, file = process_data(file, NUM_CLASSES, sub_set_classes=include_classes)
                    for result in total_results:
                        total_stats[key_val].append([result, model, dataset])
                    if dataset == "Cora":
                        for i, class_result in zip(include_classes, class_results):
                            for result in class_result:
                                label = class_graphs_labels[sorted_classes.index(i)]
                                class_stats_cora[key_val].append([result, model, "Class "+str(label)])
                    elif dataset == "Coauthor":
                        for i, class_result in zip(include_classes, class_results):
                            for result in class_result:
                                label = class_graphs_labels[sorted_classes.index(i)]
                                print(i)
                                print(label)
                                class_stats_coauthor[key_val].append([result, model, "Class "+str(label)])
                    elif dataset == "Arvix" and key_val in arvix_metrics:
                        for i, class_result in zip(include_classes, class_results):
                            for result in class_result:
                                label = class_graphs_labels[sorted_classes.index(i)]
                                class_stats_arvix[key_val].append([result, model, "Class "+str(label)])

                                
print("total", len(total_stats["aligned cosine"]))
print("cora", len(class_stats_cora["aligned cosine"]))
print("coauthor", len(class_stats_coauthor["aligned cosine"]))
print("arvix", len(class_stats_arvix["aligned cosine"]))
            
# Create dataframes and create plots 
metrics = ["aligned cosine", "jaccard", "2nd order cosine", "unaligned cosine", "distance"]
title_maps = {"aligned cosine":"Aligned Cosine Similarity", "jaccard":"20-NN Jaccard Similarity", "2nd order cosine":"2nd Order Cosine Similarity", "unaligned cosine":"Cosine Similarity", "distance":"Euclidean Distance"}
for metric in metrics:
    # Change axis based on which metric we use
    if metric=="distance":
        is_distance = True
    else:
        is_distance = False 

    if metric=="jaccard":
        is_jaccard = True
    else:
        is_jaccard = False 
    total_metric_results = total_stats[metric]
    df_total = pd.DataFrame(np.array(total_metric_results), columns=['Similarity', 'Model', 'Dataset'])
    df_total = df_total.sort_values(by=['Model', 'Dataset'])
    # Ensure that the simlarity value is a float
    df_total = df_total.explode('Similarity')
    df_total['Similarity'] = df_total['Similarity'].astype('float')
    create_plot_sns(df_total, title_maps[metric], filename=metric+"_total", x_val='Dataset', hue_val='Model', distance=is_distance, jaccard=is_jaccard)

    cora_metric_results = class_stats_cora[metric]
    df_cora = pd.DataFrame(np.array(cora_metric_results), columns=['Similarity', 'Model', 'Class'])
    df_cora = df_cora.sort_values(by=['Model', 'Class'])
    # Ensure that the simlarity value is a float
    df_cora = df_cora.explode('Similarity')
    df_cora['Similarity'] = df_cora['Similarity'].astype('float')
    create_plot_sns(df_cora, title_maps[metric], filename=metric+"_cora", x_val='Class', hue_val='Model', distance=is_distance, jaccard=is_jaccard)

    coauthor_metric_results = class_stats_coauthor[metric]
    df_coauthor = pd.DataFrame(np.array(coauthor_metric_results), columns=['Similarity', 'Model', 'Class'])
    df_coauthor = df_coauthor.sort_values(by=['Model', 'Class'])
    # Ensure that the simlarity value is a float
    df_coauthor = df_coauthor.explode('Similarity')
    df_coauthor['Similarity'] = df_coauthor['Similarity'].astype('float')
    create_plot_sns(df_coauthor, title_maps[metric], filename=metric+"_coauthor", x_val='Class', hue_val='Model', distance=is_distance, jaccard=is_jaccard)

    if metric in arvix_metrics:
        arvix_metric_results = class_stats_arvix[metric]
        df_arvix = pd.DataFrame(np.array(arvix_metric_results), columns=['Similarity', 'Model', 'Class'])
        df_arvix = df_arvix.sort_values(by=['Model', 'Class'])
        # Ensure that the simlarity value is a float
        df_arvix = df_arvix.explode('Similarity')
        df_arvix['Similarity'] = df_arvix['Similarity'].astype('float')
        create_plot_sns(df_arvix, title_maps[metric], filename=metric+"_arvix", x_val='Class', hue_val='Model', distance=is_distance, jaccard=is_jaccard)