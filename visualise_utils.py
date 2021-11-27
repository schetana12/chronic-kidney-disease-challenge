"""
    Contains the methods used in visualisation of factors identified for subtypes.
    Heatmaps are created for categorical features based on disease class.
    Scatterplot as pairs are created for numerical features colored on disease class.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os

def visualise_categories_pca(data, target_class, feature_set, output_path):
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        n=3
        m = int(np.ceil(len(feature_set)/n))
        fig, axes = plt.subplots(m,n, squeeze=False,figsize=(15, 15))
        fig.suptitle('\nDistribution of each categorical factor identified to be probable subtype based on the target disease class')
        for i in range(m):
            for j in range(n):    
                if i*n+j == len(feature_set):
                    break
                categ = feature_set[i*n+j]
                sns.heatmap(pd.crosstab(data[categ],data[target_class]),
                            ax=axes[i,j],
                            cmap='Blues',
                            square='True',
                            cbar=False,
                            annot=True,
                        fmt='d')

                axes[i,j].set_ylabel(categ)
                axes[i,j].set_xlabel("Target Labels")
        plt.savefig(f'{output_path}/pca_categories.png')


def visualise_numerics_pca(data, target_class, feature_set, output_path):
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
            
        imp_features = feature_set + [target_class]
        data_x = data[imp_features]
        pairplot = sns.pairplot(data_x, corner=True, hue=target_class)
        pairplot.fig.suptitle("\nDistribution of each numerical factor identified to be probable subtype colored on the target disease class",
        y=1.08)
        pairplot.savefig(f'{output_path}/pca_numerics.png') 