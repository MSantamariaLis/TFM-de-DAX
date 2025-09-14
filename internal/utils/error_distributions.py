import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

class ErrorDistributionPlotter:
    
    def __init__(self, errors_dict: dict):
        self.errors_dict = errors_dict

    def plot_error_distributions(self):

        """Generate a single combined plot image for all models."""
        num_models = len(self.errors_dict)
        fig, axes = plt.subplots(num_models, 3, figsize=(15, 5 * num_models))

        # Handle case where there's only one model (axes won't be 2D)
        if num_models == 1:
            axes = axes.reshape(1, -1)

        for row_idx, (model, metrics) in enumerate(self.errors_dict.items()):
            df = pd.DataFrame(metrics)

            for col_idx, metric in enumerate(['mae', 'mape', 'rmse']):
                ax = axes[row_idx, col_idx]
                sns.histplot(df[metric], kde=True, bins=20, color=['blue', 'green', 'red'][col_idx], ax=ax)

                mean_val = df[metric].mean()
                median_val = df[metric].median()

                ax.axvline(mean_val, color='black', linestyle='--', label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='orange', linestyle='-', label=f'Median: {median_val:.2f}')

                ax.set_title(f'{model} - {metric.upper()} Distribution')
                ax.set_xlabel(metric.upper())
                ax.legend()

        plt.tight_layout()

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)  
        plt.close(fig)

        return img_buf 
