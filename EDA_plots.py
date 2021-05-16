import matplotlib.pyplot as plt


def plot_missing_values(df):
    perc_missing = []
    for var in df.columns:
        perc_missing.append(df[var].isna().sum() / len(df[var]))
        
    labels = list(df.columns)
    labels = [label for _, label in sorted(zip(perc_missing, labels))]
    perc_missing.sort()
    values = perc_missing
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(20,8))
    ax.bar(x, values)
    ax.set_title('Percentage of missing values', fontsize=20)
    ax.set_ylabel('Percentage of values missing', fontsize=16)
    ax.set_xlabel('Attributes', fontsize=16)
    plt.xticks(rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.savefig()