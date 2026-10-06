import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

file_path = './results/exp1/exp1.csv'

def shade_map(ks, cmap, lo=0.4, hi=0.9):
    """Map each k to a shade of one colormap (light = small k, dark = large k)."""
    ks = list(ks)
    if len(ks) == 1:
        shades = [0.5 * (lo + hi)]
    else:
        shades = np.linspace(lo, hi, len(ks))
    return dict(zip(ks, cmap(shades)))


def plot_ols_fit(df, theta_hat, threshold, output_path='./results/exp1/exp1_analysis/ols_fit.png'):
    k_values = np.sort(df['k'].unique())
    k_to_col = {k: i for i, k in enumerate(k_values)}
    n_k = len(k_values)

    beta_stable = theta_hat[n_k]
    beta_growing = theta_hat[n_k + 1]

    stable_ks = k_values[k_values <= threshold]
    growing_ks = k_values[k_values > threshold]

    colors = {
        **shade_map(stable_ks, plt.cm.Blues),
        **shade_map(growing_ks, plt.cm.Oranges),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    handles = {}

    for k in k_values:
        data = df[df['k'] == k].sort_values('L')
        L = data['L'].to_numpy()
        y = data['log2_mean_stash'].to_numpy()

        beta = beta_stable if k <= threshold else beta_growing
        y_hat = theta_hat[k_to_col[k]] + beta * L

        line, = ax.plot(L, y_hat, color=colors[k], label=f'$k={k}$')
        ax.scatter(L, y, color=colors[k], zorder=3)
        handles[k] = line

    ax.set_xlabel(r'$L$', fontsize=16)
    ax.set_ylabel(r'$\log_2 \mu_{L,k}$', fontsize=16)
    ax.set_xticks(np.arange(df['L'].min(), df['L'].max() + 1, 1))
    ax.tick_params(labelsize=13)

    # One legend per group, so the grouping is labelled as well as coloured
    leg_stable = ax.legend(
        handles=[handles[k] for k in stable_ks],
        title=rf'$k \leq {threshold}$',
        fontsize=12, title_fontsize=12,
        loc='upper left',
    )
    ax.add_artist(leg_stable)
    ax.legend(
        handles=[handles[k] for k in growing_ks],
        title=rf'$k > {threshold}$',
        fontsize=12, title_fontsize=12,
        loc='lower right',
    )

    ax.grid(True)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    print("\nsaved:\n"f"  {output_path}\n")

# def plot_ols_fit(df, theta_hat, threshold, output_path='./results/exp1/exp1_analysis/ols_fit.png'):
#     k_values = np.sort(df['k'].unique())

#     k_to_col = {k: i for i, k in enumerate(k_values)}
#     n_k = len(k_values)

#     beta_stable = theta_hat[n_k]
#     beta_growing = theta_hat[n_k + 1]

#     plt.figure(figsize=(8, 5))

#     for i, k in enumerate(k_values):
#         data = df[df['k'] == k].sort_values('L')

#         L = data['L'].to_numpy()
#         y = data['log2_mean_stash'].to_numpy()

#         a_k = theta_hat[k_to_col[k]]

#         if k <= threshold:
#             beta = beta_stable
#         else:
#             beta = beta_growing

#         y_hat = a_k + beta * L

#         line, = plt.plot(
#             L,
#             y_hat,
#             label=f'$k={k}$'
#         )

#         plt.scatter(
#             L,
#             y,
#             color=line.get_color()
#         )

#     plt.xlabel(r'$L$', fontsize=16)
#     plt.ylabel(r'$\log_2 \mu_{L,k}$', fontsize=16)

#     plt.xticks(np.arange(df['L'].min(), df['L'].max() + 1, 1), fontsize=13)
#     plt.yticks(fontsize=13)

#     plt.legend(fontsize=13)

#     plt.grid(True)
#     plt.tight_layout()

#     plt.savefig(
#         output_path,
#         bbox_inches='tight'
#     )

#     plt.close()


def ols(X, y):
    theta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    y_hat = X @ theta_hat
    residual = y - y_hat
    rss = np.sum(residual ** 2)

    return theta_hat, y_hat, rss

def main():
    csv_path = file_path
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        return
    
    if (df['mean_stash'] <= 0).any():
        raise ValueError(
            'mean_stash must be positive before taking log2.'
        )

    df = df.copy()
    df['log2_mean_stash'] = np.log2(df['mean_stash'])

    k_values = np.sort(df['k'].unique())
    k_to_col = {k: i for i, k in enumerate(k_values)}
    n_k = len(k_values)

    # candidate thresholds
    candid_t = df['k'].unique()
    candid_t.sort()
    candid_t = candid_t[:-1]

    # keep track of the best fit
    rss = float("inf")
    threshold = None
    y_hat = None
    theta_hat = None

    # estimate the constants
    for t in candid_t:
        # Parameters are: one log2(C_k) for each k, beta_stable, beta_growing
        # this will be the columns for X
        X = []
        # y is the log2 means
        y = []

        # populate rows of X and corresponding y
        for _, row in df.iterrows():
            x = np.zeros(n_k + 2)
            L = row['L']
            k = row['k']
            y.append(row['log2_mean_stash'])
            # we match with the corresponding coefficients
            x[k_to_col[k]] = 1.0
            if k <= t:
                x[n_k] = L
            else:
                x[n_k + 1] = L
            X.append(x)

        X = np.array(X)
        y = np.array(y)

        # we have the matrix y = X * (theta_hat)
        theta_hat_new, y_hat_new, rss_new = ols(X, y)

        if rss_new <= rss:
            rss = rss_new
            threshold = t
            y_hat = y_hat_new
            theta_hat = theta_hat_new

    print(f"threshold = {threshold}")
    plot_ols_fit(df, theta_hat, threshold)


        

if __name__ == '__main__':
    main()