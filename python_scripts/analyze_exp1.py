import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = '../results/exp1/exp1.csv'

def plot_ols_fit(df, theta_hat, threshold, output_path='ols_fit.png'):
    k_values = np.sort(df['k'].unique())

    beta_stable = theta_hat[5]
    beta_growing = theta_hat[6]

    plt.figure(figsize=(8, 5))

    for i, k in enumerate(k_values):
        data = df[df['k'] == k].sort_values('L')

        L = data['L'].to_numpy()
        y = data['log2_mean_stash'].to_numpy()

        a_k = theta_hat[i]

        if k <= threshold:
            beta = beta_stable
        else:
            beta = beta_growing

        y_hat = a_k + beta * L

        line, = plt.plot(
            L,
            y_hat,
            label=f'$k={k}$'
        )

        plt.scatter(
            L,
            y,
            color=line.get_color()
        )

    plt.xlabel(r'$L$', fontsize=16)
    plt.ylabel(r'$\log_2 \mu_{L,k}$', fontsize=16)

    plt.xticks(np.arange(df['L'].min(), df['L'].max() + 1, 1), fontsize=13)
    plt.yticks(fontsize=13)

    plt.legend(fontsize=13)

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_path,
        bbox_inches='tight'
    )

    plt.close()

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
        # Parameters are: [log2(C_3), log2(C_4), log2(C_5), log2(C_6), log2(C_7), beta_stable, beta_growing]
        # this will be the columns for X
        X = []
        # y is the log2 means
        y = []

        # populate rows of X and corresponding y
        for _, row in df.iterrows():
            x = np.zeros(7)
            L = row['L']
            k = row['k']
            y.append(row['log2_mean_stash'])
            # we match with the corresponding coefficients
            x[k - 3] = 1.0
            if k <= t:
                x[5] = L
            else:
                x[6] = L
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