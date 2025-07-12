
"""
RQ1 Statistical Analysis: Mann-Whitney U Test with FDR Correction

This script compares interactions and view counts between non-partisan (Neither) and partisan
(D-leaning or R-leaning) TikTok videos using the Mann–Whitney U test. It adjusts for multiple
comparisons using the Benjamini-Hochberg False Discovery Rate correction.
"""

import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def compute_mwu_effect_size(group1, group2):
    """
    Compute the Mann–Whitney U test and effect size.

    Effect size is calculated as:
        1 - (2 * U) / (n1 * n2), where U is the test statistic.

    Returns:
        p-value and effect size
    """
    u_statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    effect_size = 1 - (2 * u_statistic) / (len(group1) * len(group2))
    return p_value, effect_size

def run_mwu_with_fdr(df, group_col, group1_label, group2_labels, test_cols):
    """
    Run Mann-Whitney U tests across multiple features and apply FDR correction.

    Parameters:
        df (pd.DataFrame): Data with group and feature columns
        group_col (str): Column name indicating group membership
        group1_label (str): Label for baseline group (e.g., 'Neither')
        group2_labels (list): Labels for comparison groups (e.g., ['Democrat', 'Republican'])
        test_cols (list): Features to compare (e.g., ['interactions', 'views'])

    Returns:
        pd.DataFrame: Test results including raw and corrected p-values
    """
    results = []

    group1 = df[df[group_col] == group1_label]
    group2 = df[df[group_col].isin(group2_labels)]

    for col in test_cols:
        p_val, eff_size = compute_mwu_effect_size(group1[col], group2[col])
        results.append({
            'feature': col,
            'p_value': p_val,
            'effect_size': eff_size
        })

    results_df = pd.DataFrame(results)
    results_df['corrected_p_value'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
    return results_df


# ========== USAGE EXAMPLE ==========

if __name__ == "__main__":
    # Load the preprocessed TikTok dataset (must contain 'partisan_leaning', 'interactios', 'views')
    df = pd.read_csv("/path/to/your/data.csv")

    features_to_test = ['interactions', 'views']

    results_rq1 = run_mwu_with_fdr(
        df=df,
        group_col='partisan_leaning',
        group1_label='Neither',
        group2_labels=['D-leaning', 'R-leaning'],
        test_cols=features_to_test
    )

    print(results_rq1)
    # Optionally save to CSV
    # results_rq1.to_csv("RQ1_MWU_results.csv", index=False)
