
import pandas as pd
import ast
import glob
import os

def parse_demographic_column(demographic_entry):
    """Parse and normalize DeepFace demographic output from a list or dict."""
    if demographic_entry:
        return pd.json_normalize(demographic_entry).iloc[0]
    return pd.Series(dtype='object')

def aggregate_video_keyframe_features(csv_path):
    """
    Aggregate frame-level features (RGB + DeepFace) into a single video-level summary.

    Parameters:
        csv_path (str): Path to a CSV file containing keyframe-level features.

    Returns:
        dict: Aggregated features for a video.
    """
    keyframe_df = pd.read_csv(csv_path)
    keyframe_df['demographics'] = keyframe_df['demographics'].apply(ast.literal_eval)

    demographics_df = keyframe_df['demographics'].apply(parse_demographic_column)
    keyframe_df = pd.concat([keyframe_df.drop(columns=['demographics']), demographics_df], axis=1)

    video_id = os.path.basename(csv_path)[:19]
    summary = {
        'video_id': video_id,
        'keyframe_count': len(keyframe_df),
        'mean_red': keyframe_df['r_mean'].mean(),
        'mean_green': keyframe_df['g_mean'].mean(),
        'mean_blue': keyframe_df['b_mean'].mean(),
        'all_races': keyframe_df.get('dominant_race', []).tolist(),
        'all_emotions': keyframe_df.get('dominant_emotion', []).tolist(),
        'all_ages': keyframe_df.get('age', []).tolist(),
        'all_genders': keyframe_df.get('dominant_gender', []).tolist(),
    }

    return summary

def compute_proportions(value_list, target_value):
    """Compute the proportion of a target value in a list, ignoring NaNs."""
    clean_values = [v for v in value_list if pd.notna(v)]
    return clean_values.count(target_value) / len(clean_values) if clean_values else 0

def compute_mean(values):
    """Compute mean of a list while ignoring NaNs."""
    clean_values = [v for v in values if pd.notna(v)]
    return sum(clean_values) / len(clean_values) if clean_values else 0

def summarize_keyframe_features(directory_path):
    """
    Summarize all keyframe CSVs in a directory into a DataFrame of video-level features.

    Parameters:
        directory_path (str): Directory containing per-video CSV files.

    Returns:
        pd.DataFrame: Aggregated video-level features.
    """
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    video_summaries = [aggregate_video_keyframe_features(f) for f in csv_files]
    summary_df = pd.DataFrame(video_summaries)

    summary_df['Proportion_White'] = summary_df['all_races'].apply(lambda x: compute_proportions(x, 'white'))
    summary_df['Age'] = summary_df['all_ages'].apply(compute_mean)
    summary_df['Proportion_Angry'] = summary_df['all_emotions'].apply(lambda x: compute_proportions(x, 'angry'))
    summary_df['Proportion_Sad'] = summary_df['all_emotions'].apply(lambda x: compute_proportions(x, 'sad'))
    summary_df['Proportion_Happy'] = summary_df['all_emotions'].apply(lambda x: compute_proportions(x, 'happy'))
    summary_df['Proportion_Fearful'] = summary_df['all_emotions'].apply(lambda x: compute_proportions(x, 'fear'))
    summary_df['Proportion_Male'] = summary_df['all_genders'].apply(lambda x: compute_proportions(x, 'Man'))
    summary_df['Proportion_Female'] = summary_df['all_genders'].apply(lambda x: compute_proportions(x, 'Woman'))

    return summary_df

# ========== USAGE EXAMPLE ==========

# Example usage for processing all keyframe feature CSVs in a directory:
if __name__ == "__main__":
    input_directory = "/path/to/keyframe_csvs"

    # Aggregate video-level features
    summary_df = summarize_keyframe_features(input_directory)
