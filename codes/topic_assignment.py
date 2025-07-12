
import pandas as pd
import ast
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# ---------- BERTopic Stage: Generate initial candidate keywords ----------

def generate_bertopic_model(texts, n_gram_range=(1, 2), top_n_topics=50):
    """
    Run BERTopic on the corpus and return topic model with candidate keywords.

    Parameters:
        texts (List[str]): A list of cleaned transcripts or text documents.
        n_gram_range (tuple): n-gram range for vectorization.
        top_n_topics (int): Number of most frequent topics to return.

    Returns:
        pd.DataFrame: Top `top_n_topics` with their representative keywords.
    """
    vectorizer_model = CountVectorizer(ngram_range=n_gram_range, stop_words="english")
    topic_model = BERTopic(vectorizer_model=vectorizer_model, verbose=True)
    topics, _ = topic_model.fit_transform(texts)

    topic_info = topic_model.get_topic_info()
    top_topic_ids = topic_info.iloc[1:top_n_topics+1]['Topic']  # exclude topic -1
    keywords_df = pd.DataFrame({
        "topic_id": [],
        "keywords": []
    })

    for tid in top_topic_ids:
        words = [w for w, _ in topic_model.get_topic(tid)]
        keywords_df = pd.concat([
            keywords_df,
            pd.DataFrame({"topic_id": [tid], "keywords": [words]})
        ], ignore_index=True)

    return keywords_df


# ---------- Keyword-based Topic Assignment ----------

def load_topic_rules(csv_path):
    """
    Load topic definitions from a CSV with 'topic', 'keywords', and 'type' columns.
    """
    df = pd.read_csv(csv_path)
    df['keywords'] = df['keywords'].apply(ast.literal_eval)
    return df.to_dict('records')

def assign_topics_to_text(text, topic_rules):
    """
    Assign topics to a transcript using keyword overlap logic from topic rules.

    Rules supported:
    - type == "OR" → Assign topic if any 1 keyword is found
    - type == "2"  → Assign topic if ≥2 keywords match, or if all keywords match when fewer than 2
    - type == "AND x NOT y" → Assign if ≥3 keywords match, 'x' is in text, and 'y' is not
    - default      → Assign if ≥3 keywords match, or all match when keyword list has <3 items
    """

    """
    Assign topics to a transcript using keyword overlap logic from topic rules.
    """
    if pd.isna(text):
        return []

    tokens = set(text.lower().split())
    assigned_topics = []

    for rule in topic_rules:
        topic = rule['topic']
        keywords = [k.lower() for k in rule['keywords']]
        type_field = str(rule['type']).strip()
        keyword_count = len(tokens.intersection(keywords))

        # Rule 1: Match if ANY keyword is found
        if type_field == 'OR':
            if keyword_count > 0:
                assigned_topics.append(topic)

        # Rule 2: Match if ≥2 keywords found, or if all keywords match when total <2
        elif type_field == '2':
            if keyword_count >= 2 or (keyword_count == len(keywords) and len(keywords) < 2):
                assigned_topics.append(topic)

        # Rule 3: Match if ≥3 keywords found AND required terms are present AND excluded terms are absent
        elif "and" in type_field.lower() and "not" in type_field.lower():
            include_terms = re.findall(r"AND\\s+'(.*?)'", type_field)
            exclude_terms = re.findall(r"NOT\\s+'(.*?)'", type_field)
            if keyword_count >= 3 and all(term in tokens for term in include_terms) and all(term not in tokens for term in exclude_terms):
                assigned_topics.append(topic)

        # Default Rule: Match if ≥3 keywords found, or if all keywords match when total <3
        else:
            if keyword_count >= 3 or (keyword_count == len(keywords) and len(keywords) < 3):
                assigned_topics.append(topic)

    return assigned_topics

def apply_topic_assignment(df, text_column, rule_csv_path):
    """
    Apply all topic rules to a DataFrame and return with 'topic_keywords' column.
    """
    topic_rules = load_topic_rules(rule_csv_path)
    df['topic_keywords'] = df[text_column].apply(lambda txt: assign_topics_to_text(txt, topic_rules))
    return df


# ========== USAGE EXAMPLE ==========

if __name__ == "__main__":
    # Step 1: Generate candidate topics with BERTopic (run only once)
    # transcripts = pd.read_csv("path_to_transcripts.csv")["transcription"].dropna().tolist()
    # keywords_df = generate_bertopic_model(transcripts)
    # keywords_df.to_csv("bertopic_keywords_raw.csv", index=False)

    # Step 2: Apply final topic rules
    df = pd.read_csv("/path/to/transcripts.csv")  # Must include 'transcription' column
    df = apply_topic_assignment(df, text_column='transcription', rule_csv_path='topic-keywords.csv')

    print(df[['id', 'topic_keywords']].head())
    # df.to_csv("transcripts_with_final_topics.csv", index=False)
