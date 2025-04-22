"""
This script processes three datasets containing text titles. It identifies and replaces low-frequency words
(without dashes and not in common stopword lists) across all datasets with a placeholder token (e.g., "<unk>").
The modified titles are saved to new CSV files.

Dependencies:
- pandas
- re
- nltk.corpus.stopwords
"""

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords

def save_words(s):
    """
    Tokenizes a string into words and filters out stopwords in English, Spanish, and French.
    Parentheses are removed from the words.

    Args:
        s (str): Input sentence.

    Returns:
        List[str]: List of filtered, cleaned words.
    """
    stop_words = set(stopwords.words("english"))
    stop_words.update(set(stopwords.words("spanish")))
    stop_words.update(set(stopwords.words("french")))
    words = s.split()
    filtered = [word.replace("(","").replace(")","") for word in words if word.lower() not in stop_words]
    return filtered    

def get_low_freq_words(dfs, min_occ=1):
    """
    Identifies low-frequency words across multiple datasets, ignoring words that contain dash characters.

    Args:
        dfs (List[pd.DataFrame]): List of DataFrames with a 'Words' column containing lists of tokens.
        min_occ (int): Minimum number of occurrences for a word to be considered low-frequency.

    Returns:
        List[str]: Words that occur exactly `min_occ` times and contain no dash characters.
    """
    count_words = {}
    
    for df in dfs:
        for word_list in df["Words"]:
            for word in word_list:
                count_words[word] = count_words.get(word, 0) + 1

    word_freq = pd.DataFrame.from_dict(count_words, orient="index", columns=["freq"]).sort_values("freq", ascending=False)

    words_without_dashes = [
        word for word in word_freq[word_freq["freq"] == min_occ].index
        if not dash_regex.search(word)
    ]
    
    return words_without_dashes

def clean_low_freq_words(df, words_to_replace, new_char="<unk>"):
    """
    Replaces low-frequency words in the 'Title' column of a DataFrame with a placeholder string.

    Args:
        df (pd.DataFrame): DataFrame with a 'Title' column to process.
        words_to_replace (List[str]): List of full words to be replaced.
        new_char (str): String to replace the low-frequency words with.

    Returns:
        None: Modifies DataFrame in-place, adding a 'Modified_Title' column.
    """
    if not words_to_replace:
        df["Modified_Title"] = None
        return
    
    pattern = r'\b(' + '|'.join(re.escape(word) for word in words_to_replace) + r')\b'
    regex = re.compile(pattern)

    def replace_word(title):
        if regex.search(title):
            return regex.sub(new_char, title)
        return None

    df['Modified_Title'] = df['Title'].apply(replace_word)

def replace_modified_captions(df):
    """
    Finalizes replacement of low-frequency words by updating the 'Title' column with 'Modified_Title' values.

    Args:
        df (pd.DataFrame): DataFrame with a 'Modified_Title' column.

    Returns:
        pd.DataFrame: A copy of the DataFrame with updated 'Title' values and without 'Modified_Title'.
    """

    modified_df = df.copy()
    no_processed_caption = np.where(~df.Modified_Title.isna())[0]
    modified_df.loc[no_processed_caption, "Title"] = modified_df[modified_df.Modified_Title.notna()].Modified_Title
    modified_df.drop(columns='Modified_Title', inplace=True)
    return modified_df

if __name__ == "__main__":
    # Load datasets
    df_train = pd.read_csv("Augmented_XL\clean_mapping_train_aug_25.csv")
    print("Augmented_XL\clean_mapping_train_aug_25.csv")
    df_val = pd.read_csv("clean_mapping_validation.csv")
    #df_test = pd.read_csv("clean_mapping_test.csv")

    # Define dash characters and compile regex
    DASH_CHARS = r"[\u002D\u058A\u05BE\u1400\u1806\u2010\u2011\u2012\u2013\u2014\u2015\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]"
    dash_regex = re.compile(DASH_CHARS)

    # Preprocess titles to extract meaningful words
    df_train["Words"] = df_train.Title.apply(save_words)
    df_val["Words"] = df_val.Title.apply(save_words)
    #df_test["Words"] = df_test.Title.apply(save_words)

    # Identify low-frequency words across all datasets
    #all_dfs = [df_train, df_val, df_test]
    all_dfs = [df_train, df_val]
    low_freq_words = get_low_freq_words(all_dfs, min_occ=1)

    # Replace low-frequency words in each dataset
    for df in all_dfs:
        clean_low_freq_words(df, low_freq_words, new_char="<unk>")

    # Finalize and export cleaned datasets
    df_train_clean = replace_modified_captions(df_train)
    df_val_clean = replace_modified_captions(df_val)
    #df_test_clean = replace_modified_captions(df_test)

    df_train_clean.drop(columns="Words", inplace=True)
    df_val_clean.drop(columns="Words", inplace=True)
    #df_test_clean.drop(columns="Words", inplace=True)

    df_train_clean.to_csv("Augmented_XL\Processed\clean_mapping_train_aug_25.csv", index=False)
    df_val_clean.to_csv("Augmented_XL\Processed\clean_mapping_validation_25_processed.csv", index=False)
    #df_test_clean.to_csv("clean_mapping_test_processed.csv", index=False)
