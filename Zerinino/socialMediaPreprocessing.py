import re
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import pandas as pd

#remove emojis, @user_acc, links, quotemarks ""
#stuff like &lt;pout&gt *sigh* () check, vise razmaka -> jedan

# def remove_emojis(text):
#     emoji_pattern = re.compile(
#         "["
#         "\U0001F600-\U0001F64F"  # Emoticons
#         "\U0001F300-\U0001F5FF"  # Symbols & pictographs
#         "\U0001F680-\U0001F6FF"  # Transport & map symbols
#         "\U0001F1E0-\U0001F1FF"  # Flags
#         "\U00002500-\U00002BEF"  # Chinese characters
#         "\U00002702-\U000027B0"
#         "\U00002702-\U000027B0"
#         "\U000024C2-\U0001F251"
#         "\U0001f926-\U0001f937"
#         "\U00010000-\U0010ffff"
#         "\u2640-\u2642"
#         "\u2600-\u2B55"
#         "\u200d"
#         "\u23cf"
#         "\u23e9"
#         "\u231a"
#         "\ufe0f"  # Dingbats
#         "\u3030"
#         "]+", flags=re.UNICODE
#     )
#     return emoji_pattern.sub(r'', text)

def remove_user_mentions(text):
    #return re.sub(r'@\w+', '', text)
    return re.sub(r'@\S+', '', text)

def remove_links(text):
    return re.sub(r'http\S+|www\.\S+', '', text)

def remove_quotes(text):
    if not isinstance(text, str):
        return text
    return re.sub(r'["“”‘’\'`]', '', text)


# def remove_markup_and_actions(text):
#     # Remove things like &lt;tag&gt;&amp and *sigh* and multiple dots w one dot
#     text = re.sub(r'&lt;.*?&gt;', '', text)      # HTML-like tags
#     text = re.sub(r'\*.*?\*', '', text)          # *sigh*, *thinking*, etc.
#     text = re.sub(r'\.{2,}', '.', text)
#     return text
def remove_markup_and_actions(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'&lt;.*?&gt;', '', text)         # Remove <tags>
    text = re.sub(r'\*[^*]+\*', '', text)           # Remove *sigh*, *text*
    text = re.sub(r'&\w+;', '', text)               # Remove HTML entities like &amp;, &quot;
    text = re.sub(r'\.{2,}', '.', text)             # Replace multiple dots with one
    return text


def normalize_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()
def remove_emojis1(text):
    if isinstance(text, str):
        # Remove all characters that are not basic ASCII (i.e., remove emojis and other symbols)
        return re.sub(r'[^\x00-\x7F]+', '', text)
    return text

MEANINGLESS_STRINGS = {'nan', 'null', 'na', 'n/a', '_'}

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def is_meaningful(text):
    if not isinstance(text, str):
        return False
    text_lower = text.strip().lower()
    if text_lower in MEANINGLESS_STRINGS:
        return False
    if len(text_lower) < 5:
        return False
    if not is_english(text_lower):
        return False
    return True

def clean_text(text):
    text = remove_emojis1(text)
    text = remove_user_mentions(text)
    text = remove_links(text)
    text = remove_quotes(text)
    text = remove_markup_and_actions(text)
    text = normalize_spaces(text)
    return text

# DATA
tweets_data = pd.read_csv('kaggle-susTweets.csv')
instagram_data = pd.read_csv('instagram_data.csv')

#print(instagram_data.columns)

#merge relevant info from both datasets into social_media.csv (messages, set label column to -1)

tweet_messages = tweets_data[['message']].copy()
insta_messages = instagram_data[['caption']].copy()
insta_messages.columns = ['message']

all_messages = pd.concat([tweet_messages, insta_messages], ignore_index=True)
all_messages['label'] = -1
all_messages.to_csv('social_media.csv', index=False)

social_media_data = pd.read_csv('social_media.csv')
social_media_data = social_media_data.iloc[:-1]

social_media_data.to_csv('social_media.csv', index=False)

#preprocess the data
social_media_data['message'] = social_media_data['message'].astype(str).apply(clean_text)
# keep only data in English
social_media_data = social_media_data[social_media_data['message'].apply(is_meaningful)]

social_media_data.to_csv('social_media.csv', index=False)


