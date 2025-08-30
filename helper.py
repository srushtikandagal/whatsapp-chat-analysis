# helper.py
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import re

# ---- optional emoji lib (v1.x / v2.x both) ----
try:
    import emoji
    _HAS_EMOJI = True
except Exception:
    emoji = None
    _HAS_EMOJI = False

extract = URLExtract()

# Common WhatsApp media placeholders (handles variants + case)
_MEDIA_PATTERNS = (
    "<media omitted>", "<image omitted>", "<video omitted>",
    "<document omitted>", "<audio omitted>", "<sticker omitted>",
    "media omitted", "image omitted", "video omitted", "sticker omitted",
    "document omitted", "audio omitted", "gif omitted"
)

_URL_RE = re.compile(r"https?://\S+")

# Fallback regex for emojis when `emoji` package is missing / fails
_EMOJI_REGEX = re.compile(
    r"[\U0001F1E6-\U0001F1FF]"   # flags
    r"|[\U0001F300-\U0001F5FF]"  # symbols & pictographs
    r"|[\U0001F600-\U0001F64F]"  # emoticons
    r"|[\U0001F680-\U0001F6FF]"  # transport & map
    r"|[\U0001F700-\U0001F77F]"
    r"|[\U0001F780-\U0001F7FF]"
    r"|[\U0001F800-\U0001F8FF]"
    r"|[\U0001F900-\U0001F9FF]"
    r"|[\U0001FA00-\U0001FAFF]"
    r"|[\u2600-\u26FF]"          # misc symbols
    r"|[\u2700-\u27BF]"          # dingbats
)

def _is_media(msg: str) -> bool:
    if not isinstance(msg, str):
        return False
    s = msg.strip().lower()
    return any(pat in s for pat in _MEDIA_PATTERNS)

def fetch_stats(selected_user, df):
    """Returns (num_messages, total_words, num_media_messages, num_links)."""
    if df is None or df.empty:
        return 0, 0, 0, 0

    use_df = df if selected_user == 'Overall' else df[df['user'] == selected_user]
    if use_df.empty:
        return 0, 0, 0, 0

    msgs = use_df['message'].astype(str)

    # total messages (non-empty)
    num_messages = int((msgs.str.strip() != "").sum())

    # total words (skip media placeholders)
    words = 0
    for m in msgs:
        if not _is_media(m):
            words += len(m.split())

    # media messages
    num_media_messages = int(sum(_is_media(m) for m in msgs))

    # links
    links = 0
    for m in msgs:
        # use urlextract (robust) and also catch plain regex
        try:
            links += len(extract.find_urls(m))
        except Exception:
            links += len(_URL_RE.findall(m))

    return num_messages, words, num_media_messages, links

def most_busy_users(df):
    if df is None or df.empty:
        return pd.Series(dtype="int64"), pd.DataFrame(columns=['name', 'percent'])
    tmp = df.copy()
    tmp['user'] = tmp['user'].astype(str).str.strip()
    tmp = tmp[(tmp['user'] != '') & (tmp['user'] != 'group_notification')]
    x = tmp['user'].value_counts().head()
    pct = round((tmp['user'].value_counts() / max(1, tmp.shape[0])) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'}
    )
    return x, pct

def create_wordcloud(selected_user, df):
    # load stopwords as a set
    try:
        with open('stop_hinglish.txt', 'r', encoding='utf-8', errors='ignore') as f:
            stop_words = set(w.strip().lower() for w in f.read().split())
    except Exception:
        stop_words = set()

    use_df = df if selected_user == 'Overall' else df[df['user'] == selected_user]
    temp = use_df[(use_df['user'] != 'group_notification')].copy()
    # drop media placeholders
    temp = temp[~temp['message'].astype(str).str.lower().apply(_is_media)]

    def remove_stop_words(message: str) -> str:
        if not isinstance(message, str):
            return ""
        cleaned = []
        for word in message.lower().split():
            if word not in stop_words:
                cleaned.append(word)
        return " ".join(cleaned)

    temp['message'] = temp['message'].astype(str).apply(remove_stop_words)
    corpus = temp['message'].str.cat(sep=" ").strip()
    if not corpus:
        return None

    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(corpus)
    return df_wc

def most_common_words(selected_user, df):
    # load stopwords
    try:
        with open('stop_hinglish.txt', 'r', encoding='utf-8', errors='ignore') as f:
            stop_words = set(w.strip().lower() for w in f.read().split())
    except Exception:
        stop_words = set()

    use_df = df if selected_user == 'Overall' else df[df['user'] == selected_user]
    temp = use_df[(use_df['user'] != 'group_notification')].copy()
    # drop media placeholders
    temp = temp[~temp['message'].astype(str).str.lower().apply(_is_media)]

    words = []
    for message in temp['message'].astype(str):
        for w in message.lower().split():
            if w and (w not in stop_words):
                words.append(w)

    return pd.DataFrame(Counter(words).most_common(20))

def _extract_emojis(text: str):
    """Return a list of emojis from text, robust across emoji versions or without the pkg."""
    if not isinstance(text, str) or not text:
        return []
    if _HAS_EMOJI:
        # Works on emoji v1.x and v2.x
        try:
            return [d['emoji'] for d in emoji.emoji_list(text)]
        except Exception:
            try:
                return [ch for ch in text if hasattr(emoji, "is_emoji") and emoji.is_emoji(ch)]
            except Exception:
                pass
    # Fallback: regex ranges
    return _EMOJI_REGEX.findall(text)

def emoji_helper(selected_user, df):
    use_df = df if selected_user == 'Overall' else df[df['user'] == selected_user]
    if use_df is None or use_df.empty:
        return pd.DataFrame(columns=[0, 1])

    emojis = []
    for message in use_df['message'].astype(str):
        emojis.extend(_extract_emojis(message))

    cnt = Counter(emojis)
    if not cnt:
        return pd.DataFrame(columns=[0, 1])

    # Keep same shape your app expects: column 0=emoji, 1=count
    return pd.DataFrame(cnt.most_common(len(cnt)))

def monthly_timeline(selected_user, df):
    use_df = df if selected_user == 'Overall' else df[df['user'] == selected_user]
    if use_df is None or use_df.empty:
        return pd.DataFrame(columns=['year', 'month_num', 'month', 'message', 'time'])

    timeline = use_df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = [f"{timeline.loc[i, 'month']}-{timeline.loc[i, 'year']}" for i in range(timeline.shape[0])]
    timeline['time'] = time
    return timeline

def daily_timeline(selected_user, df):
    use_df = df if selected_user == 'Overall' else df[df['user'] == selected_user]
    if use_df is None or use_df.empty:
        return pd.DataFrame(columns=['only_date', 'message'])
    return use_df.groupby('only_date').count()['message'].reset_index()

def week_activity_map(selected_user, df):
    use_df = df if selected_user == 'Overall' else df[df['user'] == selected_user]
    if use_df is None or use_df.empty:
        return pd.Series(dtype="int64")
    return use_df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    use_df = df if selected_user == 'Overall' else df[df['user'] == selected_user]
    if use_df is None or use_df.empty:
        return pd.Series(dtype="int64")
    return use_df['month'].value_counts()

def activity_heatmap(selected_user, df):
    """
    Your original code pivots on 'period'. Some preprocessors name the hour column 'hour'.
    This supports both:
      - if 'period' exists → use it
      - else derive 'period' from 'datetime'.hour
    """
    use_df = df if selected_user == 'Overall' else df[df['user'] == selected_user]
    if use_df is None or use_df.empty:
        return pd.DataFrame()

    tmp = use_df.copy()
    if 'period' not in tmp.columns:
        # derive from hour if possible
        if 'datetime' in tmp.columns:
            tmp['period'] = tmp['datetime'].dt.hour
        elif 'hour' in tmp.columns:
            tmp['period'] = tmp['hour']
        else:
            return pd.DataFrame()

    return tmp.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
