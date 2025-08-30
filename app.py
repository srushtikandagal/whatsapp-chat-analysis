import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from datetime import datetime

# Try importing your original preprocessor; it's optional now
try:
    import preprocessor as user_pre
except Exception:
    user_pre = None

import helper

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.sidebar.title("WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file (.txt export from WhatsApp, without media)", type=["txt"])

# ---------- Utilities ----------

def try_decode(raw: bytes) -> str:
    """Decode bytes with sensible fallbacks."""
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    # last resort
    return raw.decode("utf-8", errors="ignore")

HEADER_PATTERNS = [
    # ANDROID (en-XX, 24h): "12/08/2025, 22:15 - Name: Message"
    (r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}) - ([^:]+): (.*)$",
     "%d/%m/%Y", "%H:%M"),
    # ANDROID (en-XX, 12h): "12/08/2025, 10:15 pm - Name: Message"
    (r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}\s?[APap][Mm]) - ([^:]+): (.*)$",
     "%d/%m/%Y", "%I:%M %p"),
    # iOS style: "[12/08/2025, 22:15] Name: Message"
    (r"^\[(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}(?::\d{2})?)\] ([^:]+): (.*)$",
     "%d/%m/%Y", None),  # time fmt decided dynamically
    # iOS alt with AM/PM: "[12/08/2025, 10:15:03 PM] Name: Message"
    (r"^\[(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}(?::\d{2})?\s?[APap][Mm])\] ([^:]+): (.*)$",
     "%d/%m/%Y", None),
]

SYSTEM_MARKERS = (
    "added", "removed", "created group", "changed", "joined", "left", "deleted",
    "messages to this group are now", "group description", "end-to-end encryption"
)

def detect_header(line: str):
    """Return (pattern, date_fmt, time_fmt) if the line is a header, else None."""
    for pat, d_fmt, t_fmt in HEADER_PATTERNS:
        if re.match(pat, line):
            return (re.compile(pat), d_fmt, t_fmt)
    return None

def robust_preprocess(text: str) -> pd.DataFrame:
    """
    Universal WhatsApp parser:
    - Detects header style
    - Handles multi-line messages
    - Builds a tidy DataFrame with derived time fields
    """
    lines = [ln.rstrip("\n\r") for ln in text.splitlines() if ln.strip()]

    if not lines:
        return pd.DataFrame()

    # Find the first real header to pick a pattern
    header_info = None
    for ln in lines[:200]:
        header_info = detect_header(ln)
        if header_info:
            break
    if not header_info:
        # Could be different date order (m/d/Y). Try swapping day/month on the fly later.
        # Still proceed and collect messages best-effort.
        pass

    # Parse loop
    records = []
    cur = {"date": None, "time": None, "user": None, "message": []}

    def flush():
        if cur["date"] and cur["time"]:
            msg = "\n".join(cur["message"]).strip()
            if cur["user"] is None:
                user = "group_notification"
            else:
                user = cur["user"].strip() or "group_notification"
            records.append([cur["date"], cur["time"], user, msg])

    for ln in lines:
        info = detect_header(ln)
        if info:
            # Start of a new message
            pat, d_fmt, t_fmt = info
            m = pat.match(ln)
            date_raw, time_raw, user_raw, msg = m.group(1), m.group(2), m.group(3), m.group(4)

            # dynamic time fmt (for iOS: with/without seconds, with/without AM/PM)
            if t_fmt is None:
                t_fmt = "%H:%M:%S" if (":" in time_raw and len(time_raw.split(":")) == 3 and not ("AM" in time_raw.upper() or "PM" in time_raw.upper())) else \
                        ("%I:%M:%S %p" if (":" in time_raw and len(time_raw.split(":")) == 3) and ("AM" in time_raw.upper() or "PM" in time_raw.upper())
                         else ("%H:%M" if "AM" not in time_raw.upper() and "PM" not in time_raw.upper() else "%I:%M %p"))

            # parse date (assume d/m/Y first; fallback to m/d/Y)
            parsed_dt = None
            for date_fmt_try in (d_fmt, "%m/%d/%Y"):
                try:
                    d = datetime.strptime(date_raw, date_fmt_try if len(date_raw.split("/")[-1]) == 4 else date_fmt_try.replace("%Y", "%y"))
                    t = datetime.strptime(time_raw.upper(), t_fmt)
                    parsed_dt = datetime(d.year, d.month, d.day, t.hour, t.minute, getattr(t, "second", 0))
                    break
                except Exception:
                    continue

            # Flush previous
            flush()
            # Start new
            cur = {
                "date": parsed_dt,
                "time": parsed_dt,
                "user": user_raw.strip() if user_raw else None,
                "message": [msg] if msg else []
            }
        else:
            # Continuation of previous message
            if cur["message"] is not None:
                cur["message"].append(ln)
            else:
                # orphaned line before first header; ignore
                pass

    # flush last
    flush()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records, columns=["datetime", "time_obj", "user", "message"])

    # Derivatives
    df["date"] = df["datetime"].dt.date
    df["year"] = df["datetime"].dt.year
    df["month_num"] = df["datetime"].dt.month
    df["month"] = df["datetime"].dt.strftime("%B")
    df["day_name"] = df["datetime"].dt.strftime("%A")
    df["only_date"] = pd.to_datetime(df["date"])
    df["message"] = df["message"].fillna("").astype(str)

    return df

def build_user_list(df: pd.DataFrame):
    user_series = df["user"].dropna().astype(str)
    users = sorted({u.strip() for u in user_series.unique() if u and u.strip()})
    if "group_notification" in users:
        users.remove("group_notification")
    return ["Overall"] + users

def safe_df(x, cols_required=None):
    if x is None or not isinstance(x, pd.DataFrame) or x.empty:
        return None
    if cols_required and any(c not in x.columns for c in cols_required):
        return None
    return x

def plot_empty(msg: str):
    st.info(msg)

# ---------- App flow ----------
if uploaded_file is None:
    st.write("⬅️ Upload your exported WhatsApp chat (.txt, **without media**) to begin.")
    st.caption("Tip: In WhatsApp → Export Chat → WITHOUT media. For iPhone, use Mail/Files; for Android, use Drive/Share.")
else:
    raw = uploaded_file.getvalue()
    text = try_decode(raw)

    # Quick diagnostics panel
    with st.expander("🛠 Diagnostics (open if parsing fails)"):
        st.write("**First 15 non-empty lines** of your file (to verify header format):")
        preview_lines = [ln for ln in text.splitlines() if ln.strip()][:15]
        st.code("\n".join(preview_lines) if preview_lines else "(file seems empty)")
        # Show which header matches first
        match = None
        for ln in preview_lines:
            info = detect_header(ln)
            if info:
                match = info[0].pattern
                break
        st.write("**Detected Header Regex:**", f"`{match}`" if match else "_No standard WhatsApp header detected_")

    # 1) Try user preprocessor first (if present)
    df = None
    if user_pre is not None:
        try:
            df = user_pre.preprocess(text)
        except Exception as e:
            st.warning(f"Your preprocessor raised an error, falling back to universal parser:\n\n{e}")

    # 2) Fallback universal parser if needed
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or ("user" not in df.columns or "message" not in df.columns):
        df = robust_preprocess(text)

    # Final validation
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.error("Parsed data is empty. Check your exported chat file format (try exporting WITHOUT media).")
        st.stop()
    if "user" not in df.columns or "message" not in df.columns:
        st.error("Parsed data missing required columns ('user', 'message').")
        st.stop()

    # Build user list and UI
    user_list = build_user_list(df)
    selected_user = st.sidebar.selectbox("Show analysis with respect to", user_list)

    if st.sidebar.button("Show Analysis"):
        # ---------- Stats ----------
        try:
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        except Exception as e:
            st.error(f"fetch_stats failed: {e}")
            st.stop()

        st.title("Top Statistics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.header("Total Messages")
            st.title(int(num_messages) if pd.notna(num_messages) else 0)
        with c2:
            st.header("Total Words")
            st.title(int(words) if pd.notna(words) else 0)
        with c3:
            st.header("Media Shared")
            st.title(int(num_media_messages) if pd.notna(num_media_messages) else 0)
        with c4:
            st.header("Links Shared")
            st.title(int(num_links) if pd.notna(num_links) else 0)

        # ---------- Monthly Timeline ----------
        st.title("Monthly Timeline")
        try:
            timeline = helper.monthly_timeline(selected_user, df)
        except Exception as e:
            timeline = None
            st.warning(f"monthly_timeline failed: {e}")

        if safe_df(timeline, ["time", "message"]) is None:
            plot_empty("No monthly timeline data available.")
        else:
            fig, ax = plt.subplots()
            ax.plot(timeline["time"], timeline["message"])
            ax.set_xlabel("Month")
            ax.set_ylabel("Messages")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # ---------- Daily Timeline ----------
        st.title("Daily Timeline")
        try:
            daily_timeline = helper.daily_timeline(selected_user, df)
        except Exception as e:
            daily_timeline = None
            st.warning(f"daily_timeline failed: {e}")

        if safe_df(daily_timeline, ["only_date", "message"]) is None:
            plot_empty("No daily timeline data available.")
        else:
            fig, ax = plt.subplots()
            ax.plot(daily_timeline["only_date"], daily_timeline["message"])
            ax.set_xlabel("Date")
            ax.set_ylabel("Messages")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # ---------- Activity Map ----------
        st.title("Activity Map")
        cA, cB = st.columns(2)

        with cA:
            st.header("Most busy day")
            try:
                busy_day = helper.week_activity_map(selected_user, df)
            except Exception as e:
                busy_day = None
                st.warning(f"week_activity_map failed: {e}")

            if busy_day is None or len(busy_day) == 0:
                plot_empty("No busy day data available.")
            else:
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values)
                ax.set_xlabel("Day")
                ax.set_ylabel("Messages")
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

        with cB:
            st.header("Most busy month")
            try:
                busy_month = helper.month_activity_map(selected_user, df)
            except Exception as e:
                busy_month = None
                st.warning(f"month_activity_map failed: {e}")

            if busy_month is None or len(busy_month) == 0:
                plot_empty("No busy month data available.")
            else:
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values)
                ax.set_xlabel("Month")
                ax.set_ylabel("Messages")
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

        # ---------- Heatmap ----------
        st.title("Weekly Activity Heatmap")
        try:
            user_heatmap = helper.activity_heatmap(selected_user, df)
        except Exception as e:
            user_heatmap = None
            st.warning(f"activity_heatmap failed: {e}")

        if safe_df(user_heatmap) is None or user_heatmap.shape[0] == 0:
            plot_empty("No heatmap data available.")
        else:
            fig, ax = plt.subplots()
            sns.heatmap(user_heatmap, ax=ax)
            st.pyplot(fig)

        # ---------- Busiest Users ----------
        if selected_user == "Overall":
            st.title("Most Busy Users")
            try:
                x, new_df = helper.most_busy_users(df)
            except Exception as e:
                x, new_df = None, None
                st.warning(f"most_busy_users failed: {e}")

            c1, c2 = st.columns(2)
            with c1:
                if x is None or len(x) == 0:
                    plot_empty("No busiest users data.")
                else:
                    fig, ax = plt.subplots()
                    ax.bar(x.index, x.values)
                    ax.set_xlabel("User")
                    ax.set_ylabel("Messages")
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
            with c2:
                if safe_df(new_df) is None:
                    plot_empty("No user contribution table.")
                else:
                    st.dataframe(new_df)

        # ---------- Wordcloud ----------
        st.title("Wordcloud")
        try:
            df_wc = helper.create_wordcloud(selected_user, df)
        except Exception as e:
            df_wc = None
            st.warning(f"create_wordcloud failed: {e}")

        if df_wc is None:
            plot_empty("No wordcloud available.")
        else:
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            ax.axis("off")
            st.pyplot(fig)

        # ---------- Common Words ----------
        st.title("Most common words")
        try:
            most_common_df = helper.most_common_words(selected_user, df)
        except Exception as e:
            most_common_df = None
            st.warning(f"most_common_words failed: {e}")

        if safe_df(most_common_df) is None or most_common_df.shape[0] == 0:
            plot_empty("No common words to display.")
        else:
            cols = list(most_common_df.columns)
            word_col, count_col = cols[0], cols[1]
            fig, ax = plt.subplots()
            ax.barh(most_common_df[word_col], most_common_df[count_col])
            ax.set_xlabel("Count")
            ax.set_ylabel("Word")
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # ---------- Emoji ----------
        st.title("Emoji Analysis")
        try:
            emoji_df = helper.emoji_helper(selected_user, df)
        except Exception as e:
            emoji_df = None
            st.warning(f"emoji_helper failed: {e}")

        if safe_df(emoji_df) is None or emoji_df.shape[0] == 0:
            plot_empty("No emoji data to display.")
        else:
            e_cols = list(emoji_df.columns)
            emoji_col, count_col = e_cols[0], e_cols[1]
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(emoji_df)
            with col2:
                fig, ax = plt.subplots()
                ax.pie(emoji_df[count_col].head(), labels=emoji_df[emoji_col].head(), autopct="%0.2f")
                st.pyplot(fig)
