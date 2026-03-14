"""
Sloka Wisdom Engine — Telegram Bot
Send your life situation, get Sanskrit wisdom.

Setup:
1. Create a bot via @BotFather on Telegram, get your token
2. Set TELEGRAM_BOT_TOKEN in .env
3. Run: python telegram_bot.py
"""
import json
import os
import logging
from textwrap import dedent

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

try:
    from telegram import Update
    from telegram.ext import (
        ApplicationBuilder,
        CommandHandler,
        MessageHandler,
        ContextTypes,
        filters,
    )
except ImportError:
    print("Install python-telegram-bot: pip install python-telegram-bot")
    exit(1)

load_dotenv()

DB_DIR = "vector_db"
TOP_K = 3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Available paths
PATHS = {
    "karma_yoga": "Selfless Action (career, duty)",
    "bhakti_yoga": "Devotion (relationships, love)",
    "jnana_yoga": "Knowledge (clarity, wisdom)",
    "dhyana_yoga": "Meditation (peace, focus)",
    "sankhya_yoga": "Wisdom (grief, resilience)",
    "raja_yoga": "Discipline (health, habits)",
}

# Initialize ChromaDB
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(name="slokas", embedding_function=ef)
logger.info(f"Loaded {collection.count()} slokas.")


def search_slokas(query: str, path_filter: str = None):
    where_filter = {"path": path_filter} if path_filter else None
    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        where=where_filter,
        include=["metadatas", "distances"],
    )
    slokas = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        s = dict(meta)
        s["life_themes"] = json.loads(s["life_themes"])
        s["relevance_score"] = round(1 - dist, 3)
        slokas.append(s)
    return slokas


def format_sloka_message(query: str, slokas: list) -> str:
    difficulty_icon = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}
    lines = [f"🙏 *Wisdom for:* _{query}_\n"]

    for i, s in enumerate(slokas, 1):
        path_label = s["path"].replace("_", " ").title()
        icon = difficulty_icon.get(s["difficulty"], "🟢")
        match_pct = int(s["relevance_score"] * 100)
        themes = ", ".join(t.replace("_", " ") for t in s["life_themes"][:3])

        lines.append(f"━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"*\\[{i}\\] {s['source']} — {s['id']}*")
        lines.append(f"📿 {path_label}  •  {icon} {s['difficulty'].title()}  •  {match_pct}% match\n")
        lines.append(f"*Sanskrit:*")
        lines.append(f"_{s['sanskrit']}_\n")
        lines.append(f"*Pronunciation:*")
        lines.append(f"_{s['transliteration']}_\n")
        lines.append(f"*Meaning:*")
        lines.append(f"{s['english_meaning']}\n")
        lines.append(f"💡 *How this helps:*")
        lines.append(f"{s['when_to_use']}\n")
        lines.append(f"🏷 {themes}\n")

    lines.append("━━━━━━━━━━━━━━━━━━━━━")
    lines.append("💬 Send another situation or /paths to filter by yoga path")
    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = dedent("""\
        🙏 *Sloka Wisdom Engine*

        Welcome\\! I find ancient Sanskrit wisdom for your life situations\\.

        *How to use:*
        Just describe what you're going through, and I'll find the most relevant slokas from:
        • Bhagavad Gita
        • Yoga Sutras
        • Chanakya Niti
        • Subhashitas
        • Upanishads

        *Examples:*
        → _I'm struggling with career growth_
        → _I feel anxious and can't find peace_
        → _I want to overcome anger_

        *Commands:*
        /paths — View and filter by yoga paths
        /help — Show this message
    """)
    await update.message.reply_text(text, parse_mode="MarkdownV2")


async def paths_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = ["🧘 *Available Yoga Paths*\n"]
    lines.append("Send `/path <name>` to filter results\\.\n")
    for key, desc in PATHS.items():
        lines.append(f"• `{key}` — {desc}")
    lines.append(f"\nSend `/path clear` to remove filter\\.")
    await update.message.reply_text("\n".join(lines), parse_mode="MarkdownV2")


async def set_path(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await paths_command(update, context)
        return

    path_name = context.args[0].lower()
    if path_name == "clear":
        context.user_data.pop("path_filter", None)
        await update.message.reply_text("✅ Path filter cleared.")
    elif path_name in PATHS:
        context.user_data["path_filter"] = path_name
        await update.message.reply_text(f"✅ Filtering by: {PATHS[path_name]}\nSend /path clear to remove.")
    else:
        await update.message.reply_text(f"❌ Unknown path. Use /paths to see available options.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.strip()
    if not query:
        return

    await update.message.reply_text("🔍 Searching sacred texts...")

    path_filter = context.user_data.get("path_filter")
    slokas = search_slokas(query, path_filter=path_filter)

    if not slokas:
        await update.message.reply_text("No matching slokas found. Try rephrasing.")
        return

    response = format_sloka_message(query, slokas)
    # Telegram has a 4096 char limit, split if needed
    if len(response) <= 4096:
        await update.message.reply_text(response, parse_mode="Markdown")
    else:
        # Send without markdown if too long (avoids parsing issues)
        await update.message.reply_text(response)


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: Set TELEGRAM_BOT_TOKEN in .env file")
        print("Get one from @BotFather on Telegram")
        exit(1)

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("paths", paths_command))
    app.add_handler(CommandHandler("path", set_path))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Telegram bot is running... Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
