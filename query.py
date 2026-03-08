"""
Sloka Recommendation Engine — Interactive CLI
Ask about any life situation and get relevant Sanskrit slokas.
"""
import json
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

DB_DIR = "vector_db"
TOP_K = 5  # Number of slokas to retrieve


def get_collection():
    """Connect to the ChromaDB collection."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_collection(name="slokas", embedding_function=ef)


def retrieve_slokas(collection, user_query, top_k=TOP_K, path_filter=None):
    """Retrieve the most relevant slokas for a user's life situation."""
    where_filter = None
    if path_filter:
        where_filter = {"path": path_filter}

    results = collection.query(
        query_texts=[user_query],
        n_results=top_k,
        where=where_filter,
        include=["metadatas", "distances"]
    )

    slokas = []
    for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
        sloka = dict(meta)
        sloka["life_themes"] = json.loads(sloka["life_themes"])
        sloka["relevance_score"] = round(1 - distance, 3)  # cosine similarity
        slokas.append(sloka)

    return slokas


def generate_response_with_llm(user_query, slokas):
    """
    Use an LLM to craft a thoughtful response combining the user's situation
    with the retrieved slokas. Falls back to formatted output if no API key.
    """
    provider = os.getenv("LLM_PROVIDER", "").lower()

    # Try Anthropic
    if provider == "anthropic" or os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            return _call_anthropic(client, user_query, slokas)
        except Exception as e:
            print(f"  (Anthropic unavailable: {e}, using local format)\n")

    # Try OpenAI
    if provider == "openai" or os.getenv("OPENAI_API_KEY"):
        try:
            import openai
            client = openai.OpenAI()
            return _call_openai(client, user_query, slokas)
        except Exception as e:
            print(f"  (OpenAI unavailable: {e}, using local format)\n")

    # Fallback: formatted output without LLM
    return format_slokas_locally(user_query, slokas)


def _build_prompt(user_query, slokas):
    """Build the prompt for the LLM."""
    sloka_context = ""
    for i, s in enumerate(slokas, 1):
        sloka_context += f"""
--- Sloka {i} ---
ID: {s['id']} | Source: {s['source']} Ch.{s['chapter']} V.{s['verse']}
Sanskrit: {s['sanskrit']}
Transliteration: {s['transliteration']}
Meaning: {s['english_meaning']}
Path: {s['path']} — {s['path_description']}
When to use: {s['when_to_use']}
Difficulty: {s['difficulty']}
Relevance: {s['relevance_score']}
"""

    return f"""You are a wise and compassionate guide who recommends Sanskrit slokas for life situations.

A person has come to you with this situation:
"{user_query}"

Based on their situation, here are the most relevant slokas retrieved from sacred texts:
{sloka_context}

Your task:
1. Acknowledge their situation with empathy (1-2 sentences)
2. Recommend the top 2-3 most relevant slokas from the ones provided
3. For each recommended sloka:
   - Show the Sanskrit text in Devanagari
   - Show the transliteration (for easy learning/pronunciation)
   - Explain the meaning in simple words
   - Explain specifically HOW this sloka applies to their situation
   - Mention which path it belongs to
4. End with a brief practical suggestion for incorporating these teachings

Keep the tone warm, wise, and practical — not preachy. Make it easy for a beginner to understand and start learning these slokas."""


def _call_anthropic(client, user_query, slokas):
    """Call Claude API."""
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": _build_prompt(user_query, slokas)}]
    )
    return message.content[0].text


def _call_openai(client, user_query, slokas):
    """Call OpenAI API."""
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=2000,
        messages=[{"role": "user", "content": _build_prompt(user_query, slokas)}]
    )
    return response.choices[0].message.content


def format_slokas_locally(user_query, slokas):
    """Format slokas without an LLM — pure retrieval output."""
    lines = [
        f"\n{'='*60}",
        f"  Recommended Slokas for your situation",
        f"{'='*60}\n",
    ]

    for i, s in enumerate(slokas[:3], 1):
        lines.append(f"  [{i}] {s['id']} — {s['source']} (Ch.{s['chapter']}, V.{s['verse']})")
        lines.append(f"  Path: {s['path'].replace('_', ' ').title()}")
        lines.append(f"  Relevance: {s['relevance_score']}")
        lines.append(f"")
        lines.append(f"  Sanskrit:")
        lines.append(f"    {s['sanskrit']}")
        lines.append(f"")
        lines.append(f"  Transliteration:")
        lines.append(f"    {s['transliteration']}")
        lines.append(f"")
        lines.append(f"  Meaning:")
        lines.append(f"    {s['english_meaning']}")
        lines.append(f"")
        lines.append(f"  Why this helps you:")
        lines.append(f"    {s['when_to_use']}")
        lines.append(f"")
        lines.append(f"  {'—'*40}")
        lines.append(f"")

    return "\n".join(lines)


def get_available_paths():
    """Return the available learning paths."""
    return {
        "karma_yoga": "Path of Selfless Action (career, duty, purpose)",
        "bhakti_yoga": "Path of Devotion (relationships, love, faith)",
        "jnana_yoga": "Path of Knowledge (clarity, wisdom, self-awareness)",
        "dhyana_yoga": "Path of Meditation (peace, anxiety, focus)",
        "sankhya_yoga": "Path of Wisdom (grief, loss, resilience)",
        "raja_yoga": "Path of Discipline (health, habits, self-control)",
    }


def interactive_cli():
    """Run the interactive CLI."""
    print("\n" + "=" * 60)
    print("  🙏 Sanskrit Sloka Wisdom Engine")
    print("  Ask about any life situation, get ancient wisdom.")
    print("=" * 60)
    print()
    print("  Available Paths (optional filter):")
    for key, desc in get_available_paths().items():
        print(f"    • {key}: {desc}")
    print()
    print("  Commands:")
    print("    Type your situation naturally (e.g., 'I am struggling with career growth')")
    print("    Type 'path:<name>' to filter by path (e.g., 'path:karma_yoga')")
    print("    Type 'quit' to exit")
    print()

    collection = get_collection()
    current_path_filter = None

    while True:
        user_input = input("  You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\n  🙏 May these teachings bring you peace. Namaste.\n")
            break

        # Check for path filter
        if user_input.lower().startswith("path:"):
            path_name = user_input.split(":", 1)[1].strip()
            if path_name in get_available_paths():
                current_path_filter = path_name
                print(f"\n  ✓ Filtering by: {get_available_paths()[path_name]}")
                print(f"    (Type 'path:clear' to remove filter)\n")
            elif path_name == "clear":
                current_path_filter = None
                print(f"\n  ✓ Path filter cleared.\n")
            else:
                print(f"\n  ✗ Unknown path. Available: {', '.join(get_available_paths().keys())}\n")
            continue

        # Retrieve relevant slokas
        print(f"\n  Searching sacred texts...\n")
        slokas = retrieve_slokas(collection, user_input, top_k=TOP_K, path_filter=current_path_filter)

        if not slokas:
            print("  No matching slokas found. Try rephrasing your situation.\n")
            continue

        # Generate response
        response = generate_response_with_llm(user_input, slokas)
        print(response)
        print()


if __name__ == "__main__":
    interactive_cli()
