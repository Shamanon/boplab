from html.parser import HTMLParser
import os
import sqlite3
import zipfile

EPUB_PATH = "books/rucker_ware_tetralogy_CC.epub"  # Update path if needed
DB_PATH = "db/brain.db"
CHUNK_SIZE = 600  # Character length per lore chunk


class HTMLTextExtractor(HTMLParser):

    def __init__(self):
        super().__init__()
        self.result = []

    def handle_data(self, data):
        self.result.append(data)

    def get_text(self):
        return "".join(self.result)


def extract_text_from_epub(epub_path):
    full_text = []
    with zipfile.ZipFile(epub_path, "r") as z:
        # EPUB text chapters are stored in HTML/XHTML files
        html_files = [
            f
            for f in z.namelist()
            if f.endswith(".html")
            or f.endswith(".xhtml")
            or f.endswith(".htm")
        ]
        html_files.sort()  # Sort to keep chapter order

        for file in html_files:
            with z.open(file) as f:
                content = f.read().decode("utf-8", errors="ignore")
                parser = HTMLTextExtractor()
                parser.feed(content)
                text = parser.get_text().strip()
                if text:
                    full_text.append(text)

    return "\n\n".join(full_text)


def init_lore_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS lore 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, chunk TEXT)""")
    conn.commit()
    conn.close()


def ingest_epub(title, epub_path):
    if not os.path.exists(epub_path):
        print(f"[!] Error: Could not find EPUB file at '{epub_path}'")
        return

    print(f"[*] Extracting text from '{epub_path}'...")
    text = extract_text_from_epub(epub_path)

    # Clean up whitespace and linebreaks
    clean_text = " ".join(text.split())

    # Chunk text into readable segments
    chunks = [
        clean_text[i : i + CHUNK_SIZE]
        for i in range(0, len(clean_text), CHUNK_SIZE)
    ]

    init_lore_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    print(f"[*] Writing {len(chunks)} chunks to '{DB_PATH}'...")
    for chunk in chunks:
        c.execute(
            "INSERT INTO lore (title, chunk) VALUES (?, ?)", (title, chunk)
        )

    conn.commit()
    conn.close()
    print(f"[✓] Successfully ingested '{title}' into lore database!")


if __name__ == "__main__":
    ingest_epub("Ware Tetralogy", EPUB_PATH)
