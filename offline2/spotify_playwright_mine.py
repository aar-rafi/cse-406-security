from playwright.sync_api import sync_playwright
import json, pathlib, time, sys

COOKIES_FILE="spotify_cookies.json"
OUT_FILE="my_liked_songs.json"

def load_context(browser):
    ctx = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
    )
    cookies = json.loads(pathlib.Path(COOKIES_FILE).read_text())
    ctx.add_cookies(cookies)
    return ctx

def scroll_to_bottom(page):
    """Scroll until no new tracks load."""
    last_count = 0
    while True:
        # count visible track rows
        rows = page.locator('div[data-testid="tracklist-row"]').all()
        if len(rows) == last_count:
            break
        last_count = len(rows)
        page.keyboard.press("End")
        page.wait_for_timeout(1500)   # give Spotify time to fetch next batch

def extract_tracks(page):
    rows = page.locator('div[data-testid="tracklist-row"]').all()
    tracks = []
    for row in rows:
        title  = row.locator('a[data-testid="internal-track-link"]').inner_text()
        artist = row.locator('a[data-testid="internal-artist-link"]').inner_text()
        album  = row.locator('a[data-testid="internal-album-link"]').inner_text()
        uri    = row.get_attribute('aria-rowindex')  # fallback, see note¹
        tracks.append({"title": title, "artist": artist, "album": album, "uri": uri})
    return tracks

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=100)
        ctx  = load_context(browser)
        page = ctx.new_page()
        page.goto("https://open.spotify.com/collection/tracks")
        input("Press ENTER after you’ve logged in…")
        page.wait_for_selector('div[data-testid="tracklist-row"]', timeout=10000)

        scroll_to_bottom(page)
        songs = extract_tracks(page)

        pathlib.Path(OUT_FILE).write_text(json.dumps(songs, indent=2, ensure_ascii=False))
        print(f"Archived {len(songs)} tracks → {OUT_FILE}")

        browser.close()

if __name__ == "__main__":
    if not pathlib.Path(COOKIES_FILE).exists():
        sys.exit("Cookies not found. Run save_cookies() first.")
    main()
