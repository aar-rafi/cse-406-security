#!/usr/bin/env python3
"""
One-shot script:
1. interactive login (or cookie refresh)
2. headless scrape of Liked Songs
"""

import json, pathlib, sys, time
from playwright.sync_api import sync_playwright
from playwright._impl._errors import TimeoutError as PlaywrightTimeoutError

OUT_FILE = "my_liked_songs.json"

# ------------------------------------------------------------
# 1. Interactive login / cookie refresh
# ------------------------------------------------------------
def capture_cookies():
    print("Opening browser for manual login …")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=50)
        ctx = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
        )
        page = ctx.new_page()
        # page.goto("https://accounts.spotify.com/login")
        # input("Log in inside the browser, then press ENTER here … ")
        #
        # # Accept cookies banner if present
        # if page.locator('button[id="onetrust-accept-btn-handler"]').is_visible(timeout=3000):
        #     page.locator('button[id="onetrust-accept-btn-handler"]').click()

        # Navigate to Liked Songs to ensure cookies are valid
        page.goto("https://open.spotify.com/collection/tracks")
        page.wait_for_selector('main[role="main"]', timeout=15000)

        cookies = ctx.cookies()
        browser.close()
        pathlib.Path("spotify_cookies.json").write_text(json.dumps(cookies, indent=2))
        print("Cookies saved → spotify_cookies.json")
        return cookies

# ------------------------------------------------------------
# 2. Re-use cookies in headless mode and scrape
# ------------------------------------------------------------
def scrape_liked_songs(cookies):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=50)
        ctx = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
        )
        ctx.add_cookies(cookies)

        page = ctx.new_page()
        page.goto("https://open.spotify.com/collection/tracks")
        
        # 1. Wait until the page is really the Liked Songs page
        page.wait_for_url("**/collection/tracks", timeout=15000)
        
        # 2. Wait for at least one track row OR the empty-state message
        try:
            page.wait_for_selector('div[data-testid="tracklist-row"]', timeout=15000)
        except PlaywrightTimeoutError:
            # Try alternative selector with role attributes
            try:
                page.wait_for_selector('[role="grid"] [role="row"]', timeout=5000)
                print("Found tracks using role-based selectors")
            except PlaywrightTimeoutError:
                # Take debug screenshot if still failing
                page.screenshot(path="debug_spotify.png")
                print("Debug screenshot saved to debug_spotify.png")
                
                # Check for empty list message
                if page.locator('span:has-text("Songs you like will appear here")').count() > 0:
                    print("Liked Songs list is empty.")
                    browser.close()
                    return []
                else:
                    browser.close()
                    raise   # re-raise the timeout if something else went wrong
        
        # Wait for loading spinner to disappear if it exists
        try:
            page.wait_for_selector('[data-testid="loading-spinner"]', state="detached", timeout=5000)
        except PlaywrightTimeoutError:
            # Loading spinner might not be present, continue
            pass

        # Scroll to load every track
        last_count = 0
        stable_count = 0
        max_attempts = 5
        
        while stable_count < max_attempts:
            # Try primary selector first, fallback to role-based
            rows = page.locator('div[data-testid="tracklist-row"]').all()
            if not rows:
                rows = page.locator('[role="grid"] [role="row"]').all()
            
            current_count = len(rows)
            
            if current_count == last_count:
                stable_count += 1
            else:
                stable_count = 0
                last_count = current_count
                
            page.keyboard.press("End")
            time.sleep(1.5)

        # Extract - get final row count with both selectors
        rows = page.locator('div[data-testid="tracklist-row"]').all()
        if not rows:
            rows = page.locator('[role="grid"] [role="row"]').all()
            print(f"Using role-based selectors, found {len(rows)} rows")
        tracks = []
        
        for i, row in enumerate(rows):
            try:
                title_elem = row.locator('a[data-testid="internal-track-link"]')
                artist_elem = row.locator('a[data-testid="internal-artist-link"]')
                album_elem = row.locator('a[data-testid="internal-album-link"]')
                
                title = title_elem.inner_text() if title_elem.count() > 0 else "Unknown Title"
                artist = artist_elem.inner_text() if artist_elem.count() > 0 else "Unknown Artist"
                album = album_elem.inner_text() if album_elem.count() > 0 else "Unknown Album"
                uri = row.get_attribute('aria-rowindex') or str(i + 1)  # fallback to index
                
                tracks.append({"title": title, "artist": artist, "album": album, "uri": uri})
            except Exception as e:
                print(f"Warning: Failed to extract data from row {i + 1}: {e}")
                continue

        pathlib.Path(OUT_FILE).write_text(json.dumps(tracks, indent=2, ensure_ascii=False))
        print(f"Scraped {len(tracks)} tracks → {OUT_FILE}")
        browser.close()
        return tracks

# ------------------------------------------------------------
# 3. Main flow
# ------------------------------------------------------------
if __name__ == "__main__":
    COOKIES_FILE = "spotify_cookies.json"
    
    # Check if cookies file exists, if not, capture cookies first
    if not pathlib.Path(COOKIES_FILE).exists():
        print(f"Cookies file {COOKIES_FILE} not found. Capturing cookies first...")
        cookies = capture_cookies()
    else:
        try:
            cookies = json.loads(pathlib.Path(COOKIES_FILE).read_text())
            print(f"Loaded cookies from {COOKIES_FILE}")
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading cookies file: {e}")
            print("Capturing new cookies...")
            cookies = capture_cookies()
    
    # Scrape liked songs
    try:
        tracks = scrape_liked_songs(cookies)
        if tracks:
            print(f"Success! Scraped {len(tracks)} tracks.")
        else:
            print("No tracks found or liked songs list is empty.")
    except Exception as e:
        print(f"Error during scraping: {e}")
        print("You may need to refresh cookies by running capture_cookies() first.")
