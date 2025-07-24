from playwright.sync_api import sync_playwright
import json, pathlib

def save_cookies():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)   # interactive
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto("https://accounts.spotify.com/login")
        input("Press ENTER after you’ve logged in…")
        cookies = ctx.cookies()
        pathlib.Path("spotify_cookies.json").write_text(json.dumps(cookies))
        browser.close()

if __name__ == "__main__":
    save_cookies()
