import time
import json
import os
import signal
import sys
import random
import traceback
import socket
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import database
from database import Database

WEBSITES = [
    # websites of your choice
    "https://cse.buet.ac.bd/moodle/",
    "https://google.com",
    "https://prothomalo.com",
]

TRACES_PER_SITE = 1000
FINGERPRINTING_URL = "http://localhost:5000" 
OUTPUT_PATH = "dataset.json"

# Initialize the database to save trace data reliably
database.db = Database(WEBSITES)
database.db.init_database()

""" Signal handler to ensure data is saved before quitting. """
def signal_handler(sig, frame):
    print("\nReceived termination signal. Exiting gracefully...")
    try:
        database.db.export_to_json(OUTPUT_PATH)
    except:
        pass
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


"""
Some helper functions to make your life easier.
"""

def is_server_running(host='127.0.0.1', port=5000):
    """Check if the Flask server is running."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def setup_webdriver():
    """Set up the Selenium WebDriver with Chrome options."""
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")  # Recommended for Linux
    chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems
    
    # Set the correct Chrome binary path for Arch Linux
    chrome_options.binary_location = "/usr/bin/google-chrome-stable"
    
    # Try to get the correct chromedriver
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        print(f"ChromeDriverManager failed: {e}")
        print("Trying manual chromedriver setup...")
        
        # Try to use system chromedriver if available
        try:
            service = Service("/usr/bin/chromedriver")
            driver = webdriver.Chrome(service=service, options=chrome_options)
            return driver
        except:
            print("Manual chromedriver not found either.")
            raise e

def retrieve_traces_from_backend(driver):
    """Retrieve traces from the backend API."""
    traces = driver.execute_script("""
        return fetch('/api/get_results')
            .then(response => response.ok ? response.json() : {traces: []})
            .then(data => data.traces || [])
            .catch(() => []);
    """)
    
    count = len(traces) if traces else 0
    print(f"  - Retrieved {count} traces from backend API" if count else "  - No traces found in backend storage")
    return traces or []

def clear_trace_results(driver, wait):
    """Clear all results from the backend by pressing the button."""
    clear_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Clear all results')]")
    clear_button.click()

    wait.until(EC.text_to_be_present_in_element(
        (By.XPATH, "//div[@role='alert']"), "Cleared"))
    
def is_collection_complete():
    """Check if target number of traces have been collected."""
    current_counts = database.db.get_traces_collected()
    remaining_counts = {website: max(0, TRACES_PER_SITE - count) 
                      for website, count in current_counts.items()}
    return sum(remaining_counts.values()) == 0

"""
Your implementation starts here.
"""

def collect_single_trace(driver, wait, website_url):
    """ Implement the trace collection logic here. 
    1. Open the fingerprinting website
    2. Click the button to collect trace
    3. Open the target website in a new tab
    4. Interact with the target website (scroll, click, etc.)
    5. Return to the fingerprinting tab and close the target website tab
    6. Wait for the trace to be collected
    7. Return success or failure status
    """
    try:
        print(f"  - Collecting trace for: {website_url}")
        
        # Step 1: Open fingerprinting website if not already open
        if FINGERPRINTING_URL not in driver.current_url:
            driver.get(FINGERPRINTING_URL)
            time.sleep(2)
        
        # Step 2: Open target website in new tab
        driver.execute_script("window.open('', '_blank');")
        driver.switch_to.window(driver.window_handles[1])
        driver.get(website_url)
        time.sleep(3)  # Wait for page to load
        
        # Step 3: Interact with the target website (simulate user activity)
        for _ in range(5):  # Scroll a few times
            scroll_amount = random.randint(300, 800)
            driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
            time.sleep(random.uniform(0.5, 1.5))
        
        # Random interactions
        try:
            # Try to click on some elements
            clickable_elements = driver.find_elements(By.TAG_NAME, "a")[:3]
            if clickable_elements:
                element = random.choice(clickable_elements)
                if element.is_displayed():
                    driver.execute_script("arguments[0].click();", element)
                    time.sleep(1)
        except:
            pass  # Ignore click errors
            
        # Step 4: Switch back to fingerprinting tab
        driver.switch_to.window(driver.window_handles[0])
        
        # Step 5: Start trace collection
        collect_button = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(text(), 'Collect Trace')]")
        ))
        collect_button.click()
        
        # Step 6: Wait for collection to complete (about 10 seconds)
        print("    - Trace collection started, waiting 12 seconds...")
        time.sleep(12)
        
        # Wait for success message
        wait.until(EC.text_to_be_present_in_element(
            (By.XPATH, "//div[@role='alert']"), "complete"
        ))
        
        # Step 7: Close the target website tab
        if len(driver.window_handles) > 1:
            driver.switch_to.window(driver.window_handles[1])
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
        
        print("    - Trace collection completed successfully")
        return True
        
    except Exception as e:
        print(f"    - Error collecting trace: {str(e)}")
        # Clean up any extra tabs
        while len(driver.window_handles) > 1:
            driver.switch_to.window(driver.window_handles[-1])
            driver.close()
        driver.switch_to.window(driver.window_handles[0])
        return False

def collect_fingerprints(driver, target_counts=None):
    """ Implement the main logic to collect fingerprints.
    1. Calculate the number of traces remaining for each website
    2. Open the fingerprinting website
    3. Collect traces for each website until the target number is reached
    4. Save the traces to the database
    5. Return the total number of new traces collected
    """
    wait = WebDriverWait(driver, 20)
    total_collected = 0
    
    try:
        # Step 1: Calculate remaining traces needed
        current_counts = database.db.get_traces_collected()
        if target_counts is None:
            target_counts = {website: TRACES_PER_SITE for website in WEBSITES}
        
        remaining_counts = {website: max(0, target_counts[website] - current_counts.get(website, 0))
                          for website in WEBSITES}
        
        print(f"Remaining traces needed: {remaining_counts}")
        
        # Step 2: Open fingerprinting website
        driver.get(FINGERPRINTING_URL)
        time.sleep(3)
        
        # Step 3: Collect traces for each website
        for website in WEBSITES:
            needed = remaining_counts[website]
            if needed <= 0:
                print(f"Skipping {website} - already have enough traces")
                continue
                
            print(f"\nCollecting {needed} traces for {website}")
            
            for i in range(needed):
                print(f"  Trace {i+1}/{needed}")
                
                # Collect single trace
                success = collect_single_trace(driver, wait, website)
                
                if success:
                    # Step 4: Retrieve and save trace data
                    traces = retrieve_traces_from_backend(driver)
                    if traces:
                        # Save the most recent trace
                        latest_trace = traces[-1]
                        database.db.insert_trace(
                            website=website,
                            trace_data=latest_trace['trace'],
                            timestamp=latest_trace['timestamp']
                        )
                        total_collected += 1
                        print(f"    - Saved trace to database (total: {total_collected})")
                    
                    # Clear traces from backend to free memory
                    try:
                        clear_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Clear Results')]")
                        clear_button.click()
                        time.sleep(1)
                    except:
                        pass
                
                # Small delay between collections
                time.sleep(random.uniform(1, 3))
        
        return total_collected
        
    except Exception as e:
        print(f"Error in collect_fingerprints: {str(e)}")
        traceback.print_exc()
        return total_collected

def main():
    """ Implement the main function to start the collection process.
    1. Check if the Flask server is running
    2. Initialize the database
    3. Set up the WebDriver
    4. Start the collection process, continuing until the target number of traces is reached
    5. Handle any exceptions and ensure the WebDriver is closed at the end
    6. Export the collected data to a JSON file
    7. Retry if the collection is not complete
    """
    
    # Step 1: Check if Flask server is running
    if not is_server_running():
        print("Error: Flask server is not running on localhost:5000")
        print("Please start the server with: python app.py")
        return
    
    print("Flask server is running. Starting data collection...")
    
    driver = None
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries and not is_collection_complete():
        try:
            retry_count += 1
            print(f"\n=== Collection Attempt {retry_count}/{max_retries} ===")
            
            # Step 2 & 3: Initialize database and setup WebDriver
            print("Setting up WebDriver...")
            driver = setup_webdriver()
            
            # Step 4: Start collection process
            print(f"Target: {TRACES_PER_SITE} traces per website")
            total_new = collect_fingerprints(driver)
            
            print(f"\nCollected {total_new} new traces in this session")
            
            # Step 6: Export data
            database.db.export_to_json(OUTPUT_PATH)
            print(f"Data exported to {OUTPUT_PATH}")
            
            # Check if we're done
            if is_collection_complete():
                print("\n✅ Collection target reached!")
                break
            else:
                current_counts = database.db.get_traces_collected()
                print(f"Current counts: {current_counts}")
                print("Retrying to collect remaining traces...")
                
        except Exception as e:
            print(f"Error in main collection loop: {str(e)}")
            traceback.print_exc()
            
        finally:
            # Step 5: Ensure WebDriver is closed
            if driver:
                try:
                    driver.quit()
                except:
                    pass
                driver = None
            
            if retry_count < max_retries and not is_collection_complete():
                print(f"Waiting 5 seconds before retry...")
                time.sleep(5)
    
    # Final export
    try:
        database.db.export_to_json(OUTPUT_PATH)
        final_counts = database.db.get_traces_collected()
        print(f"\n=== Final Results ===")
        print(f"Total traces collected: {final_counts}")
        print(f"Data saved to: {OUTPUT_PATH}")
    except Exception as e:
        print(f"Error in final export: {str(e)}")

if __name__ == "__main__":
    main()
