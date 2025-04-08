import mwparserfromhell
import requests

# Send the API request for the "Dire wolf" page
response = requests.get(
    "https://en.wikipedia.org/w/api.php",
    params={
        "action": "query",
        "format": "json",
        "titles": "2025 stock market crash",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",  # Required for newer MediaWiki versions
    },
).json()

# Get the page content
page = next(iter(response["query"]["pages"].values()))

# Check if the page exists
if "revisions" not in page:
    print("Page not found.")
else:
    # Access the wikitext from the correct slot
    wikicode = page["revisions"][0]["slots"]["main"]["*"]
    parsed_wikicode = mwparserfromhell.parse(wikicode)
    plain_text = parsed_wikicode.strip_code().strip()

    # Save to a text file
    with open("2025 Myanmar earthquake", "w", encoding="utf-8") as file:
        file.write(plain_text)

    print("2025_earthquake.txt")
