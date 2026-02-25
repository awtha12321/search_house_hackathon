# Rental Listing Scraper – Docker Setup

A Docker-based web scraper that extracts rental listings from websites such as [irisrent.hu](https://www.irisrent.hu/) using **Playwright** for browser automation and the **Gemini API** for AI-powered data extraction.

---

## Project Structure

```
/
├── docker/
│   └── Dockerfile           # Container image definition
├── docker-compose.yml       # Compose file for easy deployment
├── scraper/
│   ├── scraper.py           # Main scraper script
│   ├── config.yaml          # Scraping configuration
│   └── requirements.txt     # Python dependencies
├── data/                    # Output directory (mounted volume)
├── .env.example             # Environment variable template
├── .gitignore
└── README.md
```

---

## Prerequisites

- **Docker Desktop for Windows** with the **WSL2 backend** enabled  
  → [Install guide](https://docs.docker.com/desktop/install/windows-install/)
- **WSL2** (Windows Subsystem for Linux v2)  
  → Enable in PowerShell: `wsl --install`
- A **Gemini API key** from [Google AI Studio](https://aistudio.google.com/app/apikey)

> All other dependencies (Python, Playwright, libraries) are installed automatically inside the container.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/awtha12321/search_house_hackathon.git
cd search_house_hackathon
```

### 2. Set up your API key

```bash
cp .env.example .env
```

Open `.env` and replace `your_api_key_here` with your actual Gemini API key:

```
GEMINI_API_KEY=AIzaSy...your_key_here
OUTPUT_DIR=./data
```

### 3. Build the Docker image

```bash
docker-compose build
```

This installs Python, Playwright (with Chromium), and all required Python packages inside the container.  
**This step only needs to be done once** (or after updating dependencies).

---

## Running the Scraper

```bash
docker-compose run --rm scraper
```

The scraper will:
1. Open a headless Chromium browser inside the container.
2. Navigate to the configured target URL(s) and paginate through listings.
3. Send each page's HTML to the Gemini API for intelligent data extraction.
4. Save the results as a timestamped CSV file in the `./data/` directory.

### Run with custom options

Override the target URL or max pages without editing `config.yaml`:

```bash
# Scrape a different URL
docker-compose run --rm scraper python scraper.py --url https://www.ingatlan.com/kiado-lakas

# Limit to 3 pages
docker-compose run --rm scraper python scraper.py --max-pages 3

# Use a custom config file
docker-compose run --rm scraper python scraper.py --config /app/config.yaml
```

---

## CSV Output

CSV files are written to the `./data/` directory on your host machine (mounted as a volume).

Each file is named `rental_listings_YYYYMMDD_HHMMSS.csv` and contains the following columns:

| Column | Description |
|---|---|
| `scraped_at` | UTC timestamp of when the row was scraped |
| `title` | Property name / headline |
| `price` | Monthly rent (numeric, local currency) |
| `location` | Address or district |
| `size_m2` | Floor area in square metres |
| `num_rooms` | Number of rooms/bedrooms |
| `amenities` | JSON array of features |
| `description` | Full listing description |
| `image_urls` | JSON array of photo URLs |
| `contact_info` | Phone, email or agent name |
| `listing_url` | URL of the listing detail page |

The file is UTF-8 encoded and handles Hungarian characters correctly.

---

## Configuration

Edit `scraper/config.yaml` to customise scraping behaviour:

```yaml
scraper:
  target_urls:
    - "https://www.irisrent.hu/kiado-lakas"
    - "https://www.ingatlan.com/kiado-lakas"   # add more sites here
  max_pages: 10          # pages to paginate per site
  request_delay: 2.0     # seconds between requests
  headless: true         # set to false to see the browser (requires display)

gemini:
  model: "gemini-1.5-flash"   # or gemini-1.5-pro for higher accuracy

output:
  directory: "./data"
  filename: "rental_listings"
  encoding: "utf-8"
```

### Configuring for a different website

1. Add the new URL under `scraper.target_urls` in `config.yaml`.
2. If the site uses a non-standard pagination scheme, update the `_pagination_url()` function in `scraper/scraper.py`.
3. The Gemini-powered extraction is site-agnostic — it analyses whatever HTML it receives.

---

## Getting a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with your Google account.
3. Click **Create API key**.
4. Copy the key and paste it into your `.env` file.

The free tier is sufficient for moderate scraping volumes.

---

## Troubleshooting

### Docker Desktop / WSL2 issues

| Problem | Solution |
|---|---|
| `docker: command not found` in WSL2 | Ensure Docker Desktop is running and WSL2 integration is enabled in Settings → Resources → WSL Integration |
| Permission denied on `./data/` | Run `chmod 777 ./data` in your WSL2 terminal |
| Container exits immediately | Check logs with `docker-compose logs scraper` |
| Playwright browser fails to launch | Verify all system dependencies are installed; rebuild with `docker-compose build --no-cache` |

### Scraper issues

| Problem | Solution |
|---|---|
| `GEMINI_API_KEY not set` | Ensure `.env` exists and has the correct key |
| No listings extracted | The site may have changed its HTML structure; check with `headless: false` and inspect the page |
| Rate limit / 429 errors | Increase `request_delay` in `config.yaml` |
| JSON parse errors from Gemini | Try switching to `gemini-1.5-pro` for better extraction reliability |

---

## Notes

- This scraper is intended for **personal / educational use** to populate a vector database with rental data.
- Always respect the target website's `robots.txt` and terms of service.
- The default `request_delay: 2.0` seconds helps avoid overloading the server.
- The scraper runs **on-demand** — it does not run on a schedule.
