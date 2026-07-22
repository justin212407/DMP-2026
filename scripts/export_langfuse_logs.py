from pathlib import Path
import json
import time

from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv(".env")

OUTPUT_DIR = Path("data/raw_logs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "observations.jsonl"

PAGE_SIZE = 100
MAX_PAGES = 1000      # change whenever needed

client = Langfuse()

assert client.auth_check()

downloaded = 0

with OUTPUT_FILE.open("a", encoding="utf8") as f:

    for page in range(1, MAX_PAGES + 1):

        print(f"\nPage {page}")

        resp = client.api.legacy.observations_v1.get_many(
            page=page,
            limit=PAGE_SIZE,
        )

        if len(resp.data) == 0:
            print("No more observations.")
            break

        for obs in resp.data:

            f.write(
                json.dumps(
                    obs.model_dump(),
                    ensure_ascii=False,
                    default=str,
                )
            )

            f.write("\n")

        downloaded += len(resp.data)

        print(
            f"Downloaded {downloaded:,} observations"
        )

        time.sleep(0.2)

print()
print("Done.")
print(OUTPUT_FILE)