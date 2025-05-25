import os
import urllib.request

os.makedirs('bach-cello', exist_ok=True)
os.chdir('bach-cello')

urls = [
    "cs1-1pre.mid","cs1-2all.mid","cs1-3cou.mid","cs1-4sar.mid","cs1-5men.mid","cs1-6gig.mid",
    "cs2-1pre.mid","cs2-2all.mid","cs2-3cou.mid","cs2-4sar.mid","cs2-5men.mid","cs2-6gig.mid",
    "cs3-1pre.mid","cs3-2all.mid","cs3-3cou.mid","cs3-4sar.mid","cs3-5bou.mid","cs3-6gig.mid",
    "cs4-1pre.mid","cs4-2all.mid","cs4-3cou.mid","cs4-4sar.mid","cs4-5bou.mid","cs4-6gig.mid",
    "cs5-1pre.mid","cs5-2all.mid","cs5-3cou.mid","cs5-4sar.mid","cs5-5gav.mid","cs5-6gig.mid",
    "cs6-1pre.mid","cs6-2all.mid","cs6-3cou.mid","cs6-4sar.mid","cs6-5gav.mid","cs6-6gig.mid"
]
base_url = "http://www.jsbach.net/midi"

for file in urls:
    print(f"Downloading {file}...")
    urllib.request.urlretrieve(f"{base_url}/{file}", file)
print("🚀 Done downloading all files!")
