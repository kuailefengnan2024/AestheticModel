import time
print("Starting import...")
start = time.time()
from volcenginesdkarkruntime import Ark
end = time.time()
print(f"Import took {end - start:.2f} seconds")
