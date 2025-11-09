import glob
import json
import os
import numpy as np

# Pattern for your logdirs (adjust if you used a different name)
LOG_PATTERN = "logs/GN-GAN_CIFAR10_RES_seed*/eval.txt"

paths = sorted(glob.glob(LOG_PATTERN))
if not paths:
    raise SystemExit(f"No eval.txt files found with pattern: {LOG_PATTERN}")

print("Found eval files:")
for p in paths:
    print("  ", p)

results = []

for path in paths:
    with open(path, "r") as f:
        lines = f.readlines()
        if not lines:
            continue
        # Last evaluation line = final metrics for that seed
        data = json.loads(lines[-1])
        # Prefer EMA metrics (usually better / reported in paper)
        IS = data.get("IS_EMA", data["IS"])
        IS_std = data.get("IS_std_EMA", data["IS_std"])
        FID = data.get("FID_EMA", data["FID"])
        results.append((IS, IS_std, FID))

if not results:
    raise SystemExit("No valid metric lines found in eval.txt files.")

IS_vals = np.array([r[0] for r in results])
IS_std_vals = np.array([r[1] for r in results])
FID_vals = np.array([r[2] for r in results])

print("\nPer-seed results (EMA if available):")
for i, (IS, IS_std, FID) in enumerate(results):
    print(f"  Seed {i}: IS={IS:.3f} (±{IS_std:.3f}), FID={FID:.3f}")

print("\n=== Aggregated over seeds ===")
print(f"Inception Score: {IS_vals.mean():.3f} ± {IS_vals.std(ddof=1):.3f}")
print(f"FID:             {FID_vals.mean():.3f} ± {FID_vals.std(ddof=1):.3f}")
