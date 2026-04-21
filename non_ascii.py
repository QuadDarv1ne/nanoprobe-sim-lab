with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "rb") as f:
    data = f.read()

non_ascii = [b for b in data if b > 127]
print(f"Number of non-ASCII bytes: {len(non_ascii)}")
if non_ascii:
    print(f"First few non-ASCII bytes: {non_ascii[:20]}")
    # Let's see where they occur
    # We'll find the positions
    positions = [i for i, b in enumerate(data) if b > 127]
    print(f"Positions of non-ASCII bytes: {positions[:20]}")
    # Now, let's look at the context of the first few
    for pos in positions[:5]:
        start = max(0, pos - 10)
        end = min(len(data), pos + 10)
        context = data[start:end]
        print(f"Context around position {pos}: {context!r}")
        # Try to decode as windows-1252
        try:
            print(f"  As windows-1252: {context.decode('windows-1252')!r}")
        except Exception:
            pass
else:
    print("No non-ASCII bytes found.")
