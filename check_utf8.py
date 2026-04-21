with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "rb") as f:
    data = f.read()
try:
    data.decode("utf-8")
    print("The entire file is valid UTF-8")
except UnicodeDecodeError as e:
    print(f"Invalid UTF-8 at byte index {e.start}: {e.reason}")
    print(f"  Start: {e.start}, End: {e.end}")
    # Show the invalid bytes
    print(f"  Invalid bytes: {data[e.start:e.end]!r}")
    # Show context
    start = max(0, e.start - 20)
    end = min(len(data), e.end + 20)
    context = data[start:end]
    print(f"  Context: {context!r}")
