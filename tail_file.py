with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "rb") as f:
    lines = f.readlines()
print(f"Total lines: {len(lines)}")
# Print last 30 lines
for i, line in enumerate(lines[-30:], start=len(lines) - 29):
    print(f"{i:4}: {line!r}")
