with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "rb") as f:
    data = f.read()

lines = data.split(b"\n")
print(f"Total lines (after split): {len(lines)}")
# Print last 10 lines
for i in range(max(0, len(lines) - 10), len(lines)):
    print(f"{i+1:3}: {lines[i]!r}")
