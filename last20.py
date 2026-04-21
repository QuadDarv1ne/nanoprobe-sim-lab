with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "rb") as f:
    lines = f.readlines()
print(f"Total lines: {len(lines)}")
for i, line in enumerate(lines[-20:], start=len(lines) - 19):
    print(f"{i:3}: {repr(line)}")
