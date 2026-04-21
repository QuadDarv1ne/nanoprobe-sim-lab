with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
for i, line in enumerate(lines):
    if i >= 260:  # Show from line 260 onward
        print(f"{i+1:3}: {repr(line)}")
