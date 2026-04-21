with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
print(f"Number of lines (read with UTF-8): {len(lines)}")
for i, line in enumerate(lines[-10:], start=len(lines) - 9):
    print(f"{i:3}: {repr(line)}")
