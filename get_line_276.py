with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Number of lines: {len(lines)}")
if len(lines) >= 276:
    line = lines[275]  # 0-indexed
    print(f"Line 276 (1-indexed): {repr(line)}")
else:
    print(f"File has only {len(lines)} lines, so line 276 does not exist.")
