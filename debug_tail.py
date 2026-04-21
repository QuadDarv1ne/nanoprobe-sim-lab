with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "rb") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
for i in range(len(lines) - 10, len(lines)):
    line = lines[i]
    print(f"{i+1:3}: {line}")
    try:
        line.decode("utf-8")
    except UnicodeDecodeError as e:
        print(f"   UnicodeDecodeError: {e}")
        print(f"   Raw bytes: {line!r}")
