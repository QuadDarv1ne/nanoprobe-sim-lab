with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "rb") as f:
    data = f.read()

lines = data.split(b"\n")
print(f"Number of lines (split by \\n): {len(lines)}")
for i, line in enumerate(lines):
    if i >= 270 and i < 280:
        print(f"{i:3}: {line!r}")
        try:
            line.decode("utf-8")
        except UnicodeDecodeError as e:
            print(f"   UnicodeDecodeError: {e}")
