with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "rb") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if i >= 270 and i <= 280:
        print(f"{i:3}: {line}")
        try:
            line.decode("utf-8")
        except UnicodeDecodeError as e:
            print(f"   UnicodeDecodeError: {e}")
            print(f"   Raw bytes: {line!r}")
