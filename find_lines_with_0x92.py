with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "rb") as f:
    data = f.read()

lines = data.split(b"\n")
print(f"Total lines (after split by b'\\n'): {len(lines)}")
for i, line in enumerate(lines):
    if b"\x92" in line:
        print(f"Line {i+1} (0-indexed {i}) contains 0x92:")
        print(f"  Raw bytes: {line!r}")
        # Try to decode as UTF-8
        try:
            print(f"  As UTF-8: {line.decode('utf-8')!r}")
        except UnicodeDecodeError as e:
            print(f"  Cannot decode as UTF-8: {e}")
        # Try to decode as windows-1252
        try:
            print(f"  As windows-1252: {line.decode('windows-1252')!r}")
        except Exception as e:
            print(f"  Cannot decode as windows-1252: {e}")
