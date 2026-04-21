with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "rb") as f:
    data = f.read()

# Find all occurrences of byte 0x92
positions = [i for i, b in enumerate(data) if b == 0x92]
print(f"Found {len(positions)} occurrences of byte 0x92 at positions: {positions}")

for pos in positions:
    # Show context
    start = max(0, pos - 20)
    end = min(len(data), pos + 20)
    context = data[start:end]
    print(f"Context around position {pos}: {context!r}")
    # Try to decode the context as UTF-8
    try:
        print(f"  As UTF-8: {context.decode('utf-8')!r}")
    except UnicodeDecodeError as e:
        print(f"  Cannot decode as UTF-8: {e}")
    # Try to decode as windows-1252
    try:
        print(f"  As windows-1252: {context.decode('windows-1252')!r}")
    except Exception as e:
        print(f"  Cannot decode as windows-1252: {e}")
    # Also, let's see what character 0x92 is in windows-1252
    if pos < len(data):
        print(
            f"  Byte 0x92 at position {pos} is windows-1252: {bytes([0x92]).decode('windows-1252')!r}"
        )
