with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "rb") as f:
    data = f.read()

# Find all occurrences of byte 0x92
positions = [i for i, b in enumerate(data) if b == 0x92]
print(f"Found {len(positions)} occurrences of byte 0x92 at positions: {positions}")

for pos in positions:
    # Show context - 30 bytes before and after
    start = max(0, pos - 30)
    end = min(len(data), pos + 30)
    context = data[start:end]
    print(f"\nContext around position {pos}:")
    print(f"  Raw bytes: {context!r}")

    # Try to decode as UTF-8
    try:
        print(f"  As UTF-8: {context.decode('utf-8')!r}")
    except UnicodeDecodeError as e:
        print(f"  Cannot decode as UTF-8: {e}")

    # Try to decode as windows-1252
    try:
        print(f"  As windows-1252: {context.decode('windows-1252')!r}")
    except Exception as e:
        print(f"  Cannot decode as windows-1252: {e}")

    # Show the exact byte
    print(f"  Byte at position {pos}: 0x{data[pos]:02x} ({data[pos]})")
