with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "rb") as f:
    # Read the file line by line
    lines = f.readlines()

# Since we read with 'rb', each line is bytes and includes the newline character at the end (if present).
# We want to show line 276 (1-indexed) which is index 275 in 0-indexed.
line_number = 276
index = line_number - 1
if index < len(lines):
    line = lines[index]
    print(f"Line {line_number} (1-indexed) as bytes: {line!r}")
    # Try to decode as UTF-8
    try:
        decoded = line.decode("utf-8")
        print(f"Decoded as UTF-8: {repr(decoded)}")
    except UnicodeDecodeError as e:
        print(f"Failed to decode as UTF-8: {e}")
        # Show the problematic part
        print(f"  Start: {e.start}, End: {e.end}")
        # Show the invalid bytes
        invalid_bytes = line[e.start : e.end]
        print(f"  Invalid bytes: {invalid_bytes!r}")
        # Try to decode as windows-1252
        try:
            print(f"  As windows-1252: {invalid_bytes.decode('windows-1252')!r}")
        except Exception as e2:
            print(f"  Failed to decode as windows-1252: {e2}")
else:
    print(f"Line {line_number} does not exist in the file (only {len(lines)} lines).")
