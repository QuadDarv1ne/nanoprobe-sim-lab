def find_invalid_utf8(filepath):
    with open(filepath, "rb") as f:
        data = f.read()

    try:
        data.decode("utf-8")
        print("No invalid UTF-8 found")
        return
    except UnicodeDecodeError as e:
        print(f"Invalid UTF-8 at byte index {e.start}: {e.reason}")
        # Show some context
        start = max(0, e.start - 20)
        end = min(len(data), e.end + 20)
        context = data[start:end]
        print(f"Context (bytes): {context!r}")
        # Try to decode as windows-1252 for the invalid part
        try:
            invalid_part = data[e.start : e.end]
            print(f"Invalid bytes: {invalid_part!r}")
            print(f"As windows-1252: {invalid_part.decode('windows-1252')!r}")
        except Exception as e2:
            print(f"Could not decode invalid part as windows-1252: {e2}")
        # Now, let's find the line number
        # Count newlines up to the invalid byte
        newline_count = data[: e.start].count(b"\n")
        # Line number is newline_count + 1 (if we count lines starting at 1)
        print(f"This byte is at line number {newline_count + 1} (1-indexed)")
        # Let's also show the line
        lines = data.split(b"\n")
        if newline_count < len(lines):
            line = lines[newline_count]
            print(f"Line content (bytes): {line!r}")
            try:
                line.decode("utf-8")
            except UnicodeDecodeError:
                print("This line cannot be decoded as UTF-8")
            # Try to decode as windows-1252
            try:
                print(f"Line as windows-1252: {line.decode('windows-1252')!r}")
            except Exception as e2:
                print(f"Line cannot be decoded as windows-1252: {e2}")


if __name__ == "__main__":
    find_invalid_utf8(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py")
