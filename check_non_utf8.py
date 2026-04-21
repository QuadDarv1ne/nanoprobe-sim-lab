import sys


def main():
    file_path = r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py"
    with open(file_path, "rb") as f:
        data = f.read()

    # Find non-UTF-8 bytes
    try:
        data.decode("utf-8")
        print("The entire file is valid UTF-8")
        return
    except UnicodeDecodeError as e:
        print(f"Invalid UTF-8 at position {e.start}: {e.reason}")
        # Show context
        start = max(0, e.start - 10)
        end = min(len(data), e.end + 10)
        context = data[start:end]
        print(f"Context: {context!r}")
        # Try to decode as windows-1252 to see what the characters are
        try:
            decoded = data[e.start : e.end].decode("windows-1252")
            print(f"Invalid bytes as windows-1252: {decoded!r}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
