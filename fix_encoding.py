import sys


def main():
    file_path = r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py"
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        # Try to decode as UTF-8
        try:
            data.decode("utf-8")
            print("File is already UTF-8")
            return
        except UnicodeDecodeError:
            pass
        # Try windows-1252
        try:
            decoded = data.decode("windows-1252")
            print("Successfully decoded as windows-1252")
        except UnicodeDecodeError as e:
            print(f"Failed to decode as windows-1252: {e}")
            return
        # Write back as UTF-8
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(decoded)
        print("File rewritten in UTF-8")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
