with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "rb") as f:
    data = f.read()

print(f"Total bytes: {len(data)}")
print("First 100 bytes:", repr(data[:100]))
print("Last 100 bytes:", repr(data[-100:]))

# Count different types of line endings
crlf_count = data.count(b"\r\n")
lf_count = data.count(b"\n") - crlf_count  # Approximate LF count (not counting those in CRLF)
cr_count = data.count(b"\r") - crlf_count  # Approximate CR count (not counting those in CRLF)

print(f"CRLF (\\r\\n) count: {crlf_count}")
print(f"LF (\\n) count: {lf_count}")
print(f"CR (\\r) count: {cr_count}")

# Let's split by different line endings and see what we get
lines_by_split = data.split(b"\n")
print(f"Lines when split by b'\\n': {len(lines_by_split)}")

lines_by_split_crlf = data.split(b"\r\n")
print(f"Lines when split by b'\\r\\n': {len(lines_by_split_crlf)}")

# Let's read the file as text with different newline interpretations
try:
    text_utf8 = data.decode("utf-8")
    print(f"Successfully decoded as UTF-8, length: {len(text_utf8)} chars")

    # Now split the text into lines
    lines_text = text_utf8.splitlines()
    print(f"Lines when using splitlines() on UTF-8 text: {len(lines_text)}")

    # Show last 10 lines
    print("Last 10 lines:")
    for i, line in enumerate(lines_text[-10:], start=len(lines_text) - 9):
        print(f"  {i:3}: {repr(line)}")

except Exception as e:
    print(f"Failed to decode as UTF-8: {e}")

# Let's also try to read with universal newlines
try:
    text_universal = data.decode("utf-8")
    lines_universal = text_universal.splitlines(True)  # Keep line endings
    print(f"Lines with splitlines(True): {len(lines_universal)}")
    print("Last 5 lines with endings:")
    for i, line in enumerate(lines_universal[-5:], start=len(lines_universal) - 4):
        print(f"  {i + len(lines_universal) - 4:3}: {repr(line)}")
except Exception as e:
    print(f"Failed in universal newlines approach: {e}")
