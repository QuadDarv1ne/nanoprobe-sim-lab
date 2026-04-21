with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "rb") as f:
    data = f.read()

# Count the number of newline characters (LF and CRLF)
lf_count = data.count(b"\n")
crlf_count = data.count(b"\r\n")
print(f"LF count: {lf_count}")
print(f"CRLF count: {crlf_count}")

# The number of lines is the number of newline characters + 1 if the file does not end with a newline.
# But let's just count the lines by splitting.
lines = data.split(b"\n")
print(f"Number of lines after split by b'\\n': {len(lines)}")
if data.endswith(b"\n"):
    print("File ends with a newline, so the last line is empty.")
else:
    print("File does not end with a newline.")

# Let's also split by b'\r\n' to see if that gives a different count.
lines_crlf = data.split(b"\r\n")
print(f"Number of lines after split by b'\\r\\n': {len(lines_crlf)}")
if data.endswith(b"\r\n"):
    print("File ends with a CRLF, so the last line is empty.")
else:
    print("File does not end with a CRLF.")

# Now, let's try to read the file as text with UTF-8 and see how many lines we get.
try:
    text = data.decode("utf-8")
    lines_text = text.splitlines()
    print(f"Number of lines after decoding as UTF-8 and splitlines(): {len(lines_text)}")
    lines_text_keep = text.splitlines(True)
    print(f"Number of lines after decoding as UTF-8 and splitlines(True): {len(lines_text_keep)}")
except Exception as e:
    print(f"Failed to decode as UTF-8: {e}")
