with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "rb") as f:
    data = f.read()

# Split by newline, but keep the newline characters for accurate line numbers?
# We'll split and then add the newline back for each line except the last.
lines = data.split(b"\n")
print(f"Number of lines (after split): {len(lines)}")

# Now, let's reconstruct the lines with newline to get the original line endings.
# We'll iterate and add b'\n' for each line except the last if the original ended with newline.
# But for simplicity, we'll just check each line as split (without newline) and then note that the line number in the error is 1-indexed.

for i, line in enumerate(lines):
    # The line number in the error is 1-indexed, and the line content does not include the newline.
    # However, the error might be referring to the line in the original file with newline.
    # We'll check the line as split (without newline) and also the line with newline if we can.
    # But note: the split removes the newline, so the line we are looking at is the content without the newline.
    # The error says line 276, so we look at index 275 (0-indexed) in the lines array.
    if i >= 270 and i < 280:
        try:
            line.decode("utf-8")
        except UnicodeDecodeError as e:
            print(f"Line {i+1} (0-indexed {i}) has invalid UTF-8: {e}")
            print(f"  Raw bytes: {line!r}")
            # Also show the line with the newline if we can reconstruct
            # We don't have the newline here, but we can note that the original line ended with b'\n' or b'\r\n'
            # Let's look at the original data to see the newline for this line.
            # We'll find the start and end of this line in the original data.
            # This is a bit complex. Let's just print the line number and the raw bytes.

# Let's also try to decode the entire file line by line as the original would be read.
# We'll split by b'\n' and then try to decode each part.
# But note: the file might have been saved with CRLF or LF.
# We'll do a simple approach: split by b'\n' and then try to decode each part.

print("\nNow trying to decode each line as UTF-8 (without newline):")
for i, line in enumerate(lines):
    if i >= 270 and i < 280:
        try:
            line.decode("utf-8")
        except UnicodeDecodeError as e:
            print(f"Line {i+1} (0-indexed {i}) fails: {e}")
            print(f"  Content: {line!r}")

# Now, let's look at the exact line that pytest complained about: line 276 (1-indexed) -> index 275
print(f"\nExamining line 276 (1-indexed) which is index 275 in 0-indexed array:")
line_276 = lines[275] if len(lines) > 275 else None
if line_276 is not None:
    print(f"  Raw bytes: {line_276!r}")
    try:
        line_276.decode("utf-8")
        print("  Decodes as UTF-8 successfully")
    except UnicodeDecodeError as e:
        print(f"  FAILS to decode as UTF-8: {e}")
else:
    print("  Line 276 does not exist in the split array.")

# Let's also check the line in the original data by finding the 276th line (1-indexed) by counting newlines.
# We'll iterate through the data and count newlines.
print("\nAlternative: counting newlines in the original data to find the 276th line:")
newline_count = 0
start = 0
for i, byte in enumerate(data):
    if byte == ord("\n"):
        newline_count += 1
        if (
            newline_count == 275
        ):  # We want the 276th line, so we look for the 275th newline (0-indexed count of newlines)
            # The line starts at 'start' and ends at i (exclusive of the newline)
            line_content = data[start:i]
            print(f"  Found the 276th line (1-indexed) at byte index {start} to {i}")
            print(f"  Raw bytes: {line_content!r}")
            try:
                line_content.decode("utf-8")
                print("  Decodes as UTF-8 successfully")
            except UnicodeDecodeError as e:
                print(f"  FAILS to decode as UTF-8: {e}")
            break
        start = i + 1
else:
    print("  Did not find the 276th line by counting newlines.")
