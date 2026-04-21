with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "rb") as f:
    data = f.read()

print(f"Total bytes: {len(data)}")
# Let's look at the last 100 bytes
print(f"Last 100 bytes: {data[-100:]:!r}")

# Let's split by newline and see the lines
lines = data.split(b"\n")
print(f"Number of lines after split by b'\\n': {len(lines)}")
for i, line in enumerate(lines):
    if i >= len(lines) - 10:
        print(f"{i:3}: {line!r}")

# Now, let's try to decode each line and see if any fail
for i, line in enumerate(lines):
    try:
        line.decode("utf-8")
    except UnicodeDecodeError as e:
        print(f"Line {i} (0-indexed) failed to decode: {e}")
        print(f"  Content: {line!r}")
