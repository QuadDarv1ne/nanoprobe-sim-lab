with open(r"M:\GitHub\nanoprobe-sim-lab\tests\test_admin_api.py", "rb") as f:
    data = f.read()
try:
    compile(data, "tests/test_admin_api.py", "exec")
    print("Compiled successfully")
except SyntaxError as e:
    print(f"SyntaxError: {e}")
    print(f"Line {e.lineno}: {e.text}")
    if e.offset:
        print(f"Offset: {e.offset}")
        print(" " * (e.offset - 1) + "^")
