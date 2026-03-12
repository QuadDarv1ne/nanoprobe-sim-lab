import subprocess
import sys
result = subprocess.run(
    [r"C:\Users\maksi\AppData\Local\Programs\Python\Python313\python.exe", "-m", "pytest", "tests/test_sdr_interface.py", "-v", "--tb=short", "-o", "asyncio_mode=auto"],
    capture_output=True, text=True
)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
sys.exit(result.returncode)
