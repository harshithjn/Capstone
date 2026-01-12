import subprocess, time

print("Starting PROD batch...")

web = subprocess.Popen(
    ["python", "workloads/web/app.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

time.sleep(5)

try:
    subprocess.run(["python", "scripts/batch_runner_prod.py"], check=True)
finally:
    web.terminate()
    web.wait()

print("Batch completed successfully.")
