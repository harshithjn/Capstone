import subprocess, time, os

CPU_CORES  = [2, 4, 8, 16]
MEMORY_GB = [4, 8, 16, 32]
REPETITIONS = 6

WORKLOADS = [
    {
        "type": "ML",
        "name": "mlperf_resnet50",
        "command": (
            "python workloads/mlperf/inference/vision/classification_and_detection/run.py "
            "--backend pytorch --model resnet50 --scenario Offline --device cpu"
        )
    },
    {
        "type": "ML",
        "name": "mlperf_bert",
        "command": (
            "python workloads/mlperf/inference/language/bert/run.py "
            "--backend pytorch --scenario Offline --device cpu"
        )
    },
    {
        "type": "DB",
        "name": "tpch_q3",
        "command": "psql tpch_db -f workloads/tpch/q3.sql"
    },
    {
        "type": "DB",
        "name": "tpch_q6",
        "command": "psql tpch_db -f workloads/tpch/q6.sql"
    },
    {
        "type": "WEB",
        "name": "wrk_50",
        "command": "wrk -t2 -c50 -d20s http://localhost:5000"
    },
    {
        "type": "WEB",
        "name": "wrk_200",
        "command": "wrk -t4 -c200 -d20s http://localhost:5000"
    }
]

MODE = os.environ["MODE"]              # prod
BATCH_ID = os.environ["BATCH_ID"]      # batch_01
COLLECTOR = os.environ["COLLECTOR_ID"]

OUTPUT = f"data/{MODE}_{COLLECTOR}_{BATCH_ID}.csv"

for r in range(REPETITIONS):
    for w in WORKLOADS:
        for cpu in CPU_CORES:
            for mem in MEMORY_GB:
                print(f"{MODE.upper()} | {w['name']} | CPU={cpu} RAM={mem} | REP={r+1}")
                subprocess.run([
                    "python","scripts/run_workload.py",
                    "--command", w["command"],
                    "--mode", MODE,
                    "--workload_type", w["type"],
                    "--workload_name", w["name"],
                    "--cpu_cores", str(cpu),
                    "--mem_total_gb", str(mem),
                    "--output_csv", OUTPUT,
                    "--collector_id", COLLECTOR,
                    "--batch_id", BATCH_ID,
                    "--cloud_region", "azure",
                    "--vm_sku", "Standard_D8s_v5"
                ], check=True)
                time.sleep(5)
