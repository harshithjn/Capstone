import argparse, csv, os, time, subprocess, uuid
from datetime import datetime
import psutil

def collect_metrics(proc):
    cpu, mem = [], []
    disk_start = psutil.disk_io_counters()

    while proc.poll() is None:
        cpu.append(psutil.cpu_percent(interval=1))
        mem.append(psutil.virtual_memory().used / (1024**3))

    disk_end = psutil.disk_io_counters()
    return {
        "cpu_avg": sum(cpu) / len(cpu),
        "mem_avg": sum(mem) / len(mem),
        "disk_read": (disk_end.read_bytes - disk_start.read_bytes) / (1024**2),
        "disk_write": (disk_end.write_bytes - disk_start.write_bytes) / (1024**2),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--command", required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--workload_type", required=True)
    p.add_argument("--workload_name", required=True)
    p.add_argument("--cpu_cores", type=int, required=True)
    p.add_argument("--mem_total_gb", type=int, required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--collector_id", required=True)
    p.add_argument("--batch_id", required=True)
    p.add_argument("--cloud_region", required=True)
    p.add_argument("--vm_sku", required=True)
    args = p.parse_args()

    start = time.time()
    proc = subprocess.Popen(args.command, shell=True)
    m = collect_metrics(proc)
    runtime = time.time() - start

    row = [
        str(uuid.uuid4()), args.mode,
        args.workload_type, args.workload_name,
        args.cpu_cores, args.mem_total_gb,
        round(m["cpu_avg"], 2), round(m["mem_avg"], 2),
        round(m["disk_read"], 2), round(m["disk_write"], 2),
        round(runtime, 2),
        args.collector_id, args.batch_id,
        args.cloud_region, args.vm_sku,
        datetime.utcnow().isoformat()
    ]

    header = [
        "run_id","mode","workload_type","workload_name",
        "cpu_cores","mem_total_gb",
        "cpu_avg_pct","mem_avg_gb",
        "disk_read_mb","disk_write_mb",
        "runtime_sec",
        "collector_id","batch_id",
        "cloud_region","vm_sku","timestamp"
    ]

    write_header = not os.path.exists(args.output_csv)
    with open(args.output_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

if __name__ == "__main__":
    main()
