"""
Training Manager for pipe-inspector
- Start/stop YOLO segmentation training as subprocess
- Monitor results.csv for real-time metrics
- List datasets, runs, serve result images
"""
import os
import sys
import json
import time
import signal
import csv
import subprocess
import threading
import uuid
from pathlib import Path
from datetime import datetime

# ── globals ──────────────────────────────────────────────────────
BASE_DIR = Path(os.environ.get("PIPE_INSPECTOR_DIR", "/home/intu/projects/pipe-inspector-electron"))
DATASETS_DIR = BASE_DIR / "datasets"
RUNS_DIR = BASE_DIR / "runs" / "segment"
PYTHON_BIN = str(BASE_DIR / ".venv" / "bin" / "python3")

_active_jobs: dict = {}  # job_id -> job_state


# ── helpers ──────────────────────────────────────────────────────
def _parse_results_csv(csv_path: str) -> list[dict]:
    """Parse Ultralytics results.csv into list of dicts."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean = {}
                for k, v in row.items():
                    k = k.strip()
                    try:
                        clean[k] = float(v.strip())
                    except (ValueError, AttributeError):
                        clean[k] = v.strip() if isinstance(v, str) else v
                rows.append(clean)
    except Exception:
        pass
    return rows


def _monitor_training(job_id: str):
    """Background thread: poll results.csv + process stdout."""
    job = _active_jobs.get(job_id)
    if not job:
        return

    proc = job["process"]
    results_csv = os.path.join(job["run_dir"], "results.csv")
    last_epoch = -1

    while proc.poll() is None:
        time.sleep(3)
        # parse results.csv for latest metrics
        rows = _parse_results_csv(results_csv)
        if rows:
            latest = rows[-1]
            epoch = int(latest.get("epoch", last_epoch))
            if epoch > last_epoch:
                last_epoch = epoch
                job["current_epoch"] = epoch + 1
                job["metrics_history"] = rows
                job["latest_metrics"] = latest
                pct = min(100, int((epoch + 1) / job["total_epochs"] * 100))
                job["progress_pct"] = pct

    # Process finished
    rc = proc.returncode
    job["status"] = "completed" if rc == 0 else "failed"
    job["finished_at"] = datetime.now().isoformat()
    job["return_code"] = rc

    # Final read of results
    rows = _parse_results_csv(results_csv)
    if rows:
        job["metrics_history"] = rows
        job["latest_metrics"] = rows[-1]

    # Read best metrics
    run_dir = Path(job["run_dir"])
    if (run_dir / "weights" / "best.pt").exists():
        job["best_model"] = str(run_dir / "weights" / "best.pt")
        job["best_model_size_mb"] = round((run_dir / "weights" / "best.pt").stat().st_size / 1024 / 1024, 1)

    # List result images
    result_images = []
    for ext in ["*.png", "*.jpg"]:
        for f in run_dir.glob(ext):
            if f.name != "labels.jpg":
                result_images.append(f.name)
    job["result_images"] = sorted(result_images)

    print(f"[TRAINING] Job {job_id} finished with code {rc}")


# ── public API ───────────────────────────────────────────────────
def list_datasets() -> list[dict]:
    """List available datasets in datasets/ dir."""
    datasets = []
    if not DATASETS_DIR.exists():
        return datasets
    for d in sorted(DATASETS_DIR.iterdir()):
        if not d.is_dir():
            continue
        yaml_file = d / "dataset.yaml"
        info = {"name": d.name, "path": str(d)}
        if yaml_file.exists():
            info["has_yaml"] = True
            # Count images
            train_dir = d / "images" / "train"
            val_dir = d / "images" / "val"
            info["train_images"] = len(list(train_dir.glob("*"))) if train_dir.exists() else 0
            info["val_images"] = len(list(val_dir.glob("*"))) if val_dir.exists() else 0
            # Parse yaml for class info
            try:
                with open(yaml_file) as f:
                    content = f.read()
                for line in content.split("\n"):
                    if line.strip().startswith("nc:"):
                        info["num_classes"] = int(line.split(":")[1].strip())
                    if "names:" in line:
                        pass  # could parse but keep simple
            except Exception:
                pass
        else:
            info["has_yaml"] = False
        # Disk size
        try:
            result = subprocess.run(["du", "-sh", str(d)], capture_output=True, text=True, timeout=5)
            info["disk_size"] = result.stdout.split()[0] if result.stdout else "?"
        except Exception:
            info["disk_size"] = "?"
        datasets.append(info)
    return datasets


def list_runs() -> list[dict]:
    """List completed/in-progress training runs."""
    runs = []
    if not RUNS_DIR.exists():
        return runs
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        info = {"name": d.name, "path": str(d)}

        # Check if active job
        for jid, job in _active_jobs.items():
            if job.get("run_dir") == str(d):
                info["job_id"] = jid
                info["status"] = job["status"]
                break
        else:
            info["status"] = "completed" if (d / "weights" / "best.pt").exists() else "unknown"

        # Parse args.yaml
        args_file = d / "args.yaml"
        if args_file.exists():
            try:
                with open(args_file) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("model:"):
                            info["model"] = line.split(":", 1)[1].strip()
                        elif line.startswith("epochs:"):
                            info["epochs"] = int(line.split(":")[1].strip())
                        elif line.startswith("imgsz:"):
                            info["imgsz"] = int(line.split(":")[1].strip())
                        elif line.startswith("batch:"):
                            info["batch"] = int(line.split(":")[1].strip())
                        elif line.startswith("data:"):
                            info["data"] = line.split(":", 1)[1].strip()
            except Exception:
                pass

        # Best model info
        best_pt = d / "weights" / "best.pt"
        if best_pt.exists():
            info["best_model_size_mb"] = round(best_pt.stat().st_size / 1024 / 1024, 1)

        # Final metrics from results.csv
        results_csv = d / "results.csv"
        rows = _parse_results_csv(str(results_csv))
        if rows:
            last = rows[-1]
            info["total_epochs_run"] = len(rows)
            info["final_metrics"] = {
                "box_mAP50": last.get("metrics/mAP50(B)", 0),
                "box_mAP50_95": last.get("metrics/mAP50-95(B)", 0),
                "mask_mAP50": last.get("metrics/mAP50(M)", 0),
                "mask_mAP50_95": last.get("metrics/mAP50-95(M)", 0),
            }

        # Result images
        result_images = []
        for ext in ["*.png", "*.jpg"]:
            for f in d.glob(ext):
                if f.name not in ("labels.jpg",):
                    result_images.append(f.name)
        info["result_images"] = sorted(result_images)

        runs.append(info)
    return runs


def get_run_image(run_name: str, filename: str) -> str | None:
    """Get full path to a run result image."""
    path = RUNS_DIR / run_name / filename
    if path.exists() and path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        return str(path)
    return None


def start_training(config: dict) -> dict:
    """Start a YOLO segmentation training job."""
    dataset_name = config.get("dataset", "")
    model_arch = config.get("model", "yolov8m-seg.pt")
    epochs = int(config.get("epochs", 100))
    batch = int(config.get("batch", 8))
    imgsz = int(config.get("imgsz", 640))
    patience = int(config.get("patience", 20))
    run_name = config.get("name", f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    lr0 = float(config.get("lr0", 0.01))
    augment = config.get("augment", True)

    # Validate dataset
    dataset_dir = DATASETS_DIR / dataset_name
    yaml_path = dataset_dir / "dataset.yaml"
    if not yaml_path.exists():
        return {"error": f"Dataset YAML not found: {yaml_path}"}

    # Check GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        free_mb = int(result.stdout.strip().split("\n")[0])
        if free_mb < 2000:
            return {"error": f"GPU memory too low: {free_mb}MB free (need ≥2GB)"}
    except Exception:
        pass

    # Build training command
    train_script = f"""
import sys
sys.path.insert(0, '{BASE_DIR}')
from ultralytics import YOLO
model = YOLO('{model_arch}')
model.train(
    data='{yaml_path}',
    epochs={epochs},
    batch={batch},
    imgsz={imgsz},
    patience={patience},
    lr0={lr0},
    project='{RUNS_DIR}',
    name='{run_name}',
    exist_ok=True,
    verbose=True,
    augment={augment},
    device=0,
    workers=4,
    plots=True,
)
"""
    script_path = f"/tmp/train_{run_name}.py"
    with open(script_path, "w") as f:
        f.write(train_script)

    # Start subprocess
    job_id = str(uuid.uuid4())[:8]
    run_dir = str(RUNS_DIR / run_name)
    os.makedirs(run_dir, exist_ok=True)

    proc = subprocess.Popen(
        [PYTHON_BIN, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(BASE_DIR),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    )

    job = {
        "job_id": job_id,
        "status": "running",
        "pid": proc.pid,
        "process": proc,
        "run_name": run_name,
        "run_dir": run_dir,
        "config": config,
        "total_epochs": epochs,
        "current_epoch": 0,
        "progress_pct": 0,
        "latest_metrics": {},
        "metrics_history": [],
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "result_images": [],
    }
    _active_jobs[job_id] = job

    # Start monitoring thread
    t = threading.Thread(target=_monitor_training, args=(job_id,), daemon=True)
    t.start()

    return {
        "job_id": job_id,
        "run_name": run_name,
        "run_dir": run_dir,
        "pid": proc.pid,
        "status": "running",
    }


def get_training_status(job_id: str) -> dict | None:
    """Get status of a training job."""
    job = _active_jobs.get(job_id)
    if not job:
        return None

    # Sanitize for JSON (remove process object)
    safe = {k: v for k, v in job.items() if k != "process"}
    return safe


def stop_training(job_id: str) -> dict:
    """Stop a running training job."""
    job = _active_jobs.get(job_id)
    if not job:
        return {"error": "Job not found"}

    proc = job.get("process")
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGINT)  # graceful stop
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        job["status"] = "stopped"
        job["finished_at"] = datetime.now().isoformat()
        return {"status": "stopped", "job_id": job_id}

    return {"status": job.get("status", "unknown"), "job_id": job_id}


def list_active_jobs() -> list[dict]:
    """List all active/recent training jobs."""
    result = []
    for jid, job in _active_jobs.items():
        safe = {k: v for k, v in job.items() if k != "process"}
        result.append(safe)
    return result
