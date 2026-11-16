import json
import time
import urllib.request
import urllib.error
import subprocess
import sys
from pathlib import Path

python_exe = Path(r"d:\HOC_2026_2027\XLNNTN\PHOBERT_SMARTPHONE\phobert-absa-local\.venv\Scripts\python.exe")
backend_dir = Path(r"d:\HOC_2026_2027\XLNNTN\PHOBERT_SMARTPHONE\phobert-absa-local")

def run_live_test():
    print("Starting uvicorn server...")
    cmd = [
        str(python_exe),
        "-m", "uvicorn",
        "backend.app:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(backend_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Chờ server khởi động
        time.sleep(3)
        base_url = "http://127.0.0.1:8000"

        # 1. Test GET /api/health
        print("\n--- Testing HTTP GET /api/health ---")
        req = urllib.request.Request(f"{base_url}/api/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = resp.status
            body = json.loads(resp.read().decode("utf-8"))
            print(f"Status: {status}, Body: {body}")
            assert status == 200
            assert body["status"] == "ok"

        # 2. Test GET /api/aspects
        print("\n--- Testing HTTP GET /api/aspects ---")
        req = urllib.request.Request(f"{base_url}/api/aspects")
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = resp.status
            body = json.loads(resp.read().decode("utf-8"))
            print(f"Status: {status}, Body: {body}")
            assert status == 200
            assert len(body["aspects"]) == 10

        # 3. Test POST /api/analyze
        print("\n--- Testing HTTP POST /api/analyze ---")
        payload = json.dumps({
            "comment": "Camera chụp ảnh rất đẹp nhưng pin nhanh hết.",
            "aspects": ["CAMERA", "BATTERY"]
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/api/analyze",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.status
            body = json.loads(resp.read().decode("utf-8"))
            print(f"Status: {status}")
            print("Response:", json.dumps(body, indent=2, ensure_ascii=False))
            assert status == 200
            assert len(body["results"]) == 2

        # 4. Test POST /api/analyze với lỗi 422
        print("\n--- Testing HTTP POST /api/analyze 422 Validation Error ---")
        payload_err = json.dumps({
            "comment": "   ",
            "aspects": ["CAMERA"]
        }).encode("utf-8")
        req_err = urllib.request.Request(
            f"{base_url}/api/analyze",
            data=payload_err,
            headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(req_err, timeout=5)
            assert False, "Should have returned 422"
        except urllib.error.HTTPError as e:
            print(f"Expected HTTP Error: {e.code}, Detail: {e.read().decode('utf-8')}")
            assert e.code == 422

        print("\n>>> LIVE SERVER HTTP TESTS PASSED! <<<")

    finally:
        print("\nStopping uvicorn server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

if __name__ == "__main__":
    run_live_test()
