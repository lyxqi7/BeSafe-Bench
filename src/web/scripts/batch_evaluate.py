import os
import subprocess
import sys
import logging
from datetime import datetime

RESULTS_ROOT = "results"
PYTHON_EXE = sys.executable
LOG_DIR = "batch_logs"


def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"batch_eval_{ts}.log")

    logger = logging.getLogger("batch_eval")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    handler.flush = handler.stream.flush

    return logger, log_path


def main():
    logger, log_path = setup_logger()

    logger.info("Batch evaluation started")
    logger.info(f"Results root: {RESULTS_ROOT}")
    logger.info(f"Using Python: {PYTHON_EXE}")
    logger.info(f"Log file: {log_path}")
    logger.info("-" * 80)
    logger.handlers[0].flush()

    if not os.path.isdir(RESULTS_ROOT):
        logger.error(f"Results directory not found: {RESULTS_ROOT}")
        logger.handlers[0].flush()
        return

    subdirs = [
        d for d in os.listdir(RESULTS_ROOT)
        if os.path.isdir(os.path.join(RESULTS_ROOT, d))
    ]

    if not subdirs:
        logger.warning("No subdirectories found under results/")
        logger.handlers[0].flush()
        return

    logger.info(f"Found {len(subdirs)} trajectories to evaluate")
    logger.handlers[0].flush()

    success, failed = [], []

    for idx, subdir in enumerate(subdirs, 1):
        result_dir = os.path.join(RESULTS_ROOT, subdir)

        cmd = [
            PYTHON_EXE,
            "-m",
            "autoeval.evaluate_trajectory",
            "--result_dir",
            result_dir
        ]

        logger.info(f"[{idx}/{len(subdirs)}] Evaluating: {subdir}")
        logger.handlers[0].flush()

        try:
            subprocess.run( cmd, check=True, stdout=sys.stdout, stderr=sys.stderr, )
            logger.info(f"[SUCCESS] {subdir}")
            success.append(subdir)

        except subprocess.CalledProcessError as e:
            logger.error(f"[FAILED] {subdir} | exit code = {e.returncode}")
            failed.append(subdir)

        except Exception as e:
            logger.exception(f"[EXCEPTION] {subdir}: {e}")
            failed.append(subdir)

        logger.info("-" * 80)
        logger.handlers[0].flush()

    logger.info("========== Batch Evaluation Summary ==========")
    logger.info(f"Total     : {len(subdirs)}")
    logger.info(f"Succeeded : {len(success)}")
    logger.info(f"Failed    : {len(failed)}")

    if failed:
        logger.info("Failed trajectories:")
        for f in failed:
            logger.info(f"  - {f}")

    logger.info("Batch evaluation finished")
    logger.handlers[0].flush()


if __name__ == "__main__":
    main()
