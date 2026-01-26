import subprocess
import sys
import os


def main():
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    websites = ["shopping", "shopping_admin", "gitlab", "reddit"]

    for website in websites:
        print(f"\n{'#' * 40}")
        print(f"### {website}")
        print(f"{'#' * 40}\n")

        try:
            subprocess.run([
                "python", "-u", "pipeline.py",
                "--website", website
            ], check=True, env=env)


        except subprocess.CalledProcessError as e:
            pass

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()