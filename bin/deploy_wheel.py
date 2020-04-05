import subprocess
import glob
import tempfile

BUCKET_URI = "s3://s3.boppypi.registry"


def print_output(stdout, stderr):
    output = (stdout + b"\n" + stderr).decode("UTF-8")
    for line in output.split("\n"):
        print(line)


def execute(cmd):
    print(f"> Executing: {cmd}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print_output(stdout, stderr)


def create_package():
    execute(["python", "setup.py", "sdist"])


def get_package_path(output_dir):
    return glob.glob(f"{output_dir}/*")[0]


def update_index_file(package_name):
    temp_file = f"{tempfile.mkdtemp()}/index.html"
    print(f"> Creating template file [{temp_file}] for package [{package_name}]")
    with open(temp_file, "w") as fl:
        fl.write(
            f"<html><body><a href=\"{package_name}\">{package_name}</a></body></html>"
        )

    execute(["aws", "s3", "cp", temp_file, f"{BUCKET_URI}/index.html"])


def upload_package(package_path, s3_filepath_key):
    execute(["aws", "s3", "cp", package_path, f"{BUCKET_URI}/{s3_filepath_key}"])


if __name__ == "__main__":
    output_dir = "dist"
    print("> Starting pypi deployment process")
    create_package()
    package_path = get_package_path(output_dir=output_dir)
    package_name = package_path.replace(f"{output_dir}/", "")
    update_index_file(package_name=package_name)
    upload_package(package_path=package_path, s3_filepath_key=package_name)
