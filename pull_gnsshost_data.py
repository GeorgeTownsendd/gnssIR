#!/usr/bin/env python3
import os
import sys
import subprocess
import zipfile
import glob
from pathlib import Path
import shutil


def get_working_directory():
    working_dir = '/home/george/Scripts/gnssIR/field_tests/field_test_2/'
    if not os.path.isdir(working_dir):
        print(f"Error: Directory {working_dir} does not exist")
        sys.exit(1)
    return working_dir


def sync_host(host_num, password):
    host_dir = f"gnss-host{host_num}"
    os.makedirs(host_dir, exist_ok=True)

    print(f"\nTrying {host_dir}...")
    try:
        cmd = [
            "sshpass", "-p", password,
            "rsync", "-avz",
            "-e", "ssh -o ConnectTimeout=30",
            f"gnss@gnss-host{host_num}:~/rtkbase/data/",
            f"{host_dir}/"
        ]

        process = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if process.returncode == 0:
            print(f"Success: {host_dir}")
            return True
        else:
            print(f"Failed: {host_dir}")
            print(f"Error: {process.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"Timeout: {host_dir} is not responding")
        return False
    except KeyboardInterrupt:
        print(f"\nSkipping {host_dir}")
        return False


def extract_zips(directory):
    print(f"\nExtracting zip files in {directory}")
    zip_files = glob.glob(os.path.join(directory, "**/*.zip"), recursive=True)
    if zip_files:
        print(f"Found {len(zip_files)} zip files to extract")
        for zip_path in zip_files:
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(zip_path))
                    os.remove(zip_path)
            except Exception as e:
                print(f"Error extracting {os.path.basename(zip_path)}: {e}")


def organize_files(host_dir):
    print(f"\nOrganizing files in {host_dir}")

    junk_dir = os.path.join(host_dir, "junk")
    metadata_dir = os.path.join(host_dir, "metadata")
    os.makedirs(junk_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    small_files_count = 0
    tag_files_count = 0

    for root, _, files in os.walk(host_dir):
        if root in [junk_dir, metadata_dir]:
            continue

        for file in files:
            file_path = os.path.join(root, file)

            try:
                if os.path.getsize(file_path) < 512:
                    shutil.move(file_path, os.path.join(junk_dir, file))
                    small_files_count += 1
                elif file.endswith('.ubx.tag'):
                    shutil.move(file_path, os.path.join(metadata_dir, file))
                    tag_files_count += 1
            except Exception as e:
                print(f"Error processing {file}: {e}")

    print(f"Moved {small_files_count} small files to junk")
    print(f"Moved {tag_files_count} tag files to metadata")


def sync_all_hosts():
    working_dir = get_working_directory()
    os.chdir(working_dir)
    print(f"Changed to directory: {working_dir}")

    password = input("Enter SSH password for gnss hosts: ")
    successful_hosts = []
    for host_num in range(1, 5):
        if sync_host(host_num, password):
            successful_hosts.append(host_num)
    for host_num in successful_hosts:
        host_dir = os.path.join(working_dir, f"gnss-host{host_num}")
        extract_zips(host_dir)
        organize_files(host_dir)
    return successful_hosts


if __name__ == "__main__":
    sync_all_hosts()