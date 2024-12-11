import os
import shutil
from pathlib import Path



def copy_arcs(station, year, doy, dest_dir):
    """
    Copy GNSS-IR arc files to destination directory.

    Args:
        station (str): Station name
        year (int): Year
        doy (int): Day of year
        dest_dir (str): Destination directory for arc files

    Returns:
        bool: True if files were successfully copied, False otherwise
    """
    refl_code = os.environ.get('REFL_CODE')
    if refl_code is None:
        print('Warning: REFL_CODE environment variable not set, cannot copy arc files')
        return False

    # Source arc directory (following GNSS-IR convention)
    arc_dir = Path(refl_code) / str(year) / 'arcs' / station / f'{doy:03d}'

    if not arc_dir.exists():
        print(f'Warning: No arc directory found at {arc_dir}')
        return False

    # Create destination directory
    os.makedirs(dest_dir, exist_ok=True)

    # Copy all arc files from main directory
    files_copied = False
    for arc_file in arc_dir.glob('sat*.txt'):
        shutil.copy2(arc_file, dest_dir)
        files_copied = True

    # Check for and copy failed QC files if they exist
    failqc_dir = arc_dir / 'failQC'
    if failqc_dir.exists():
        failqc_dest_dir = os.path.join(dest_dir, 'failQC')
        os.makedirs(failqc_dest_dir, exist_ok=True)
        for arc_file in failqc_dir.glob('sat*.txt'):
            shutil.copy2(arc_file, failqc_dest_dir)
            files_copied = True

    return files_copied