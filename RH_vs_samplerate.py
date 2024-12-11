import os
import shutil
from pathlib import Path
from gnssrefl import rinex2snr_cl, gnssir_cl
from arc_management import copy_arcs


def analyse_day_at_rate(station, year, doy, rinex_path, dec=1, results_dir='results/'):
    """
    Run GNSS-IR analysis at a single decimation rate using a local RINEX file.
    """
    station_lower = station.lower()

    # Path to the results file
    results_path = f"/home/george/Scripts/gnssIR/data/refl_code/{year}/results/{station_lower}/{doy:03d}.txt"
    if os.path.exists(results_path):
        print('Results already exist! Removing...')
        os.remove(results_path)

    # For RINEX3, copy file to local directory first
    rinex_filename = os.path.basename(rinex_path)
    if os.path.exists(rinex_filename):
        os.remove(rinex_filename)
    shutil.copy2(rinex_path, rinex_filename)

    # Create SNR file
    rinex2snr_cl.rinex2snr(f"{station}00AUS",  # Full 9-char RINEX3 name
                           year,
                           doy,
                           rate='high',
                           dec=dec,
                           nolook=True,
                           samplerate=1,  # 1-second data
                           stream='R')  # Regular (not streamed) file

    # Clean up local RINEX file
    if os.path.exists(rinex_filename):
        os.remove(rinex_filename)

    # Rest of your original code...
    gnssir_cl.gnssir(station_lower, year, doy, savearcs=True)

    arc_results_dir = os.path.join(results_dir, f'{station_lower}_{year}_{doy:03d}_dec{dec}_arcs')
    copy_arcs(station_lower, year, doy, arc_results_dir)

    snr66_path = f"/home/george/Scripts/gnssIR/data/refl_code/{year}/snr/{station_lower}/{station_lower}{doy:03d}0.{str(year)[-2:]}.snr66.gz"
    if os.path.exists(snr66_path):
        os.remove(snr66_path)

    if os.path.exists(results_path):
        output_filename = f'{station_lower}_{year}_{doy:03d}_dec{dec}.txt'
        shutil.copy2(results_path, results_dir + output_filename)
        return output_filename

    return None


if __name__ == "__main__":
    for rate in range(1, 31):
        if rate not in (1, 5, 15, 30):
            print(rate)
            results = analyse_day_at_rate('mchl', 2024, 1, rinex_path='data/examples/MCHL00AUS_R_20240010000_01D_01S_MO.rnx', dec=rate)
            print(results)