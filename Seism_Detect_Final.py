from obspy.signal.invsim import cosine_taper 
from obspy.signal.filter import highpass
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import numpy as np
import pandas as pd
import datetime
from obspy import read
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import cm
import glob
import os
# This program detects seisms present on mseed files
start = False
while not start:
    body = input("The data is from which celestial body? Type 'Moon' or 'Mars': ")
    
    if body == "Mars" or body == "Moon":
        pack = input("Which data set? Type 'Training' or 'Test': ")
        
        if body == "Mars":
            minfreq = 0.1
            maxfreq = 9.9
            sta_len = 120
            lta_len = 500
            thr_on = 3
            thr_off = 0.5
            a = 10000
            normalize_value = 1e-5 
            output_catalog_file = './output/catalog_mars.csv'

            if pack == "Test":
                mseed_directory = './data/mars/test/data/'
                start = True
            elif pack == "Training":
                mseed_directory = './data/mars/training/data/'
                start = True
            else:
                print("Invalid data set. Please choose 'Training' or 'Test'.")
                continue

        elif body == "Moon":
            minfreq = 0.1
            maxfreq = 3.2
            sta_len = 500
            lta_len = 10000
            thr_on = 4
            thr_off = 0.5
            normalize_value = 1e-9
            a = 3e-18
            output_catalog_file = './output/catalog_moon.csv'

            if pack == "Training":
                mseed_directory = './data/lunar/training/data/S12_GradeA/'
                start = True
            elif pack == "Test":
                package = input("Which package? Type 'S12_GradeB', 'S15_GradeA', 'S15_GradeB', 'S16_GradeA' or 'S16_GradeB': ")
                valid_packages = ['S12_GradeB', 'S15_GradeA', 'S15_GradeB', 'S16_GradeA', 'S16_GradeB']
                if package in valid_packages:
                    mseed_directory = f'./data/lunar/training/data/{package}/'
                    start = True
                else:
                    print(f"Invalid package. Please choose one of the following: {', '.join(valid_packages)}")
                    continue
            else:
                print("Invalid data set. Please choose 'Training' or 'Test'.")
                continue

    else:
        print("Invalid celestial body. Please choose 'Moon' or 'Mars'.")

output_figures_dir = f'./output/figures/{body}/'

if not os.path.exists(output_figures_dir):
    os.makedirs(output_figures_dir)  # Create the directory if it doesn't exist

if os.path.exists(output_catalog_file):
    os.remove(output_catalog_file)

# Get all .mseed files in the directory
mseed_files = glob.glob(os.path.join(mseed_directory, '*.mseed'))

# Flag to control header writing
header_written = False

for mseed_file in mseed_files:
    full_path = os.path.join(mseed_directory, mseed_file)
    evid = mseed_file.split('_').pop().split('.').pop(0)

    try:
        st = read(mseed_file)
        print(f'Analyzing file: {os.path.basename(mseed_file)}')

        st[0].stats
        fname = os.path.basename(mseed_file)  # Get only the filename

        # Since there is only one trace, we can directly use it
        tr = st.traces[0].copy()
        tr_times = tr.times()
        tr_data = tr.data

        starttime = tr.stats.starttime.datetime

        # Determine data type based on the channel name
        if tr.stats.channel.endswith('H'):
            label_y = 'Amplitude (Filt. Cts./s)'
            label_cbar = 'Power ((Filt. Cts./s)^2/sqrt(Hz))'
        else:
            label_y = 'Velocity (m/s)'
            label_cbar = 'Power ((m/s)^2/sqrt(Hz))'

        # Filter and compute spectrogram
        st_filt = st.copy()
        st_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
        tr_filt = st_filt.traces[0].copy()
        tr_times_filt = tr_filt.times()
        tr_data_filt = tr_filt.data

        # Set the density of the readings
        nperseg = 128
        # Compute the spectrogram for the original trace
        f, t, sxx = signal.spectrogram(tr_data, tr.stats.sampling_rate, nperseg=nperseg)  # Use original data

        # Check if the length of the filtered trace is sufficient for STA/LTA
        if len(tr_data_filt) >= (lta_len * tr_filt.stats.sampling_rate):
            # Run classic STA/LTA
            cft = classic_sta_lta(tr_data_filt, int(sta_len * tr_filt.stats.sampling_rate), int(lta_len * tr_filt.stats.sampling_rate))
            on_off = np.array(trigger_onset(cft, thr_on, thr_off))

            # Initialize valid trigger list
            valid_triggers = []

            # Calculate the power of the filtered trace
            power_filt = tr_data_filt ** 2  # Power is the square of the amplitude

            # Iterate through detection times and compile valid triggers
            for triggers in on_off:
                # Check if the power at the detection time is greater than half of 'a'
                if any(power_filt[triggers[0] + i] > 0.5*a for i in range(-200, 201)):
                    valid_triggers.append(triggers)

            # Convert valid_triggers to a NumPy array for easier indexing
            valid_triggers = np.array(valid_triggers)
            
        else:
            print(f"Skipping STA/LTA for {fname}: Not enough data points.")
            continue  # Skip to the next file


        # Create a 2x3 grid for 6 total plots
        fig, axs = plt.subplots(2, 3, figsize=(20, 10))

        # Plotting the original Trace graph
        ax_bhv = axs[0, 0]
        ax_bhv.plot(tr_times, tr_data, label='Seismogram Data')
        for triggers in valid_triggers:
            ax_bhv.axvline(x=tr_times[triggers[0]], color='red', label='Detection' if triggers[0] == valid_triggers[0][0] else "")
            ax_bhv.axvline(x=tr_times[triggers[1]], color='purple', label='Trig. Off' if triggers[1] == valid_triggers[0][1] else "")
        ax_bhv.set_xlim([min(tr_times), max(tr_times)])  # Fit entire activity
        ax_bhv.set_ylabel(label_y)
        ax_bhv.set_xlabel('Time (s)')
        ax_bhv.set_title('Trace', fontweight='bold')  # Updated title
        ax_bhv.legend()

        # Plotting the Filtered Trace graph
        ax_bhu = axs[0, 1]
        ax_bhu.plot(tr_times, tr_data_filt, label='Seismogram Data (Filtered)', color='orange')  # Using filtered data
        for triggers in valid_triggers:
            ax_bhu.axvline(x=tr_times[triggers[0]], color='red', label='Detection' if triggers[0] == valid_triggers[0][0] else "")
            ax_bhu.axvline(x=tr_times[triggers[1]], color='purple', label='Trig. Off' if triggers[1] == valid_triggers[0][1] else "")
        ax_bhu.set_xlim([min(tr_times), max(tr_times)])  # Fit entire activity
        ax_bhu.set_ylabel(label_y)
        ax_bhu.set_xlabel('Time (s)')
        ax_bhu.set_title('Trace (Filtered)', fontweight='bold')  # Updated title
        ax_bhu.legend()

        # Ensure all values are positive for log scale (using filtered data)
        tr_data_filt_positive = tr_data_filt - np.min(tr_data_filt) + normalize_value
        # Plotting the Log Scale Trace graph
        ax_bhw = axs[0, 2]
        ax_bhw.plot(tr_times_filt, tr_data_filt_positive, label='Seismogram Data', color='green')  # Using filtered data
        ax_bhw.set_yscale('log')  # Set log scale
        ax_bhw.set_xlim([min(tr_times), max(tr_times)])  # Fit entire activity
        ax_bhw.set_ylabel(label_y)
        ax_bhw.set_xlabel('Time (s)')
        ax_bhw.set_title('Trace (Log Scale)', fontweight='bold')  # Updated title
        ax_bhw.legend()

        # Plotting the Spectrogram
        ax_h = axs[1, 0]
        im = ax_h.pcolormesh(t, f, 10 * np.log10(sxx), shading='gouraud', cmap=cm.viridis)
        ax_h.set_ylabel('Frequency (Hz)')
        ax_h.set_xlabel('Time (s)')
        ax_h.set_title('Spectrogram', fontweight='bold')  # Updated title
        fig.colorbar(im, ax=ax_h, label='Intensity (dB)')

        # Plotting STA/LTA
        ax_sta_lta = axs[1, 1]
        ax_sta_lta.plot(tr_times_filt, cft, color='blue', label='STA/LTA')
        ax_sta_lta.axhline(thr_on, color='red', linestyle='--', label='Threshold On')
        ax_sta_lta.axhline(thr_off, color='orange', linestyle='--', label='Threshold Off')
        ax_sta_lta.set_xlim([min(tr_times), max(tr_times)])  # Fit entire activity
        ax_sta_lta.set_ylim([-0.1, 3.5])
        ax_sta_lta.set_ylabel('STA/LTA')
        ax_sta_lta.set_xlabel('Time (s)')
        ax_sta_lta.set_title('STA/LTA', fontweight='bold')  # Updated title
        ax_sta_lta.legend()

        f_bhu, t_bhu, sxx_bhu = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate, nperseg=nperseg)  # Using filtered data
        ax2_bhu = axs[1, 2]
        vals_bhu = ax2_bhu.pcolormesh(t_bhu, f_bhu, sxx_bhu, cmap=cm.jet, vmax = a)  # Using filtered data
        ax2_bhu.set_xlim([min(tr_times_filt), max(tr_times_filt)])
        ax2_bhu.set_xlabel('Time (s)', fontweight='bold')
        ax2_bhu.set_ylabel('Frequency (Hz)', fontweight='bold')
        ax2_bhu.set_title('Spectrogram (Filtered)', fontweight='bold')  # Added title
        for triggers in valid_triggers:
            ax2_bhu.axvline(x=tr_times[triggers[0]], color='red')
            ax2_bhu.axvline(x=tr_times[triggers[1]], color='purple')
        cbar_bhu = plt.colorbar(vals_bhu, ax=ax2_bhu, orientation='horizontal')
        cbar_bhu.set_label(label_cbar, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_figures_dir, f'{evid}_spectrogram.png'))  # Save the figure
        plt.close(fig)
        detections = []
        if valid_triggers is not None:  # Only append if a mean trigger time is available
            for triggers in valid_triggers:
                event_time = starttime + datetime.timedelta(seconds=tr_times[triggers[0]])
                detection_data = {
                    'filename': [fname],
                    'time_abs(%Y-%m-%dT%H:%M:%S.%f)': [event_time.strftime('%Y-%m-%dT%H:%M:%S.%f')],
                    'time_rel(sec)': [tr_times[triggers[0]]],
                    'evid': [evid]
                }
                detection_df = pd.DataFrame(detection_data)
                detections.append(detection_df)

            # Append to catalog file
            for detection in detections:
                if header_written:
                    detection.to_csv(output_catalog_file, mode='a', header=False, index=False)
                else:
                    detection_df.to_csv(output_catalog_file, mode='w', header=True, index=False)
                    header_written = True
    except Exception as e:
        print(f"Error processing file {fname}: {str(e)}")
