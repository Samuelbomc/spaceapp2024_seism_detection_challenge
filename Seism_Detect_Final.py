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

start = False
while start is False:
    body = input("The data is from which celestial body? Type Moon or Mars")
    if body == "Mars":
        start = True
        minfreq = 0.1
        maxfreq = 9.9
        sta_len = 120
        lta_len = 500
        thr_on = 3
        thr_off = 0.5
        mseed_directory = './data/mars/test/data/'

    elif body == "Moon":
        start = True
        minfreq = 0.1
        maxfreq = 3
        sta_len = 120
        lta_len = 600
        thr_on = 3.4
        thr_off = 0.5
        mseed_directory = './data/lunar/training/data/S12_GradeA/'

    else:
        print("Please try again, celestial body not recognized")
output_figures_dir = './output/figures/'
output_catalog_file = './output/catalog.csv'

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

        # Compute the spectrogram for the original trace
        nperseg = 128
        f, t, sxx = signal.spectrogram(tr_data, tr.stats.sampling_rate, nperseg=nperseg)  # Use original data

        # Run classic STA/LTA
        cft = classic_sta_lta(tr_data_filt, int(sta_len * tr_filt.stats.sampling_rate), int(lta_len * tr_filt.stats.sampling_rate))
        on_off = np.array(trigger_onset(cft, thr_on, thr_off))

        # Initialize valid trigger list
        valid_triggers = []

        # Iterate through detection times and compile valid triggers
        for triggers in on_off:
            valid_triggers.append(triggers)

        # Convert valid_triggers to a NumPy array for easier indexing
        valid_triggers = np.array(valid_triggers)

        # Calculate the mean position of triggers
        if valid_triggers.size > 0:  # Use .size to check if there are any triggers
            trigger_times = tr_times[valid_triggers[:, 0].astype(int)]  # Ensure integer indexing
            mean_trigger_time = np.mean(trigger_times)
        else:
            print('No valid triggers found.')
            mean_trigger_time = None  # Ensure this is defined

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
        tr_data_filt_positive = tr_data_filt - np.min(tr_data_filt) + 1e-5 
        # Plotting the Log Scale Trace graph
        ax_bhw = axs[0, 2]
        ax_bhw.plot(tr_times_filt, tr_data_filt_positive, label='Seismogram Data', color='green')  # Using filtered data
        ax_bhw.set_yscale('log') 
        for triggers in valid_triggers:
            ax_bhw.axvline(x=tr_times[triggers[0]], color='red', label='Detection' if triggers[0] == valid_triggers[0][0] else "")
            ax_bhw.axvline(x=tr_times[triggers[1]], color='purple', label='Trig. Off' if triggers[1] == valid_triggers[0][1] else "")
        ax_bhw.set_xlim([min(tr_times), max(tr_times)])  # Fit entire activity
        ax_bhw.set_ylabel(f'Log {label_y}')
        ax_bhw.set_xlabel('Time (s)')
        ax_bhw.set_title('Trace (Log Scale)', fontweight='bold')  # Updated title
        ax_bhw.legend()

        # Spectrogram 1 (original)
        ax2_bhv = axs[1, 0]
        vals_bhv = ax2_bhv.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=10000)  # Using original data
        ax2_bhv.set_xlim([min(tr_times), max(tr_times)])  # Use original trace time
        ax2_bhv.set_xlabel('Time (s)', fontweight='bold')
        ax2_bhv.set_ylabel('Frequency (Hz)', fontweight='bold')
        ax2_bhv.set_title('Spectrogram', fontweight='bold')  # Added title
        for triggers in valid_triggers:
            ax2_bhv.axvline(x=tr_times[triggers[0]], color='red')
            ax2_bhv.axvline(x=tr_times[triggers[1]], color='purple')
        cbar_bhv = plt.colorbar(vals_bhv, ax=ax2_bhv, orientation='horizontal')
        cbar_bhv.set_label(label_cbar, fontweight='bold')

        # Spectrogram 2 (using filtered trace)
        f_bhu, t_bhu, sxx_bhu = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate, nperseg=nperseg)  # Using filtered data
        ax2_bhu = axs[1, 1]
        vals_bhu = ax2_bhu.pcolormesh(t_bhu, f_bhu, sxx_bhu, cmap=cm.jet, vmax=10000)  # Using filtered data
        ax2_bhu.set_xlim([min(tr_times_filt), max(tr_times_filt)])
        ax2_bhu.set_xlabel('Time (s)', fontweight='bold')
        ax2_bhu.set_ylabel('Frequency (Hz)', fontweight='bold')
        ax2_bhu.set_title('Spectrogram (Filtered)', fontweight='bold')  # Added title
        for triggers in valid_triggers:
            ax2_bhu.axvline(x=tr_times[triggers[0]], color='red')
            ax2_bhu.axvline(x=tr_times[triggers[1]], color='purple')
        cbar_bhu = plt.colorbar(vals_bhu, ax=ax2_bhu, orientation='horizontal')
        cbar_bhu.set_label(label_cbar, fontweight='bold')

        # Spectrogram 3 (Log scale)
        ax2_bhw = axs[1, 2]
        sxx_bhw_log = np.log(sxx + 1e-10)  # Avoid log(0)
        vals_bhw = ax2_bhw.pcolormesh(t, f, sxx_bhu, cmap=cm.jet, vmax=10000)  # Using log scale
        ax2_bhw.set_xlim([min(tr_times), max(tr_times)])  # Use original trace time
        ax2_bhw.set_xlabel('Time (s)', fontweight='bold')
        ax2_bhw.set_ylabel('Frequency (Hz)', fontweight='bold')
        ax2_bhw.set_title('Spectrogram (Log Scale)', fontweight='bold')  # Added title
        for triggers in valid_triggers:
            ax2_bhw.axvline(x=tr_times[triggers[0]], color='red')
            ax2_bhw.axvline(x=tr_times[triggers[1]], color='purple')
        cbar_bhw = plt.colorbar(vals_bhw, ax=ax2_bhw, orientation='horizontal')
        cbar_bhw.set_label('Log Power', fontweight='bold')

        plt.tight_layout()

        # Save the plots to a file
        plt.savefig(os.path.join(output_figures_dir, f'{evid}_spectrogram.png'), dpi=300)
        plt.close()
        detections = []
        if mean_trigger_time is not None:  # Only append if a mean trigger time is available
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
        print(f'Error processing file {full_path}: {e}')
