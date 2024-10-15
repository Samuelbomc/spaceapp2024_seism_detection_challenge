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
nperseg = 256

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
    os.makedirs(output_figures_dir)

if os.path.exists(output_catalog_file):
    os.remove(output_catalog_file)

mseed_files = glob.glob(os.path.join(mseed_directory, '*.mseed'))

header_written = False

for mseed_file in mseed_files:
    full_path = os.path.join(mseed_directory, mseed_file)
    evid = mseed_file.split('_').pop().split('.').pop(0)

    try:
        st = read(mseed_file)
        print(f'Analyzing file: {os.path.basename(mseed_file)}')

        st[0].stats
        fname = os.path.basename(mseed_file)

        tr = st.traces[0].copy()
        tr_times = tr.times()
        tr_data = tr.data

        starttime = tr.stats.starttime.datetime

        if tr.stats.channel.endswith('H'):
            label_y = 'Amplitude (Filt. Cts./s)'
            label_cbar = 'Power ((Filt. Cts./s)^2/sqrt(Hz))'
        else:
            label_y = 'Velocity (m/s)'
            label_cbar = 'Power ((m/s)^2/sqrt(Hz))'

        st_filt = st.copy()
        st_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
        tr_filt = st_filt.traces[0].copy()
        tr_times_filt = tr_filt.times()
        tr_data_filt = tr_filt.data

        f, t, sxx = signal.spectrogram(tr_data, tr.stats.sampling_rate, nperseg=nperseg)

        if len(tr_data_filt) >= (lta_len * tr_filt.stats.sampling_rate):
            cft = classic_sta_lta(tr_data_filt, int(sta_len * tr_filt.stats.sampling_rate), int(lta_len * tr_filt.stats.sampling_rate))
            on_off = np.array(trigger_onset(cft, thr_on, thr_off))

            valid_triggers = []

            power_filt = tr_data_filt ** 2

            for triggers in on_off:
                if any(power_filt[triggers[0] + i] > 0.5*a for i in range(-200, 201)):
                    valid_triggers.append(triggers)

            valid_triggers = np.array(valid_triggers)
            
        else:
            print(f"Skipping STA/LTA for {fname}: Not enough data points.")
            continue

        fig, axs = plt.subplots(2, 3, figsize=(20, 10))

        ax_v = axs[0, 0]
        ax_v.plot(tr_times, tr_data, label='Seismogram Data')
        for triggers in valid_triggers:
            ax_v.axvline(x=tr_times[triggers[0]], color='red', label='Detection' if triggers[0] == valid_triggers[0][0] else "")
            ax_v.axvline(x=tr_times[triggers[1]], color='purple', label='Trig. Off' if triggers[1] == valid_triggers[0][1] else "")
        ax_v.set_xlim([min(tr_times), max(tr_times)])
        ax_v.set_ylabel(label_y)
        ax_v.set_xlabel('Time (s)')
        ax_v.set_title('Trace', fontweight='bold')
        ax_v.legend()

        ax_u = axs[0, 1]
        ax_u.plot(tr_times, tr_data_filt, label='Seismogram Data (Filtered)', color='orange')
        for triggers in valid_triggers:
            ax_u.axvline(x=tr_times[triggers[0]], color='red', label='Detection' if triggers[0] == valid_triggers[0][0] else "")
            ax_u.axvline(x=tr_times[triggers[1]], color='purple', label='Trig. Off' if triggers[1] == valid_triggers[0][1] else "")
        ax_u.set_xlim([min(tr_times), max(tr_times)])
        ax_u.set_ylabel(label_y)
        ax_u.set_xlabel('Time (s)')
        ax_u.set_title('Trace (Filtered)', fontweight='bold')
        ax_u.legend()

        tr_data_filt_positive = tr_data_filt - np.min(tr_data_filt) + normalize_value
        ax_w = axs[0, 2]
        ax_w.plot(tr_times_filt, tr_data_filt_positive, label='Seismogram Data', color='green')
        ax_w.set_yscale('log')
        ax_w.set_xlim([min(tr_times), max(tr_times)])
        ax_w.set_ylabel(label_y)
        ax_w.set_xlabel('Time (s)')
        ax_w.set_title('Trace (Log Scale)', fontweight='bold')
        ax_w.legend()

        ax2_h = axs[1, 0]
        im = ax2_h.pcolormesh(t, f, 10 * np.log10(sxx), shading='gouraud', cmap=cm.viridis)
        ax2_h.set_ylabel('Frequency (Hz)')
        ax2_h.set_xlabel('Time (s)')
        ax2_h.set_title('Spectrogram', fontweight='bold')
        fig.colorbar(im, ax=ax2_h, label='Intensity (dB)')

        ax2_sta_lta = axs[1, 1]
        ax2_sta_lta.plot(tr_times_filt, cft, color='blue', label='STA/LTA')
        ax2_sta_lta.axhline(thr_on, color='red', linestyle='--', label='Threshold On')
        ax2_sta_lta.axhline(thr_off, color='orange', linestyle='--', label='Threshold Off')
        ax2_sta_lta.set_xlim([min(tr_times), max(tr_times)])
        ax2_sta_lta.set_ylim([-0.1, 10])
        ax2_sta_lta.set_ylabel('STA/LTA')
        ax2_sta_lta.set_xlabel('Time (s)')
        ax2_sta_lta.set_title('STA/LTA', fontweight='bold')
        ax2_sta_lta.legend()

        ax2 = axs[1, 2]
        vals2 = ax2.pcolormesh(t, f, sxx, cmap=cm.jet, shading='auto', vmax=a)
        ax2.set_xlim([min(t), max(t)])
        if np.all(f > 0):
            ax2.set_yscale('log')
        ax2.set_ylabel('Freq (Hz)')
        ax2.set_xlabel('Time (s)')
        ax2.set_title('Spectrogram', fontweight='bold')
        fig.colorbar(vals2, ax=ax2, label=label_cbar)


        output_file_path = os.path.join(output_figures_dir, fname + '_trace.png')
        fig.suptitle(f'{evid},  Start: {starttime}', fontweight='bold', fontsize=20)
        plt.tight_layout()
        plt.savefig(output_file_path, bbox_inches='tight')
        plt.close()

        detections = []
        if valid_triggers is not None:
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

            for detection in detections:
                if header_written:
                    detection.to_csv(output_catalog_file, mode='a', header=False, index=False)
                else:
                    detection.to_csv(output_catalog_file, mode='w', header=True, index=False)
                    header_written = True

    except Exception as e:
        print(f"Skipping {fname}: {e}")

