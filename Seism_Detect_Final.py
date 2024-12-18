from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy.ndimage import gaussian_filter1d
import numpy as np
from obspy import read
from scipy import signal
from matplotlib import cm
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import glob
import os
import noisereduce as nr

start = False
nperseg = 256
plots = input("Create plots? Type 'Y' or 'N': ")
while not start:
    body = input("The data is from which celestial body? Type 'Moon' or 'Mars': ")
    
    if body == "Mars" or body == "Moon":
        pack = input("Which data set? Type 'Training' or 'Test': ")
        
        if body == "Mars":
            minfreq = 0.1
            maxfreq = 9.9
            sta_len = 60
            lta_len = 250
            thr_on = 1.5
            thr_off = 1
            a = 10000
            spl = 50000
            normalize_value = 1e-5
            noise_mean = 90
            noise_std = 18
            min_trigger_difference = 100
            n_fft=2048
            win_length=1024
            hop_length=512
            n_std_thresh_stationary=2
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
            sta_len = 100
            lta_len = 1000
            thr_on = 1.5
            thr_off = 1
            normalize_value = 1e-9
            spl = 50000
            a = 3e-18
            noise_mean = 5e-10
            noise_std = 1e-10
            min_trigger_difference = 3000
            n_fft=2048,
            win_length=1024,
            hop_length=512,
            n_std_thresh_stationary=2
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

        st[0].stats
        fname = os.path.basename(mseed_file)
           
        tr = st.traces[0].copy()
        tr_times = tr.times()
        tr_data = tr.data
        starttime = tr.stats.starttime.datetime
        st_filt = st.copy()
        st_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
        tr_filt = st_filt.traces[0].copy()
        tr_times_filt = tr_filt.times()
        tr_data_filt = tr_filt.data
        tr_data_filt = gaussian_filter1d(tr_data_filt, sigma=1)
        tr_data_filt = nr.reduce_noise(
            y = tr_data_filt, 
            sr = spl,
            n_fft = n_fft,
            win_length = win_length,
            hop_length = hop_length,
            n_std_thresh_stationary = n_std_thresh_stationary
        )
        window_size = 100
        std_threshold = 2 

        adjusted_data = tr_data_filt.copy()

        i = window_size
        while i < len(tr_data_filt) - window_size:

            local_window = np.concatenate((tr_data_filt[i - window_size:i], tr_data_filt[i + 1:i + window_size + 1]))
            local_mean = np.mean(local_window)
            local_std = np.std(local_window)

            if abs(tr_data_filt[i] - local_mean) > std_threshold * local_std:

                start = i
                while i < len(tr_data_filt) - window_size and abs(tr_data_filt[i] - local_mean) > std_threshold * local_std:
                    i += 1
                end = i  

                adjusted_data[start:end] = (tr_data_filt[start - 1] + tr_data_filt[end]) / 2
            else:
                i += 1

        tr_data_filt = adjusted_data

        noise = np.random.normal(noise_mean, noise_std, size=tr_data_filt.shape)
        tr_data_filt += noise

        
        tr_filt.data = tr_data_filt
        if tr.stats.channel.endswith('H'):
            label_y = 'Amplitude (Filt. Cts./s)'
            label_cbar = 'Power ((Filt. Cts./s)^2/sqrt(Hz))'
        else:
            label_y = 'Velocity (m/s)'
            label_cbar = 'Power ((m/s)^2/sqrt(Hz))'

        f, t, sxx = signal.spectrogram(tr_data_filt, tr.stats.sampling_rate, nperseg=nperseg)


        if len(tr_data_filt) >= (lta_len * tr_filt.stats.sampling_rate):
            cft = classic_sta_lta(tr_data_filt, int(sta_len * tr_filt.stats.sampling_rate), int(lta_len * tr_filt.stats.sampling_rate))
            on_off = np.array(trigger_onset(cft, thr_on, thr_off))

            valid_triggers = []

            power = tr_data ** 2

            for triggers in on_off:
                start_idx, end_idx = triggers
                num = 0
                for i in range (0, 200):
                    if power[triggers[0] + i] > 0.8*a:
                        num += 1
                if num >= 10:
                    if (end_idx - start_idx) >= min_trigger_difference:
                        valid_triggers.append(triggers)

            valid_triggers = np.array(valid_triggers)
            
        else:
            print(f"Skipping STA/LTA for {fname}: Not enough data points.")
            continue

        if plots == "Y" or plots == "y":
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
