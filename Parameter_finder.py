from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy.ndimage import gaussian_filter1d
import numpy as np
from obspy import read
from scipy import signal
from matplotlib import cm
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import glob
import os
import random
import noisereduce as nr

lowest_difference = 1000000000
randomness = 0
nperseg = 258
knowledge = []
difference_value = 0
plots = input("Draw plots? type Y or N: ")

def data_analysis(sta_len, lta_len, thr_on, thr_off, body, mseed_directory, output_catalog_file, minfreq, maxfreq, normalize_value, a, spl):
    detections = []
    output_figures_dir = f'./output/figures/{body}/'
    global plots
    if not os.path.exists(output_figures_dir):
        os.makedirs(output_figures_dir)

    if os.path.exists(output_catalog_file):
        os.remove(output_catalog_file)

    mseed_files = glob.glob(os.path.join(mseed_directory, '*.mseed'))

    for mseed_file in mseed_files:
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
                y=tr_data_filt, 
                sr=spl,
                n_fft=2048,
                win_length=1024,
                hop_length=512,
                n_std_thresh_stationary=2
            )

            if tr.stats.channel.endswith('H'):
                label_y = 'Amplitude (Filt. Cts./s)'
                label_cbar = 'Power ((Filt. Cts./s)^2/sqrt(Hz))'
            else:
                label_y = 'Velocity (m/s)'
                label_cbar = 'Power ((m/s)^2/sqrt(Hz))'

            tr_filt.data = tr_data_filt
            
            f, t, sxx = signal.spectrogram(tr_data_filt, tr.stats.sampling_rate, nperseg=nperseg)

            if len(tr_data_filt) >= (lta_len * tr_filt.stats.sampling_rate):
                cft = classic_sta_lta(tr_data_filt, int(sta_len * tr_filt.stats.sampling_rate), int(lta_len * tr_filt.stats.sampling_rate))
                on_off = np.array(trigger_onset(cft, thr_on, thr_off))

                valid_triggers = []

                power_filt = tr_data_filt ** 2

                for triggers in on_off:
                    if any(power_filt[triggers[0] + i] > 0.5 * a for i in range(-50, 300)):
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

            if valid_triggers is not None:
                for triggers in valid_triggers:
                    event_time = starttime + datetime.timedelta(seconds=tr_times[triggers[0]])
                    if body == "Mars":
                        detection_data = {
                            'filename': [fname],
                            'time_abs(%Y-%m-%dT%H:%M:%S.%f)': [event_time.strftime('%Y-%m-%dT%H:%M:%S.%f')],
                            'time_rel(sec)': [tr_times[triggers[0]]],
                            'evid': [evid],
                        }
                    else:
                        detection_data = {
                            'filename': [fname],
                            'time_abs(%Y-%m-%dT%H:%M:%S.%f)': [event_time.strftime('%Y-%m-%dT%H:%M:%S.%f')],
                            'time_rel(sec)': [tr_times[triggers[0]]],
                            'evid': [evid],
                            'mq_type': "deep_mq" 
                        }
                    detection_df = pd.DataFrame(detection_data)
                    detections.append(detection_df)
            
        except Exception as e:
            print(f"Skipping {fname}: {e}")

    if len(detections) > 1:
        all_detections = pd.concat(detections, ignore_index=True)
        all_detections = all_detections.sort_values(by='evid')
        all_detections.to_csv(output_catalog_file, mode='w', header=True, index=False)
    elif len(detections) > 0:
        all_detections = detections[0]
        all_detections.to_csv(output_catalog_file, mode='w', header=True, index=False)
    else:
        if body == "Mars":
            empty_data = {
                'filename': [],
                'time_abs(%Y-%m-%dT%H:%M:%S.%f)': [],
                'time_rel(sec)': [],
                'evid': []
            }
        else:
            empty_data = {
                'filename': [],
                'time_abs(%Y-%m-%dT%H:%M:%S.%f)': [],
                'time_rel(sec)': [],
                'evid': [],
                'mq_type': []
            }
        empty_df = pd.DataFrame(empty_data)
        empty_df.to_csv(output_catalog_file, mode='w', header=True, index=False)

def save_training_data(sta_len, lta_len, thr_on, body, output_training_file):
    training_data = {
        'sta_len': [sta_len],
        'lta_len': [lta_len],
        'thr_on': [thr_on],
        'body': [body],
        'difference_value': [difference_value]
    }
    training_df = pd.DataFrame(training_data)

    os.makedirs(os.path.dirname(output_training_file), exist_ok=True)
    training_df.to_csv(output_training_file, mode='a', header=not os.path.exists(output_training_file), index=False)

def best_value(sta_len, lta_len, thr_on, thr_off, body, mseed_directory, output_catalog_file, minfreq, maxfreq, normalize_value, a, spl):
    global difference_value
    global lowest_difference
    global randomness
    loop = True
    data3 = [sta_len, lta_len, thr_on, thr_off, body, mseed_directory, output_catalog_file, minfreq, maxfreq, normalize_value, a, spl]

    while loop is True:
        data_analysis(data3[0], data3[1], data3[2], data3[3], data3[4], data3[5], data3[6], data3[7], data3[8], data3[9], data3[10], data[11])

        if body == "Moon":
            catalog_path = './data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'
        else:
            catalog_path = './data/mars/training/catalogs/Mars_InSight_training_catalog_final.csv'

        st = pd.read_csv(catalog_path)
        st_output = pd.read_csv(output_catalog_file)

        difference_value = 0

        catalog_evids = st['evid'].unique()

        difference_values = {}

        for evid in catalog_evids:

            catalog_rows = st[st['evid'] == evid]
            output_rows = st_output[st_output['evid'] == evid]

            if not output_rows.empty:
 
                difference_values[evid] = abs(catalog_rows.iloc[0, 2] - output_rows.iloc[0, 2])
            else:

                catalog_times = catalog_rows.iloc[:, 2].values
                
                if catalog_times.size > 0:
                    closest_diff = float('inf')

                    for time in catalog_times:

                        if not output_rows.empty:
                            closest_value = output_rows.iloc[(output_rows.iloc[:, 2] - time).abs().argsort()[:1], 2].values
                            
                            if closest_value.size > 0:
                                closest_diff = min(closest_diff, abs(time - closest_value[0]))
                    
                    if closest_diff < float('inf'):
                        difference_values[evid] = closest_diff

        # Now, sum up the differences for the overall difference_value
        difference_value += sum(difference_values.values())

        if body == "Moon":
            pen = 1000
        else:
            pen = 50
        difference_value += abs(len(st) - len(st_output)) * pen
        if body == "Moon":
            max_difference = 1000
        else:
            max_difference = 20
        if difference_value < max_difference: 
            optimal_lta_len = lta_len
            optimal_sta_len = sta_len
            optimal_thr_on = thr_on
            output_training_file = './output/training_data.csv'
            optimal_values = [optimal_sta_len, optimal_lta_len, optimal_thr_on]
            print(f'Lowest difference is {lowest_difference}, with an sta len of {optimal_sta_len}, an lta len of {optimal_lta_len} and a thr on of {optimal_thr_on}')
            save_training_data(optimal_sta_len, optimal_lta_len, optimal_thr_on, body, output_training_file)
            loop = False
        else:
            sta_len1 = sta_len
            lta_len1 = lta_len
            thr_on1 = thr_on
            sta_len += random.randint(-randomness, randomness)
            lta_len += random.randint(-randomness, randomness)
            thr_on += random.randint(-1, 1)
            if ([sta_len, lta_len, thr_on] in knowledge):
                while ([sta_len, lta_len, thr_on] in knowledge):
                    sta_len = sta_len1
                    lta_len = lta_len1
                    thr_on = thr_on1
                    sta_len += random.randint(-randomness, randomness)
                    lta_len += random.randint(-randomness, randomness)
                    thr_on += random.randint(-1, 1)
            data3[0] = sta_len
            data3[1] = lta_len
            data3[2] = thr_on
            knowledge.append([sta_len, lta_len, thr_on])
            if difference_value < lowest_difference:
                lowest_difference = difference_value
                optimal_lta_len = lta_len
                optimal_sta_len = sta_len
                optimal_thr_on = thr_on
                print(f'Lowest difference is {lowest_difference}, with an sta len of {optimal_sta_len}, an lta len of {optimal_lta_len} and a thr on of {optimal_thr_on}')
            sta_len = sta_len1
            lta_len = lta_len1
            thr_on = thr_on1
    return optimal_values

def data_base():
    global randomness
    start = False
    sta_len = 0
    lta_len = 0
    thr_on = 0
    thr_off = 0
    while not start:
        body = input("The data is from which celestial body? Type 'Moon' or 'Mars': ")
            
        if body == "Mars" or body == "Moon":
                
            if body == "Mars":
                minfreq = 0.1
                maxfreq = 9.9
                sta_len += 121
                lta_len += 495
                thr_on += 3
                thr_off += 0.5
                a = 10000
                randomness += 100
                normalize_value = 1e-5 
                spl = 100000
                # sta_len 121, lta_len 495, thr on 3
                output_catalog_file = './output/catalog_mars.csv'
                mseed_directory = './data/mars/training/data/'
                start = True

            elif body == "Moon":
                minfreq = 0.1
                maxfreq = 3.2
                sta_len = 511
                lta_len = 9580
                thr_on = 3
                thr_off = 0.5
                randomness += 50
                normalize_value = 1e-9
                spl = 50000
                # sta len 519, lta len 9599 thr on of 4
                a = 3e-18
                output_catalog_file = './output/catalog_moon.csv'
                mseed_directory = './data/lunar/training/data/S12_GradeA/'
                start = True

            else:
                print("Invalid celestial body. Please choose 'Moon' or 'Mars'.")
    
    
    return [sta_len, lta_len, thr_on, thr_off, body, mseed_directory, output_catalog_file, minfreq, maxfreq, normalize_value, a, spl]


data = data_base()
print(best_value(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11]))
