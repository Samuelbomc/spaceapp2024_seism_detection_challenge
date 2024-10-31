# spaceapp2024_seism_detection_challenge

To use the program its necessary to install the data (Found here https://wufs.wustl.edu/SpaceApps/data/space_apps_2024_seismic_detection.zip) and output folders and the file Seism_Detect_Final.py. Then, when you run this last program, its going to first ask you if you want to analyze data from either Mars or from the Moon. After that its going to ask which dataset to use. Finally, if applicable, it will ask which package to use. The graphs will be stored in the folder ./output/figures/(Celestial body chosen) and the catalog with the detections will be placed at ./output/. There is also a file called "Parameter_finder" that compares the results of the parameters given with the correct detection, looking for the configuration that get the lowest difference score.

The implementation of Training_Data.py is pending, we plan to implement in the future a machine learning algorythm that given large databases of noise (Sounds other than seisms found on celestial bodies surfaces) can then eliminate simmilar behaving patterns on the raw data before its analysis its done.

// Proyect Description

The Informed Seism Detector (ISD) is a seismic data processing program designed to identify seismic events obscured by various noise sources across Earth, the Moon, and Mars. By locally analyzing the data and transmitting only the relevant information, the ISD optimizes bandwidth usage, which is essential given the vast distances involved in transmitting data from space.

The program adapts its approach based on the origin of the seismic data. For instance, Mars data is analyzed in a frequency range from 0.1 to 9.9 Hz, using shorter windows for Short-Term Average (STA) and Long-Term Average (LTA) measurements (60 and 250 seconds, respectively) with a STA/LTA ratio threshold of 1.5 to indicate potential seismic activity. In contrast, lunar data is processed over a narrower range of 0.1 to 3.2 Hz, using extended STA and LTA windows (100 and 1000 seconds, respectively) due to the unique seismic properties of the Moon.

Once the data is filtered using a bandpass filter, the ISD applies a noise reduction step, using Gaussian filtering and a Short-Term Fourier Transform (STFT) with parameters tailored for each dataset. Noise is further reduced by an adaptive algorithm that identifies outliers based on local mean and standard deviation, replacing them with averaged values.

The STA/LTA algorithm is then applied, detecting seismic events when the STA-to-LTA ratio exceeds the threshold. The ISD also checks event power using a thresholding method to ensure detected events are significant; valid detections are recorded only if the power exceeds a set level. A minimum time gap between detections is enforced to avoid redundant triggers.

If enabled, the ISD creates visual outputs, including time series, filtered traces, logarithmic scale traces, STA/LTA plots, and spectrograms. For lunar and Martian data, it uses specific label configurations based on the source characteristics. Spectrograms are generated and scaled to enhance the visibility of seismic events, with logarithmic scaling on trace plots to highlight seismic waveforms against background noise.

For each detected event, the ISD logs details such as the absolute and relative event time, filename, and event ID in a catalog file, stored in CSV format. This catalog, along with the visual outputs saved as PNG files, provides a comprehensive record of the seismic events, facilitating efficient data analysis and selective transmission across space.
