import pandas as pd
from obspy import read
import os

# Step 1: Load the catalog CSV file
catalog_file_path = './data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'
catalog_data = pd.read_csv(catalog_file_path)

# Step 2: Define a function to extract event details from catalog and MiniSEED files
def process_event(row, mseed_dir):
    event_info = {}
    
    # Extract start time and event type
    start_time = row['time_abs(%Y-%m-%dT%H:%M:%S.%f)']
    event_type = row['mq_type']
    mseed_file = row['filename']
    
    event_info['start_time'] = start_time
    event_info['event_type'] = event_type
    
    # Step 3: Load the corresponding MiniSEED file
    mseed_file_path = os.path.join(mseed_dir, mseed_file)
    
    try:
        st = read(mseed_file_path)
        # Assuming you want to use the first trace for amplitude calculation
        trace = st[0]
        
        # Step 4: Calculate amplitude (Filtered Counts per Second)
        # This is a simple max amplitude extraction, more advanced filters can be added
        amplitude = max(abs(trace.data))  # You can apply a bandpass filter here if needed
        event_info['amplitude (cts/s)'] = amplitude
    
    except Exception as e:
        print(f"Error processing {mseed_file}: {e}")
        event_info['amplitude (cts/s)'] = None
    
    return event_info

# Directory where MiniSEED files are located
mseed_directory = 'path_to_mseed_files/'

# Step 5: Iterate through the catalog and extract information for each event
events = []
for index, row in catalog_data.iterrows():
    event_details = process_event(row, mseed_directory)
    events.append(event_details)

# Step 6: Create a DataFrame with the extracted data and save it as CSV
event_df = pd.DataFrame(events)
event_df.to_csv('extracted_event_info.csv', index=False)

print("Event extraction completed.")
