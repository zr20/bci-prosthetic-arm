import mne

path = "/Users/zarakhursheed/Downloads/NYU/DATASETS/Dataset 2b BCICIV/Training/B0101T.gdf"
raw = mne.io.read_raw_gdf(path, preload=True)
print(raw.info)
print(raw.ch_names)

events, event_id = mne.events_from_annotations(raw)
print(event_id)
print(events[:10])
