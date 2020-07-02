sample_rate = 44100
window_size = 2048
hop_size = 512      # So that there are 64 frames per second
mel_bins = 40
fmin = 50       # Hz
fmax = 14000    # Hz

# MFCC setings
mfcc_frames = 431
n_mfcc = 40
mfcc_hop_size = 512

# gammatonegram settings
gamm_frames = 499
n_gamm = 64

frames_per_second = sample_rate // hop_size
audio_duration = 10     # Audio recordings in DCASE2019 Task1 are all 10 seconds
frames_num = frames_per_second * audio_duration
total_samples = sample_rate * audio_duration
total_frames = (sample_rate * audio_duration) // hop_size

labels = ['airport', 'shopping_mall', 'metro_station', 'street_pedestrian', 
    'public_square', 'street_traffic', 'tram', 'bus', 'metro', 'park', 'unknown']
    
# classes_num = len(labels)
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}