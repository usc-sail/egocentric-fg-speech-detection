import os
import torchaudio
import torch
import argparse
import sys
import yaml
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob

def get_file_paths(ap):
    if os.path.isfile(ap):
        audio_paths = [x.rstrip() for x in open(ap, 'r').readlines()]
        audio_paths = [x for x in audio_paths if os.path.isfile(x)]
    elif os.path.isdir(ap):
        audio_paths = glob.glob(ap + '/*wav')
    else:
        sys.exit("audio_paths argument must be a) text-file containing COMPLETE paths to valid audio files on each line, or b) directory containing valid audio files")

    assert len(audio_paths)>0, "No valid audio files found, check if corrupted"
    
    return audio_paths

class FGDataset(Dataset):
    def __init__(self, audio_paths, audio_conf):
        self.data = audio_paths
        self.melbins = audio_conf['num_mel_bins']
        self.target_length = audio_conf['target_length']
        self.norm_mean = audio_conf['spec_mean']
        self.norm_std = audio_conf['spec_std']
        self.sample_rate = audio_conf['sample_frequency']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fileid = os.path.basename(self.data[index]).split('.wav')[0]
        waveform, sr = torchaudio.load(self.data[index])
        if sr != self.sample_rate:
            transform = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = transform(waveform)

        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=self.sample_rate, use_energy=False,
                                                    window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)

        p = self.target_length - fbank.shape[0]
        if p>0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p<0:
            start_id = np.random.randint(0, -p)
            fbank = fbank[start_id: start_id+self.target_length, :]
        
        fbank = (fbank-self.norm_mean)/(self.norm_std*2)
        fbank = torch.transpose(fbank, 0, 1).unsqueeze(0)
        return fileid, fbank


def inference(dataloader, model, outfile, device):
    smax = torch.nn.Softmax(dim=1)
    model.eval()
    out_data = []

    with torch.no_grad():
        for audio_file, audio_input in tqdm(dataloader):
            audio_input = audio_input.to(device)
            audio_output = audio_model(audio_input)
            audio_output_smax = smax(audio_output)
            posterior = audio_output_smax[:, 1].cpu().numpy()
            label = torch.argmax(audio_output_smax, dim=1).cpu().numpy()
            out_data.extend(list(zip(audio_file, posterior, label)))
    
    out_data = pd.DataFrame(out_data, columns=['filename', 'posterior', 'label'])
    out_data.to_csv(outfile, index=False)


if __name__ == '__main__':
    usage_example = '''example usage:
        python foreground.py wav_files.txt results.csv
        python foreground.py /path/to/wav/files/ results.csv
        python foreground.py --config config.yaml /path/to/wav/directory/ results.csv'''
                 
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="Foreground-Speech Detection", epilog=usage_example)
    parser.add_argument("audio_paths", type=str, help='text-file containing complete paths to audio-files on each line OR directory containing audio-files')
    parser.add_argument("out_file", type=str, help='Output text-file name')
    parser.add_argument("--config", type=str, help='config file (typically "config.yaml")', default='config.yaml')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    audio_paths = get_file_paths(args.audio_paths)
    config = yaml.load(open(args.config, 'r'), yaml.FullLoader)
   
    ## Create Dataloader
    dataloader = torch.utils.data.DataLoader(FGDataset(audio_paths, config['data']), batch_size=config['model']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'], pin_memory=True)
    
    ## Define and load model
    audio_model = torch.load(config['model']['pretrained_path'], map_location=device)
    if device.type == 'cpu':
        audio_model = audio_model.module
    
    ## Run inference
    inference(dataloader, audio_model, args.out_file, device)
