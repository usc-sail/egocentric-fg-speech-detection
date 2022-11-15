# egocentric-fg-speech-detection

Foreground speech detection in egocentric audio. Current pipeline supports 30s audio files, can extend to support to longer audio files in the future.



### Setup instructions
1. Clone github repo:
```
git clone https://github.com/usc-sail/egocentric-fg-speech-detection.git
cd egocentric-fg-speech-detection
```

2. Install [Anaconda](https://www.anaconda.com/) and python dependencies
```
conda create -n fgenv python=3.7
conda activate fgenv
pip install -r requirements.txt
```

3. Download pretrained model(s) [here](https://drive.google.com/drive/folders/1pcuWFoXgB_hjw6vo4k_eNj8P_XrNW4ci). Make sure to move them under the **pretrained_models** directory. Edit the **config.yaml** to reflect the path of the pretrained model you want to use.

4. Run inference! You can process input audio files in either of two ways:
      <ul>
        <li> Provide a single path to directory containing all of the audio-files (if they are all in one place) </li>
        <li> Provide a single text file containing complete paths to audio files, one on each line (if they are distributed) </li>
      </ul>

  Foreground label predictions are written in a csv file in the format <fileid, posterior, label> - posterior is a confidence measure in the range [0,1] 
  and label is a binary (0/1)  value for the absence/presence of foreground speech in the audio-snippet.

```python
  python foreground.py -h
  Usage: foreground.py [-h] [--config CONFIG] audio_paths out_file
  
  Foreground-Speech Detection
  positional arguments:
    audio_paths         text-file containing complete paths to audio-files on each line
                        OR directory containing audio-files
    out_file            Output text-file name
  
  optional arguments:
    -h, --help       show this help message and exit
    --config CONFIG  config file (typically "config.yaml")

example usage:
        python foreground.py wav_files.txt results.csv
        python foreground.py /path/to/wav/files/ results.csv
        python foreground.py --config config.yaml /path/to/wav/files/ results.csv
        
```
