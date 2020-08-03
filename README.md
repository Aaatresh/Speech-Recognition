# Speech-Recognition
Speech Recognition and comparison of DTW, HMM and DeepSpeech.

<img src = https://wevolver-project-images.s3.amazonaws.com/0.8ieefm4xu361_ChocH_eUxil5eaeXIsd3rw.png width = 600px />
Image Source: wevolver-project-images.s3.amazonaws.com

# Data
Any audio data can be fed into this system as long as it consists of speech. The audio data can be stored in directory, and models can be run separately.

# Running the code
To run mfcc + dtw:
```
run all cells in ./dtw/speech_recog_dtw_time.ipynb
```

To run HMM:
```
python3 ./HMM/demo_ghmm.py
python3 ./HMM/demo_gmmhmm.py
```

To run deepspeech:
```
run ./deepspeech/run_model.py
```
