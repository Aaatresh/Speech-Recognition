# Speech-Recognition
Speech Recognition and comparison of DTW, HMM and DeepSpeech.

<img src = https://wevolver-project-images.s3.amazonaws.com/0.8ieefm4xu361_ChocH_eUxil5eaeXIsd3rw.png width = 600px />
Image Source: wevolver-project-images.s3.amazonaws.com


## Description
A study to compare speech recognition algorithms for real-time human voice command detection. These algorithms can be further used for controlling robotics movements by performing real-time inference on RaspberryPi computers. These algorithms were used for the Voice Controlled Robot project as part of IEEE NITK.

## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Data
Any audio data can be fed into this system as long as it consists of speech. The audio data can be stored in directory, and models can be run separately.

### Running the code
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

## Authors
Contributors names and contact info
* Anirudh Aatresh (aaa.171ec106@nitk.edu.in)

## Version History
* 0.1
    * Initial Release

## License

This project is licensed under the GPL-3.0 License - see the LICENSE.md file for details

## Acknowledgements
Special thanks to IEEE NITK for funding and guiding us through this project. 
Thank you https://github.com/MyrtleSoftware for aiding us in the implementation of "deepspeech".
