# Speech-Recognition
Speech Recognition and comparison of DTW, HMM and DeepSpeech.

<img src = https://wevolver-project-images.s3.amazonaws.com/0.8ieefm4xu361_ChocH_eUxil5eaeXIsd3rw.png width = 600px />
Image Source: wevolver-project-images.s3.amazonaws.com


## Description
A study to compare speech recognition algorithms for real-time human voice command detection. These algorithms can be further used for controlling robotics movements by performing real-time inference on RaspberryPi computers. These algorithms were used for the Voice Controlled Robot project as part of IEEE NITK.

## Getting Started

### Dependencies

* TensorFlow==1.x.
* RaspberryPi 3
* Python==2.x

### Installing
```
git clone Aaatresh/Speech-Recognition
```

### Data
The data used in this project consisted of multiple audio recordings for each command to be followed. 

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
* Vishal Shukla

## Version History
* 0.1
    * Initial Release

## License

This project is licensed under the GPL-3.0 License - see the LICENSE.md file for details

## Acknowledgements
Special thanks to IEEE NITK for funding and guiding us through this project. <br>
Thank you https://github.com/MyrtleSoftware for aiding us in the implementation of "deepspeech".
