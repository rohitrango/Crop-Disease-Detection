## Crop Disease Detector

Second Runners Up at Microsoft Code.Fun.Do hackathon, IIT Bombay

This application uses a modified AlexNet architecture to classify among 38 classes of healthy and diseased crops. We use the *PlantVillage* dataset for training the network. 

To improve upon the accuracy, we use a modified version of the aforementioned network which also takes the name of the crop, since farmers are expected to know the crops of which they want to know the disease (or the lack of disease). We achieved around *91%* accuracy on the held-out test set.

All this is rolled into an API, and an Android app that uses this backend. Check the _images_ folder for screenshots.

## TODO
- [ ] Find better and robust datasets
- [ ] Improve epidemic detection based on temporal data.

Feel free to report bugs by opening an issue. You can also add features by adding a PR.