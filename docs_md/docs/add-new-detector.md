# How to add a new detector

![DeepClawLogo](asset/fig-DeepClaw.png)
The structure of `dcp` folder is as following.

    -dcp
      +docs
      +gui
      +utils
      __init__.py
      demo.py
      functions.py
      GUI.py
      ManipulationTasks.py
      Master.py
      Watcher.py
 In `utils` folder, you can find the file `VISPEstimator.py`. We implement a sample detector class called `VISPEstimator`. Developer can create new detectors and save codes in `utils` folder as the same as `VISPEstimator.py`.
 The instantiation of the new detector should be written at `demo.py`. Developer should replace the default parameter `estimator` in `DataCollectionPlatform` class.
 

    def  init_devices(self):
        self.estimator = YOUR_ESTIMATOR(parameters)


 The most important modification is in `ManipulationTasks.py`. In the `obtain_data` function, we call the detecting function to obtain 6D poses by inputting rgb and other information. Developer should add the related output adaptation function to match the data format in the process.
 

    def  obtain_data(self, original_data, parameters):
    color_image, depth_image, point_cloud = original_data['RGB'], original_data['DepthInfo'], original_data['PointCloud']
    results = parameters['estimator'].YOUR_ESTIMATOR_DETECTING_FUNCTION(depth_image, color_image)
    markers = ADAPTATION_FUNCTION(results)

