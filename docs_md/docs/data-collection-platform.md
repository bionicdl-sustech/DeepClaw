# Data Collection Platform

![DeepClawLogo](asset/fig-DeepClaw.png)

## Design Notes
This data collection platform design include two parts: the physical assembling, and the software installation. In hardware part, we'll introduce how to assemble as the same as the robot system introduced by the paper[]. As for software, we'll provide the instruments how to collect your own data step by step.

### Platform Assembling
The physical part in the data collection platform includes a robot station, a pair of modified tongs, and a computational equipment. The robot station contains a aluminum frame, a RGB-D camera(we used Realsense D435), two lights, and a aluminum desk. Based on our open-source file[need updated by WANG Teng], the pair of modified tongs can be printed by 3D-printer. The computational equipment should be prepare by yourself. The recommended configurations about the CPU, GPU, and memory are higher than Intel Core i5, Intel UHD Graphics, and 4 GB memory. The assembly manual of the aluminum frame and aluminum desk can be found here[need updated by WANG Teng].

### Software Installation
The instruction of DeepClaw installation can be found at [https://bionicdl-sustech.github.io/DeepClaw/install/].
All of the algorithm modules and GUI are integrated in the last released version. 
Users need to install DeepClaw from the source.
To verify the installation:

    $ python dcp/demo.py
If you can see data collection platform GUI is rendered on the screen as following, then your installation is successful.
![DCP-GUI](asset/fig-GUI.png)

