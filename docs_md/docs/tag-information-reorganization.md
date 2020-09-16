# Reorganization of Tag Information

![DeepClawLogo](asset/fig-DeepClaw.png)
In the `obtain_data` function of `ManipulationTasks`, we can obtain 6D pose information of all tags by the detector. The next step is to divide all tags into several groups corresponding to the gripper design. In the figure 3 (b) of our paper, there are 6 tags arranged in a symmetrical structure. The roles of tag position setting is that every rigid link needs at least one tag. According to this role, we arrange one tag to each tong link in our design. In our program, we can use filter to obtain the certain tag pose as following.

    def  apriltag_filter(tags, min_id, max_id):
       target_tags = []
       for tag in tags:
           if min_id <= tag['id'] <= max_id:
           target_tags.append(tag)
           return target_tags

Then we can make a logical combination of several specific tags. For example, the TCP pose is the midpoint of the line of two tag points in tongs.
All above operations are achieve in the `obtain_data` function in `ManipulationTasks.py` file. Developers should modify the processing logic there.

