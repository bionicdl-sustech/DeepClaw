import argparse
from input_output.Configuration import *

parser = argparse.ArgumentParser()
parser.add_argument("robot", type=str, choices=['ur10e', 'ur5', 'franka'], help="name of robot arm")
parser.add_argument("gripper", type=str, choices=['rg6'], help="name of robot gripper")
parser.add_argument("sensor", type=str, choices=['realsense'], help="name of sensor")
parser.add_argument("task", type=str, choices=['test', 'io-test'], help="task name")
args = parser.parse_args()

ROBOT_NAME = args.robot
GRIPPER_NAME = args.gripper
SENSOR_NAME = args.sensor
TASK_NAME = args.task


def initial_sensors(sensor_name):
    if sensor_name == "realsense":
        from driver.sensors.camera.RealsenseController import RealsenseController
        cfg = readConfiguration('/config/sensors/realsense.yaml')
        realsense = RealsenseController(cfg)
        # realsense = ''
        return realsense
    else:
        print("Not support for this sensor.")
        return None


def initial_robot(robot_name):
    if robot_name == "ur10e":
        from driver.arms.UR10eController import UR10eController
        cfg = readConfiguration('/config/arms/ur10e.yaml')
        robot = UR10eController(cfg)
        return robot
    elif robot_name == "ur5":
        from driver.arms.UR5Controller import UR5Controller
        cfg = readConfiguration('/config/arms/ur5.yaml')
        robot = UR5Controller(cfg)
        return robot
    elif robot_name == "franka":
        from driver.arms.FrankaController import FrankaController
        cfg = readConfiguration('/config/arms/franka.yaml')
        robot = FrankaController(cfg)
        return robot
    else:
        print("Don't support this robot!")
        return None


def initial_task(task_name, perception_system, manipulation_system, is_debug=False):
    if task_name == "test":
        from examples.TestCase import TestCase
        test_case = TestCase(perception_system, manipulation_system, is_debug)
        return test_case
    elif task_name == "io-test":
        from examples.IOTest import IOTest
        iotest = IOTest(perception_system, manipulation_system, is_debug)
        return iotest
    else:
        print('No such task.')
        return None


if __name__ == '__main__':
    sensor = initial_sensors(SENSOR_NAME)
    robot = initial_robot(ROBOT_NAME)
    # robot.matrix_load()

    if robot is not None and sensor is not None:
        perception_system = {'Camera': sensor, 'Recorder': sensor}
        manipulation_system = {'Arm': robot, 'Gripper': robot}

        task = initial_task(TASK_NAME, perception_system, manipulation_system, is_debug=True)
        if task is not None:
            task.task_display()
