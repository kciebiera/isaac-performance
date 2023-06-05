from omni.isaac.kit import SimulationApp

RENDER = True
MAP_SIZE = 256

simulation_app = SimulationApp({"headless": not RENDER})
import random

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka import Franka, KinematicsSolver

from enum import Enum

RobotState = Enum('RobotState', ['OPEN', 'DOWN', 'CLOSE', 'UP'])

def single_run(dx: float = 0, dy: float = 0):
    """
    Runs single simulation run with given dx and dy offsets
    this function is called from main() function MAP_SIZE^2 times
    """
    random.seed(42)
    robot_state = RobotState.OPEN
    eps = 0.001
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    fancy_cube =  world.scene.add(
        DynamicCuboid(
            name = "target_cube",
            prim_path="/World/target_cube",
            position=np.array([0.3, 0, 0.1]),
            scale=np.array([0.03, 0.03, 0.03]),
            color=np.array([0, 0, 1.0]),
        ))
    for i in range(20):
        world.scene.add(
        DynamicCuboid(
            name=f"random_cube_{i}",
            prim_path=f"/World/random_cube_{i}",
            position=np.array([0.1 + 0.4 * random.random(), -0.2 + random.random() * 0.4, 0.1]),
            scale=np.array([0.03, 0.03, 0.03]),
            color=np.array([0, 0, 1.0]),
        ))
    my_franka = world.scene.add(Franka(prim_path="/World/Franka", name="my_franka"))
    my_controller = KinematicsSolver(my_franka)
    articulation_controller = my_franka.get_articulation_controller()
    world.reset()

    # wait for the world to stabilize
    for i in range(30):
        world.step(render=RENDER)
        
    target_position = fancy_cube.get_current_dynamic_state().position
    target_position[0] += dx
    target_position[1] += dy

    for i in range(1500): # 1500 steps at most, just in case
        world.step(render=RENDER)
        if world.is_playing():
            gripper_positions = my_franka.gripper.get_joint_positions()
            if robot_state == RobotState.OPEN:
                my_franka.gripper.apply_action(
                    ArticulationAction(joint_positions=[gripper_positions[0] + (0.005), gripper_positions[1] + (0.005)])
                )
                if gripper_positions[1] >= 0.04 - eps:
                    robot_state = RobotState.DOWN
            elif robot_state in [RobotState.DOWN, RobotState.UP] :
                actions, succ = my_controller.compute_inverse_kinematics(
                    target_position=target_position,
                    target_orientation=np.array([0,0,1,0])# state.orientation
                )
                if succ:
                    articulation_controller.apply_action(actions)
                else:
                    print(dx, dy, 100) # can't reach the target_position
                    break
                end_effector_poistion = my_controller.compute_end_effector_pose(position_only=True)[0]
                dist = np.linalg.norm(end_effector_poistion-target_position)
                if dist < eps: # end effector is close enough
                    if robot_state == RobotState.UP: # simmulation have reached the end
                        # print the result
                        print(dx, dy, gripper_positions[0] + gripper_positions[1])
                        # exit the loop
                        break
                    robot_state = RobotState.CLOSE
                    close_steps_to_go = 150 #empirical number of steps to close the gripper
            if robot_state == RobotState.CLOSE:
                my_franka.gripper.apply_action(
                    ArticulationAction(joint_positions=[gripper_positions[0] - (0.005), gripper_positions[1] - (0.005)])
                )
                close_steps_to_go -= 1
                if close_steps_to_go <= 0:
                    target_position[2] += 0.1 # we want to move the object up
                    robot_state = RobotState.UP
    world.clear_instance()

def main():
    offsets = [-0.1, 0, 0.1]
    offsets = [0.2 * (i / (MAP_SIZE - 1)) - 0.1 for i in range(MAP_SIZE)]

    for dx in offsets:
        for dy in offsets:
            single_run(dx, dy)

main()
simulation_app.close()