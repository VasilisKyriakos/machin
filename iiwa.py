from importlib.util import set_loader
import numpy as np
import RobotDART as rd
import dartpy # OSX breaks if this is imported before RobotDART
import copy
from utils import damped_pseudoinverse, AdT, enforce_joint_limits


tee = 0

class Iiwa_World:

    velocity = 0

    def __init__(self):

        time_step = 0.001

        self.simu = rd.RobotDARTSimu(time_step)

        graphics = rd.gui.Graphics()

        self.simu.set_graphics(graphics)

        graphics.look_at([0., 2.5, 0.5], [0., 0., 0.])


        ########## Create robot ##########

        self.robot = rd.Iiwa()

        self.robot.fix_to_world()

        body_names = self.robot.dof_names()

        print(body_names)

        self.robot.set_actuator_types("servo")

        #self.start_state = self.robot.set_positions()

        #self.current_state = self.start_state

        self.done = False

        self.simu.add_robot(self.robot)


        self.target_positions = copy.copy(self.robot.positions())

        self.target_positions[0] = -2.

        self.target_positions[3] = -np.pi / 2.0

        self.target_positions[5] = np.pi / 2.0

        self.robot.set_positions(self.target_positions)

        eef_link_name = "iiwa_link_ee"


        self.goal_robot = rd.Robot.create_ellipsoid(dims=[0.1,0.1,0.1], 

                            pose=self.robot.body_pose_vec(eef_link_name), color=[0., 1., 0., 0.5], 
                            
                                ellipsoid_name="target")

        self.simu.add_visual_robot(self.goal_robot)

        self.goal_robot.fix_to_world()

        self.simu.add_checkerboard_floor()
        
        self.tee = self.robot.body_pose(eef_link_name).translation()




    """         while True:

            if self.simu.step_world():

                 break

            pos = self.robot.positions()
            
            print(pos)

            break """




        #self.robot.set_commands([-5])

    def reset(self):

        positions = self.target_positions+np.random.rand(self.robot.num_dofs())*np.pi/3-np.pi/6.

        self.robot.set_positions(enforce_joint_limits(self.robot, positions))

        current = np.array(copy.copy(self.robot.positions()))  # pos, vel

        state = self.target_positions- current

        eef_link_name = "iiwa_link_ee"

        cur_eef = self.robot.body_pose(eef_link_name).translation()

 

        return state



    def reward(self,eefpose):
        #print((np.linalg.norm(eefpose- self.tee)))

        return -(np.linalg.norm(eefpose- self.tee))



    def step(self,action):

        self.done = False

        self.robot.set_commands(enforce_joint_limits(self.robot,action))

        #print(action)

        self.simu.step_world()

        current = copy.copy(self.robot.positions())

        state = self.target_positions - current

        #print(f"****state*****{state}")

        eef_link_name = "iiwa_link_ee"

        cur_eef = self.robot.body_pose(eef_link_name).translation()

        rew = self.reward(cur_eef)
            
        return state, rew, self.done


