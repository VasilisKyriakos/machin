import numpy as np
import RobotDART as rd
import dartpy # OSX breaks if this is imported before RobotDART



class World:

    velocity = 0

    def __init__(self):

        time_step = 0.001

        self.simu = rd.RobotDARTSimu(time_step)

        graphics = rd.gui.Graphics()

        self.simu.set_graphics(graphics)

        graphics.look_at([0., 2.5, 0.5], [0., 0., 0.])


        ########## Create robot ##########

        self.robot = rd.Robot("pendulum.urdf")

        self.robot.fix_to_world()

        body_names = self.robot.dof_names()

        #print(body_names)

        self.robot.set_actuator_types("torque")

        self.start_state = self.robot.set_positions([np.pi/2])

        self.current_state = self.start_state

        self.done = False

        self.simu.add_robot(self.robot)


    """         while True:

                if simu.step_world():
                    break

                pos = self.robot.positions()
                
                print(pos)

                #self.robot.set_commands([-5]) """




    def reward2(self,angle):
    
        r = min(angle,2*np.pi-angle)

        if r < np.pi/8:
            return [10]
        
        else:
            return [0]


    def reward(self,angle,acc):

        ang = min(angle,2*np.pi-angle)
        ac=acc
        #torque= torque[0]/25
    
        r = -(ang**2 + 0.1*ac)

        return r


    def step(self,action):

        prev = self.robot.positions()

        prev = prev%(2*np.pi)

        self.robot.set_commands(action)

        #print(action)

        self.simu.step_world()

        current = self.robot.positions()

        #current = current%(2*np.pi)

        

        #print("current: ",current)

        #print("reward: ",rew)

        posx = np.cos(current[0])
        posy = np.sin(current[0])


        vel = current - prev

        self.acc = vel - self.velocity

        self.velocity = vel

        state = [posx,posy,vel[0]*200]

        #state = [(current[0]/np.pi)-1,vel[0]*200]


        rew = self.reward(current,self.acc)

        #print(vel)

        return state, rew, self.done
    

    def reset(self):

        temp = np.random.rand()*2*np.pi

        temp = 0
        
        current = self.robot.set_positions([temp])  

        state = np.array([np.cos(temp),np.sin(temp), 0.])  # pos, vel

        #state = np.array([0., 0.])  # pos, vel

        self.done = False

        return state

        
    ########## Give a small push to the robot ##########
    #robot.set_external_torque(robot.body_name(1), [0., 0.01, 0.])


