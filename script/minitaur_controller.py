
import numpy as np
import gym, pybullet_envs, math
import time

def frac(x):
    return x - np.floor(x)

def sawtooth(t, freq, phase=0):
    T = 1/float(freq)
    y = frac(t/T + phase)
    return y

def mytest():
    env = gym.make("MinitaurBulletEnv-v0", render=False)
    env.reset()
    for i in range(1000):
        a = env.action_space.sample()
        print(a)
        env.step(a)
        time.sleep(1)

def SinePolicyExample():
    from pybullet_envs.bullet import minitaur_gym_env
    import argparse
    from pybullet_envs.bullet import minitaur_env_randomizer
    """An example of minitaur walking with a sine gait."""
    randomizer = None#(minitaur_env_randomizer.MinitaurEnvRandomizer())
    environment = minitaur_gym_env.MinitaurBulletEnv(render=False,
                                                   motor_velocity_limit=np.inf,
                                                   pd_control_enabled=True,
                                                   hard_reset=False,
                                                   env_randomizer=randomizer,
                                                   on_rack=False,
                                                   accurate_motor_model_enabled=True)

    sum_reward = 0
    steps = 20000
    amplitude_1_bound = 0.3
    amplitude_2_bound = 0.2
    speed = 10
    environment.minitaur.SetFootFriction(1.0)
    for step_counter in range(steps):
        time_step = 0.01
        t = step_counter * time_step

        amplitude1 = amplitude_1_bound
        amplitude2 = amplitude_2_bound
        steering_amplitude = 0
        if t < 10:
          steering_amplitude = 0.2
        elif t < 15:
          steering_amplitude = 0
        else:
          steering_amplitude = -0.2

        # Applying asymmetrical sine gaits to different legs can steer the minitaur.
        fl = math.sin(t * speed) * (amplitude1+0.2)
        br = math.sin(t * speed) * (amplitude1-0.2)
        fr = math.sin(t * speed + math.pi) * (amplitude1-0.2)
        bl = math.sin(t * speed + math.pi) * (amplitude1+0.2)
        a3 = math.sin(t * speed + math.pi/2) * amplitude2
        a4 = math.sin(t * speed + math.pi + math.pi/2) * amplitude2
        # action = [a1, a2, a2, a1, a3, a4, a4, a3]
        action = [fl, bl, fr, br, a3, a4, a4, a3]
        # action = [swing_front_left, swing_back_left, swing_front_right, swing_back_right, a3, a4, a4, a3]
        _, reward, done, _ = environment.step(action)
        sum_reward += reward
        if done:
          break
    environment.reset()

class ControllerSine () :

    def __init__(self, params=None, array_dim=100):
        self.array_dim = array_dim
        self._params = None

    def nextCommand(self, t):
        # Control parameters
        # steer = 0.0 #Move in different directions
        # step_size = 1.0 # Walk with different step_size forward or backward
        # leg_extension = 1.0 #Walk on different terrain
        # leg_extension_offset = -1.0

        steer = self._params[0] #Move in different directions
        step_size = self._params[1] # Walk with different step_size forward or backward
        leg_extension = self._params[2] #Walk on different terrain
        leg_extension_offset = self._params[3]

        # Robot specific parameters
        swing_limit = 0.5
        extension_limit = 0.4
        speed = 2

        A = np.clip(step_size + steer, -1, 1)
        B = np.clip(step_size - steer, -1, 1)
        extension = extension_limit * (leg_extension+1.0) * 0.5
        max_extension = np.clip(extension + extension_limit*leg_extension_offset, 0, extension)
        min_extension = np.clip(-extension + extension_limit*leg_extension_offset, -extension, 0)

        #We want legs to move sinusoidally, smoothly
        fl = math.sin(t * speed) * (swing_limit * A)
        br = math.sin(t * speed) * (swing_limit * B)
        fr = math.sin(t * speed + math.pi) * (swing_limit * B)
        bl = math.sin(t * speed + math.pi) * (swing_limit * A)

        #We can legs to reach extreme extension as quickly as possible: More like a smoothed square wave
        e1 = np.clip(3.0 * math.sin(t * speed + math.pi/2), min_extension, max_extension)
        e2 = np.clip(3.0 * math.sin(t * speed + math.pi + math.pi/2), min_extension, max_extension)
        # [swing_front_left, swing_back_left, swing_front_right, swing_back_right, a3, a4, a4, a3]
        return np.array([fl,bl,fr,br,e1,e2,e2,e1])
        # return np.array([fl,bl,fr,br,-1,-1,-1,-1])
        # return np.array([1.,0.,0.,0.,0.5,0.5,0.5,0.5])

    def setParams(self, params, array_dim=100):
        self._params = params

    def setRandom(self):
        self._random = True
        self.setParams(np.random.rand(4) * 2.0 - 1.0)

    def getParams(self):
        return self._params

class ControllerSineGallop() :

    def __init__(self, params=None, array_dim=100):
        self.array_dim = array_dim
        self._params = None

    def nextCommand(self, t):
        # Control parameters
        # steer = 0.0 #Move in different directions
        # step_size = 1.0 # Walk with different step_size forward or backward
        # leg_extension = 1.0 #Walk on different terrain
        # leg_extension_offset = -1.0

        steer = self._params[0] #Move in different directions
        step_size = self._params[1] # Walk with different step_size forward or backward
        leg_extension = self._params[2] #Walk on different terrain
        leg_extension_offset = self._params[3]

        # Robot specific parameters
        swing_limit = -0.3
        extension_limit = 0.9
        speed = 21 #22 is best

        A = np.clip(step_size + steer, -1, 1)
        B = np.clip(step_size - steer, -1, 1)
        extension = extension_limit * (leg_extension+1.0) * 0.5
        max_extension = np.clip(extension + extension_limit*leg_extension_offset, 0, extension)
        min_extension = np.clip(-extension + extension_limit*leg_extension_offset, -extension, 0)

        #We want legs to move sinusoidally, smoothly
        fl = math.sin(t * speed) * (swing_limit * A)
        fr = math.sin(t * speed) * (swing_limit * B)
        bl = math.sin(t * speed + math.pi/2) * (swing_limit * A)
        br = math.sin(t * speed + math.pi/2) * (swing_limit * B)

        #We can legs to reach extreme extension as quickly as possible: More like a smoothed square wave
        ef =  math.sin(t * speed + np.pi/2) * extension
        eb = math.sin(t * speed + math.pi) * extension
        # [swing_front_left, swing_back_left, swing_front_right, swing_back_right, a3, a4, a4, a3]
        return np.array([fl,bl,fr,br,ef,eb,ef,eb])
        # return np.array([1.,0.,0.,0.,0.5,0.5,0.5,0.5])

    def setParams(self, params, array_dim=100):
        self._params = params

    def setRandom(self):
        self._random = True
        self.setParams(np.random.rand(4) * 2.0 - 1.0)

    def getParams(self):
        return self._params

class ControllerSaw () :

    def __init__(self, params=None, array_dim=100):
        self.array_dim = array_dim
        self._params = None

    def nextCommand(self, t):
        # Control parameters
        # steer = 0.0 #Move in different directions
        # step_size = 1.0 # Walk with different step_size forward or backward
        # leg_extension = 1.0 #Walk on different terrain
        # leg_extension_offset = -1.0

        steer = self._params[0] #Move in different directions
        step_size = self._params[1] # Walk with different step_size forward or backward
        leg_extension = self._params[2] #Walk on different terrain
        leg_extension_offset = self._params[3]

        # Robot specific parameters
        swing_limit = 0.6
        extension_limit = 0.4
        speed = 3.0 #cycle per second

        # Steer modulates only the magnitude (abs) of step_size
        A = np.clip(abs(step_size) + steer, 0, 1) if step_size >= 0 else -np.clip(abs(step_size) + steer, 0, 1)
        B = np.clip(abs(step_size) - steer, 0, 1) if step_size >= 0 else -np.clip(abs(step_size) - steer, 0, 1)

        #We want legs to move sinusoidally, smoothly
        fl = math.sin(t * speed*2*np.pi) * (swing_limit * A)
        br = math.sin(t * speed*2*np.pi) * (swing_limit * B)
        fr = math.sin(t * speed*2*np.pi + math.pi) * (swing_limit * B)
        bl = math.sin(t * speed*2*np.pi + math.pi) * (swing_limit * A)

        # Sawtooth for faster contraction
        e1 = -(2*sawtooth(t , speed , 0.25) - 1.) * extension_limit + 0.3
        e2 = -(2*sawtooth(t , speed , 0.5 + 0.25) - 1.)* extension_limit + 0.3
        # print(e1)
        # [swing_front_left, swing_back_left, swing_front_right, swing_back_right, a3, a4, a4, a3]
        return np.array([fl,bl,fr,br,e1,e2,e2,e1])
        # print(fl)
        # return np.array([0,0,0,0,-1,0,0,0])

    def setParams(self, params, array_dim=100):
        self._params = params

    def setRandom(self):
        self._random = True
        self.setParams(np.random.rand(4) * 2.0 - 1.0)

    def getParams(self):
        return self._params

def SinePolicyAdvanced():
    import pybullet
    import fast_adaptation_embedding.env.minitaur_env as minitaur_gym_env
    import argparse
    # from pybullet_envs.bullet import minitaur_env_randomizer
    """An example of minitaur walking with a sine gait."""
    env = minitaur_gym_env.MinitaurBulletEnv(render=False,
                                                   motor_velocity_limit=np.inf,
                                                   pd_control_enabled=True,
                                                   hard_reset=False,
                                                   env_randomizer=None,
                                                   on_rack=False,
                                                   accurate_motor_model_enabled=False)

    steps = 20000
    env.set_floor_friction(2.0)
    controller = ControllerSaw()
    controller.setParams(np.array([-0.0, 1.0, 0, 0]))
    time_step = 0.002
    control_step=0.01
    for step_counter in range(steps):
        t = step_counter * control_step
        # qKey = ord('q')
        lKey = pybullet.B3G_LEFT_ARROW
        rKey = pybullet.B3G_RIGHT_ARROW
        fKey = pybullet.B3G_UP_ARROW
        bKey = pybullet.B3G_DOWN_ARROW
        params = controller.getParams()
        keys = pybullet.getKeyboardEvents()
        if lKey in keys and keys[lKey] & pybullet.KEY_WAS_TRIGGERED:
            params[0] = -1.
            controller.setParams(params)
        elif rKey in keys and keys[rKey] & pybullet.KEY_WAS_TRIGGERED:
            params[0] = 1.
            controller.setParams(params)
        elif fKey in keys and keys[fKey] & pybullet.KEY_WAS_TRIGGERED:
            params[0] = 0.
            params[1] = abs(params[1])
            controller.setParams(params)
        elif bKey in keys and keys[bKey] & pybullet.KEY_WAS_TRIGGERED:
            params[0] = 0.
            params[1] = -abs(params[1])
            controller.setParams(params)

        print(controller.nextCommand(t))
        env.step(np.clip(controller.nextCommand(t),-1, 1))

SinePolicyAdvanced()
# import fast_adaptation_embedding.env.minitaur_env as Minitaur_env
# environment = Minitaur_env.MinitaurBulletEnv(render=True,
#                                                 motor_velocity_limit=np.inf,
#                                                 pd_control_enabled=True,
#                                                 hard_reset=False,
#                                                 env_randomizer=None,
#                                                 on_rack=False)
