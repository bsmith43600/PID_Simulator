import numpy as np
import matplotlib.pyplot as plt

# Create object class for a spring-mass-damper system model. This object contains methods to execute
# caculations of position, acceleration, and velocity based on appropriate inputs. The calculations
# are executed in discrete time steps with the assumption of constant acceleration across the step. 
# The smaller the time step used, the better the model will approximate continuous behavior.

class SMD_System_Sim:

    # Gather spring-mass-damper system constants as part of object initilization. mass = mass, 
    # k = spring constant, and c = damping coefficient. This model is unit agnostic, so long as
    # the input parameters are within the same system of measurement.
    
    def __init__(self, mass, k, c):
        self.mass = mass
        self.k = k
        self.c = c

    # Create object method to return acceleration based upon input force, initial velocity, and 
    # initial position.
    def get_acceleration(self, force, v0, x0):
        return (force - self.k*x0 - self.c * v0) / self.mass

    # Create object method to return velocity based upon acceleration, initial velocity, and
    # duration of acceleration. This method assumes that acceleration is constant and does 
    # not change as a result of the changing velocity and position. The error due to this 
    # assumption approaches 0 as the time step becomes smaller, and can be considered negligible
    # with an appropriately small time step.
    def get_velocity(self, a, v0, time_step):
        return a * time_step + v0

    # Create object method to return position based on acceleration, initial velocity, initial
    # position, and time step size. This method assumes that acceleration is constant and does 
    # not change as a result of the changing velocity and position. The error due to this 
    # assumption approaches 0 as the time step becomes smaller, and can be considered negligible
    # with an appropriately small time step.
    def get_position(self, a, v0, x0, time_step):
        return a*time_step**2 + v0*time_step + x0


#################################################################################################################


# Create object class for PID_Controller. Object contains the PID cofficients, a method for calculating
# output signal based on system state, a method for running a system simulation based on the controller signal
# and a method for autotuning the attributes to optimize controller performance for an input system. 
class PID_Controller:
    
    # Initialize controller attributes when object is created. kp, ki, and kd are the constants for the
    # proportional, integral, and derivative elements of the controller, respectively. Target is an input attribute
    # which establishes the goal the controller is trying to achieve. Target is not set at object intiliazation, but
    # is rather set as part of "running" the controller. Signal is effectively an output attribute which represents 
    # the "current" signal being emitted from the controller. Accumulator is a attribute used to execute the integral 
    # calculation, and last_reading is using to determine the slope for application of the derivative calculation.
    # Max_signal is used to limit the signal request that the controller can make, and the sample_rate establishes how
    # frequently the controller can adjust the signal. In a practical application this would equate to a 
    # maximum speed/torque/force of a motor or actuator element and the sampling rate of a digital controller device.

    def __init__(self, kp=1, ki=0, kd=0, max_signal=1000):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.target = 0
        self.signal = 0

        self.accumulator = 0
        self.last_reading = 0

        self.sample_rate = 0.1
        self.max_signal = max_signal
    
    # Set_new_target sets the target attribute and resets the held accumulator value. Resetting the accumulator
    # is necessary for the integral controller to function properly.
    def set_new_target(self, target):
        self.accumulator = 0
        self.target = target

    # Adjust_signal is the method that performs the actual function of the controller. This method calculates a 
    # new signal based on the feedback_value, accumulator value, last_reading, and target. It then sets the new
    # signal to the self.signal attribute.
    def adjust_signal(self, feedback_value):
        # Calculate the error - difference between target and feedback_value (measurement)
        error = self.target - feedback_value

        # Add error to accumulator
        self.accumulator += error

        # Calculate signal based on PID coefficients
        self.signal = self.kp * error + self.ki * self.accumulator + self.kd * (feedback_value - self.last_reading)/self.sample_rate
        
        # If caluclated signal exceeds max_signal, then set signal to max_signal value. Do this in both positive
        # and negative directions
        if self.signal > self.max_signal:
            self.signal = self.max_signal
        elif self.signal < -self.max_signal:
            self.signal = -self.max_signal

        # Save current feedback_value as last_reading attribute for use in calulating next signal.
        self.last_reading = feedback_value

    
    # Run_simulation is a controller method which will simulate the output of a system object "system_simulator"
    # over a time_period "duration."
    def run_simulation(self, system_simulator, duration, commands):
        # system_simulator - system object that will be controlled by the controller
        # duration - float, period over which to iterate system_function - this parameter is unit agnostic in execution of this
        #                   code. This method assumes that duration is of the same units as system attributes of the system_simulator
        # commands - list of times and command values to feed to the controller, each list entry is of the form [time, target]

        # Initialize index for cycling through the commands list
        command_idx = 0

        # Initialize x (position) and v (velocity) lists. Establish that initial position and velocity are both 0.
        x = [0]
        v = [0]

        # If the first command is at time 0, immediately set new target in controller, and increment command index
        if commands[0][0] == 0:
            self.set_new_target(commands[0][1])
            command_idx += 1
        
        # Initialize list of target values, this will be returned by the method for use in plotting the commands vs. time
        target_list = [self.target]

        # Adjust signal based upon the target at time = 0
        self.adjust_signal(x[0])
        signal_list = [self.signal]

        # Initialize a (acceleration) list and calculate the acceleration at time 0 based on controller signal. Within the
        # context of this model, the acceleration is considered to change instantaneously with the change in controller
        # signal.
        a = [system_simulator.get_acceleration(signal_list[0], v[0], x[0])]

        # Establish time step size for simulation. Time_step_size should always be smaller than self.sample_rate.
        time_step_size = 0.01

        # Build list of time increments based upon the time_step_size. This list will be returned by the method to be used as the 
        # x-axis when plotting signal, target, or position
        time_list = np.linspace(0,duration, int(np.ceil(duration/time_step_size)))
        
        # Starting with index 1, loop through every time position and calculate acceleration, velocity, and position
        for i in range(1, len(time_list)):

            # Verify that command_idx does not exceed length of commands input
            if command_idx <= len(commands)-1:
                # Check if current time step is in commands list, if so set the target to new command. Use <= in case floating
                # point error results in bypassing the command time.
                if commands[command_idx][0] <= round(time_list[i], 2):
                    self.set_new_target(commands[command_idx][1])
                    command_idx += 1

            # Add current target to target list
            target_list.append(self.target)

            # Calculate x[t] and v[t] based on current controller settings and acceleration from previous time step. x is calculated first
            # so that the last element in v is the initial velocity for the new value of x.
            x.append(system_simulator.get_position(a[-1], v[-1], x[-1], time_step_size))
            v.append(system_simulator.get_velocity(a[-1], v[-1], time_step_size))


            # Adjust signal based on new position of x
            if time_list[i] % self.sample_rate:
                self.adjust_signal(x[i])

            # Add signal to signal list
            signal_list.append(self.signal)

            # Use newly calculated x, v, and signal to determine new a - a changes instantaneously at the start of each interval based on signal,
            # x and v change at the conclusion of the interval based on a 
            a.append(system_simulator.get_acceleration(signal_list[-1], v[-1], x[-1]))


        return time_list, signal_list, x, target_list


    # get_error method returns the cumulative difference between each element in and target_list at the same index. This is used for evaluation
    # of PID coefficients in auto-tuning method.
    def get_error(self, x, target_list):
        return sum([abs(x[i] - target_list[i]) for i in range(len(x))])


    # The auto_tune method will automatically adjust the PID coefficients of the controller object to minimize the total error in response to
    # a step change in the target value. The min_input and max_input define minimum and maximum size of the step commands that the system is 
    # expected to receive. The method will calculate a midpoint between these values and optimize for minimum error across all 3 responses. 
    # This is to ensure that the tuning is balanced across the entire range of expected commands.
    #
    # This method works by iterating each coefficient forwards and backwards while also decreasing the size of the steps until the coefficient
    # has been determined to a pre-determined accuracy. The method will iterate forward at the initial step size until the error increases rather
    # then decreases. When this occurs the step size will be reduced by a factor of 10 and the method will reverse direction until the error
    # error increases rather than decreases. It will continue to zig-zag back and forth, decreasing the size of the steps until eventually the
    # step size is less than the predetermined minimum step size for that coefficient. It will then execute the same algorithm on the next 
    # coefficient. The coefficients are evaluated and set in the following order: kp, ki, kd.
    def auto_tune(self, system_simulator, min_input, max_input):
        
        # Set maximum iterations. Tuning will stop if this number of iterations is reached.
        max_iterations = 1000

        # Initialize kp at 1 - intializing at 0 will result in no response from the controller
        self.kp = 1

        # Reset ki and kd values to 0 for tuning
        self.ki = 0
        self.kd = 0

        # direction variable determines whether to iterate the current coefficient up or down. Initialize in the
        # positive (up) direction.
        direction = 1
        
        # Initialize current_error as None, the prevents an error when setting the "previous_error" variable on the first iteration of the loop.
        current_error = None

        # plot_current_iteration is used at the end of each loop to determine whether or not the current interation should be plotted. Initialize
        # as True to print initial loop at kp = 1.
        plot_current_iteration = True

        # Initialize x as list. Every element of x list will be a list of values output from run_simulation
        # for each input in check_inputs.
        x = []

        # Create list of target steps to check, these will be used as input steps for the error optimization.
        check_targets = [min_input, (min_input + max_input)/2, max_input]

        
        # Because of the different scales of kp, ki, and kd, it is important to track which is being optimized. This variable holds that information
        # as an integer which can be easily incremented. 0 = kp, 1 = ki, 2 = kd
        constant_to_optimize = 0

        # Initialize the step size and minimum step_size for kp
        step_size = 1000
        min_step_size = 10

        ######## Begin optimization loop. ########
        for i in range(max_iterations):
            
            # Unless this is the first loop, increase coefficient by step_size value, multiply by direction to appropriately increment vs. decrement,
            # round values to compensate for floating point errors. Change minimum step size to scale based upon size of coefficient.
            if i != 0:
                if constant_to_optimize == 0:
                    self.kp += step_size*direction
                    self.kp = round(self.kp)
                    if self.kp != 0:
                        min_step_size = abs(self.kp/10000)
                elif constant_to_optimize == 1:
                    self.ki += step_size*direction
                    self.ki = round(self.ki, 6)
                    if self.ki != 0:
                        min_step_size = abs(self.ki/10000)
                elif constant_to_optimize == 2:
                    self.kd += step_size*direction
                    self.kd = round(self.kd)
                    if self.kd != 0:
                        min_step_size = abs(self.kd/10000)

            # Hold error from last iteration as "previous_error", re-initialize current error as 0
            previous_error = current_error
            current_error = 0
            
            # Reset x to empty list
            x = []

            # Reset target_idx to 0
            target_idx = 0

            # Loop through targets and calculate total error of all targets
            for target in check_targets:
                # Run System Simulation, execute both step up and step down to balanace directional behavior as the directions
                # are not equivalent due to spring loading.
                time_list, signal_list, temp_x, target_list = self.run_simulation(system_simulator, 500, [[0,target], [250, 0]])
                
                # Append temp_x list to x. This will result in 3 lists of x values, one for each target. These are held
                # onto in order to be plotted at the end of the loop.
                x.append(temp_x)

                # calculate current error
                current_error += self.get_error(x[target_idx], target_list)

                target_idx += 1
                
            # Print auto-tuning information to terminal
            print('Iteration {}, Kp = {}, Ki = {}, Kd = {}, Error = {}'.format(i, self.kp, self.ki, self.kd, current_error))

            # If the plot_current_iteration flag is true, or if it is loop index 1, plot the system response with the current 
            # PID coefficients for each target.
            if plot_current_iteration or (i == 1):
                fig, axes = plt.subplots(len(check_targets), sharex=True, constrained_layout = True)

                fig.suptitle('Iteration = ' + str(i) + ', Kp = ' + str(self.kp) + ', Ki = ' + str(self.ki) + ', Kd = ' + str(self.kd))

                for ii in range(len(check_targets)):
                    #axes[ii].plot(time_list[list(time_list <= 100)], np.array(x[ii])[list(time_list <=100)], color="#296ca3")
                    axes[ii].plot(time_list, x[ii], color="#296ca3")
                    axes[ii].set_title('X Position, Step Size = ' + str(check_targets[ii]))
                
                plt.show(block=False)

                # Reset plot_current_iteration to false to prevent plotting every interation
                plot_current_iteration = False
                    

            # On first iteration through the loop, there is no previous_error to compare to, immediately return to top of loop.
            if i == 0:
                continue

            
            # If the error grew in the last iteration, reverse direction, and decrease the step size.
            if current_error >= previous_error:
                step_size /= 10
                direction *= -1
                # Plot the iterations on which the step size changes to visualize progress of the auto-tuner
                plot_current_iteration = True

            # If step_size is less than the minimum step size, then change which coefficient is being iterated, and reset 
            # step_size and min_step_size
            if step_size < min_step_size:
                # When the step size becomes lower than the minimum step size, the current coefficient setting will have
                # a larger error than the previous. Set current coefficient to previous value before changing
                # constant_to_optimize.
                if constant_to_optimize == 0:
                    self.kp += 10*step_size*direction
                    self.kp = round(self.kp)
                elif constant_to_optimize == 1:
                    self.ki += 10*step_size*direction
                    self.ki = round(self.ki, 3)
                elif constant_to_optimize == 2:
                    self.kd += 10 * step_size*direction
                    self.kd = round(self.kd)
                
                # Increment constant_to_optimize and change step_size, direction, and min_step_size based on the new constant.
                constant_to_optimize += 1
                plot_current_iteration = True
                if constant_to_optimize == 1:
                    step_size = 1
                    direction = 1
                    min_step_size = 0.001
                elif constant_to_optimize == 2:
                    step_size = 100000
                    direction = -1
                    min_step_size = 100     
            
            # If constant_to_optimize = 3, then all of the coefficients have been set and the loop can be stopped.
            # Plot final iteration, then break.
            if constant_to_optimize == 3:
                print('Final Values: Kp = {}, Ki = {}, Kd = {}, Error = {}'.format(i, self.kp, self.ki, self.kd, current_error))

                fig, axes = plt.subplots(len(check_targets), sharex=True, constrained_layout = True)

                fig.suptitle('Iteration = ' + str(i) + ', Kp = ' + str(self.kp) + ', Ki = ' + str(self.ki) + ', Kd = ' + str(self.kd))

                for ii in range(len(check_targets)):
                    #axes[ii].plot(time_list[list(time_list <= 100)], np.array(x[ii])[list(time_list <=100)], color="#296ca3")
                    axes[ii].plot(time_list, x[ii], color="#296ca3")
                    axes[ii].set_title('X Position, Step Size = ' + str(check_targets[ii]))
                
                plt.show(block=False)

                # Reset plot_current_iteration to false to prevent plotting every interation
                plot_current_iteration = False
                
                break

        return self


                

##############################################################################################################################################
# Begin main code

# Create Controller object
controller = PID_Controller()

# Create system object. Inputs are randomized, and scaled. Scaling factors were determined experimentally as a design space that generally aligns
# well with the test command set in command_list.
system_simulator = SMD_System_Sim(mass = np.random.rand()*100+50, k = np.random.rand()*5, c = np.random.rand()*10)

# Tune the control for step sizes of 1 to 50
controller = controller.auto_tune(system_simulator, 1, 50)

# Establish command list for test run of controller
command_list = [[0,50], [20,49], [40,25],[60,40],[80,15]]

# Simulate the controllers response to the command_list
time_list, signal_list, x, target_list = controller.run_simulation(system_simulator, 100, command_list)

# Plot target, response, and signal of test run
fig, (ax1, ax2) = plt.subplots(2, sharex=True, constrained_layout=True)

ax1.plot(time_list, target_list, color='#7c8994', label='Target')
ax1.plot(time_list, x, color='#296ca3', alpha=0.7, label='Position')

ax1.legend()
ax1.set_ylabel('Position')
ax1.set_title('Target and Position')

ax2.plot(time_list, signal_list, color='#296ca3')
ax2.set_ylabel('Signal')
ax2.set_title('Controller Signal, max = ' + str(controller.max_signal))

ax2.set_xlabel('Time')

fig.suptitle('Simulated System: Mass = {}, k = {},  c = {}'.format(round(system_simulator.mass, 2), round(system_simulator.k, 2), round(system_simulator.c,2)))

plt.show(block=True)

