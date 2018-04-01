from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter8
import rospy
import math
import csv

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
M = 0.09949492
B = -0.05454824

class Controller(object):
    def __init__(self, dbw_enabled,
                vehicle_mass, fuel_capacity,
                brake_deadband, decel_limit,
                accel_limit, wheel_radius,
                wheel_base, steer_ratio,
                max_lat_accel, max_steer_angle):
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = -decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        #self.wheel_base = wheel_base
        #self.steer_ratio = steer_ratio
        #self.max_lat_accel = max_lat_accel
        #self.max_steer_angle = max_steer_angle

        min_speed = 0.0
        self.yaw_controller = YawController(wheel_base, steer_ratio,
                                            min_speed, max_lat_accel,
                                            max_steer_angle)
        kp = 0.5
        ki = 1.0
        kd = 0.0
        self.pid_throttle = PID(kp, ki, kd, mn=0, mx=accel_limit)

        self.lowpass_a = LowPassFilter8(1., 1., 1., 1., 1., 1., 1., 1.)
        self.lowpass_clv = LowPassFilter8(1., 1., 1., 1., 1., 1., 1., 1.)

    def control(self, dbw_enabled, goal_linear_v, goal_angular_v, stop_a, current_linear_v, dt):
        if not dbw_enabled:
            self.pid_throttle.reset()

            self.lowpass_a.reset()
            self.lowpass_clv.reset()
            return 0., 0., 0.

        if stop_a > self.brake_deadband:
            current_linear_v = self.lowpass_clv.filt(current_linear_v)
            stop_a = self.lowpass_a.filt(stop_a)
            a_add_on = self.accel_add_on(current_linear_v)
            stop_a -= a_add_on
            brake = max(0, stop_a)*self.vehicle_mass*self.wheel_radius
            throttle = 0
        else:
            self.lowpass_a.reset()
            self.lowpass_clv.reset()
            brake = 0
            error = goal_linear_v - current_linear_v
            throttle = self.pid_throttle.step(error, dt)

        if goal_linear_v < 0.01 and current_linear_v < 0.01:
            steer = 0

        else:
            steer = self.yaw_controller.get_steering(goal_linear_v,
                                                     goal_angular_v,
                                                     current_linear_v)

        return throttle, brake, steer


    def accel_add_on(self, current_linear_v):
        # M and B were calculated by fitting a polynomial to data collected
        # by speeding ego up to high speed and then tracking delta v and dt
        # when applying no throttle or brake. On flat ground, the car in the simulator
        # decelerates linearly with respect to speed with no throttle or speed.
        return M*current_linear_v + B
