param map = localPath('/home/anjali/carla_sim/models/Town01.xodr')
param carla_map = 'Town01'
#param weather =  'MidRainyNight'
#param weather = { 100.0, 100.0, 90.0, 100.0, -1.0, -90.0, 100.0, 0.75, 0.1, 100.0, 1.0, 0.03, 0.0331, 0.0}
#param weather = {'cloudiness':0}
param weather = {'sun_altitude_angle':90}
model scenic.simulators.carla.model

## CONSTANTS
EGO_MODEL = "vehicle.lincoln.mkz_2017"
EGO_SPEED = 10
EGO_BRAKING_THRESHOLD = 12

LEAD_CAR_SPEED = 10
LEADCAR_BRAKING_THRESHOLD = 10
LEAD_CAR_MODEL = "vehicle.nissan.micra"

BRAKE_ACTION = 1.0

## DEFINING BEHAVIORS
# EGO BEHAVIOR: Follow lane, and brake after passing a threshold distance to the leading car
behavior EgoBehavior(speed=10):
    try:
        do FollowLaneBehavior(speed)

    interrupt when withinDistanceToAnyCars(self, EGO_BRAKING_THRESHOLD):
        take SetBrakeAction(BRAKE_ACTION)

# LEAD CAR BEHAVIOR: Follow lane, and brake after passing a threshold distance to obstacle
behavior LeadingCarBehavior(speed=10):
    try: 
        do FollowLaneBehavior(speed)

    interrupt when withinDistanceToAnyObjs(self, LEADCAR_BRAKING_THRESHOLD):
        take SetBrakeAction(BRAKE_ACTION)

## DEFINING SPATIAL RELATIONS
# Please refer to scenic/domains/driving/roads.py how to access detailed road infrastructure
# 'network' is the 'class Network' object in roads.py

# make sure to put '*' to uniformly randomly select from all elements of the list, 'lanes'
lane = Uniform(*network.lanes)

obstacle = new Barrel on lane.centerline

leadCar = new Car following roadDirection from obstacle for Range(-50, -30),
        with blueprint LEAD_CAR_MODEL,
        with color (0,0,1),
        with behavior LeadingCarBehavior(LEAD_CAR_SPEED)
        

randomCar = new Car following roadDirection from obstacle for Range(-10, 10),
        with behavior LeadingCarBehavior(LEAD_CAR_SPEED)

ego = new Car following roadDirection from leadCar for Range(-15, -10),
        with blueprint EGO_MODEL,
        with behavior EgoBehavior(EGO_SPEED)


require (distance to intersection) > 80
terminate when ego.speed < 0.1 and (distance to obstacle) < 30