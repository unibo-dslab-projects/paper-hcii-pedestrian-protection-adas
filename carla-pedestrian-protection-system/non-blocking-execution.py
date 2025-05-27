


import carla
import cv2
import numpy as np
import pygame
import math
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import paho.mqtt.client as mqtt





import manual_control as mc
from importlib import reload
reload(mc)

steering_wheel = False


# import manual_control_steeringwheel as mc
# from importlib import reload
# reload(mc)

# steering_wheel = True



client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()
spectator = world.get_spectator()

def spawn_camera(attach_to=None, transform=carla.Transform(carla.Location(x=1.2, z=1.2), carla.Rotation(pitch=-10)), fov=90.0, width=800, height=600, sensor_tick=0.0):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(width))
    camera_bp.set_attribute('image_size_y', str(height))
    camera_bp.set_attribute('fov', str(fov))
    camera_bp.set_attribute('sensor_tick', str(sensor_tick))
    camera = world.spawn_actor(camera_bp, transform, attach_to=attach_to)
    return camera

def remove_all(world: carla.World):
    '''
    Remove all actors and sensors from the world.

    Args:
        world: the world to remove actors and sensors from
    '''
    for a in world.get_actors().filter('vehicle.*'):
        a.destroy()
    for a in world.get_actors().filter('sensor.*'):
        a.destroy()
    for a in world.get_actors().filter('walker.pedestrian.*'):
        a.destroy()
    for a in world.get_actors().filter('controller.ai.walker'):
        a.destroy()

def process_image(image, res):
    '''
    Process the input image, detect lanes, interpolate and draw the echidistant lane.
    If an error occurs, the original image with no relevations is put in the result queue.

    Args:
        image: the input image
        res: the queue to put the result
    '''
    res.put({'img': image, 'left_lane': None, 'right_lane': None})


class GameLoop(object):
    def __init__(self, args):
        self.args = args
        pygame.init()
        pygame.font.init()
        self.world = None
        self.original_settings = None
        self.fps = args.maxfps

        try:
            self.sim_world = client.get_world()
            if args.sync:
                self.original_settings = self.sim_world.get_settings()
                settings = self.sim_world.get_settings()
                if not settings.synchronous_mode:
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = 0.05
                self.sim_world.apply_settings(settings)

                traffic_manager = client.get_trafficmanager()
                traffic_manager.set_synchronous_mode(True)

            if not self.sim_world.get_settings().synchronous_mode:
                print('WARNING: You are currently in asynchronous mode and could '
                    'experience some issues with the traffic simulation')

            self.display = pygame.display.set_mode(
                (args.width, args.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.display.fill((0,0,0))
            pygame.display.flip()

            hud = mc.HUD(args.width, args.height)
            self.world = mc.World(self.sim_world, hud, args)
            self.controller = None
            if steering_wheel:
                self.controller = mc.DualControl(self.world)
            else:
                self.controller = mc.KeyboardControl()

            if args.sync:
                self.sim_world.tick()
            else:
                self.sim_world.wait_for_tick()
        except Exception:
            mc.logging.exception('Error creating the world')

    def get_speed(self, vehicle: carla.Vehicle):
        '''
        Get the speed of the vehicle.

        Args:
            vehicle: the vehicle to get the speed from

        Returns:
            the speed of the vehicle
        '''
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return speed

    def render(self, clock: pygame.time.Clock):
        '''
        Render the world and make the simulation tick.

        Args:
            clock: the clock to control the frame rate
        '''
        self.world.tick(clock)
        self.world.render(self.display)
        pygame.display.flip()

    def start(self, processed_output, autopilot=False, detection_center=100, threshold=20):
        '''
        Starts the application loop.

        Args:
            processed_output: the queue to get the processed image
            autopilot: if True, the player will be controlled by the autopilot
            detection_center: the center of the detection area
            threshold: the threshold to detect if the player is going to cross the lane
        '''
        self.world.player.set_autopilot(autopilot)
        try:
            clock = pygame.time.Clock()
            while True:
                if self.args.sync:
                    self.sim_world.tick()
                clock.tick_busy_loop(self.fps)

                if self.controller.parse_events(self.world, clock):
                    return

                # Show processed camera output
                try:
                    # output = processed_output.get_nowait()
                    output = processed_output.get()


                    cv2.imshow('Processed image', output['img'])
                except Exception as e:
                    pass

                self.render(clock)

                cv2.waitKey(1)
        finally:

            if self.original_settings:
                self.sim_world.apply_settings(self.original_settings)

            if self.world is not None:
                self.world.destroy()

            pygame.quit()




def setup():
    argparser = mc.argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1", "2", "All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    argparser.add_argument(
        '--maxfps',
        default=30,
        type=int,
        help='Fps of the client (default: 30)')
    args = argparser.parse_args()

    # Set max fps
    # args.maxfps = 60

    # Set window resolution
    args.res = '1280x720'
    args.width, args.height = [int(x) for x in args.res.split('x')]

    # Set vehicle filter
    args.filter = 'vehicle.mercedes.coupe_2020'

    # Set synchronous mode
    args.sync = True

    log_level = mc.logging.DEBUG if args.debug else mc.logging.INFO
    mc.logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    mc.logging.info('listening to server %s:%s', args.host, args.port)

    print(mc.__doc__)

    return GameLoop(args)




import random
import time
import threading

def spawn_walker(world: carla.World):
    """
    Spawns a pedestrian walker in the given CARLA world and makes it walk to a random destination.
    Args:
        world (carla.World): The CARLA world object where the walker will be spawned.
    Returns:
        tuple: A tuple containing the walker actor and its controller actor.
               If spawning fails, returns (None, None).
    """
    blueprint_library = world.get_blueprint_library()

    walker_blueprints = list(blueprint_library.filter('walker.pedestrian.*'))

    if not walker_blueprints:
        print("No pedestrian blueprints available.")
        return None, None

    walker_bp = random.choice(walker_blueprints)

    if walker_bp.has_attribute("speed"):
        speed = random.uniform(0.8, 2.0)
        walker_bp.set_attribute('speed', str(speed))
    else:
        speed = 1.0

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("Warning: No spawn points available for pedestrians. Cannot spawn walker.")
        return None, None

    spawn_point = random.choice(spawn_points)

    walker = world.try_spawn_actor(walker_bp, spawn_point)
    if walker is None:
        print("Failed to spawn walker at the chosen spawn point.")
        return None, None

    controller_bp = blueprint_library.find('controller.ai.walker')
    if controller_bp is None:
        print("No walker controller blueprint found.")
        return walker, None

    controller = world.try_spawn_actor(controller_bp, carla.Transform(), walker)
    if controller:
        controller.start()
        # Make the walker walk randomly by continuously assigning new random destinations
        def random_walk():
            while controller.is_alive and walker.is_alive:
                destination = world.get_random_location_from_navigation()
                if destination:
                    controller.go_to_location(destination)
                controller.set_max_speed(speed)
                # Wait for a random time before assigning a new destination
                time.sleep(random.uniform(5, 15))
        threading.Thread(target=random_walk, daemon=True).start()

    return walker, controller


walkers = []
controllers = []
for _ in range(100):
    walker, controller = spawn_walker(world)
    if walker and controller:
        walkers.append(walker)
        controllers.append(controller)











camera_width = 800
camera_height = 600

crop_height = 240
height_adjust = 50


processed_output = Queue()

# setup the simulation environment
game_loop = setup()

# get the vehicle and attach the camera
vehicle = world.get_actors().filter('vehicle.*')[0]
front_camera = spawn_camera(attach_to=vehicle, transform=carla.Transform(
        carla.Location(x=0.3, y=0.0, z=1.5), # posizione dello specchietto retrovisore
        carla.Rotation(pitch=-10.0)
    ),
    # sensor_tick=0.1,
    width=camera_width, height=camera_height
)

cv2.namedWindow('Processed image', cv2.WINDOW_NORMAL)

# callback for the camera
def camera_callback(image):
    '''
    Callback for the camera.

    Args:
        image: the image captured by the camera
    '''
    try:
        video_output = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        video_output = video_output[:, :, :3]

        # Crop the image (zoom)
        # start_x = (video_output.shape[1] - crop_width) // 2
        # start_y = (video_output.shape[0] - crop_height) // 2 - height_adjust
        # cropped_img = video_output[start_y:start_y + crop_height, start_x:start_x + crop_width]

        # process the image in a separate thread
        with ThreadPoolExecutor() as executor:
            executor.submit(lambda: process_image(video_output, processed_output))

    except Exception as e:
        print(e.with_traceback())

# attach the callback to the camera
front_camera.listen(lambda image: camera_callback(image))

def cleanup():
    remove_all(world)

# start the game loop
try:
    game_loop.start(processed_output, autopilot=False, detection_center=100, threshold=7)
except KeyboardInterrupt:
    cleanup()
finally:
    cleanup()