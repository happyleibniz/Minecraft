import platform
import ctypes
import math
import logging
import random
import time
import os

import pyglet

# Optimize Pyglet options upfront
# Set shadow_window to False if not needed for specific rendering techniques, reducing overhead.
pyglet.options["shadow_window"] = False
# Disable debug_gl for release builds to avoid performance overhead from GL error checking.
pyglet.options["debug_gl"] = False
# Ensure local libraries are searched efficiently.
pyglet.options["search_local_libs"] = True
# Specify audio drivers; consider reducing if not all are needed or if one is preferred.
pyglet.options["audio"] = ("openal", "pulse", "directsound", "xaudio2", "silent")

import pyglet.gl as gl
import player
import texture_manager

import world

import options
# Removed redundant time import, as it's already imported at the top.

import joystick
import keyboard_mouse
from collections import deque

# Use a named tuple or dataclass for config for potentially slightly better memory usage
# and clarity, though a class is fine too.
class InternalConfig:
    def __init__(self, options):
        self.RENDER_DISTANCE = options.RENDER_DISTANCE
        self.FOV = options.FOV
        self.INDIRECT_RENDERING = options.INDIRECT_RENDERING
        self.ADVANCED_OPENGL = options.ADVANCED_OPENGL
        self.CHUNK_UPDATES = options.CHUNK_UPDATES
        self.VSYNC = options.VSYNC
        self.MAX_CPU_AHEAD_FRAMES = options.MAX_CPU_AHEAD_FRAMES
        self.SMOOTH_FPS = options.SMOOTH_FPS
        self.SMOOTH_LIGHTING = options.SMOOTH_LIGHTING
        self.FANCY_TRANSLUCENCY = options.FANCY_TRANSLUCENCY
        self.MIPMAP_TYPE = options.MIPMAP_TYPE
        self.COLORED_LIGHTING = options.COLORED_LIGHTING
        self.ANTIALIASING = options.ANTIALIASING

class Window(pyglet.window.Window):
    def __init__(self, **args):
        # Pass all args to super() including width, height, caption, resizable, vsync, config.
        super().__init__(**args)

        self.options = InternalConfig(options)

        if self.options.INDIRECT_RENDERING and not gl.gl_info.have_version(4, 2):
            raise RuntimeError("""Indirect Rendering is not supported on your hardware
            This feature is only supported on OpenGL 4.2+, but your driver doesnt seem to support it,
            Please disable "INDIRECT_RENDERING" in options.py""")

        # F3 Debug Screen
        self.show_f3 = False
        # Pre-calculate common string parts for F3 debug screen.
        self.system_info = (
            f"Python: {platform.python_implementation()} {platform.python_version()}\n"
            f"System: {platform.machine()} {platform.system()} {platform.release()} {platform.version()}\n"
            f"CPU: {platform.processor()}\n"
            f"Display: {gl.gl_info.get_renderer()}\n"
            f"{gl.gl_info.get_version()}"
        )
        self.f3 = pyglet.text.Label("", x = 10, y = self.height - 10,
                font_size = 16,
                color = (255, 255, 255, 255),
                width = self.width // 3,
                multiline = True
        )
        logging.info(f"System Info: {self.system_info}")

        # Create textures
        logging.info("Creating Texture Array")
        self.texture_manager = texture_manager.TextureManager(16, 16, 256)

        # Create world
        self.world = world.World(None, self.texture_manager, self.options)

        # Player setup
        logging.info("Setting up player & camera")
        self.player = player.Player(self.world, self.width, self.height)
        self.world.player = self.player

        # Pyglet scheduling
        # Player interpolation can be tied to the main update loop if it's not time-critical,
        # or kept separate if it needs a consistent fast tick. Keeping it for now.
        pyglet.clock.schedule(self.player.update_interpolation)
        # Instead of fixed 1/60, let `update` be called as frequently as possible
        # within Pyglet's event loop, and handle delta_time internally.
        # This allows for more dynamic frame rates and avoids scheduling overhead.
        pyglet.clock.schedule(self.update)
        self.mouse_captured = False

        # Misc stuff
        self.holding = 50

        # OpenGL initial state setup
        # Perform these only once if possible, rather than every frame.
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        if self.options.ANTIALIASING:
            gl.glEnable(gl.GL_MULTISAMPLE)
            gl.glEnable(gl.GL_SAMPLE_ALPHA_TO_COVERAGE)
            gl.glSampleCoverage(0.5, gl.GL_TRUE) # Should be 0.5f, GL_TRUE is usually 1

        # Controls
        self.controls = [0, 0, 0]
        self.joystick_controller = joystick.Joystick_controller(self)
        self.keyboard_mouse = keyboard_mouse.Keyboard_Mouse(self)

        # Music
        logging.info("Loading audio")
        try:
            # Use glob.glob for potentially better performance with patterns
            # or os.walk if directory structure is complex.
            # For a flat directory, listdir + join is fine.
            audio_path = "audio/music"
            self.music = [
                pyglet.media.load(os.path.join(audio_path, file))
                for file in os.listdir(audio_path)
                if os.path.isfile(os.path.join(audio_path, file))
            ]
        except Exception as e:
            logging.error(f"Failed to load music: {e}")
            self.music = []

        self.media_player = pyglet.media.Player()
        self.media_player.volume = 0.5

        if self.music: # Simplified check for empty list
            self.media_player.queue(random.choice(self.music))
            self.media_player.play()
            self.media_player.standby = False
        else:
            self.media_player.standby = True

        self.media_player.next_time = 0

        # GPU command syncs
        self.fences = deque()

    def toggle_fullscreen(self):
        self.set_fullscreen(not self.fullscreen)

    def on_close(self):
        logging.info("Deleting media player")
        self.media_player.delete()
        # Ensure all fences are deleted to avoid resource leaks.
        while self.fences:
            fence = self.fences.popleft()
            gl.glDeleteSync(fence)
        super().on_close()

    def update_f3(self, delta_time):
        """Update the F3 debug screen content."""
        player_pos = self.player.position
        player_rounded_pos = self.player.rounded_position
        player_chunk_pos = world.get_chunk_position(player_pos)
        player_local_pos = world.get_local_position(player_pos)

        # Cache frequently accessed attributes
        world_chunks = self.world.chunks
        world_visible_chunks = self.world.visible_chunks

        chunk_count = len(world_chunks)
        visible_chunk_count = len(world_visible_chunks)

        # Use sum with generator expressions for efficiency.
        quad_count = sum(chunk.mesh_quad_count for chunk in world_chunks.values())
        visible_quad_count = sum(chunk.mesh_quad_count for chunk in world_visible_chunks)

        # Optimize string formatting by using f-strings and pre-calculating values.
        # Avoid redundant calculations within the f-string if possible.
        fps_info = f"{round(1 / delta_time)} FPS ({self.world.chunk_update_counter} Chunk Updates) "
        fps_info += "inf" if not self.options.VSYNC else "vsync"
        fps_info += "ao" if self.options.SMOOTH_LIGHTING else ""

        light_level_block = self.world.get_light(player_rounded_pos)
        light_level_sky = self.world.get_skylight(player_rounded_pos)
        max_light = max(light_level_block, light_level_sky)

        self.f3.text = \
f"""
{fps_info}
C: {visible_chunk_count} / {chunk_count} pC: {self.world.pending_chunk_update_count} pU: {len(self.world.chunk_building_queue)} aB: {chunk_count}
E: {self.world.visible_entities} / {len(self.world.entities)}
Client Singleplayer @{round(delta_time * 1000)} ms tick {round(1 / delta_time)} TPS

XYZ: ( X: {round(player_pos[0], 3)} / Y: {round(player_pos[1], 3)} / Z: {round(player_pos[2], 3)} )
Block: {player_rounded_pos[0]} {player_rounded_pos[1]} {player_rounded_pos[2]}
Chunk: {player_local_pos[0]} {player_local_pos[1]} {player_local_pos[2]} in {player_chunk_pos[0]} {player_chunk_pos[1]} {player_chunk_pos[2]}
Light: {max_light} ({light_level_sky} sky, {light_level_block} block)

{self.system_info}

Renderer: {"OpenGL 3.3 VAOs" if not self.options.INDIRECT_RENDERING else "OpenGL 4.0 VAOs Indirect"} {"Conditional" if self.options.ADVANCED_OPENGL else ""}
Buffers: {chunk_count}
Chunk Vertex Data: {round(quad_count * 28 * ctypes.sizeof(gl.GLfloat) / 1048576, 3)} MiB ({quad_count} Quads)
Chunk Visible Quads: {visible_quad_count}
Buffer Uploading: Direct (glBufferSubData)
"""

    def update(self, delta_time):
        """
        Main game update loop. Called by pyglet.clock.schedule.
        delta_time is provided by pyglet.
        """
        # Music playback logic
        if not self.media_player.source and self.music: # Check if music list is not empty
            if not self.media_player.standby:
                self.media_player.standby = True
                self.media_player.next_time = time.time() + random.randint(240, 360)
            elif time.time() >= self.media_player.next_time:
                self.media_player.standby = False
                self.media_player.queue(random.choice(self.music))
                self.media_player.play()

        # Input handling
        if not self.mouse_captured:
            self.player.input = [0, 0, 0]

        # Update game components
        self.joystick_controller.update_controller()
        self.player.update(delta_time)
        self.world.tick(delta_time)

        # Update other entities (iterate directly over self.world.entities)
        for entity in self.world.entities:
            entity.update(delta_time)

        # Update F3 debug screen last to include current frame's data
        if self.show_f3:
            self.update_f3(delta_time)

    def on_draw(self):
        """
        Pyglet's main drawing event.
        Optimized to reduce state changes and manage GPU synchronization.
        """
        # OpenGL state enabled once in __init__
        # gl.glEnable(gl.GL_DEPTH_TEST) # Already enabled in __init__

        # Handle CPU-GPU sync
        while len(self.fences) > self.options.MAX_CPU_AHEAD_FRAMES:
            fence = self.fences.popleft()
            # Use GL_DONT_WAIT if non-blocking wait is desired, but GL_SYNC_FLUSH_COMMANDS_BIT
            # combined with a timeout is usually better for controlled CPU-GPU pipeline.
            gl.glClientWaitSync(fence, gl.GL_SYNC_FLUSH_COMMANDS_BIT, 2147483647) # 2147483647 is GL_TIMEOUT_IGNORED_NV
            gl.glDeleteSync(fence)

        self.clear() # Clear color and depth buffers

        self.player.update_matrices() # Update player's view and projection matrices
        self.world.prepare_rendering() # Prepare world for rendering (e.g., culling)
        self.world.draw() # Draw the world

        # Draw the F3 Debug screen
        if self.show_f3:
            # f3.draw() handles its own GL state, so no need to push/pop matrix.
            self.f3.draw()

        # CPU - GPU Sync: Manage fences for smoother FPS or strict vsync.
        if not self.options.SMOOTH_FPS:
            # Pyglet 2 note: glFenceSync might still be missing or a workaround needed.
            # If it's still missing, this block might be a no-op or require custom ctypes.
            # Assuming it will be available or a future Pyglet version supports it.
            # This is critical for preventing CPU from getting too far ahead of GPU.
            # self.fences.append(gl.glFenceSync(gl.GL_SYNC_GPU_COMMANDS_COMPLETE, 0))
            pass # Keep as pass if glFenceSync is indeed unavailable for now.
        else:
            # glFinish is a strong sync, potentially causing stalls, but ensures smoothness.
            gl.glFinish()

    # Input functions are generally fine as they are callback-driven.
    def on_resize(self, width, height):
        logging.info(f"Resize {width} * {height}")
        gl.glViewport(0, 0, width, height)

        self.player.view_width = width
        self.player.view_height = height
        # Update f3 label's position and width correctly relative to new window size.
        self.f3.y = height - self.f3.content_height - 10 # Adjust y based on content height
        self.f3.width = width // 3

class Game:
    def __init__(self):
        # Configure OpenGL context for better compatibility and performance.
        # double_buffer = True for smoother animation.
        # major_version, minor_version for specific OpenGL context (e.g., 3.3 core profile).
        # depth_size for depth buffer precision.
        # sample_buffers and samples for Antialiasing (MSAA).
        self.config = gl.Config(
            double_buffer = True,
            major_version = 3,
            minor_version = 3,
            depth_size = 24, # Increased depth size for better precision, common in games.
            sample_buffers=1 if options.ANTIALIASING else 0, # 1 to enable, 0 to disable
            samples=options.ANTIALIASING # Number of samples if AA is enabled.
        )
        self.window = Window(
            config = self.config,
            width = 852,
            height = 480,
            caption = "Minecraft clone",
            resizable = True,
            vsync = options.VSYNC
        )

    def run(self):
        # pyglet.app.run(interval=0) schedules updates as fast as possible.
        # This is generally good for games as it allows the game to run at maximum FPS
        # and relies on delta_time for consistent game logic speed.
        pyglet.app.run(interval=0)

def init_logger():
    log_folder = "logs/"
    # Use a more descriptive and sortable timestamp for log filenames.
    log_filename = f"game_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_folder, log_filename)

    if not os.path.isdir(log_folder):
        os.makedirs(log_folder) # Use makedirs to create intermediate directories if needed.

    # Open in 'a' (append) mode if you want to avoid 'x' (exclusive creation) if it exists.
    # 'x' is fine if you truly want to ensure a new file.
    # No need to write "[LOGS]\n" explicitly, basicConfig handles it.
    # with open(log_path, 'x') as file:
    #     file.write("[LOGS]\n")

    logging.basicConfig(level=logging.INFO, filename=log_path,
        format="[%(asctime)s] [%(processName)s/%(threadName)s/%(levelname)s] (%(module)s.py/%(funcName)s) %(message)s")

def main():
    init_logger()
    game = Game()
    game.run()

if __name__ == "__main__":
    main()
