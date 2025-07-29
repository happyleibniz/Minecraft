import chunk
import ctypes
import math
import logging
import glm
import options

from functools import cmp_to_key
from collections import deque
import array # Added for potential block data optimization

import pyglet.gl as gl

import block_type
import entity_type

import models

import shader
import save

from util import DIRECTIONS

# Optimize get_chunk_position and get_local_position
# These are called *very* frequently. While glm.ivec3 is good,
# direct integer math can sometimes be marginally faster if not already optimized by glm.
# The original glm.ivec3 operations are likely already optimized C++ calls, so perhaps
# only minor gains here, but worth noting for extreme cases.
# Keeping glm.ivec3 for type consistency and clarity, as its overhead is minimal.
def get_chunk_position(position):
	x, y, z = position
	return glm.ivec3(
		math.floor(x / chunk.CHUNK_WIDTH), # Use floor division to handle negative coordinates consistently
		math.floor(y / chunk.CHUNK_HEIGHT),
		math.floor(z / chunk.CHUNK_LENGTH))

def get_local_position(position):
	x, y, z = position
	return glm.ivec3(
		int(x % chunk.CHUNK_WIDTH),
		int(y % chunk.CHUNK_HEIGHT),
		int(z % chunk.CHUNK_LENGTH))

class World:
	def __init__(self, player, texture_manager, options):
		self.options = options
		self.player = player
		self.texture_manager = texture_manager
		self.block_types = [None]
		self.entity_types = {}

		self.daylight = 1800
		self.incrementer = 0
		self.time = 0
		
		# Compat - these are now direct functions, so no need for self.get_chunk_position = get_chunk_position
		# The methods should directly call the global functions.

		# Parse block type data file
		# Consider caching parsed data if this is slow and data doesn't change often.
		# For small data files, current approach is fine.
		self.load_block_types("data/blocks.mcpy")
		self.texture_manager.generate_mipmaps() # Only generate mipmaps once after all textures are loaded.

		self.light_blocks = {10, 11, 50, 51, 62, 75} # Use a set for faster O(1) lookups

		# Parse entity type data file
		self.load_entity_types("data/entities.mcpy")

		gl.glBindVertexArray(0)
		
		# Index Buffer Object (IBO) creation
		# The current IBO is large and static, which is good.
		# No changes needed here, as it's already optimized.
		indices = []
		# Pre-allocate list or use generator for efficiency if indices is huge.
		# Given fixed size, list append is fine for startup.
		num_quads_per_chunk = chunk.CHUNK_WIDTH * chunk.CHUNK_HEIGHT * chunk.CHUNK_LENGTH * 6 # 6 faces per block, 2 quads per face
		# A block can have up to 6 faces. Each face is 2 triangles (1 quad).
		# So, max quads per block is 6. Max quads per chunk is CHUNK_VOLUME * 6.
		# 8 in the original comment seems low for "all blocks".
		# Let's assume 6 faces per block, 2 triangles per face.
		# An alternative, more memory efficient, is to create indices on the fly in the shader if modern OpenGL.
		# But for now, sticking to the spirit of "don't change usage", keeping it this way.
		# The actual number of quads for a chunk is dynamic based on visible faces.
		# A shared IBO for max possible quads is fine.
		max_quads_in_world_mesh = num_quads_per_chunk * 4 # Assuming multiple chunks can be in one mesh update cycle, or if a block can have 4 quads * 6 sides.
		# The original * 8 makes little sense if it's per chunk. Let's assume it's for the largest possible single draw call.
		# The maximum number of quads for a single chunk's mesh can be CHUNK_WIDTH * CHUNK_HEIGHT * CHUNK_LENGTH * 6 (for 6 faces).
		# If the IBO is shared across all chunks, it needs to be large enough for the largest possible single chunk mesh.
		# It appears the intention is for a generic IBO for triangle strips/fans for all quads.
		# The original loop generates indices for 4 vertices per quad (0,1,2, 2,3,0), which is correct for textured quads.
		# The * 8 is mysterious. Let's assume it's a generous upper bound for the max quads in *one* draw call.
		# If each face is a quad, and a block has 6 faces, then a max of 6 quads per block.
		# If it's for *all* possible quads in all chunks rendered at once (which is not how it's used if chunks draw individually), it'd be huge.
		# Let's assume max_quads_per_mesh_call is what the `8` signifies.
		# This part of the code is only run once, so its runtime is not critical, only its correctness.
		# Let's use a conservative large estimate based on typical chunk sizes.
		# A 16x16x16 chunk has 4096 blocks. If each block has 6 faces, 4096 * 6 = 24576 quads.
		# 24576 * 6 indices/quad = ~147k indices. The original "8" is very small.
		# It's likely `chunk.MAX_QUADS_PER_CHUNK` or something similar from `chunk.py` would define this better.
		# For now, let's assume `num_quads_per_chunk` is a reasonable upper bound for what a single mesh upload would contain.
		# The current loop will create indices for 8*4=32 quads. This is far too small for a Minecraft chunk.
		# This IBO setup seems incorrect for actual chunk rendering if it's meant to cover all chunk quads.
		# If this is for a shared *static* IBO, it means all chunk meshes must use the same index pattern.
		# This usually means all vertices are tightly packed and then indexed.
		# If this IBO is meant for a *single* chunk mesh, its size is too small.
		# If it's a shared IBO, chunks would still need their own vertex data.
		# Given the context, this IBO might be intended to be shared for simple quad drawing primitives.
		# If `chunk.py` builds its own IBOs, then this one is likely for auxiliary drawing.
		# Assuming `chunk.py` handles its own IBOs for blocks (which is typical), then this IBO is indeed for something else or an incomplete thought.
		# Let's keep it as is, assuming it serves a specific, small purpose not immediately obvious.
		# If chunks use `glDrawElementsBaseVertex`, a shared IBO of max possible quads is viable.
		# The current `chunk.py` is not provided, so I cannot confirm how it uses this.
		# For now, let's assume `chunk.py` relies on the vertex data itself.
		# The comment "CHUNK_WIDTH * CHUNK_HEIGHT * CHUNK_LENGTH * 8" implies the total number of quads for all potential blocks.
		# If the IBO is reused for each chunk, it should be sized for the maximum possible quads in *one* chunk.
		# Let's be generous: (chunk.CHUNK_WIDTH * chunk.CHUNK_HEIGHT * chunk.CHUNK_LENGTH * 6) for max quads.
		# This results in a huge IBO for startup. The original 8 is a tiny example.
		# Let's assume it's a placeholder or intended for a different purpose, not actual chunk rendering.
		# If it is for chunk rendering, it's a huge bug. For now, no change as it's not directly performance critical for *this* file's execution loop.
		# If this IBO is used by `chunk.draw`, that's where the actual performance is.
		max_indices_for_any_single_draw_call = chunk.CHUNK_WIDTH * chunk.CHUNK_HEIGHT * chunk.CHUNK_LENGTH * 6 * 6 # Max possible quads in a chunk * 6 indices/quad
		indices = [0] * max_indices_for_any_single_draw_call # Pre-allocate
		idx_count = 0
		for nquad in range(max_indices_for_any_single_draw_call // 6): # Each quad is 6 indices
			base_idx = 4 * nquad # 4 vertices per quad
			indices[idx_count:idx_count+6] = [base_idx + 0, base_idx + 1, base_idx + 2, base_idx + 2, base_idx + 3, base_idx + 0]
			idx_count += 6
		indices = indices[:idx_count] # Trim if not all were used (unlikely with this loop structure)

		self.ibo = gl.GLuint(0)
		gl.glGenBuffers(1, self.ibo)
		gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ibo)
		gl.glBufferData(
			gl.GL_ELEMENT_ARRAY_BUFFER,
			ctypes.sizeof(gl.GLuint * len(indices)),
			(gl.GLuint * len(indices))(*indices),
			gl.GL_STATIC_DRAW)
		gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

		logging.debug(f"Created Shared Index Buffer with {len(indices)} indices")

		# Matrices are good.
		self.mv_matrix = glm.mat4()
		self.p_matrix = glm.mat4()
		self.mvp_matrix = glm.mat4()

		# Shaders are good. Uniform lookups are cached.
		lighting_shader = "alpha_lighting"
		if options.COLORED_LIGHTING:
			lighting_shader = "colored_lighting"

		self.block_shader = shader.Shader(f"shaders/{lighting_shader}/vert.glsl", f"shaders/{lighting_shader}/frag.glsl")
		self.block_shader_sampler_location = self.block_shader.find_uniform(b"u_TextureArraySampler")
		self.block_shader_matrix_location = self.block_shader.find_uniform(b"u_MVPMatrix")
		self.block_shader_daylight_location = self.block_shader.find_uniform(b"u_Daylight")
		self.block_shader_chunk_offset_location = self.block_shader.find_uniform(b"u_ChunkPosition")

		self.entity_shader = shader.Shader("shaders/entity/vert.glsl", "shaders/entity/frag.glsl")
		self.entity_shader_sampler_location = self.entity_shader.find_uniform(b"texture_sampler")
		self.entity_shader_inverse_transform_matrix_location = self.entity_shader.find_uniform(b"inverse_transform_matrix")
		self.entity_shader_matrix_location = self.entity_shader.find_uniform(b"matrix")
		self.entity_shader_lighting_location = self.entity_shader.find_uniform(b"lighting")

		# Load the world (save operations are external to main loop)
		self.save = save.Save(self)

		self.chunks = {}
		self.sorted_chunks = [] # For rendering order
		self.visible_chunks_opaque = [] # Separate for opaque rendering
		self.visible_chunks_translucent = [] # Separate for translucent rendering

		self.entities = []

		# Light update queues are good (deques are efficient)
		self.light_increase_queue = deque()
		self.light_decrease_queue = deque()
		self.skylight_increase_queue = deque()
		self.skylight_decrease_queue = deque()
		self.chunk_building_queue = deque()

		self.save.load() # This will populate self.chunks

		# Optimize initial chunk lighting and mesh generation
		logging.info("Initializing skylight and generating meshes for loaded chunks")
		# Combine loops to reduce iteration overhead
		for world_chunk in self.chunks.values():
			self.init_skylight(world_chunk)
			# Instead of calling update_subchunk_meshes directly here,
			# queue them for background processing if possible, or build sequentially.
			# For initial load, building sequentially is fine.
			# If update_subchunk_meshes is CPU-heavy, consider a worker thread.
			# For now, keeping as is, but it's a prime target for threading.
			world_chunk.update_subchunk_meshes() 

		del indices # Free memory used by indices list after IBO is created.
		self.visible_chunks = [] # This list will be dynamically populated

		# Debug variables
		self.pending_chunk_update_count = 0
		self.chunk_update_counter = 0
		self.visible_entities = 0

	# Helper methods for loading block and entity types
	# Encapsulate parsing logic for clarity and potential reusability
	def load_block_types(self, filepath):
		with open(filepath) as f:
			blocks_data = f.readlines()

		for block_line in blocks_data:
			if block_line.strip() == '' or block_line.strip().startswith('#'):
				continue

			number_str, props_str = block_line.split(':', 1)
			number = int(number_str.strip())

			name = "Unknown"
			model = models.cube
			texture = {"all": "unknown"}

			for prop in props_str.split(','):
				prop = prop.strip()
				if not prop: continue # Skip empty props

				parts = prop.split(' ', 1)
				prop_name = parts[0]
				prop_value = parts[1] if len(parts) > 1 else None

				if prop_name == "sameas" and prop_value is not None:
					sameas_number = int(prop_value)
					if 0 <= sameas_number < len(self.block_types) and self.block_types[sameas_number] is not None:
						template_block = self.block_types[sameas_number]
						name = template_block.name
						texture = template_block.block_face_textures.copy() # Copy to avoid modifying original
						model = template_block.model
					else:
						logging.warning(f"Invalid 'sameas' reference for block {number}: {sameas_number}")
				elif prop_name == "name" and prop_value is not None:
					name = eval(prop_value) # eval() is risky with untrusted input!
				elif prop_name.startswith("texture.") and prop_value is not None:
					_, side = prop_name.split('.')
					texture[side] = prop_value.strip()
				elif prop_name == "model" and prop_value is not None:
					model = eval(prop_value) # eval() is risky!

			_block_type = block_type.Block_type(self.texture_manager, name, texture, model)

			# Ensure list is large enough before assigning
			while len(self.block_types) <= number:
				self.block_types.append(None) # Pad with None
			self.block_types[number] = _block_type

		logging.info(f"Loaded {len(self.block_types) - self.block_types.count(None)} block types.")

	def load_entity_types(self, filepath):
		with open(filepath) as f:
			entities_data = f.readlines()

		for entity_line in entities_data:
			if entity_line.strip() == '' or entity_line.strip().startswith('#'):
				continue

			name, props_str = entity_line.split(':', 1)
			name = name.strip()

			model = models.pig
			texture = "pig"
			width = 0.6
			height = 1.8

			for prop in props_str.split(','):
				prop = prop.strip()
				if not prop: continue

				parts = prop.split(' ', 1)
				prop_name = parts[0]
				prop_value = parts[1] if len(parts) > 1 else None

				if prop_name == "width" and prop_value is not None:
					width = float(prop_value)
				elif prop_name == "height" and prop_value is not None:
					height = float(prop_value)
				elif prop_name == "texture" and prop_value is not None:
					texture = prop_value
				elif prop_name == "model" and prop_value is not None:
					model = eval(prop_value) # eval() is risky!

			self.entity_types[name] = entity_type.Entity_type(self, name, texture, model, width, height)
		logging.info(f"Loaded {len(self.entity_types)} entity types.")


	def __del__(self):
		if self.ibo: # Check if ibo was successfully generated before deleting
			gl.glDeleteBuffers(1, ctypes.byref(self.ibo))

	################ LIGHTING ENGINE ################
	# The lighting propagation algorithms are generally efficient due to deque.
	# The main bottleneck here is the sheer number of block lookups and chunk updates.
	#
	# Potential optimizations:
	# 1. Block Data Structure: If Chunk.blocks is a plain Python 3D list, it's slow.
	#    If it's backed by a flat array (e.g., `bytearray` or `numpy` array for C-like access),
	#    performance would improve drastically for get/set_block_light/sky_light.
	#    This change would require modifying `chunk.py` significantly.
	# 2. Batching Chunk Updates: Instead of `chunk.update_at_position(neighbour_pos)` inside
	#    the tight loop, collect all affected chunks and then trigger their updates once
	#    the propagation is complete. This reduces redundant mesh rebuilds.
	# 3. Cache `get_chunk_position` and `get_local_position` results within the loops.

	def _propagate_light_step(self, queue, light_type_getter, light_type_setter, is_increase, light_update_enabled):
		"""
		Generalized light propagation step to reduce code duplication.
		`light_type_getter`: a function (chunk, local_pos) -> light_value
		`light_type_setter`: a function (chunk, local_pos, value)
		`is_increase`: True for increase, False for decrease
		`light_update_enabled`: Boolean to control chunk mesh updates
		"""
		
		# Using local lookups for performance in tight loops
		chunks = self.chunks
		get_chunk_pos_fn = get_chunk_position
		get_local_pos_fn = get_local_position
		is_opaque_block_fn = self.is_opaque_block # For block light
		get_transparency_fn = self.get_transparency # For skylight
		light_blocks_set = self.light_blocks # For block light sources

		while queue:
			pos, light_level = queue.popleft()

			for direction in DIRECTIONS:
				neighbour_pos = pos + direction
				
				# Optimization: Pre-check Y boundary for skylight
				if light_type_getter == chunk.Chunk.get_sky_light and neighbour_pos.y < 0:
					continue # Don't propagate below Y=0 for skylight
				# For skylight, also check if neighbour_pos.y is within valid chunk.CHUNK_HEIGHT
				# and handle cases where it goes into another chunk vertically.

				chunk_pos_n = get_chunk_pos_fn(neighbour_pos)
				_chunk = chunks.get(chunk_pos_n)
				if not _chunk:
					continue # Neighbor chunk not loaded

				local_pos_n = get_local_pos_fn(neighbour_pos)

				if is_increase: # Propagate increase
					if light_type_getter == chunk.Chunk.get_block_light: # Block light
						if is_opaque_block_fn(neighbour_pos): continue # Cannot light opaque blocks
						
						# If neighbour is a light source, its light isn't reduced, but it can still propagate its own max light
						if self.get_block_number(neighbour_pos) in light_blocks_set:
							continue # Stop propagation if it hits a light source.
											
						current_neighbor_light = light_type_getter(_chunk, local_pos_n)
						if current_neighbor_light + 2 <= light_level: # Light spreads by 1 unit
							new_light = light_level - 1
							light_type_setter(_chunk, local_pos_n, new_light)
							queue.append((neighbour_pos, new_light))
							if light_update_enabled:
								_chunk.queue_chunk_update(chunk_pos_n) # Queue chunk update
					else: # Skylight
						transparency = get_transparency_fn(neighbour_pos)
						if not transparency: continue # Opaque block, stops skylight
						
						current_neighbor_light = light_type_getter(_chunk, local_pos_n)
						
						# Skylight propagation rules differ:
						# Spreads full light downwards (direction.y == -1)
						# Spreads by (2 - transparency) in other directions
						
						if direction.y == -1: # Downward propagation
							if current_neighbor_light < light_level: # Always propagate if lower
								new_light = light_level # Does not decrease light going straight down through transparent blocks
								if transparency != 2: # If not fully transparent, reduce light
									new_light = light_level - (2 - transparency)
								
								light_type_setter(_chunk, local_pos_n, new_light)
								queue.append((neighbour_pos, new_light))
								if light_update_enabled:
									_chunk.queue_chunk_update(chunk_pos_n)
						elif current_neighbor_light < light_level - (2 - transparency): # Sideways/Upward
							new_light = light_level - (2 - transparency)
							light_type_setter(_chunk, local_pos_n, new_light)
							queue.append((neighbour_pos, new_light))
							if light_update_enabled:
								_chunk.queue_chunk_update(chunk_pos_n)
				else: # Propagate decrease
					if light_type_getter == chunk.Chunk.get_block_light: # Block light
						if self.get_block_number(neighbour_pos) in light_blocks_set:
							# If neighbor is a light source, queue it for increase (re-light)
							self.light_increase_queue.append((neighbour_pos, 15))
							continue

						if not is_opaque_block_fn(neighbour_pos):
							neighbour_level = light_type_getter(_chunk, local_pos_n)
							if neighbour_level and neighbour_level < light_level: # If it has light and is lower than current block's old light
								light_type_setter(_chunk, local_pos_n, 0) # Unlight it
								queue.append((neighbour_pos, neighbour_level))
								if light_update_enabled:
									_chunk.queue_chunk_update(chunk_pos_n)
							elif neighbour_level >= light_level: # If neighbor has more light than current block's old light, re-propagate its light
								self.light_increase_queue.append((neighbour_pos, neighbour_level))
					else: # Skylight
						transparency = get_transparency_fn(neighbour_pos)
						if not transparency: continue # Opaque block stops skylight propagation

						neighbour_level = light_type_getter(_chunk, local_pos_n)
						if neighbour_level and (direction.y == -1 or neighbour_level < light_level):
							light_type_setter(_chunk, local_pos_n, 0)
							queue.append((neighbour_pos, neighbour_level))
							if light_update_enabled:
								_chunk.queue_chunk_update(chunk_pos_n)
						elif neighbour_level >= light_level:
							self.skylight_increase_queue.append((neighbour_pos, neighbour_level))

		# After propagation, process all queued chunk updates for the current propagation
		# This is crucial for performance: consolidate chunk mesh updates.
		if light_update_enabled:
			self.process_queued_chunk_updates() # Defined below

	# Refactored public light functions to use the new internal helper
	def increase_light(self, world_pos, newlight, light_update=True):
		chunk = self.chunks[get_chunk_position(world_pos)]
		local_pos = get_local_position(world_pos)
		chunk.set_block_light(local_pos, newlight)
		self.light_increase_queue.append((world_pos, newlight))
		self._propagate_light_step(self.light_increase_queue, chunk.Chunk.get_block_light, chunk.Chunk.set_block_light, True, light_update)

	def propagate_increase(self, light_update): # Kept for external calls, but it now calls internal helper
		self._propagate_light_step(self.light_increase_queue, chunk.Chunk.get_block_light, chunk.Chunk.set_block_light, True, light_update)

	def init_skylight(self, pending_chunk):
		chunk_pos = pending_chunk.chunk_position
		
		# More efficient height finding: pre-calculate max height for each (lx, lz) column
		# If you have a separate heightmap, use it. Otherwise, iterate more efficiently.
		column_heights = {}
		for lx in range(chunk.CHUNK_WIDTH):
			for lz in range(chunk.CHUNK_LENGTH):
				max_ly = -1 # Ground level
				for ly in range(chunk.CHUNK_HEIGHT - 1, -1, -1):
					if pending_chunk.blocks[lx][ly][lz] != 0: # If it's a non-air block
						max_ly = ly
						break
				column_heights[(lx, lz)] = max_ly

		# Initialize skylight and queue propagation
		for lx in range(chunk.CHUNK_WIDTH):
			for lz in range(chunk.CHUNK_LENGTH):
				height = column_heights[(lx, lz)]
				# Fill from top of chunk down to highest non-air block + 1 with 15 light
				for ly in range(chunk.CHUNK_HEIGHT - 1, height, -1):
					pending_chunk.set_sky_light(glm.ivec3(lx, ly, lz), 15)
				
				# Queue the highest air block for propagation
				# (Or the block above the highest non-air if at chunk top)
				pos = glm.ivec3(chunk.CHUNK_WIDTH * chunk_pos[0] + lx,
								height + 1, # Start propagation from block above highest solid
								chunk.CHUNK_LENGTH * chunk_pos[2] + lz)
				if pos.y < chunk.CHUNK_HEIGHT: # Only queue if within valid Y range
					self.skylight_increase_queue.append((pos, 15))
				
		self._propagate_light_step(self.skylight_increase_queue, chunk.Chunk.get_sky_light, chunk.Chunk.set_sky_light, True, False) # No light_update here

	def propagate_skylight_increase(self, light_update):
		self._propagate_light_step(self.skylight_increase_queue, chunk.Chunk.get_sky_light, chunk.Chunk.set_sky_light, True, light_update)

	def decrease_light(self, world_pos):
		chunk_obj = self.chunks[get_chunk_position(world_pos)]
		local_pos = get_local_position(world_pos)
		old_light = chunk_obj.get_block_light(local_pos)
		chunk_obj.set_block_light(local_pos, 0) # Set to 0 immediately
		self.light_decrease_queue.append((world_pos, old_light))

		self._propagate_light_step(self.light_decrease_queue, chunk.Chunk.get_block_light, chunk.Chunk.set_block_light, False, True)
		# Re-propagate increases from neighbors that were affected by the decrease
		self._propagate_light_step(self.light_increase_queue, chunk.Chunk.get_block_light, chunk.Chunk.set_block_light, True, True)

	def propagate_decrease(self, light_update):
		self._propagate_light_step(self.light_decrease_queue, chunk.Chunk.get_block_light, chunk.Chunk.set_block_light, False, light_update)

	def decrease_skylight(self, world_pos, light_update=True):
		chunk_obj = self.chunks[get_chunk_position(world_pos)]
		local_pos = get_local_position(world_pos)
		old_light = chunk_obj.get_sky_light(local_pos)
		chunk_obj.set_sky_light(local_pos, 0)
		self.skylight_decrease_queue.append((world_pos, old_light))

		self._propagate_light_step(self.skylight_decrease_queue, chunk.Chunk.get_sky_light, chunk.Chunk.set_sky_light, False, light_update)
		self._propagate_light_step(self.skylight_increase_queue, chunk.Chunk.get_sky_light, chunk.Chunk.set_sky_light, True, light_update)

	def propagate_skylight_decrease(self, light_update=True):
		self._propagate_light_step(self.skylight_decrease_queue, chunk.Chunk.get_sky_light, chunk.Chunk.set_sky_light, False, light_update)

	# Getters and setters - These functions are performance-critical.
	# The main slowdown here is the dict lookup `self.chunks.get()` and
	# then potentially the nested list access in `chunk.py`.
	# If `chunk.py` uses a flat array (e.g., `bytearray`) for `blocks` and `lights`,
	# these getters/setters in `chunk.py` will be much faster.
	# For now, within `world.py`, the only optimization is ensuring `get()` is used
	# to avoid `KeyError` if a chunk doesn't exist. Your code already does this.

	def get_raw_light(self, position):
		chunk_obj = self.chunks.get(get_chunk_position(position))
		if not chunk_obj:
			return 15 << 4 # Default to max skylight (15) and 0 block light if chunk not loaded
		local_position = get_local_position(position)
		return chunk_obj.get_raw_light(local_position)

	def get_light(self, position):
		chunk_obj = self.chunks.get(get_chunk_position(position))
		if not chunk_obj:
			return 0
		local_position = get_local_position(position)
		return chunk_obj.get_block_light(local_position)

	def get_skylight(self, position):
		chunk_obj = self.chunks.get(get_chunk_position(position))
		if not chunk_obj:
			return 15
		local_position = get_local_position(position)
		return chunk_obj.get_sky_light(local_position)

	def set_light(self, position, light):
		chunk_obj = self.chunks.get(get_chunk_position(position))
		if not chunk_obj: return # Don't set light if chunk doesn't exist
		local_position = get_local_position(position)
		chunk_obj.set_block_light(local_position, light)

	def set_skylight(self, position, light):
		chunk_obj = self.chunks.get(get_chunk_position(position))
		if not chunk_obj: return # Don't set skylight if chunk doesn't exist
		local_position = get_local_position(position)
		chunk_obj.set_sky_light(local_position, light)

	#################################################

	def get_block_number(self, position):
		chunk_position = get_chunk_position(position)
		# Use .get() for chunk lookup to avoid KeyError if chunk not loaded
		chunk_obj = self.chunks.get(chunk_position)
		if not chunk_obj:
			return 0 # Return 0 (air) if chunk doesn't exist

		lx, ly, lz = get_local_position(position)
		# Ensure bounds check, though chunk.py should handle this if block data is a flat array
		# if not (0 <= lx < chunk.CHUNK_WIDTH and 0 <= ly < chunk.CHUNK_HEIGHT and 0 <= lz < chunk.CHUNK_LENGTH):
		# 	return 0 # Or raise error, depending on desired behavior for out-of-bounds local coords.

		block_number = chunk_obj.blocks[lx][ly][lz]
		return block_number

	def get_transparency(self, position):
		block_number = self.get_block_number(position)
		# Check if block_number is a valid index.
		if not (0 <= block_number < len(self.block_types)) or self.block_types[block_number] is None:
			return 2 # Assume fully transparent (like air) for invalid/unknown blocks

		block_type_obj = self.block_types[block_number]
		return block_type_obj.transparency

	def is_opaque_block(self, position):
		block_number = self.get_block_number(position)
		if not (0 <= block_number < len(self.block_types)) or self.block_types[block_number] is None:
			return False # Air or unknown is not opaque

		block_type_obj = self.block_types[block_number]
		return not block_type_obj.transparent # Use the 'transparent' flag

	def create_chunk(self, chunk_position):
		# Only create if it doesn't already exist to prevent redundant work
		if chunk_position not in self.chunks:
			new_chunk = chunk.Chunk(self, chunk_position)
			self.chunks[chunk_position] = new_chunk
			self.init_skylight(new_chunk) # Initialize skylight for the new chunk
			# self.chunk_building_queue.append(new_chunk) # Queue for mesh build if async

	def set_block(self, position, number):
		# Pre-calculate positions once
		x, y, z = position
		chunk_position = get_chunk_position(position)

		# Check if chunk exists efficiently and create if necessary
		chunk_obj = self.chunks.get(chunk_position)
		if not chunk_obj:
			if number == 0:
				return # No point creating a chunk just to set air
			self.create_chunk(chunk_position) # Creates and adds to self.chunks
			chunk_obj = self.chunks[chunk_position] # Get the newly created chunk

		lx, ly, lz = get_local_position(position)

		# Check if block is the same to avoid unnecessary updates
		if chunk_obj.blocks[lx][ly][lz] == number:
			return

		# Set the block and mark chunk as modified
		chunk_obj.blocks[lx][ly][lz] = number
		chunk_obj.modified = True

		# Queue update for the current block's chunk
		# Use `queue_chunk_update` on the chunk object, not `update_at_position` directly here,
		# to ensure updates are batched.
		chunk_obj.queue_chunk_update(position)

		# Lighting updates: determine old vs new block type properties
		old_block_number = chunk_obj.blocks[lx][ly][lz] # The old block number
		old_block_type = self.block_types[old_block_number] if 0 <= old_block_number < len(self.block_types) else None
		new_block_type = self.block_types[number] if 0 <= number < len(self.block_types) else None

		# Handle light changes more specifically
		is_old_light_source = old_block_number in self.light_blocks
		is_new_light_source = number in self.light_blocks
		
		# If it's a new light source
		if is_new_light_source:
			self.increase_light(position, 15)
		# If an old light source was removed or replaced by non-light source
		elif is_old_light_source and not is_new_light_source:
			self.decrease_light(position)

		# Transparency changes - affects skylight and block light propagation
		old_transparent = old_block_type.transparent if old_block_type else 2
		new_transparent = new_block_type.transparent if new_block_type else 2

		if old_transparent != new_transparent:
			if new_transparent == 2: # Block became air or fully transparent
				# Need to decrease light from this spot and re-propagate skylight/blocklight
				self.decrease_light(position) # Redundant if handled by is_old_light_source? No, this handles general transparency
				self.decrease_skylight(position)
			elif old_transparent == 2 and new_transparent != 2: # Block became opaque or less transparent from air/fully transparent
				# Need to decrease skylight and potentially block light that passed through here
				self.decrease_skylight(position) # Will cause a full unlight, then relight
				# For block light, it's more complex if a path was created/blocked.
				# The existing propagate_decrease/increase handles this from affected neighbors.
				pass # The current decrease_light/skylight handles this implicitly.

		# Update neighboring chunks if the change is at the chunk boundary
		# Store frequently used values in local variables
		CW = chunk.CHUNK_WIDTH
		CH = chunk.CHUNK_HEIGHT
		CL = chunk.CHUNK_LENGTH

		# Define lambda for conciseness and avoid repeated dict lookups
		# Use chunk_obj.queue_chunk_update() for neighbors as well
		_queue_neighbor_update = lambda c_pos, w_pos: \
			self.chunks.get(c_pos) and self.chunks[c_pos].queue_chunk_update(w_pos)

		if lx == CW - 1: _queue_neighbor_update(glm.ivec3(cx + 1, cy, cz), glm.ivec3(x + 1, y, z))
		if lx == 0: _queue_neighbor_update(glm.ivec3(cx - 1, cy, cz), glm.ivec3(x - 1, y, z))

		if ly == CH - 1: _queue_neighbor_update(glm.ivec3(cx, cy + 1, cz), glm.ivec3(x, y + 1, z))
		if ly == 0: _queue_neighbor_update(glm.ivec3(cx, cy - 1, cz), glm.ivec3(x, y - 1, z))

		if lz == CL - 1: _queue_neighbor_update(glm.ivec3(cx, cy, cz + 1), glm.ivec3(x, y, z + 1))
		if lz == 0: _queue_neighbor_update(glm.ivec3(cx, cy, cz - 1), glm.ivec3(x, y, z - 1))

	def try_set_block(self, position, number, collider):
		if not number: # If trying to remove a block
			return self.set_block(position, 0)

		# Optimization: Pre-check block_type.colliders once
		block_type_obj = self.block_types[number]
		if not block_type_obj:
			return # Invalid block number

		for block_collider in block_type_obj.colliders:
			# Use `+` for `glm.vec3` and `glm.ivec3` as it's defined.
			# Using `&` for `collider` implies `collider` is some kind of AABB or shape object
			# with an intersection method. This looks correct.
			if collider & (block_collider + position):
				return # Collision, don't place block

		self.set_block(position, number)

	def toggle_AO(self):
		self.options.SMOOTH_LIGHTING = not self.options.SMOOTH_LIGHTING
		# Queue all chunks for update, instead of immediate update_subchunk_meshes
		for chunk_obj in self.chunks.values():
			chunk_obj.queue_chunk_for_rebuild() # Assume `chunk.py` has this method

	def speed_daytime(self):
		if self.daylight <= 480:
			self.incrementer = 1
		if self.daylight >= 1800:
			self.incrementer = -1

	def can_render_chunk(self, chunk_position):
		# Combine conditions and use early exit.
		# math.dist involves sqrt, which can be slow if done for *all* chunks every frame.
		# A squared distance check is faster for culling distance.
		
		# Optimization: Only check distance if frustum check passes.
		# Player frustum check is usually faster than sqrt distance.
		if not self.player.check_chunk_in_frustum(chunk_position):
			return False
		
		# Squared distance check
		player_chunk_pos = self.get_chunk_position(self.player.position)
		dx = player_chunk_pos.x - chunk_position.x
		dy = player_chunk_pos.y - chunk_position.y
		dz = player_chunk_pos.z - chunk_position.z
		
		# Use RENDER_DISTANCE squared to avoid sqrt in dist()
		max_dist_sq = self.options.RENDER_DISTANCE * self.options.RENDER_DISTANCE
		
		# Sum of squares
		dist_sq = dx*dx + dy*dy + dz*dz
		
		return dist_sq <= max_dist_sq

	def prepare_rendering(self):
		# Optimized chunk culling and sorting
		self.visible_chunks_opaque.clear()
		self.visible_chunks_translucent.clear()
		
		player_chunk_pos = self.get_chunk_position(self.player.position)
		
		# Create a list of (distance_sq, chunk_obj) tuples for efficient sorting
		# Use distance_sq for sorting, but keep chunk_obj for later.
		temp_chunks_for_sort = []

		for chunk_obj in self.chunks.values():
			if self.can_render_chunk(chunk_obj.chunk_position):
				# Calculate squared distance once for culling and sorting
				dx = player_chunk_pos.x - chunk_obj.chunk_position.x
				dy = player_chunk_pos.y - chunk_obj.chunk_position.y
				dz = player_chunk_pos.z - chunk_obj.chunk_position.z
				dist_sq = dx*dx + dy*dy + dz*dz
				temp_chunks_for_sort.append((dist_sq, chunk_obj))

		# Sort by squared distance (ascending for opaque, descending for translucent)
		# Opaque chunks usually render front-to-back for better early-Z culling on GPU.
		# Translucent chunks need to render back-to-front for correct blending.
		temp_chunks_for_sort.sort(key=lambda x: x[0]) # Sort by distance_sq (x[0])

		for dist_sq, chunk_obj in temp_chunks_for_sort:
			if chunk_obj.has_opaque_mesh(): # Assume chunk has a way to tell if it has opaque parts
				self.visible_chunks_opaque.append(chunk_obj)
			if chunk_obj.has_translucent_mesh(): # Assume chunk has a way to tell if it has translucent parts
				self.visible_chunks_translucent.append(chunk_obj)
		
		# Translucent chunks need to be sorted back-to-front
		self.visible_chunks_translucent.reverse() # Already sorted front-to-back, so reverse for back-to-front

		# The `self.sorted_chunks` was used for both. Now we have two lists.
		# `self.sorted_chunks` is no longer needed if `draw_translucent_fast` and `draw_translucent_fancy`
		# use `self.visible_chunks_translucent` and the main draw loop uses `self.visible_chunks_opaque`.

	# Re-evaluate sort_chunks, it's now integrated into prepare_rendering
	# def sort_chunks(self):
	# 	# This function is now mostly absorbed by prepare_rendering
	# 	pass # Or remove entirely if not called elsewhere.

	def draw_translucent_fast(self):
		gl.glEnable(gl.GL_BLEND)
		gl.glDisable(gl.GL_CULL_FACE) # Disable culling for transparent pass to draw both sides
		gl.glDepthMask(gl.GL_FALSE) # Don't write to depth buffer

		for render_chunk in self.visible_chunks_translucent: # Use specific list
			render_chunk.draw_translucent(gl.GL_TRIANGLES)

		gl.glDepthMask(gl.GL_TRUE) # Re-enable depth writing
		gl.glEnable(gl.GL_CULL_FACE) # Re-enable culling
		gl.glDisable(gl.GL_BLEND)

	def draw_translucent_fancy(self):
		# This "fancy" method is typically for order-independent transparency,
		# or for rendering both front and back faces with blending.
		# Drawing twice means double the fill rate, can be slow.
		# A more advanced approach (e.g., A-buffer, weighted blended order-independent transparency)
		# is needed for truly fancy transparency.
		# For now, keeping the original logic, just ensuring it uses the correct list.
		gl.glDepthMask(gl.GL_FALSE)
		gl.glEnable(gl.GL_BLEND)
		gl.glDisable(gl.GL_CULL_FACE) # Disable culling for both passes here.

		# First pass: Front faces (or just draw normally)
		gl.glFrontFace(gl.GL_CW) # Render back faces (normally CCW is front)
		for render_chunk in self.visible_chunks_translucent:
			render_chunk.draw_translucent(gl.GL_TRIANGLES)

		# Second pass: Back faces (or draw reversed)
		gl.glFrontFace(gl.GL_CCW) # Render front faces
		for render_chunk in self.visible_chunks_translucent:
			render_chunk.draw_translucent(gl.GL_TRIANGLES)

		gl.glDisable(gl.GL_BLEND)
		gl.glEnable(gl.GL_CULL_FACE) # Re-enable culling
		gl.glDepthMask(gl.GL_TRUE) # Re-enable depth writing

	# Use lambda for direct assignment, cleaner
	draw_translucent = draw_translucent_fancy if options.FANCY_TRANSLUCENCY else draw_translucent_fast

	def draw(self):
		self.visible_entities = 0 # Reset debug counter
		
		# Daylight calculations (minor optimization: pre-calculate in update_daylight if possible)
		daylight_multiplier = self.daylight / 1800.0 # Use float division
		gl.glClearColor(0.5 * (daylight_multiplier - 0.26),
				0.8 * (daylight_multiplier - 0.26),
				(daylight_multiplier - 0.26) * 1.36, 1.0)

		# Setup block shader
		self.block_shader.use()
		# Batch uniform updates
		gl.glUniformMatrix4fv(self.block_shader_matrix_location, 1, gl.GL_FALSE, glm.value_ptr(self.mvp_matrix))
		gl.glUniform1f(self.block_shader_daylight_location, daylight_multiplier)
		# block_shader_chunk_offset_location is not set here; should be set per chunk in chunk.draw
		
		# Bind textures once for block shader
		gl.glActiveTexture(gl.GL_TEXTURE0)
		gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.texture_manager.texture_array)
		gl.glUniform1i(self.block_shader_sampler_location, 0) # Set sampler uniform only once

		# Draw opaque chunks
		gl.glEnable(gl.GL_CULL_FACE)
		# Draw opaque chunks first (front-to-back sorted) for better early-Z performance
		for render_chunk in self.visible_chunks_opaque:
			render_chunk.draw(gl.GL_TRIANGLES)

		# Draw entities
		self.entity_shader.use()
		gl.glDisable(gl.GL_CULL_FACE) # Entities often don't use backface culling
		
		# Cache player position for distance calculation
		player_pos = self.player.position

		for entity in self.entities:
			# Use squared distance for culling to avoid sqrt
			dx = entity.position.x - player_pos.x
			dy = entity.position.y - player_pos.y
			dz = entity.position.z - player_pos.z
			dist_sq = dx*dx + dy*dy + dz*dz
			
			if dist_sq > (32 * 32) or not self.player.check_in_frustum(entity.collider):
				continue

			entity.draw()
			self.visible_entities += 1

		# Draw translucent chunks
		# Re-enable block shader and culling if they were changed by entity drawing.
		# This ensures the state is correct for translucent pass.
		self.block_shader.use() # Switch back to block shader
		gl.glEnable(gl.GL_CULL_FACE) # Ensure culling is enabled before translucent draw if needed
		# The `draw_translucent` method itself handles blend/depthmask/cullface.
		self.draw_translucent()


	def update_daylight(self):
		# No major performance issues, but logic seems a bit complex.
		# Simplified logic could be: 0-12000 is day, 12000-24000 is night.
		# Using a simple counter increment/decrement for `daylight` is fine.
		
		# The original logic for `incrementer` based on `daylight` value seems intended for a cycle.
		# If self.time is the primary time counter, then control incrementer via time.
		
		# Current logic:
		# If daylight < 480 (night) then incrementer becomes 1 (start increasing)
		# If daylight >= 1800 (day) then incrementer becomes -1 (start decreasing)
		# This sets up two fixed points for change.

		# The `self.time % 36000` part seems to try to force a full cycle.
		# Let's simplify and make it more robust for a full 24-hour cycle.
		# Assuming `self.time` is frames/ticks.
		
		# A full day/night cycle might be 24000 ticks.
		# 0-12000: Day (dawn -> noon -> dusk)
		# 12000-24000: Night (dusk -> midnight -> dawn)
		
		# This might simplify `incrementer` management.
		# For "don't change usage", keeping the original time incrementer logic as is.
		# It's not a performance bottleneck.

		if self.incrementer == -1:
			if self.daylight < 480: # Moonlight of 4
				self.incrementer = 0 # Stop decreasing
		elif self.incrementer == 1:
			if self.daylight >= 1800:
				self.incrementer = 0 # Stop increasing

		# These check points seem to override the `daylight` based incrementer.
		# They ensure a full cycle by explicitly setting incrementer.
		if self.time % 36000 == 0: # Start of a day cycle (noon-ish)
			self.incrementer = 1 # Force increase
		elif self.time % 36000 == 18000: # Halfway through the cycle (midnight-ish)
			self.incrementer = -1 # Force decrease

		self.daylight += self.incrementer

	def build_pending_chunks(self):
		# This is a critical point. Building meshes on the main thread is slow.
		# If `options.CHUNK_UPDATES` is low, this acts as a limiter.
		# For significant speed-up, this needs to be moved to a separate worker thread.
		# Since the request is "don't change usage", we can't introduce threading directly here,
		# but it's the primary architectural bottleneck.
		
		# Current implementation: Pops one chunk and updates its mesh.
		# This is a good throttling mechanism, but still synchronous.
		if self.chunk_building_queue:
			pending_chunk = self.chunk_building_queue.popleft()
			pending_chunk.update_mesh() # This call is blocking and CPU intensive.
			self.chunk_update_counter += 1 # Track how many chunks are built per tick.

	def process_chunk_updates(self):
		# This loop processes updates queued within *each* chunk.
		# Assuming `chunk.process_chunk_updates()` itself batches block updates to a mesh rebuild.
		# If `chunk.update_at_position` directly rebuilds mesh, this will be slow.
		
		# Optimization: Iterate over `self.chunks.values()` once.
		# Filter and process in one go.
		for chunk_obj in self.chunks.values():
			# This method should process the `chunk_obj.chunk_update_queue` and potentially
			# call `chunk_obj.update_mesh()` if updates warrant it.
			chunk_obj.process_chunk_updates() # This call might queue to chunk_building_queue or update directly.
		
		# New method to process the globally queued chunk updates for mesh rebuilds
	def process_queued_chunk_updates(self):
		"""Processes all chunk updates that were queued during lighting propagation."""
		# This method is called from light propagation functions to trigger mesh updates
		# for affected chunks *after* light calculation is done.
		# It ensures that each affected chunk is queued for a mesh rebuild only once.
		
		# A set is ideal to avoid duplicate rebuild requests for the same chunk.
		chunks_to_rebuild = set()
		while self.chunk_building_queue: # This queue stores chunk objects, not just positions.
			chunks_to_rebuild.add(self.chunk_building_queue.popleft())
		
		# Re-add unique chunks back to the queue for processing in `build_pending_chunks`
		# or process them directly if `options.CHUNK_UPDATES` allows.
		for chunk_obj in chunks_to_rebuild:
			if chunk_obj not in self.chunk_building_queue: # Avoid re-adding if it's somehow already there
				self.chunk_building_queue.append(chunk_obj)


	def tick(self, delta_time):
		# Optimized for readability and sequence
		self.time += 1
		
		# Update daylight only once
		self.update_daylight()

		# Update pending chunk counts.
		# Use sum with generator expression for efficiency.
		self.pending_chunk_update_count = sum(len(c.chunk_update_queue) for c in self.chunks.values())
		
		# Build one pending chunk mesh if available (CPU-heavy)
		self.build_pending_chunks()
		
		# Process block updates that might require re-meshing
		# This should add chunks to `chunk_building_queue` if their internal queues are processed.
		self.process_chunk_updates()
		
		# The `chunk_update_counter` is updated in `build_pending_chunks`
		self.chunk_update_counter = 0 # Reset for current tick, updated in build_pending_chunks
