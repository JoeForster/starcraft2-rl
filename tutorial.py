# standard library imports
import random
import datetime
import os

# sc2 imports
import sc2
from sc2 import run_game, maps, Race, Difficulty, Result
from sc2.player import Bot, Computer
import sc2.constants as scc

# other library imports
import cv2
import numpy as np

# local imports
from examples.protoss.cannon_rush import CannonRushBot


HEADLESS = False


class AttackUnit:
	def __init__(self, attack_num, defend_num):
		self.attack_num = attack_num
		self.defend_num = defend_num


class JoeBot(sc2.BotAI):

	def __init__(self):
		self.iteration = 0
		self.ITERATIONS_PER_MINUTE = 165
		self.MAX_WORKERS = 60
		self.WORKERS_PER_NEXUS = 16
		self.do_something_after = 0
		self.train_data = []
		self.train_data_folder = "train_data"

	def on_start(self):
		self.start_datetime = datetime.datetime.utcnow()
		self.start_timestamp = self.start_datetime.isoformat()
		print("JoeBot on_start at " + self.start_timestamp)
		# Assumes nobody's made a file with this name - it'll break if so!
		if not os.path.isdir(self.train_data_folder):
			os.mkdir(self.train_data_folder)

	async def on_step(self, iteration):
		self.iteration = iteration
		await self.scout()
		await self.distribute_workers()
		await self.build_workers()
		await self.build_pylons()
		await self.build_assimilators()
		await self.expand()
		await self.offensive_force_buildings()
		await self.build_offensive_force()
		await self.intel()
		await self.attack()

	def on_end(self, my_result):
		print("JoeBot on_end got result: " + str(my_result))

		if my_result == Result.Victory:
			out_path = "train_data/{}.npy".format(self.start_timestamp)
			np.save(out_path, np.array(self.train_data))

	async def intel(self):
		game_data = np.zeros(
			(self.game_info.map_size[1], self.game_info.map_size[0], 3),
			np.uint8)

		# UNIT: [SIZE, (BGR COLOR)]
		# from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY,
		# CYBERNETICSCORE, STARGATE, VOIDRAY
		draw_units = [
			# Buildings
			(scc.NEXUS, [15, (0, 255, 0)]),
			(scc.PYLON, [3, (20, 235, 0)]),
			(scc.ASSIMILATOR, [2, (55, 200, 0)]),
			(scc.GATEWAY, [3, (200, 100, 0)]),
			(scc.CYBERNETICSCORE, [3, (150, 150, 0)]),
			(scc.STARGATE, [5, (255, 0, 0)]),
			(scc.ROBOTICSFACILITY, [5, (215, 155, 0)]),
			# Units
			(scc.PROBE, [1, (55, 200, 0)]),
			(scc.VOIDRAY, [3, (255, 100, 0)]),
			(scc.OBSERVER, [3, (255, 255, 255)]),
		]

		for unit_type, unit_vals in draw_units:
			for unit in self.units(unit_type).ready:
				pos = unit.position
				cv2.circle(
					game_data,
					(int(pos[0]), int(pos[1])),
					unit_vals[0],
					unit_vals[1],
					-1)

		main_base_names = ["nexus", "supplydepot", "hatchery"]
		for enemy_building in self.known_enemy_structures:
			pos = enemy_building.position
			if enemy_building.name.lower() not in main_base_names:
				cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
		for enemy_building in self.known_enemy_structures:
			pos = enemy_building.position
			if enemy_building.name.lower() in main_base_names:
				cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

		for enemy_unit in self.known_enemy_units:
			if not enemy_unit.is_structure:
				worker_names = [
					"probe",
					"scv",
					"drone"]
				# if that unit is a PROBE, SCV, or DRONE... it's a worker
				pos = enemy_unit.position
				if enemy_unit.name.lower() in worker_names:
					cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
				else:
					cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

		for obs in self.units(scc.OBSERVER).ready:
			pos = obs.position
			cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)

		# Draw stat bars
		line_max = 50
		mineral_ratio = self.minerals / 1500
		if mineral_ratio > 1.0:
			mineral_ratio = 1.0

		vespene_ratio = self.vespene / 1500
		if vespene_ratio > 1.0:
			vespene_ratio = 1.0

		population_ratio = self.supply_left / self.supply_cap
		if population_ratio > 1.0:
			population_ratio = 1.0

		plausible_supply = self.supply_cap / 200.0

		supply_diff = (self.supply_cap - self.supply_left)

		if supply_diff == 0:
			military_weight = 1.0
		else:
			military_weight = len(self.units(scc.VOIDRAY)) / supply_diff
		if military_weight > 1.0:
			military_weight = 1.0

		cv2.line(game_data, (0, 19), (int(line_max*military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
		cv2.line(game_data, (0, 15), (int(line_max*plausible_supply), 15), (220, 200, 200), 3)  # plausible supply (supply/200.0)
		cv2.line(game_data, (0, 11), (int(line_max*population_ratio), 11), (150, 150, 150), 3)  # population ratio (supply_left/supply)
		cv2.line(game_data, (0, 7), (int(line_max*vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
		cv2.line(game_data, (0, 3), (int(line_max*mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

		# flip horizontally to make our final fix in visual representation:
		self.flipped = cv2.flip(game_data, 0)

		if not HEADLESS:
			resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)

			cv2.imshow('Intel', resized)
			cv2.waitKey(1)

	def random_location_variance(self, enemy_start_location):
		x = enemy_start_location[0]
		y = enemy_start_location[1]

		x += ((random.randrange(-20, 20)) / 100) * enemy_start_location[0]
		y += ((random.randrange(-20, 20)) / 100) * enemy_start_location[1]

		if x < 0:
			x = 0
		if y < 0:
			y = 0
		if x > self.game_info.map_size[0]:
			x = self.game_info.map_size[0]
		if y > self.game_info.map_size[1]:
			y = self.game_info.map_size[1]

		go_to = sc2.position.Point2(sc2.position.Pointlike((x, y)))
		return go_to

	async def scout(self):
		# Send idle scouts randomly towards enemy start locations
		if self.units(scc.OBSERVER).exists:
			scout = self.units(scc.OBSERVER)[0]
			if scout.is_idle:
				enemy_location = self.enemy_start_locations[0]
				move_to = self.random_location_variance(enemy_location)
				print("Move idle scout to location: " + str(move_to))
				await self.do(scout.move(move_to))

		else:
			for rf in self.units(scc.ROBOTICSFACILITY).ready.noqueue:
				if self.can_afford(scc.OBSERVER) and self.supply_left > 0:
					await self.do(rf.train(scc.OBSERVER))

	async def build_workers(self):
		# DECISION POINT: How many workers, and how many per nexus
		num_probes = len(self.units(scc.PROBE))
		if len(self.units(scc.NEXUS)) * self.WORKERS_PER_NEXUS > num_probes:
			if num_probes < self.MAX_WORKERS:
				for nexus in self.units(scc.NEXUS).ready.noqueue:
					if self.can_afford(scc.PROBE):
						await(self.do(nexus.train(scc.PROBE)))

	async def build_pylons(self):
		# DECISION POINT: When to build a pylon
		if self.supply_left < 5 and not self.already_pending(scc.PYLON):
			nexuses = self.units(scc.NEXUS).ready
			if nexuses.exists and self.can_afford(scc.PYLON):
				# Super simple build near nexus to test for now
				# DECISION POINT: WHERE DO WE BUILD A PYLON?
				await self.build(scc.PYLON, near=nexuses.first)

	async def build_assimilators(self):
		# For each ready nexus
		for nexus in self.units(scc.NEXUS).ready:
			# For each geyser that is close to that nexus
			# (MAGIC NUMBER)
			geysers = self.state.vespene_geyser.closer_than(15.0, nexus)
			for geyser in geysers:
				# Unless we can't afford it...
				if not self.can_afford(scc.ASSIMILATOR):
					break
				# Select an appropriate worker and build a geyser if not already there
				worker = self.select_build_worker(geyser.position)
				if worker is None:
					break
				if not self.units(scc.ASSIMILATOR).closer_than(1.0, geyser).exists:
					await self.do(worker.build(scc.ASSIMILATOR, geyser))

	async def expand(self):
		# Extremely simple/naive
		# DECISION POINT: WHEN AND WHERE TO EXPAND
		max_expansions = self.iteration / self.ITERATIONS_PER_MINUTE
		if self.units(scc.NEXUS).amount < max_expansions and self.can_afford(scc.NEXUS):
			await self.expand_now()

	async def offensive_force_buildings(self):
		# DECISION POINT: When do we decide to start buildings
		if self.units(scc.PYLON).ready.exists:
			# DECISION POINT: Where to we build the buildings
			pylon = self.units(scc.PYLON).ready.random
			# GOT A GATEWAY -> then get a cybernetics core
			# DECISION POINT: How many gateways do we build and when
			if self.units(scc.GATEWAY).ready.exists:
				# DECISION POINT: When might we want extra cybernetics cores (if any)?
				if not self.units(scc.CYBERNETICSCORE).exists:
					if self.can_afford(scc.CYBERNETICSCORE):
						if not self.already_pending(scc.CYBERNETICSCORE):
							await self.build(scc.CYBERNETICSCORE, near=pylon)
			# DECISION POINT: When do we build (more) gateways?
			# DECISION POINT: Where do we build (more) gateways?
			# NOT GOT ENOUGH GATEWAYS AND CAN BUILD ONE -> then build one
			# cur_max_gateways = (self.iteration / self.ITERATIONS_PER_MINUTE) / 2
			cur_max_gateways = 1
			if len(self.units(scc.GATEWAY)) <= cur_max_gateways:
				if self.can_afford(scc.GATEWAY) and not self.already_pending(scc.GATEWAY):
					await self.build(scc.GATEWAY, near=pylon)

			# Buildings that depend upon Cybernetics 
			if self.units(scc.CYBERNETICSCORE).ready.exists:
				cur_max_robotics = 1
				if len(self.units(scc.ROBOTICSFACILITY)) < cur_max_robotics:
					if self.can_afford(scc.ROBOTICSFACILITY):
						if not self.already_pending(scc.ROBOTICSFACILITY):
							await self.build(scc.ROBOTICSFACILITY, near=pylon)

				cur_max_stargates = (self.iteration / self.ITERATIONS_PER_MINUTE)  # / 2
				if len(self.units(scc.STARGATE)) < cur_max_stargates:
					if self.can_afford(scc.STARGATE) and not self.already_pending(scc.STARGATE):
						await self.build(scc.STARGATE, near=pylon)

	async def build_offensive_force(self):
		cur_max_stalkers = self.units(scc.VOIDRAY).amount
		for gw in self.units(scc.GATEWAY).ready.noqueue:
			# DECISION POINT: When to build. What to build. At what supply level.
			# TODO more generalised "can I build this unit"? (afford+supply+tech)
			# TODO Should check whether we have the tech yet
			# (currently spams errors)
			if self.units(scc.STALKER).amount <= cur_max_stalkers:
				if self.can_afford(scc.STALKER) and self.supply_left > 0:
					if self.units(scc.CYBERNETICSCORE).ready:
						await self.do(gw.train(scc.STALKER))

		for sg in self.units(scc.STARGATE).ready.noqueue:
			if self.can_afford(scc.VOIDRAY) and self.supply_left > 0:
				await self.do(sg.train(scc.VOIDRAY))

	def find_target(self):
		# DECISION POINT: Target of attack, when to attack enemy base, etc
		if len(self.known_enemy_units) > 0:
			return random.choice(self.known_enemy_units)
		elif len(self.known_enemy_structures) > 0:
			return random.choice(self.known_enemy_structures)
		else:
			return self.enemy_start_locations[0]

	async def old_attack(self):
		# DECISION POINT: When to launch an attack
		# DECISION POINT: With what units to attack
		attack_units = {
			scc.STALKER: AttackUnit(15, 5),
			scc.VOIDRAY: AttackUnit(8, 3)}

		for unit, spec in attack_units.items():
			units = self.units(unit)
			# If enough, find attack target
			if units.amount > spec.attack_num and units.amount > spec.defend_num:
				for s in units.idle:
					await self.do(s.attack(self.find_target()))

			# If some but not enough, just attack known enemy units
			elif units.amount > spec.defend_num and len(self.known_enemy_units) > 0:
				for s in units.idle:
					await self.do(s.attack(random.choice(self.known_enemy_units)))

	async def attack(self):
		if self.units(scc.VOIDRAY).idle:
			choice = random.randrange(0, 4)
			target = False
			if self.iteration > self.do_something_after:
				if choice == 0:
					# be idle for some period so that we don't choose some action next time
					wait = random.randrange(20, 165)
					self.do_something_after = self.iteration + wait

				elif choice == 1:
					# attack_unit_closest_nexus
					if len(self.known_enemy_units) > 0:
						chosen_nexus = self.units(scc.NEXUS)
						target = self.known_enemy_units.closest_to(random.choice(chosen_nexus))

				elif choice == 2:
					# attack enemy structures
					if len(self.known_enemy_structures) > 0:
						target = random.choice(self.known_enemy_structures)

				elif choice == 3:
					# attack_enemy_start
					target = self.enemy_start_locations[0]

				if target:
					for vr in self.units(scc.VOIDRAY).idle:
						await self.do(vr.attack(target))
				y = np.zeros(4)
				y[choice] = 1
				print("attack: " + str(y))
				self.train_data.append([y, self.flipped])


def get_opponent_cannon_rush():
	return Bot(Race.Protoss, CannonRushBot())


def get_opponent_terran(difficulty=Difficulty.Hard):
	return Computer(Race.Terran, difficulty)


if __name__ == '__main__':
	bot_inst = JoeBot()
	bot_inst.on_start()
	results = run_game(maps.get("AbyssalReefLE"), [
		Bot(Race.Protoss, bot_inst),
		Computer(Race.Terran, Difficulty.Hard)
	], realtime=False)
	# run_game returns a list of form e.g.
	# [<Result.Victory: 1>, <Result.Defeat: 2>]
	# if there are multiple bots, but only one item e.g.
	# <Result.Victory: 1>
	# if just one - or None if the game didn't finish properly.
	try:
		bot_result = results[0]
	except TypeError:
		bot_result = results  # Single value or None
	bot_inst.on_end(bot_result)
