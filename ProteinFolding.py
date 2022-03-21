import numpy as np
import random
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from datetime import datetime

random.seed(datetime.now())
'''
This code primarily follows standards set by the following papers
	[1] https://link.aps.org/doi/10.1103/PhysRevE.48.1469

	[2]	"Effects of Confinement on the Thermodynamics of a Collapsing
	Heteropolymer: An Off-Lattice Wang-Landau Monte Carlo Simulation
	Study"
'''

class protein:
	"""
		docstring for protein
		The protein class holds all information and fuctions
		regarding protein folding with the AB off lattice protein
		folding model. The function takes a sequence of peptides
		and attempts to calculate the lowest energy state with a
		monte carlo algorithm.

		(string) sequence - monomer code where A represents hyrophylic and B hydrophobic [2] section IIA
		(float) bond_length - distance between monomers
		(float) min_energy - lowest allowed energy for any configuration
		(float) mix_energy - greatest allowed energy for any configuration
		(float) delta_energy - energy step size
		(int) option:  
			0 to see the animation
			1 see each step in sequence
			2 to for no graph output
			
	"""
	def __init__(self, sequence="ABBA",bond_length=1,min_energy=0,max_energy=100,delta_energy=0.1,option=1):
		super(protein, self).__init__()
		self.output = 1
		if option == 0:
			self.show_animation = 0
		elif option == 1:
			self.show_animation = 1
		elif option == 2:
			self.output = 0
		else:
			print("{} is not a valid option".format(option))
			print("Now exiting")
			exit(-1)

		self.bond_length = bond_length
		self.min_energy = min_energy
		self.max_energy = max_energy
		self.delta_energy = delta_energy
		self.num_visited = np.zeros(int(np.ceil((self.max_energy-self.min_energy)/self.delta_energy)))
		#state_density is stored as ln(state density) for some reason
		#but sticking with it because thats what [2] does
		self.state_density = np.zeros_like(self.num_visited)
		self.sequence = sequence
		self.binary_sequence = np.where(np.array(list(self.sequence))=='A',1,0)# self.binary_sequence_from_sequence(self.sequence)
		self.sequence_size = self.binary_sequence.size
		self.angles = np.zeros(self.sequence_size-2) #this obviously shouldnt be random for actual folding
		# self.angles = np.pi * np.random.rand(self.sequence_size-2) #this obviously shouldnt be random for actual folding
		self.Cij = self.make_Cij(np.where(self.binary_sequence == 0,-1,1)) 
		self.x_pos = []
		self.y_pos = []
		#scale factor from [2]
		self.scale_factor = 1.064

		#Set markers for graph since this only needs to happen once
		self.A_markers = np.arange(self.binary_sequence.size)
		self.A_markers = np.where(self.binary_sequence==1,self.A_markers,-1)
		self.A_markers = self.A_markers[self.A_markers!=-1].tolist()
		
		self.B_markers = np.arange(self.binary_sequence.size)
		self.B_markers = np.where(self.binary_sequence==0,self.B_markers,-1)
		self.B_markers = self.B_markers[self.B_markers!=-1].tolist()
		

	def energy_to_index(self,energy):
		shifted_energy = energy - self.min_energy
		index = int(np.floor(shifted_energy/self.delta_energy))
		return index

	def print_protein(self):
		x_axis_name = 'x position'
		y_axis_name = 'y position'

		print(self.A_markers)
		print(self.B_markers)


		plt.xlabel(x_axis_name)
		plt.ylabel(y_axis_name)
		plt.xlim([-9,9])
		plt.ylim([-9,9])
		plt.title('AB off lattice {}-mer, Total energy: {}'.format(self.sequence_size,self.get_total_energy()))
		plt.plot(self.x_pos,self.y_pos,marker='o',markevery=self.A_markers,label="A-mers")
		plt.plot(self.x_pos,self.y_pos,marker='*',markevery=self.B_markers,label="B-mers")
		# plt.legend(["A-mers","B-mers"])
		plt.show()

	def get_positions(self,angles,start_x=0,start_y=0,start_dir="E"):
		x_pos = np.zeros(self.sequence_size)
		y_pos = np.zeros(self.sequence_size)
		x_pos[0] = start_x
		y_pos[0] = start_y

		if start_dir == "N":
			x_pos[1] += 0
			y_pos[1] += 1
		elif start_dir == "E":
			x_pos[1] += 1
			y_pos[1] += 0
		elif start_dir == "S":
			x_pos[1] += 0
			y_pos[1] += -1
		elif start_dir == "W":
			x_pos[1] += -1
			y_pos[1] += 0
		else:
			print('Please enter start_dir as "N", "S", "E", or "W" for cardinal directions')
			return -1

		theta = 0
		for i in range(2,self.sequence_size):
			theta += angles[i-2]
			x_pos[i] = x_pos[i-1] + np.cos(theta)
			y_pos[i] = y_pos[i-1] + np.sin(theta)

		self.x_pos = x_pos
		self.y_pos = y_pos

	def distance(self,pep_index1,pep_index2):
		return np.sqrt(np.power((self.x_pos[pep_index1]-self.x_pos[pep_index2]),2) + np.power((self.y_pos[pep_index1]-self.y_pos[pep_index2]),2))

	def make_Cij(self,binary_sequence):
		Cij = np.zeros((binary_sequence.size,binary_sequence.size))
		for i in range(binary_sequence.size):
			for j in range(binary_sequence.size):
				Cij[i][j] = 0.125*(1 + binary_sequence[i] + binary_sequence[j] + 5*binary_sequence[i]*binary_sequence[j])
		return Cij

	# def binary_sequence_from_sequence(self,sequence):
	# 	return_me = np.ones(len(sequence))
	# 	for index,letter in enumerate(sequence):
	# 		if letter == "B":
	# 			return_me[index] = -1
	# 	return return_me

	#from [2], histogram is flat if all spots have 20 hits
	def check_flat(self):
		over_twenty = np.where(self.num_visited>=20,1,0)
		if np.sum(over_twenty) == over_twenty.size:
			self.scle_factor = np.sqrt(self.scale_factor)
			self.num_visited = np.zeros_like(self.num_visited)
			return True
		else:
			return False


	#As defined in [2], equation (6)
	def accept_move(self,old_en_index,new_en_index):
		return random.random() < min(self.state_density[old_en_index]/self.state_density[new_en_index],1)
	
	def fold(self,min_simulation_factor=10**-8):
		old_energy = self.get_total_energy()
		while self.scale_factor > min_simulation_factor:
			old_energy = self.simple_hinge(old_energy)
				


	#for long energies this would need to change to try and get change in E instead of total E
	def get_total_energy(self): ###Use x and y positions to get disntace
		v1 = np.sum(0.25*(1-np.cos(self.angles))) #equation ___ in [1]
		 
		v2 = 0
		self.get_positions(self.angles)
		for i in range(self.sequence_size-2):
			for j in range(i+2,self.sequence_size):
				rij = self.distance(i,j) 
				v2 += 4*(np.power(rij,-12) - self.Cij[i][j]*np.power(rij,-6))

		return v1 + v2

	#This method of movement is from: 
	#"Effects of Confinement on the Thermodynamics of a Collapsing
	# Heteropolymer: An Off-Lattice Wang-Landau Monte Carlo Simulation
	# Study"
	def simple_hinge(self,old_energy):
		#a_index is random monomer index
		#b_index is ramdomly one higher or one lower index as a_index
		a_index = random.randrange(0,self.sequence_size)
		b_index = random.randrange(0,2)

		if a_index == 0: 
			b_index = 1
		elif a_index == self.sequence_size-1:
			b_index = a_index - 1
		elif b_index == 0:
			b_index = a_index - 1
		elif b_index == 1:
			b_index = a_index + 1

		if a_index == 1 and b_index == 0:
			a_index = 0
			b_index = 1
		elif a_index == self.sequence_size-2 and b_index == self.sequence_size-1:
			a_index = self.sequence_size-1
			b_index = self.sequence_size-2

		shift_angle = random.uniform(-np.pi,np.pi)

		self.angles[b_index-1] += shift_angle
		new_energy = self.get_total_energy()
		# print("old_energy: {},\tnew_energy: {},\tshift_angle: {}".format(old_energy,new_energy,shift_angle))
		if new_energy <= self.max_energy and new_energy >= self.min_energy:
			index = self.energy_to_index(new_energy)
			self.num_visited[index] += 1
			self.state_density[index] += np.log(self.scale_factor)
			self.check_flat()
			if self.accept_move(self.energy_to_index(old_energy),self.energy_to_index(new_energy)):
				print("Move accepted")
				if self.output:
					# not show_animation means show steps
					if not self.show_animation:
						self.print_protein()
				old_energy = new_energy
			else:
				self.angles[b_index-1] -= shift_angle
				print("Move rejected")
		else:
			print("Move outside energy range")
			self.angles[b_index-1] -= shift_angle
		
		self.get_positions(self.angles)
		return old_energy


#find out how long sequence will be and then fill in values for more efficent code
#don't use strings
def fibonacci_sequence(index):
	if index == 0:
		return "A"
	elif index == 1:
		return "B"
	else:
		fmin2 = "A"
		fmin1 = "B"
		f = ""
		i = 0
		while i < index-1:
			f = fmin2 + fmin1
			fmin2 = fmin1
			fmin1 = f
			i += 1
		return f

def main():
	arg = dict()

	# print(fibonacci_sequence(45))
	p = protein(sequence="ABBA",min_energy=0,max_energy=2,delta_energy=0.05,option=0)
	p.fold(min_simulation_factor=10**-3)

if __name__ == '__main__':
	main()


