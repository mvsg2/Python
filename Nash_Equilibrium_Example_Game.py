#num = 1400
import matplotlib.pyplot as plt

nums = []
nash_eq = []

for num in range(100, 500, 13):   # Try playing around with changing these initial, final points, and the step gap

	print("The initial number is:", num) 
	nums.append(num)

	while num >= 1:
	
		num *= 2
		num /= 3
	
	nash_eq.append(num)
	print("The Nash Equilibrium floating value is:", num)
	print()

plt.figure(figsize=(50,50))
plt.plot(nums, nash_eq)
plt.plot(nums, nash_eq, 'ro', markersize=4.7)
plt.title('Nash Equilibrium final floating point value for different initial seed values')
plt.show()

print("The final floating point value interestingly seems to be always greater than 0.5 (closer to 1) ...")