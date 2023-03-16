def isDivisible(m, n):
	while(m < n):
		print("Oh no... This is a proper fraction :|")
		break
	if (m % n == 0):
		print(f"Yay, {m} is divisible by {n}! :)")
	else:
		print(f"Hmm... {m} is not divisible by {n} :(")

if __name__ == "__main__":
	a = int(input("Enter your dividend: "))
	b = int(input("Enter your divisor: "))
	isDivisible(a, b)
	