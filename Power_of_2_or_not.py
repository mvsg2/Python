def test(n):
 	return n>0 and (n&(n-1)==0)

def main():
	
	n = int(input("Number: "))
	if test(n):
		print("Yes, it's a power of two!")
	else:
		print("Nah, it's not a power of two")

if __name__ == "__main__":

	main()	