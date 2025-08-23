def is_even_or_odd(number):
  """Checks if a number is even or odd.

  Args:
    number: An integer.

  Returns:
    "Even" if the number is even, "Odd" if the number is odd.
  """
  if number % 2 == 0:
    return "Even"
  else:
    return "Odd"

# Get input from the user
try:
  num = int(input("Enter an integer: "))
  result = is_even_or_odd(num)
  print(f"The number {num} is {result}.")
except ValueError:
  print("Invalid input. Please enter an integer.")