from cartpole import CartPole

# visual = True turns on animation (don ’t use this in other sections !)

example_system = CartPole(visual = True)
cart_position = 0.0
cart_velocity = 0.1
pole_angle = 0.01
pole_velocity = 0.0
state = [cart_position , cart_velocity , pole_angle , pole_velocity]

example_system.setState(state)

for _ in range(100):
    example_system.performAction()
