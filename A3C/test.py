'''
v_s_ : 0.0469627
gamma : 0.9
br [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
buffer_v_target : [6.1439894123998027, 5.7155437915553362, 5.2394931017281507, 4.7105478908090559, 4.1228309897878397, 3.4698122108753773, 2.7442357898615302, 1.9380397665128113, 1.0422664072364569]
'''

s__critic = 0.0469627
GAMMA = 0.9
buffer_r = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

buffer_c_target = []
buffer_c_target.append(s__critic)

for r in buffer_r[::-1]:
    buffer_c_target.append(
        r + (GAMMA * buffer_c_target[-1])
    )

buffer_c_target.reverse()
buffer_c_target.pop()

print(buffer_c_target)

