Y = 0
Cb = 133
Cr = 77

a = 0.2627
b = 0.6780
c = 0.0593
d = 1.8814
e = 1.4747

R = Y + e * Cr
G = Y - (a * e / b) * Cr - (c * d / b) * Cb
B = Y + d * Cb

print(R)
print(G)
print(B)