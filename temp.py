M1 = [0.5,0.54,0.7,0.46,0.48,0.48,0.46,0.6,0.56]
M2 = [0.6,0.46,0.725,0.54,0.52,0.68,0.56,0.62,0.57]
M3 = [0.1,0.2,0.25,0.2,0.12,0.24,0.16,0.17,0.17]
M4 = [0.2,0.16,0.25,0.18,0.1,0.26,0.16,0.19,0.17]
sameple_num = [10,40,50,50,50,50,50,100,100,100]

M1_avg = 0
M2_avg = 0
M3_avg = 0
M4_avg = 0

for sample_idx in range(len(M1)):
    M1_avg += M1[sample_idx]*sameple_num[sample_idx]
    M2_avg += M2[sample_idx]*sameple_num[sample_idx]
    M3_avg += M3[sample_idx]*sameple_num[sample_idx]
    M4_avg += M4[sample_idx]*sameple_num[sample_idx]

M1_avg /= sum(sameple_num)
M2_avg /= sum(sameple_num)
M3_avg /= sum(sameple_num)
M4_avg /= sum(sameple_num)

print(f"M1_avg: {M1_avg}")
print(f"M2_avg: {M2_avg}")
print(f"M3_avg: {M3_avg}")
print(f"M4_avg: {M4_avg}")