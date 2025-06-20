not_allowed_tanks = [4, 5]
allowed_combs = [(3, 5), (1, 2), (4, 5, 3), (2,3,1)]

if len(not_allowed_tanks) > 0:
    allowed_combs = [t for t in allowed_combs if not any(i in not_allowed_tanks for i in t)]

print(allowed_combs)