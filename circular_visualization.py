for factor in range(5):
    temp_arr = []
    for category in range(19):
        temp_arr.append(result[category][factor])
    for category in range(19):
        result[category][factor] = (result[category][factor] - min(temp_arr)) /
            (max(temp_arr) - min(temp_arr))
print(result)
