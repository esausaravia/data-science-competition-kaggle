import h3

# Example coordinates
latitude = 37.7749
longitude = -122.4194
resolution = 9  # H3 resolution level (higher means smaller hexagons)

# Convert lat/lon to an H3 index
h3_index = h3.latlng_to_cell(latitude, longitude, resolution)

print(f"H3 Index: {type(h3_index)} {h3_index}")


myarraytest = []

for i in range(5):
    myarraytest.append(i)

print(f"len: {len(myarraytest)}" )