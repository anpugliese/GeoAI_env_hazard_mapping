import math

# inspiration and formula from here
# https://stackoverflow.com/questions/1158909/average-of-two-angles-with-wrap-around
def degree_average(degrees):
    sin_sum = 0
    cos_sum = 0
    if len(degrees) == 1:  
        for deg in degrees: #"iterate" over the element and return it
            return deg

    for deg in degrees:
        if type(deg) == str:
            deg = float(deg)
            
        #decompose the angles
        deg_radians = math.radians(deg)
        sin_sum += math.sin(deg_radians)
        cos_sum += math.cos(deg_radians)    

    #use atan2 instead of atan
    avg_radians = math.atan2(sin_sum, cos_sum)
    avg_deg = math.degrees(avg_radians)
    avg_deg = (avg_deg + 360) % 360 # to convert the angle between  0 - 360
    return avg_deg

'''
Function to determine the section of the wind given the degree in which it is coming
if 8 sectors:
        North 337.5 - 22.5 -> sector 1
        NE 22.5 - 67.5 -> sector 2
        East 67.5 - 112.5 -> sector 3
        SE 112.5 - 157.5 -> sector 4
        South 157.5 - 202.5 -> sector 5
        SW 202.5 - 247.5 -> sector 6
        West 247.5 - 292.5 -> sector 7
        NW 292.5 - 337.5 -> sector 8
'''
def wind_sectors(degree):
    if degree >= 337.5 or degree < 22.5:
        return 1
    elif degree >= 22.5 and degree < 67.5:
        return 2
    elif degree >= 67.5 and degree < 112.5:
        return 3
    elif degree >= 112.5 and degree < 157.5:
        return 4
    elif degree >= 157.5 and degree < 202.5:
        return 5
    elif degree >= 202.5 and degree < 247.5:
        return 6
    elif degree >= 247.5 and degree < 292.5:
        return 7
    elif degree >= 292.5 and degree < 337.5:
        return 8
    else:
        return 0