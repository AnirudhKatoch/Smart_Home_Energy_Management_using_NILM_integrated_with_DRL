import datetime


def seconds_from_new_year(year, month, day, hour=0, minute=0, second=0):
    # Create a datetime object for the start of the year
    start_of_year = datetime.datetime(year, 1, 1)

    # Create a datetime object for the given date
    given_date = datetime.datetime(year, month, day, hour, minute, second)

    # Calculate the difference between the two dates
    time_difference = given_date - start_of_year

    # Convert the difference to seconds
    seconds = time_difference.total_seconds()

    return seconds


# Example usage

year = 2024
month = 11
day = 6
hour = 20
minute = 46
second = 20

seconds_1 = (seconds_from_new_year(year, month, day, hour, minute, second))

year = 2024
month = 11
day = 7
hour = 6
minute = 28
second = 44

seconds_2 = (seconds_from_new_year(year, month, day, hour, minute, second))

Time = seconds_2 - seconds_1  + (0.115 - 0.144)

print('Time', Time)

Time_difference =  96 - Time

#print('Time_difference', Time_difference)

Actual_time = 0.9803333333333333
Required_Time = Actual_time + (Time_difference/96)

#print('Required_Time', Required_Time)


# I like this one '0.9554854166666669' with this I get 95.953 almost perfect but a little less which is exactly what we want