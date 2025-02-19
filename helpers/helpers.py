from datetime import timedelta



#TODO:: Establecer un rando de fechas 
def get_week_range(date):
    start_of_week = date - timedelta(days=date.weekday())
    end_of_week = start_of_week + timedelta(days=6)
    return start_of_week.strftime("%d %B"), end_of_week.strftime("%d %B")
