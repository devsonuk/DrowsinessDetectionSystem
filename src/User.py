import time


class User:
    def __init__(self, login_time, blink_count, yawn_count, name='User', gender='Male', dob='01/01/1999'):
        self.name = name
        self.gender = gender
        selef.dob = dob
        self.login_time = login_time
        self.blink_count = blink_count
        self.yawn_count = yawn_count

    def get_total_duration(self):
        current_time = time.time()
        return current_time - self.login_time

    def get_blink_rate(self):
        try:
            return self.blink_count // (self.get_total_duration() // 60)
        except:
            return self.blink_count

    def get_yawn_rate(self):
        try:
            return self.yawn_count // (self.get_total_duration() // 60)
        except:
            return self.yawn_count
